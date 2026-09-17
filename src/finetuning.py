"""Fine-tuning de um transformer a partir do modelo base com divisão estratificada 80/20.

Executar da raiz do projeto:
    python -m src.finetuning
    python -m src.finetuning --epochs 10 --test-size 0.20 --seed 42
"""

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import openpyxl
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .email_data import LABELS, email_text

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "joeddav/xlm-roberta-large-xnli"
DEFAULT_EXCEL = ROOT / "data/emails_classificacao.xlsx"
DEFAULT_OUTPUT = ROOT / "src/models/xlm_roberta_large_xnli_finetuned"


class EmailDataset(Dataset):
    def __init__(self, examples: List[Tuple[str, str, str]], tokenizer, max_length: int):
        self.items = []
        self.uids = []
        self.labels = []
        for uid, text, label in examples:
            item = tokenizer(text, truncation=True, max_length=max_length)
            item["labels"] = LABELS.index(label)
            self.items.append(item)
            self.uids.append(uid)
            self.labels.append(label)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def find_emails_by_uid(search_dirs: List[Path]) -> Dict[str, Dict]:
    """Procura emails em formato JSON recursivamente nas pastas fornecidas e indexa por UID."""
    emails = {}
    for folder in search_dirs:
        if not folder.is_dir():
            continue
        for path in folder.glob("*.json"):
            if path.name == "summary.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                uid = str(data.get("uid") or data.get("id") or path.stem).removesuffix(".0").strip()
                if uid and uid not in emails:
                    emails[uid] = data
            except Exception:
                continue
    return emails


def load_examples(excel_path: Path, email_dirs: List[Path]) -> List[Tuple[str, str, str]]:
    """Lê as labels humanas na folha 'Revisão' do Excel e carrega o texto dos emails."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Ficheiro Excel não encontrado: {excel_path}")

    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    if "Revisão" not in wb.sheetnames:
        raise ValueError("A folha 'Revisão' não foi encontrada no Excel.")

    sheet = wb["Revisão"]
    rows = sheet.iter_rows(values_only=True)
    header = next((row for row in rows if "UID" in row and "Label correta" in row), None)
    if header is None:
        raise ValueError("Não encontrei as colunas 'UID' e 'Label correta' no Excel.")

    uid_col = header.index("UID")
    label_col = header.index("Label correta")

    human_labels = {}
    for row in rows:
        uid = str(row[uid_col] or "").removesuffix(".0").strip()
        label = str(row[label_col] or "").strip()
        if not uid or not label:
            continue
        if label not in LABELS:
            print(f"[Aviso] Label ignorada para UID {uid} por não pertencer às classes oficiais: '{label}'")
            continue
        human_labels[uid] = label

    if not human_labels:
        raise ValueError("O Excel não contém nenhuma label humana válida preenchida na coluna 'Label correta'.")

    emails_map = find_emails_by_uid(email_dirs)
    missing = human_labels.keys() - emails_map.keys()
    if missing:
        raise ValueError(f"Faltam ficheiros JSON correspondentes aos seguintes UIDs: {sorted(missing)}")

    examples = []
    for uid, label in human_labels.items():
        data = emails_map[uid]
        text = email_text(data)
        examples.append((uid, text, label))

    return examples


def stratified_split(
    examples: List[Tuple[str, str, str]], test_size: float = 0.20, seed: int = 42
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Divide a amostra em conjuntos de treino e teste respeitando a proporção de cada classe."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex[2]].append(ex)

    train, test = [], []
    for label, items in sorted(by_class.items()):
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            # Classes com apenas 1 exemplo têm de ficar no treino para o modelo aprender
            train.extend(items)
            print(f"[Aviso] A classe '{label}' tem apenas 1 exemplo e será colocada no conjunto de treino.")
        else:
            n_test = int(round(n * test_size))
            if n_test == 0 and test_size > 0:
                n_test = 1
            elif n_test >= n:
                n_test = n - 1

            test.extend(items[:n_test])
            train.extend(items[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "recall_macro": recall,
        "precision_macro": precision,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Modelo pré-treinado base de onde sempre recomeça o fine-tuning (defeito: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=DEFAULT_EXCEL,
        help=f"Caminho para o Excel com as anotações (defeito: {DEFAULT_EXCEL.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--emails-dir",
        type=Path,
        action="append",
        default=None,
        help="Diretórios contendo os ficheiros JSON (pode ser repetido). Por defeito procura em data/incoming_emails e data/extracted_emails.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Diretório onde gravar o checkpoint fine-tuned (defeito: {DEFAULT_OUTPUT.relative_to(ROOT)}).",
    )
    parser.add_argument("--epochs", type=float, default=10, help="Número de épocas de treino (defeito: 10).")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size por dispositivo (defeito: 2).")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Taxa de aprendizagem (defeito: 2e-5).")
    parser.add_argument("--max-length", type=int, default=512, help="Tamanho máximo de sequência (defeito: 512).")
    parser.add_argument("--test-size", type=float, default=0.20, help="Fração para conjunto de teste (defeito: 0.20 = 20%).")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória para reprodutibilidade (defeito: 42).")

    args = parser.parse_args()

    email_search_dirs = args.emails_dir or [
        ROOT / "data/incoming_emails",
        ROOT / "data/extracted_emails",
    ]

    print(f"A carregar emails anotados de {args.excel}...")
    examples = load_examples(args.excel, email_search_dirs)
    print(f"Total de emails rotulados: {len(examples)}")
    for label in LABELS:
        count = sum(1 for _, _, y in examples if y == label)
        print(f"  - {label}: {count}")

    # Divisão 80/20 estratificada
    train_examples, test_examples = stratified_split(examples, test_size=args.test_size, seed=args.seed)
    print(f"\nDivisão dos dados (seed={args.seed}):")
    print(f"  Treino ({len(train_examples)} emails, {len(train_examples)/len(examples)*100:.1f}%):")
    for label in LABELS:
        c = sum(1 for _, _, y in train_examples if y == label)
        print(f"    * {label}: {c}")
    print(f"  Teste  ({len(test_examples)} emails, {len(test_examples)/len(examples)*100:.1f}%):")
    for label in LABELS:
        c = sum(1 for _, _, y in test_examples if y == label)
        print(f"    * {label}: {c}")

    # Sempre recomeçar a partir do modelo base
    print(f"\nA carregar modelo base: {args.base_model} (sem data leakage)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )

    train_dataset = EmailDataset(train_examples, tokenizer, args.max_length)
    test_dataset = EmailDataset(test_examples, tokenizer, args.max_length) if test_examples else None

    # Configuração dos argumentos de treino
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch" if test_dataset else "no",
        save_strategy="no",
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics if test_dataset else None,
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
    )

    print("\nA iniciar fine-tuning...")
    trainer.train()

    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    print(f"Modelo guardado com sucesso em: {args.output}")

    # Avaliação final detalhada no conjunto de teste (20%)
    if test_dataset:
        print("\nA avaliar no conjunto de teste (20%)...")
        eval_metrics = trainer.evaluate()
        print(f"Resultados de Validação: Loss={eval_metrics.get('eval_loss', 0):.4f}, "
              f"Accuracy={eval_metrics.get('eval_accuracy', 0):.4f}, "
              f"F1 Macro={eval_metrics.get('eval_f1_macro', 0):.4f}")

        # Guardar predições de teste para auditoria
        preds_output = trainer.predict(test_dataset)
        logits = preds_output.predictions
        pred_indices = np.argmax(logits, axis=-1)

        test_results_path = args.output.parent / "test_predictions.csv"
        with test_results_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["UID", "Label_Real", "Label_Prevista", "Score"])
            for uid, true_label, pred_idx, logit in zip(test_dataset.uids, test_dataset.labels, pred_indices, logits):
                exp_logits = np.exp(logit - np.max(logit))
                probs = exp_logits / exp_logits.sum()
                score = round(float(probs[pred_idx]), 4)
                writer.writerow([uid, true_label, LABELS[pred_idx], score])

        print(f"Predições detalhadas do conjunto de teste guardadas em: {test_results_path}")
        print("\n=== Relatório de Classificação (Conjunto de Teste) ===")
        print(classification_report(test_dataset.labels, [LABELS[p] for p in pred_indices], labels=LABELS, zero_division=0))


if __name__ == "__main__":
    main()
