"""Fine-tuning diário com 5-fold cross-validation e histórico de resultados.

Executar da raiz do projeto:
    python -m src.finetuning
"""

import argparse
import csv
import gc
import hashlib
import json
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import numpy as np
import openpyxl
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from .email_data import LABELS, email_text

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_MODEL = "joeddav/xlm-roberta-large-xnli"
DEFAULT_EXCEL = ROOT / "data/emails_classificacao.xlsx"
DEFAULT_EMAILS_DIR = ROOT / "data/globalbrico_emails"
DEFAULT_OUTPUT = ROOT / "src/models/xlm_roberta_large_xnli_finetuned"
DEFAULT_RESULTS_DIR = ROOT / "data/finetuning_results"
DEFAULT_HISTORY = ROOT / "data/finetuning_history.csv"
DEFAULT_FOLDS = 5

# uid, texto, label
Example = Tuple[str, str, str]


class EmailDataset(Dataset):
    def __init__(self, examples: List[Example], tokenizer, max_length: int):
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


def save_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def find_emails_by_uid(search_dirs: List[Path]) -> Dict[str, Dict]:
    emails = {}
    for folder in search_dirs:
        if not folder.is_dir():
            continue

        #Percorre a pasta 'data/globalbrico_emails/'
        for path in folder.glob("*.json"):

            #Evita o summary.json (Existe em extracted_emails)

            if path.name == "summary.json":
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                print(f"[Aviso] JSON ignorado ({path.name}): {error}")
                continue
            uid = str(data.get("uid") or data.get("id") or path.stem).removesuffix(".0").strip()
            if not uid:
                print(f"[Aviso] JSON sem UID ignorado: {path}")
                continue
            if uid in emails and emails[uid] != data:
                raise ValueError(f"Conteúdo contraditório para o UID duplicado {uid}.")
            emails[uid] = data
    return emails

def load_examples(excel_path: Path, email_dirs: List[Path]) -> List[Example]:
    if not excel_path.exists():
        raise FileNotFoundError(f"Ficheiro Excel não encontrado: {excel_path}")
    #Abrir o excel
    workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    try:
        #Nome da folha que é 'Revisão'
        if "Revisão" not in workbook.sheetnames:
            raise ValueError("A folha 'Revisão' não foi encontrada no Excel.")
        rows = workbook["Revisão"].iter_rows(values_only=True)
        # Header = UID e Label Correta
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
                print(f"[Aviso] Label inválida ignorada para {uid}: {label!r}")
                continue
            if uid in human_labels:
                raise ValueError(f"UID repetido no Excel: {uid}")
            human_labels[uid] = label
    finally:
        workbook.close()

    if not human_labels:
        raise ValueError("O Excel não contém labels humanas válidas.")

    #Encontra os emails válidos. Extrair o texto do excel
    emails = find_emails_by_uid(email_dirs)
    missing = sorted(set(human_labels) - set(emails))
    if missing:
        raise ValueError(f"Faltam JSON para os UIDs: {missing}")

    examples = []
    skipped_replies = 0
    for uid, label in human_labels.items():
        email = emails[uid]
        if email.get("in_reply_to") not in (None, ""):
            skipped_replies += 1
            continue
        examples.append((uid, email_text(email), label))
    if skipped_replies:
        print(f"Ignorados no fine-tuning: {skipped_replies} emails com in_reply_to preenchido.")
    if not examples:
        raise ValueError("Não existem mensagens iniciais rotuladas para treinar.")
    return examples

def stratified_folds(examples: List[Example], folds: int, seed: int) -> List[List[Example]]:
    """Distribui mensagens iniciais pelos folds utilizando apenas as labels."""
    labels = [example[2] for example in examples]
    label_counts = Counter(labels)
    insufficient = {label: label_counts[label] for label in LABELS if label_counts[label] < folds}
    if insufficient:
        raise ValueError(
            f"{folds}-fold requer pelo menos {folds} mensagens iniciais por classe. "
            f"Contagens insuficientes: {insufficient}"
        )
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [
        [examples[index] for index in test_indices]
        for _train_indices, test_indices in splitter.split(np.zeros(len(examples)), labels)
    ]


def dataset_fingerprint(examples: List[Example]) -> str:
    payload = [
        {"uid": uid, "label": label, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        for uid, text, label in sorted(examples)
    ]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def make_model(base_model: str):
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)},
    )


def make_training_args(output_dir: str, args, seed: int):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        use_cpu = False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
    )


def clear_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def scores(truth: List[str], predicted: List[str]) -> Dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=LABELS, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels": list(LABELS),
            "values": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
        },
    }


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_history(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def paired_with_previous(results_dir: Path, current_rows: List[Dict], method: str) -> Dict:
    """Compara execuções da mesma metodologia nos UIDs presentes em ambas."""
    latest = results_dir / "latest.json"
    empty = {
        "n_common": 0,
        "previous_run": None,
        "previous_accuracy": None,
        "current_accuracy": None,
        "delta_accuracy": None,
        "previous_f1_macro": None,
        "current_f1_macro": None,
        "delta_f1_macro": None,
    }
    if not latest.exists():
        return empty
    pointer = json.loads(latest.read_text(encoding="utf-8"))
    previous_metrics_path = Path(pointer.get("metrics", ""))
    if not previous_metrics_path.exists():
        return empty
    previous_metrics = json.loads(previous_metrics_path.read_text(encoding="utf-8"))
    if previous_metrics.get("method") != method:
        return empty
    previous_path = Path(pointer["run_dir"]) / "predictions.csv"
    if not previous_path.exists():
        return empty
    with previous_path.open(encoding="utf-8-sig", newline="") as source:
        previous = {row["uid"]: row for row in csv.DictReader(source)}
    current = {row["uid"]: row for row in current_rows}
    common = sorted(set(previous) & set(current))
    if not common:
        return empty
    truth = [current[uid]["label_real"] for uid in common]
    previous_predicted = [previous[uid]["label_prevista"] for uid in common]
    current_predicted = [current[uid]["label_prevista"] for uid in common]
    previous_scores = scores(truth, previous_predicted)
    current_scores = scores(truth, current_predicted)
    return {
        "n_common": len(common),
        "previous_run": pointer.get("run_id"),
        "previous_accuracy": previous_scores["accuracy"],
        "current_accuracy": current_scores["accuracy"],
        "delta_accuracy": current_scores["accuracy"] - previous_scores["accuracy"],
        "previous_f1_macro": previous_scores["f1_macro"],
        "current_f1_macro": current_scores["f1_macro"],
        "delta_f1_macro": current_scores["f1_macro"] - previous_scores["f1_macro"],
    }


def main():
    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, set_seed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--emails-dir", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds tem de ser pelo menos 2.")
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_length < 32:
        parser.error("Épocas e batch size têm de ser positivos; max-length tem de ser pelo menos 32.")

    examples = load_examples(args.excel, args.emails_dir or [DEFAULT_EMAILS_DIR])
    print(f"Emails rotulados: {len(examples)}")
    for label in LABELS:
        print(f"  {label}: {sum(item[2] == label for item in examples)}")

    folds = stratified_folds(examples, args.folds, args.seed)
    for index, fold in enumerate(folds, 1):
        print(f"Fold {index}: teste={len(fold)}, treino={len(examples) - len(fold)}")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.results_dir / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = args.results_dir / f"{run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    collator = DataCollatorWithPadding(tokenizer)
    predictions = []
    fold_metrics = []
    fold_logs = {}

    for fold_index, test_examples in enumerate(folds):
        test_uids = {item[0] for item in test_examples}
        train_examples = [item for item in examples if item[0] not in test_uids]
        fold_seed = args.seed + fold_index
        set_seed(fold_seed)
        print(f"\n=== Fold {fold_index + 1}/{args.folds}: treino={len(train_examples)}, teste={len(test_examples)} ===")

        train_dataset = EmailDataset(train_examples, tokenizer, args.max_length)
        test_dataset = EmailDataset(test_examples, tokenizer, args.max_length)
        with tempfile.TemporaryDirectory(prefix=f"globalbrico-fold-{fold_index + 1}-") as temporary:
            model = make_model(args.base_model)
            trainer = Trainer(
                model=model,
                args=make_training_args(temporary, args, fold_seed),
                train_dataset=train_dataset,
                data_collator=collator,
                processing_class=tokenizer,
            )
            trainer.train()
            output = trainer.predict(test_dataset)
            logits = output.predictions
            predicted_indices = np.argmax(logits, axis=-1)
            predicted_labels = [LABELS[index] for index in predicted_indices]
            fold_score = scores(test_dataset.labels, predicted_labels)
            fold_metrics.append({
                "fold": fold_index + 1,
                "n_train": len(train_examples),
                "n_test": len(test_examples),
                "accuracy": fold_score["accuracy"],
                "precision_macro": fold_score["precision_macro"],
                "recall_macro": fold_score["recall_macro"],
                "f1_macro": fold_score["f1_macro"],
            })
            fold_logs[str(fold_index + 1)] = trainer.state.log_history
            for uid, label, predicted_index, logit in zip(
                test_dataset.uids, test_dataset.labels, predicted_indices, logits
            ):
                probabilities = np.exp(logit - np.max(logit))
                probabilities = probabilities / probabilities.sum()
                predictions.append({
                    "uid": uid,
                    "fold": fold_index + 1,
                    "label_real": label,
                    "label_prevista": LABELS[predicted_index],
                    "correto": label == LABELS[predicted_index],
                    "score": round(float(probabilities[predicted_index]), 6),
                })
            del output, logits, trainer, model, train_dataset, test_dataset
            clear_device_cache()

    if len(predictions) != len(examples) or len({row["uid"] for row in predictions}) != len(examples):
        raise RuntimeError("As previsões out-of-fold não cobrem cada email exatamente uma vez.")

    predictions.sort(key=lambda row: row["uid"])
    truth = [row["label_real"] for row in predictions]
    predicted = [row["label_prevista"] for row in predictions]
    aggregate = scores(truth, predicted)
    method = f"{args.folds}-fold-stratified-cross-validation"
    paired = paired_with_previous(args.results_dir, predictions, method)
    fold_summary = {
        metric: {
            "mean": mean(row[metric] for row in fold_metrics),
            "std": pstdev(row[metric] for row in fold_metrics),
        }
        for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    }

    write_csv(run_dir / "predictions.csv", predictions)
    write_csv(run_dir / "fold_metrics.csv", fold_metrics)
    save_json(run_dir / "folds.json", {
        "method": "stratified-by-label",
        "folds": {
            str(index + 1): sorted(item[0] for item in fold)
            for index, fold in enumerate(folds)
        },
    })

    print("\n=== Treino final de produção com 100% dos dados ===")
    set_seed(args.seed)
    full_dataset = EmailDataset(examples, tokenizer, args.max_length)
    final_model = make_model(args.base_model)
    final_trainer = Trainer(
        model=final_model,
        args=make_training_args(str(args.output), args, args.seed),
        train_dataset=full_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    final_trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    final_trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)

    previous = None
    if args.history.exists():
        with args.history.open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        previous = next(
            (row for row in reversed(rows) if row.get("method") == method),
            None,
        )

    metrics = {
        "run_id": run_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "dataset_fingerprint": dataset_fingerprint(examples),
        "base_model": args.base_model,
        "output_model": str(args.output),
        "n_total": len(examples),
        "folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        **aggregate,
        "fold_metrics": fold_metrics,
        "fold_summary": fold_summary,
        "paired_with_previous_on_common_uids": paired,
        "delta_accuracy_previous_run": (
            aggregate["accuracy"] - float(previous["accuracy"]) if previous else None
        ),
        "delta_f1_macro_previous_run": (
            aggregate["f1_macro"] - float(previous["f1_macro"]) if previous else None
        ),
        "fold_training_logs": fold_logs,
        "final_training_log": final_trainer.state.log_history,
    }
    save_json(run_dir / "metrics.json", metrics)

    history_row = {
        "run_id": run_dir.name,
        "created_at_utc": metrics["created_at_utc"],
        "method": metrics["method"],
        "dataset_fingerprint": metrics["dataset_fingerprint"],
        "n_total": metrics["n_total"],
        "n_groups": "",
        "accuracy": round(aggregate["accuracy"], 6),
        "f1_macro": round(aggregate["f1_macro"], 6),
        "precision_macro": round(aggregate["precision_macro"], 6),
        "recall_macro": round(aggregate["recall_macro"], 6),
        "fold_accuracy_std": round(fold_summary["accuracy"]["std"], 6),
        "fold_f1_macro_std": round(fold_summary["f1_macro"]["std"], 6),
        "paired_common_uids": paired["n_common"],
        "paired_delta_accuracy": (
            "" if paired["delta_accuracy"] is None else round(paired["delta_accuracy"], 6)
        ),
        "paired_delta_f1_macro": (
            "" if paired["delta_f1_macro"] is None else round(paired["delta_f1_macro"], 6)
        ),
        "delta_accuracy": "" if previous is None else round(metrics["delta_accuracy_previous_run"], 6),
        "delta_f1_macro": "" if previous is None else round(metrics["delta_f1_macro_previous_run"], 6),
    }
    append_history(args.history, history_row)
    save_json(args.results_dir / "latest.json", {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "metrics": str(run_dir / "metrics.json"),
    })

    print(
        f"\nOOF Accuracy={aggregate['accuracy']:.4f} | "
        f"OOF F1 Macro={aggregate['f1_macro']:.4f} | "
        f"F1 entre folds={fold_summary['f1_macro']['mean']:.4f} ± {fold_summary['f1_macro']['std']:.4f}"
    )
    print(classification_report(truth, predicted, labels=LABELS, zero_division=0))
    print(f"Resultados guardados em: {run_dir}")
    print(f"Histórico guardado em: {args.history}")
    print(f"Modelo final guardado em: {args.output}")


if __name__ == "__main__":
    main()
