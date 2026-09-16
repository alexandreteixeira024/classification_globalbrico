"""Fine-tuning simples com as labels humanas do Excel.

Executar da raiz do projeto:
    python3 -m research.finetuning --model joeddav/xlm-roberta-large-xnli

Instalar primeiro, se necessário: pip install transformers torch accelerate openpyxl sentencepiece
"""

import argparse
import json
import re
from pathlib import Path

from openpyxl import load_workbook
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Pedido de Informação", "Pedido de Encomenda", "SPAM"]


class EmailDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.items = []
        for text, label in examples:
            item = tokenizer(text, truncation=True, max_length=max_length)
            item["labels"] = LABELS.index(label)
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def load_examples(excel_path, email_dir):
    sheet = load_workbook(excel_path, read_only=True, data_only=True)["Revisão"]
    rows = sheet.iter_rows(values_only=True)
    header = next((row for row in rows if "UID" in row and "Label correta" in row), None)
    if header is None:
        raise ValueError("Não encontrei as colunas UID e Label correta no Excel.")
    uid_col, label_col = header.index("UID"), header.index("Label correta")
    human_labels = {}
    for row in rows:
        uid = str(row[uid_col] or "").removesuffix(".0")
        label = str(row[label_col] or "").strip()
        if not uid or not label:
            continue
        if label not in LABELS or uid in human_labels:
            raise ValueError(f"Label inválida ou UID repetido no Excel: {uid} ({label})")
        human_labels[uid] = label

    emails = {}
    for path in email_dir.glob("*.json"):
        if path.name == "summary.json":
            continue
        email = json.loads(path.read_text(encoding="utf-8"))
        uid = str(email.get("uid") or "")
        if uid in emails:
            raise ValueError(f"UID repetido nos emails: {uid}")
        emails[uid] = email

    missing = human_labels.keys() - emails.keys()
    if missing:
        raise ValueError(f"Faltam JSON para os UIDs: {sorted(missing)}")
    if set(human_labels.values()) != set(LABELS):
        raise ValueError("É necessária pelo menos uma label humana de cada classe.")

    examples = []
    for uid, label in human_labels.items():
        email = emails[uid]
        subject = re.sub(r"^\s*(?:\*+SPAM\*+|\[SPAM\]|SPAM\b)[\s:_-]*", "", email.get("subject") or "", flags=re.I)
        text = "\n".join(part for part in (
            f"Assunto: {subject}",
            f"Remetente: {email.get('from') or ''}",
            email.get("text") or email.get("html") or "",
        ) if part.strip())
        examples.append((text, label))
    return examples


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning de um transformer para classificar emails.")
    parser.add_argument("--model", default="joeddav/xlm-roberta-large-xnli")
    parser.add_argument("--excel", type=Path, default=ROOT / "research/ground_truth/ground_truth_emails.xlsx")
    parser.add_argument("--emails", type=Path, default=ROOT / "research/extracted_emails")
    parser.add_argument("--output", type=Path, default=ROOT / "src/models/xlm_roberta_large_xnli_finetuned")
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    if args.epochs <= 0 or args.max_length < 32:
        parser.error("--epochs deve ser positivo e --max-length deve ser pelo menos 32.")

    examples = load_examples(args.excel, args.emails)
    print(f"A treinar com {len(examples)} emails: " + ", ".join(f"{label}={sum(y == label for _, y in examples)}" for label in LABELS))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.output),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=2,
            learning_rate=2e-5,
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=EmailDataset(examples, tokenizer, args.max_length),
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(args.output)
    print(f"Modelo guardado em {args.output}")


if __name__ == "__main__":
    main()
