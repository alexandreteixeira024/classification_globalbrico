"""Email text and labelled examples shared by inference and evaluation."""

import json
import re
from pathlib import Path

from openpyxl import load_workbook

LABELS = ("Pedido de Informação", "Pedido de Encomenda", "SPAM")


def email_text(email):
    # Match the preprocessing used to fine-tune the saved checkpoint.
    #Identificação do SPAM por vazio no assunto do email.
    subject = re.sub(
        r"^\s*(?:\*+SPAM\*+|\[SPAM\]|SPAM\b)[\s:_-]*",
        "",
        email.get("subject") or "",
        flags=re.I,
    )   
    return "\n".join(
        part
        for part in (
            f"Assunto: {subject}",
            f"Remetente: {email.get('from') or ''}",
            email.get("text") or email.get("html") or "",
        )
        if part.strip()
    )

#Carrega os emails classificados a partir de 'emails_classificação.csv'
def load_labelled_emails(excel_path, email_dir):
    workbook = load_workbook(excel_path, read_only=True, data_only=True)
    try:
        rows = workbook["Revisão"       ].iter_rows(values_only=True)
        header = next((row for row in rows if "UID" in row and "Label correta" in row), None)
        if header is None:
            raise ValueError("Não encontrei UID e Label correta no Excel.")
        uid_col, label_col = header.index("UID"), header.index("Label correta")
        labels = {}
        for row in rows:
            uid = str(row[uid_col] or "").removesuffix(".0")
            label = str(row[label_col] or "").strip()
            if not uid or not label:
                continue
            if uid in labels or label not in LABELS:
                raise ValueError(f"UID duplicado ou label inválida: {uid}: {label}")
            labels[uid] = label
    finally:
        workbook.close()

    emails = {}
    for path in Path(email_dir).glob("*.json"):
        if path.name == "summary.json":
            continue
        email = json.loads(path.read_text(encoding="utf-8"))
        uid = str(email.get("uid") or "")
        if not uid or uid in emails:
            raise ValueError(f"UID ausente ou duplicado: {path}")
        emails[uid] = email
    missing = labels.keys() - emails.keys()
    if missing:
        raise ValueError(f"Faltam ficheiros JSON para: {sorted(missing)}")
    if not labels:
        raise ValueError("O Excel não tem labels humanas.")
    return [
        {"uid": uid, "label": label, "text": email_text(emails[uid])}
        for uid, label in labels.items()
    ]
