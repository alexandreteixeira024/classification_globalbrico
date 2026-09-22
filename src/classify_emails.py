"""Classify extracted JSON emails with the production SetFit checkpoint."""

import argparse
import csv
import json
import re
from pathlib import Path

from .email_data import LABELS, email_text

ROOT = Path(__file__).resolve().parent.parent
SPAM_IN_SUBJECT = re.compile(r"\bSPAM\b", re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/extracted_emails")
    parser.add_argument("--model", type=Path, default=ROOT / "src/models/bertimbau_setfit")
    parser.add_argument("--output", type=Path, default=ROOT / "data/classified_emails.csv")
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error(f"Modelo não encontrado: {args.model}")
    paths = sorted(p for p in args.input_dir.glob("*.json") if p.name != "summary.json")
    if not paths:
        parser.error(f"Sem emails JSON em {args.input_dir}")

    classifier = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    with args.output.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["uid", "label", "score", "source"])
        writer.writeheader()
        for path in paths:
            email = json.loads(path.read_text(encoding="utf-8"))
            if SPAM_IN_SUBJECT.search(email.get("subject") or ""):
                label, score, source = "SPAM", "", "subject_rule"
            else:
                if classifier is None:
                    from setfit import SetFitModel

                    classifier = SetFitModel.from_pretrained(str(args.model))
                probabilities = classifier.predict_proba(
                    [email_text(email)],
                    as_numpy=True,
                    show_progress_bar=False,
                )[0]
                predicted_index = int(probabilities.argmax())
                model_labels = classifier.labels or list(LABELS)
                label = model_labels[predicted_index]
                score, source = float(probabilities[predicted_index]), "setfit"
                if label not in LABELS:
                    raise ValueError(f"Label inesperada no modelo: {label}")
            writer.writerow({"uid": email["uid"], "label": label, "score": score, "source": source})
    print(f"{len(paths)} emails classificados: {args.output}")


if __name__ == "__main__":
    main()
