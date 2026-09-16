import argparse
import csv
import json
from pathlib import Path

from transformers import pipeline


#Carrega os json na pasta tests/extracted_emails
def load_emails(input_dir):
    """
    Carrega todos os ficheiros JSON do diretório de entrada, ignorando o summary.json.
    Retorna uma lista de dicionários com os emails guardados.
    """
    emails = []
    input_path = Path(input_dir)

    for file_path in sorted(input_path.glob("*.json")):
        if file_path.name == "summary.json":
            continue

        with file_path.open("r", encoding="utf-8") as f:
            emails.append(json.load(f))

    return emails


def build_email_text(email):
    """
    Constrói um texto único para classificação usando assunto + corpo + remetente.
    """
    parts = []

    if email.get("subject"):
        parts.append(f"Assunto: {email['subject']}")

    if email.get("from"):
        parts.append(f"Remetente: {email['from']}")

    if email.get("text"):
        parts.append(email["text"])
    elif email.get("html"):
        parts.append(email["html"])

    return "\n".join(parts).strip()


def classify_email(classifier, email, labels):
    """
    Usa zero-shot classification para atribuir uma etiqueta ao email.
    """
    text = build_email_text(email)

    #Se não houve texto dentro do email, é classificado como SPAM com score 0.0
    if not text:
        return {
            "uid": email.get("uid"),
            "label": "SPAM",
            "score": 0.0,
            "reason": "Texto vazio para classificação.",
        }

    result = classifier(
        text,
        candidate_labels=labels,
        multi_label=False,
    )
    '''O problema é não é multi_label, significando que só pode ser uma categoria dentro das 3'''

    top_label = result["labels"][0]
    top_score = float(result["scores"][0])

    return {
        "uid": email.get("uid"),
        "label": top_label,
        "score": round(top_score, 4),
        "raw_prediction": result,
    }


def save_results(results, output_file):
    """
    Guarda os resultados em JSON e um resumo em CSV na mesma pasta.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    csv_path = output_path.with_suffix(".csv")
    if csv_path == output_path:
        csv_path = output_path.with_name(output_path.name + ".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "label", "score", "reason"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Classifica emails guardados em JSON usando xlm-roberta-large-xnli."
    )
    parser.add_argument(
        "--input-dir",
        default="research/extracted_emails",
        help="Diretório com os emails em JSON.",
    )
    parser.add_argument(
        "--output-file",
        default="research/classified_emails.json",
        help="Ficheiro JSON onde guardar os resultados da classificação.",
    )
    parser.add_argument(
        "--model",
        default="joeddav/xlm-roberta-large-xnli",
        help="Modelo zero-shot a usar.",
    )
    args = parser.parse_args()

    labels = [
        "Pedido de Informação",
        "Pedido de Encomenda",
        "SPAM",
    ]

    # O pipeline de zero-shot classification é o método mais simples para usar este modelo.
    classifier = pipeline(
        "zero-shot-classification",
        model=args.model,
    )

    emails = load_emails(args.input_dir)

    results = []
    for email in emails:
        result = classify_email(classifier, email, labels)
        results.append(result)

    csv_path = save_results(results, args.output_file)

    print(f"Classificação concluída. {len(results)} emails processados.")
    print(f"Resultados guardados em: {args.output_file}")
    print(f"CSV guardado em: {csv_path}")


if __name__ == "__main__":
    main()
