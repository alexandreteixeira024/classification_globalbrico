"""Sincronização incremental de novos emails JSON com folha Excel de classificação e comparação de IAs.

Execução típica:
    python -m src.sync_excel
    python -m src.sync_excel --source-dir /caminho/pasta/externa
    python -m src.sync _excel --input-dir data/incoming_emails --excel data/emails_classificacao.xlsx
"""

import argparse
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from email_data import LABELS, email_text

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = ROOT / "data/globalbrico_emails"
DEFAULT_EXCEL_PATH = ROOT / "data/emails_classificacao.xlsx"
DEFAULT_MODEL_DIR = ROOT / "src/models/xlm_roberta_large_email_512"
SPAM_IN_SUBJECT = re.compile(r"\bSPAM\b", re.IGNORECASE)

COLUMNS = [
    ("UID", 15),
    ("Data", 20),
    ("Remetente", 35),
    ("Assunto", 45),
    ("Texto (Resumo)", 55),
    ("Label correta", 22),
    ("Label Outra IA", 22),
    ("Label Transformer", 22),
    ("Score Transformer", 18),
    ("Concordância Outra IA", 24),
    ("Concordância Transformer", 26),
    ("Anexos", 15),
    ("Notas", 30),
    ("Ficheiro JSON", 40),
]


def canonical_label(value: Any) -> str:
    """Normaliza o nome da categoria se corresponder a uma das labels válidas."""
    if not value:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().strip(" .:\"'`\n")
    for label in LABELS:
        if text.casefold() == label.casefold():
            return label
    # Casos comuns sem acentos ou com underscores
    simple = re.sub(r"[\W_]+", " ", text).lower()
    for label in LABELS:
        simple_label = re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKD", label).encode("ASCII", "ignore").decode("utf-8")).lower()
        if simple == simple_label:
            return label
    return text


def normalize_email(data: Dict[str, Any], filepath: Optional[Path] = None) -> Dict[str, Any]:
    """Uniformiza dados de emails provenientes de diferentes fontes (IMAP, API, etc)."""
    raw_uid = data.get("uid") or data.get("id") or (filepath.stem if filepath else "")
    uid = str(raw_uid).strip()

    subject = str(data.get("subject") or "").strip()
    sender = str(data.get("from") or data.get("sender") or "").strip()
    date = str(data.get("date") or data.get("received_at") or data.get("timestamp") or "").strip()

    text = data.get("text") or data.get("body") or data.get("html") or ""
    if isinstance(text, str):
        clean_text = " ".join(text.split())
    else:
        clean_text = str(text)

    summary_text = clean_text[:300] + ("..." if len(clean_text) > 300 else "")

    # Deteta label da outra IA se já vier registada no JSON
    outra_ia_label = (
        data.get("outra_ia")
        or data.get("outra_ia_label")
        or data.get("external_label")
        or data.get("ai_label")
        or data.get("predicted_label")
        or data.get("prediction")
        or ""
    )
    if not outra_ia_label and "label" in data and not data.get("human_verified"):
        outra_ia_label = data.get("label") or ""

    outra_ia_label = canonical_label(outra_ia_label)

    # Contagem ou lista de anexos
    attachments = data.get("attachments") or []
    if isinstance(attachments, list):
        if attachments and isinstance(attachments[0], dict):
            filenames = [att.get("filename") for att in attachments if att.get("filename")]
            anexos_desc = f"{len(attachments)} ({', '.join(filenames)})" if filenames else str(len(attachments))
        else:
            anexos_desc = str(len(attachments))
    elif attachments:
        anexos_desc = str(attachments)
    else:
        anexos_desc = "0"

    return {
        "uid": uid,
        "date": date,
        "from": sender,
        "subject": subject,
        "summary": summary_text,
        "full_text": text,
        "outra_ia_label": outra_ia_label,
        "attachments_desc": anexos_desc,
        "raw": data,
        "filepath": str(filepath) if filepath else "",
    }


def copy_new_files_from_source(source_dir: Path, target_dir: Path) -> int:
    """Copia ficheiros .json novos da pasta externa para a pasta de entrada local."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Pasta de origem não encontrada: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    copied_count = 0

    for src_file in source_dir.glob("*.json"):
        if src_file.name == "summary.json":
            continue
        dst_file = target_dir / src_file.name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
            copied_count += 1

    return copied_count


class TransformerClassifier:
    """Carrega o modelo fine-tuned para classificar novos emails."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "text-classification",
                model=str(self.model_dir),
                tokenizer=str(self.model_dir),
            )

    def predict(self, email_norm: Dict[str, Any]) -> Tuple[str, Optional[float]]:
        subject = email_norm["subject"]
        if SPAM_IN_SUBJECT.search(subject):
            return "SPAM", 1.0

        self._load_pipeline()
        formatted_text = email_text(email_norm["raw"])
        result = self._pipeline(formatted_text, truncation=True, max_length=512)[0]
        label = canonical_label(result["label"])
        score = round(float(result["score"]), 4)
        return label, score


def get_or_create_workbook(excel_path: Path) -> Tuple[openpyxl.Workbook, openpyxl.worksheet.worksheet.Worksheet, Set[str]]:
    """Abre ou inicializa o workbook na folha 'Revisão' e recolhe os UIDs já registados."""
    header_titles = [col[0] for col in COLUMNS]

    if excel_path.exists():
        wb = openpyxl.load_workbook(excel_path)
        if "Revisão" in wb.sheetnames:
            ws = wb["Revisão"]
        else:
            ws = wb.active
            ws.title = "Revisão"

        # Localizar header e UIDs existentes
        existing_uids = set()
        uid_col_idx = None
        for row_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
            if "UID" in row and "Label correta" in row:
                uid_col_idx = row.index("UID")
                continue
            if uid_col_idx is not None:
                uid_val = str(row[uid_col_idx] or "").removesuffix(".0").strip()
                if uid_val:
                    existing_uids.add(uid_val)

        return wb, ws, existing_uids

    # Criar novo workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Revisão"

    # Estilos de cabeçalho
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    ws.append(header_titles)
    ws.row_dimensions[1].height = 28

    thin_border = Border(
        left=Side(style="thin", color="D3D3D3"),
        right=Side(style="thin", color="D3D3D3"),
        top=Side(style="thin", color="D3D3D3"),
        bottom=Side(style="thin", color="D3D3D3"),
    )

    for col_idx, (col_name, col_width) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = col_width

    # Congelar painéis no cabeçalho
    ws.freeze_panes = "A2"

    return wb, ws, set()


def setup_data_validation_and_styling(ws: openpyxl.worksheet.worksheet.Worksheet, max_row: int):
    """Configura listas suspensas (dropdown) e alinhamentos para as linhas."""
    label_col_idx = [col[0] for col in COLUMNS].index("Label correta") + 1
    label_col_letter = get_column_letter(label_col_idx)

    # Validação de dados (dropdown) para Label correta
    dv = DataValidation(
        type="list",
        formula1=f'"{",".join(LABELS)}"',
        allow_blank=True,
        showDropDown=True,
    )
    dv.error = "Por favor selecione uma das categorias válidas."
    dv.errorTitle = "Categoria inválida"
    ws.add_data_validation(dv)
    dv.add(f"{label_col_letter}2:{label_col_letter}{max(max_row, 1000)}")


def sync_emails(
    input_dir: Path,
    excel_path: Path,
    source_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    predict_transformer: bool = True,
) -> int:
    """Executa o ciclo completo de sincronização incremental."""
    # 1. Copiar ficheiros da pasta externa, se especificada
    if source_dir:
        copied = copy_new_files_from_source(source_dir, input_dir)
        if copied > 0:
            print(f"[Origem Externa] {copied} ficheiros copiados para {input_dir}")

    input_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(p for p in input_dir.glob("*.json") if p.name != "summary.json")

    wb, ws, existing_uids = get_or_create_workbook(excel_path)

    # 2. Filtrar apenas emails novos
    new_emails: List[Dict[str, Any]] = []
    for filepath in json_files:
        try:
            raw_data = json.loads(filepath.read_text(encoding="utf-8"))
            norm = normalize_email(raw_data, filepath)
            if not norm["uid"]:
                print(f"[Aviso] Ficheiro sem UID ignorado: {filepath.name}")
                continue
            if norm["uid"] in existing_uids:
                continue
            new_emails.append(norm)
            existing_uids.add(norm["uid"])
        except Exception as err:
            print(f"[Erro] Falha ao ler {filepath.name}: {err}")

    if not new_emails:
        print(f"Nenhum email novo encontrado. O ficheiro Excel está atualizado ({len(existing_uids)} emails registados).")
        return 0

    print(f"Detetados {len(new_emails)} novos emails. A processar...")

    # 3. Classificação com Transformer (opcional / se disponível)
    classifier = None
    if predict_transformer and model_dir and model_dir.is_dir():
        try:
            classifier = TransformerClassifier(model_dir)
            print(f"Classificador Transformer carregado de: {model_dir}")
        except Exception as err:
            print(f"[Aviso] Não foi possível carregar o Transformer ({err}). Coluna ficará vazia.")
    elif predict_transformer and model_dir:
        print(f"[Info] Modelo Transformer não encontrado em {model_dir}. Coluna ficará vazia.")

    # Estilos das novas células
    regular_font = Font(name="Calibri", size=10)
    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")
    summary_align = Alignment(horizontal="left", vertical="center", wrap_text=False)

    thin_border = Border(
        left=Side(style="thin", color="E0E0E0"),
        right=Side(style="thin", color="E0E0E0"),
        top=Side(style="thin", color="E0E0E0"),
        bottom=Side(style="thin", color="E0E0E0"),
    )

    # Letras das colunas para fórmulas
    col_dict = {name: get_column_letter(idx + 1) for idx, (name, _) in enumerate(COLUMNS)}
    correct_letter = col_dict["Label correta"]
    other_ai_letter = col_dict["Label Outra IA"]
    transf_letter = col_dict["Label Transformer"]

    # 4. Inserir novas linhas no Excel
    start_row = ws.max_row + 1

    for idx, email_norm in enumerate(new_emails):
        current_row = ws.max_row + 1

        transf_label = ""
        transf_score_val = ""
        if classifier:
            try:
                transf_label, score_num = classifier.predict(email_norm)
                transf_score_val = score_num if score_num is not None else ""
            except Exception as err:
                print(f"[Aviso] Falha na previsão do UID {email_norm['uid']}: {err}")

        formula_outra_ia = (
            f'=IF(OR({correct_letter}{current_row}="",{other_ai_letter}{current_row}=""),"Pendente",'
            f'IF({correct_letter}{current_row}={other_ai_letter}{current_row},"Concorda","Diverge"))'
        )
        formula_transf = (
            f'=IF(OR({correct_letter}{current_row}="",{transf_letter}{current_row}=""),"Pendente",'
            f'IF({correct_letter}{current_row}={transf_letter}{current_row},"Concorda","Diverge"))'
        )

        row_values = [
            email_norm["uid"],
            email_norm["date"],
            email_norm["from"],
            email_norm["subject"],
            email_norm["summary"],
            "",  # Label correta (para ser preenchida pelo utilizador)
            email_norm["outra_ia_label"],
            transf_label,
            transf_score_val,
            formula_outra_ia,
            formula_transf,
            email_norm["attachments_desc"],
            "",  # Notas
            email_norm["filepath"],
        ]

        ws.append(row_values)
        ws.row_dimensions[current_row].height = 20

        # Formatação individual por coluna
        for c_idx in range(1, len(COLUMNS) + 1):
            cell = ws.cell(row=current_row, column=c_idx)
            cell.font = regular_font
            cell.border = thin_border
            if c_idx in (1, 2, 6, 7, 8, 9, 10, 11, 12):
                cell.alignment = center_align
            elif c_idx == 5:
                cell.alignment = summary_align
            else:
                cell.alignment = left_align

    setup_data_validation_and_styling(ws, ws.max_row)

    excel_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(excel_path)
    print(f"Sucesso: {len(new_emails)} emails adicionados a {excel_path} (Total atual: {ws.max_row - 1}).")
    return len(new_emails)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Caminho para uma pasta externa de onde copiar novos emails (.json) antes de atualizar o Excel.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Pasta local contendo os emails JSON a indexar (defeito: {DEFAULT_INPUT_DIR.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=DEFAULT_EXCEL_PATH,
        help=f"Ficheiro Excel de destino (defeito: {DEFAULT_EXCEL_PATH.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Diretório do modelo transformer fine-tuned (defeito: {DEFAULT_MODEL_DIR.relative_to(ROOT)}).",
    )
    parser.add_argument(
        "--no-transformer",
        action="store_true",
        help="Não executa a previsão automática pelo transformer.",
    )

    args = parser.parse_args()

    sync_emails(
        input_dir=args.input_dir,
        excel_path=args.excel,
        source_dir=args.source_dir,
        model_dir=args.model_dir,
        predict_transformer=not args.no_transformer,
    )


if __name__ == "__main__":
    main()
