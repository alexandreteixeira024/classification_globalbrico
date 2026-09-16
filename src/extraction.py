import argparse
import getpass
import imaplib
import json
import os
import re
import ssl
import sys
from email import policy
from email.parser import BytesParser
from pathlib import Path


def load_env(path):
    """Carrega KEY=VALUE, com aspas opcionais e valores literais (sem interpolacao)."""
    if not path.exists():
        return

    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or not all(c.isalnum() or c == "_" for c in key):
            raise ValueError(f"Formato invalido no .env, linha {number}.")

        value = value.strip()
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Aspas nao fechadas no .env, linha {number}.")
            value = value[1:-1]

        os.environ.setdefault(key, value)


def sanitize_filename(value):
    value = re.sub(r"\s+", "_", (value or "email").strip())
    value = re.sub(r"[^A-Za-z0-9_\-]", "", value)
    return value[:80] or "email"


def decode_part(part):
    try:
        content = part.get_content()
    except (LookupError, UnicodeError):
        content = part.get_payload(decode=True) or b""

    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")

    return str(content)


def strip_html(text):
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>|</div>|</li>|</tr>|</td>|</th>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def serialize_email(message):
    subject = message.get("Subject", "(sem assunto)")
    sender = message.get("From", "(remetente desconhecido)")
    to = message.get("To", "(destinatario desconhecido)")
    date = message.get("Date", "(sem data)")

    plain_text = ""
    html_text = ""

    for part in message.walk():
        if part.is_multipart():
            continue

        content_type = part.get_content_type()
        if content_type == "text/plain" and not plain_text:
            plain_text = decode_part(part)
        elif content_type == "text/html" and not html_text:
            html_text = decode_part(part)

    body_text = plain_text.strip() if plain_text.strip() else strip_html(html_text)
    body_html = html_text.strip()

    attachments = []
    for part in message.iter_attachments():
        payload = part.get_payload(decode=True) or b""
        attachments.append(
            {
                "filename": part.get_filename() or "anexo",
                "content_type": part.get_content_type(),
                "size_bytes": len(payload),
            }
        )

    return {
        "uid": message.get("X-Original-UID") or None,
        "subject": subject,
        "from": sender,
        "to": to,
        "date": date,
        "text": body_text,
        "html": body_html,
        "attachments": attachments,
    }


def load_seen_state(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def last_saved_uid(output_dir, folder):
    saved = []
    for path in output_dir.glob("[0-9]*_*.json"):
        uid = path.name.split("_", 1)[0]
        if not uid.isdigit():
            continue
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("folder", "INBOX") == folder:
                saved.append(int(uid))
        except (OSError, ValueError):
            continue
    return max(saved, default=0)


def extract_recent_emails(host, port, username, password, limit=10, timeout=60, output_dir=None, folder="INBOX"):
    if not folder.strip() or any(char in folder for char in "\r\n\x00"):
        raise ValueError("E necessario indicar uma pasta IMAP valida.")
    output_dir = Path(output_dir or Path(__file__).resolve().parent.parent / "data/extracted_emails")
    output_dir.mkdir(parents=True, exist_ok=True)

    context = ssl.create_default_context()
    print(f"A ligar a {host}:{port} por IMAP/TLS...", flush=True)

    client = imaplib.IMAP4_SSL(host, port, ssl_context=context, timeout=timeout)
    try:
        client.login(username, password)
        print("Autenticacao efetuada.", flush=True)

        mailbox = '"' + folder.replace("\\", "\\\\").replace('"', '\\"') + '"'
        status, _ = client.select(mailbox, readonly=True)
        if status != "OK":
            raise RuntimeError(f"Nao foi possivel abrir a pasta {folder!r}. Verificar o nome da pasta no servidor IMAP.")

        status, data = client.uid("search", None, "ALL")
        if status != "OK":
            raise RuntimeError("Nao foi possivel pesquisar os emails.")

        uids = sorted((data[0] or b"").split(), key=int, reverse=True)
        selected = uids[:limit] if limit else uids
        print(f"Emails na pasta {folder}: {len(uids)}. A extrair: {len(selected)}.", flush=True)

        state_path = output_dir / ".extraction_state"
        state = load_seen_state(state_path)
        state_key = f"{host}:{port}/{username}/{folder}"
        _, validity_data = client.response("UIDVALIDITY")
        uidvalidity = (validity_data or [None])[0]
        uidvalidity = uidvalidity.decode("ascii") if isinstance(uidvalidity, bytes) else str(uidvalidity)
        previous = state.get(state_key, {})
        if previous.get("uidvalidity") == uidvalidity:
            last_seen = int(previous.get("last_uid", 0))
        else:
            last_seen = last_saved_uid(output_dir, folder) if not previous else 0
        new_uids = [uid for uid in uids if int(uid) > last_seen]
        print(f"Emails novos desde a ultima verificacao: {len(new_uids)}.", flush=True)

        metadata = []

        for uid in selected:
            uid_str = uid.decode("ascii")
            status, response = client.uid("fetch", uid, "(BODY.PEEK[])")
            if status != "OK":
                raise RuntimeError(f"Falha ao ler o UID {uid_str}.")

            raw = next((item[1] for item in response if isinstance(item, tuple)), None)
            if raw is None:
                print(f"UID {uid_str}: mensagem indisponivel.")
                continue

            message = BytesParser(policy=policy.default).parsebytes(raw)
            email_payload = serialize_email(message)
            email_payload["uid"] = uid_str
            email_payload["folder"] = folder

            filename = f"{uid_str}_{sanitize_filename(email_payload['subject'])}.json"
            file_path = output_dir / filename
            file_path.write_text(json.dumps(email_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            metadata.append(
                {
                    "uid": uid_str,
                    "folder": folder,
                    "file": file_path.name,
                    "subject": email_payload["subject"],
                    "from": email_payload["from"],
                    "to": email_payload["to"],
                    "date": email_payload["date"],
                }
            )

            print(f"- Guardado: {file_path.name}")

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        state[state_key] = {"uidvalidity": uidvalidity, "last_uid": max((int(uid) for uid in uids), default=last_seen)}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\nExtracao concluida. {len(metadata)} emails guardados em {output_dir}")
        return metadata

    finally:
        try:
            client.logout()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Extrair emails de uma pasta IMAP para ficheiros JSON estruturados.")

    try:
        load_env(Path(__file__).resolve().parent.parent / ".env")
    except (OSError, ValueError):
        parser.error("Nao foi possivel carregar o .env. Verificar acesso e formato KEY=VALUE.")

    parser.add_argument("--host", default=os.getenv("Email__ImapHost") or os.getenv("IMAP_HOST", "mail.globalbrico.pt"))
    parser.add_argument("--porta", type=int, default=os.getenv("Email__ImapPort") or os.getenv("IMAP_PORT", "993"))
    parser.add_argument("--utilizador", default=os.getenv("Email__Username") or os.getenv("EMAIL", "encomendas@globalbrico.pt"))
    parser.add_argument("--limite", type=int, default=os.getenv("Email__MaxMessagesPerPoll", "10"))
    parser.add_argument("--pasta", default=os.getenv("IMAP_FOLDER", "INBOX"), help="Nome da pasta IMAP a extrair (por defeito: INBOX).")
    parser.add_argument("--timeout", type=int, default=os.getenv("Email__ConnectionTimeoutSeconds", "60"))
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "data/extracted_emails"))
    args = parser.parse_args()

    if not 1 <= args.porta <= 65535:
        parser.error("A porta deve estar entre 1 e 65535.")
    if args.limite < 0:
        parser.error("O limite nao pode ser negativo.")
    if not args.pasta.strip() or any(char in args.pasta for char in "\r\n\x00"):
        parser.error("E necessario indicar uma pasta IMAP valida.")
    if not args.host.strip() or not args.utilizador.strip():
        parser.error("E necessario indicar o servidor e o utilizador.")
    if not 1 <= args.timeout <= 600:
        parser.error("O timeout deve estar entre 1 e 600 segundos.")

    try:
        password = os.getenv("Email__Password") or os.getenv("EMAIL_PASSWORD") or getpass.getpass("Password do email: ")
        if not password:
            print("E necessario indicar a password.", file=sys.stderr)
            return 1

        extract_recent_emails(
            host=args.host,
            port=args.porta,
            username=args.utilizador,
            password=password,
            limit=args.limite,
            timeout=args.timeout,
            output_dir=args.output_dir,
            folder=args.pasta,
        )
        return 0

    except (KeyboardInterrupt, EOFError):
        print("\nOperacao cancelada.", file=sys.stderr)
        return 130
    except imaplib.IMAP4.error as error:
        message = error.args[0].decode("utf-8", errors="replace") if error.args and isinstance(error.args[0], bytes) else str(error)
        print(f"Erro IMAP: {message}", file=sys.stderr)
        return 1
    except ssl.SSLCertVerificationError:
        print("Falha na validacao do certificado TLS do servidor.", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Falha de ligacao ({type(error).__name__}): verificar host, porta e rede.", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
