import argparse
import getpass
import imaplib
import os
import ssl
import sys
from pathlib import Path

#Carrega variaveis de ambiente do .env automaticamente
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


def connect_inbox(host, port, username, password, timeout=60):
    context = ssl.create_default_context()
    print(f"A ligar a {host}:{port} por IMAP/TLS...", flush=True)

    client = imaplib.IMAP4_SSL(host, port, ssl_context=context, timeout=timeout)
    try:
        client.login(username, password)
        print("Autenticacao efetuada.", flush=True)
        status, _ = client.select("INBOX", readonly=True)
        if status != "OK":
            raise RuntimeError("Nao foi possivel abrir a INBOX.")

        print(f"Conexao IMAP estabelecida com sucesso para {username} (INBOX).", flush=True)
        return client
    except Exception:
        try:
            client.logout()
        except Exception:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description="Estabelecer ligacao IMAP/TLS a uma caixa de entrada.")

    try:
        load_env(Path(__file__).resolve().parent.parent / ".env")
    except (OSError, ValueError):
        parser.error("Nao foi possivel carregar o .env. Verificar acesso e formato KEY=VALUE.")

    parser.add_argument("--host", default=os.getenv("Email__ImapHost") or os.getenv("IMAP_HOST", "mail.globalbrico.pt"))
    parser.add_argument("--porta", type=int, default=os.getenv("Email__ImapPort") or os.getenv("IMAP_PORT", "993"))
    parser.add_argument("--utilizador", default=os.getenv("Email__Username") or os.getenv("EMAIL", "encomendas@globalbrico.pt"))
    parser.add_argument("--timeout", type=int, default=os.getenv("Email__ConnectionTimeoutSeconds", "60"))
    args = parser.parse_args()

    if not 1 <= args.porta <= 65535:
        parser.error("A porta deve estar entre 1 e 65535.")
    if not args.host.strip() or not args.utilizador.strip():
        parser.error("E necessario indicar o servidor e o utilizador.")
    if not 1 <= args.timeout <= 600:
        parser.error("O timeout deve estar entre 1 e 600 segundos.")

    try:
        password = os.getenv("Email__Password") or os.getenv("EMAIL_PASSWORD") or getpass.getpass("Password do email: ")
        if not password:
            print("E necessario indicar a password.", file=sys.stderr)
            return 1

        client = connect_inbox(args.host, args.porta, args.utilizador, password, args.timeout)
        client.logout()
        print("Ligacao terminada.")
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
