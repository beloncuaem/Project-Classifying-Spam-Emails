from __future__ import annotations

import base64
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"
DEFAULT_TOKEN_PATH = PROJECT_ROOT / "token.json"


def extract_message_hint(value: str) -> str:
    text = value.strip()
    if not text:
        return ""

    for pattern in [r"[?&]th=([^&/#]+)", r"/([a-fA-F0-9]{12,})\b", r"#(?:inbox|all|sent)/([^/?#]+)"]:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return text


def gmail_setup_status(credentials_path: Path = DEFAULT_CREDENTIALS_PATH) -> dict[str, Any]:
    missing_packages = []
    for package_name in ["googleapiclient", "google_auth_oauthlib", "google.auth.transport.requests"]:
        try:
            __import__(package_name)
        except ImportError:
            missing_packages.append(package_name)

    return {
        "ready": credentials_path.exists() and not missing_packages,
        "credentials_path": str(credentials_path),
        "credentials_exists": credentials_path.exists(),
        "missing_packages": missing_packages,
        "note": (
            "Gmail cần OAuth. Hãy tạo OAuth Client cho desktop app, tải credentials.json "
            "vào thư mục root repo và cài google-api-python-client google-auth-httplib2 google-auth-oauthlib."
        ),
    }


def _message_to_text(raw_bytes: bytes) -> str:
    message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = str(message.get("subject", "")).strip()
    body_parts = []

    if message.is_multipart():
        for part in message.walk():
            disposition = str(part.get_content_disposition() or "")
            if disposition == "attachment":
                continue
            if part.get_content_type() in {"text/plain", "text/html"}:
                try:
                    body_parts.append(part.get_content())
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    body_parts.append(payload.decode("utf-8", errors="replace"))
    else:
        try:
            body_parts.append(message.get_content())
        except Exception:
            payload = message.get_payload(decode=True) or b""
            body_parts.append(payload.decode("utf-8", errors="replace"))

    return "\n\n".join(part for part in [subject, *body_parts] if part).strip()


def read_gmail_message(
    message_id_or_link: str,
    credentials_path: Path = DEFAULT_CREDENTIALS_PATH,
    token_path: Path = DEFAULT_TOKEN_PATH,
) -> dict[str, Any]:
    status = gmail_setup_status(credentials_path)
    if not status["ready"]:
        raise RuntimeError(status["note"])

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            credentials = flow.run_local_server(port=0)
        token_path.write_text(credentials.to_json(), encoding="utf-8")

    service = build("gmail", "v1", credentials=credentials)
    message_id = extract_message_hint(message_id_or_link)
    payload = service.users().messages().get(userId="me", id=message_id, format="raw").execute()
    raw_message = base64.urlsafe_b64decode(payload["raw"].encode("utf-8"))
    return {
        "message_id": message_id,
        "text": _message_to_text(raw_message),
    }
