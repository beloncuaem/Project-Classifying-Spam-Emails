from __future__ import annotations

from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from src.explain_prediction import build_prediction_evidence
from src.gmail_reader import gmail_setup_status, read_gmail_message
from src.predict import DEFAULT_MODEL_PATH, load_pipeline, predict_email


ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".eml"}


def read_uploaded_email(file_storage) -> str:
    filename = file_storage.filename or ""
    extension = Path(filename).suffix.lower()
    raw_bytes = file_storage.read()

    if extension == ".eml":
        message = BytesParser(policy=policy.default).parsebytes(raw_bytes)
        subject = str(message.get("subject", "")).strip()
        parts = []
        if message.is_multipart():
            for part in message.walk():
                if str(part.get_content_disposition() or "") == "attachment":
                    continue
                if part.get_content_type() in {"text/plain", "text/html"}:
                    try:
                        parts.append(part.get_content())
                    except Exception:
                        payload = part.get_payload(decode=True) or b""
                        parts.append(payload.decode("utf-8", errors="replace"))
        else:
            try:
                parts.append(message.get_content())
            except Exception:
                payload = message.get_payload(decode=True) or b""
                parts.append(payload.decode("utf-8", errors="replace"))
        return "\n\n".join(part for part in [subject, *parts] if part).strip()

    return raw_bytes.decode("utf-8", errors="replace")


def make_prediction_response(text: str, model, source: str = "manual") -> dict[str, Any]:
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("Vui lòng nhập nội dung email hoặc upload file có nội dung.")

    result = predict_email(clean_text, model=model)
    result["input_source"] = source
    result["evidence"] = build_prediction_evidence(model, result, model_path=DEFAULT_MODEL_PATH)
    return result


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    app.config["MODEL"] = load_pipeline(DEFAULT_MODEL_PATH)

    @app.get("/")
    def index():
        return render_template("spam_checker.html", gmail_status=gmail_setup_status())

    @app.post("/")
    def check_email_page():
        model = app.config["MODEL"]
        email_text = request.form.get("email_text", "")
        result = None
        error = None

        try:
            if request.files.get("email_file") and request.files["email_file"].filename:
                upload = request.files["email_file"]
                extension = Path(upload.filename or "").suffix.lower()
                if extension not in ALLOWED_UPLOAD_EXTENSIONS:
                    raise ValueError("Chỉ hỗ trợ upload file .txt hoặc .eml.")
                email_text = read_uploaded_email(upload)
                result = make_prediction_response(email_text, model, source=f"upload:{upload.filename}")
            else:
                result = make_prediction_response(email_text, model, source="manual")
        except Exception as exc:
            error = str(exc)

        return render_template(
            "spam_checker.html",
            email_text=email_text,
            result=result,
            error=error,
            gmail_status=gmail_setup_status(),
        )

    @app.post("/api/check")
    def check_email_api():
        model = app.config["MODEL"]
        email_text = ""
        source = "manual"

        try:
            if request.is_json:
                email_text = (request.get_json(silent=True) or {}).get("email_text", "")
            elif request.files.get("email_file"):
                upload = request.files["email_file"]
                extension = Path(upload.filename or "").suffix.lower()
                if extension not in ALLOWED_UPLOAD_EXTENSIONS:
                    raise ValueError("Chỉ hỗ trợ upload file .txt hoặc .eml.")
                email_text = read_uploaded_email(upload)
                source = f"upload:{upload.filename}"
            else:
                email_text = request.form.get("email_text", "")

            return jsonify(make_prediction_response(email_text, model, source=source))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/gmail/check")
    def check_gmail_page():
        model = app.config["MODEL"]
        gmail_input = request.form.get("gmail_message", "")
        result = None
        error = None

        try:
            gmail_message = read_gmail_message(gmail_input)
            result = make_prediction_response(gmail_message["text"], model, source=f"gmail:{gmail_message['message_id']}")
        except Exception as exc:
            error = str(exc)

        return render_template(
            "spam_checker.html",
            gmail_input=gmail_input,
            result=result,
            error=error,
            gmail_status=gmail_setup_status(),
        )

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "model_path": str(DEFAULT_MODEL_PATH),
                "model_exists": DEFAULT_MODEL_PATH.exists(),
                "gmail": gmail_setup_status(),
            }
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
