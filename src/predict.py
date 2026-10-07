"""Dự đoán email mới bằng pipeline đã train.

File này thuộc nhiệm vụ của Tuấn Tú:
- Load pipeline `.joblib` từ thư mục models/.
- Tiền xử lý email mới bằng clean_email_text().
- Trả về nhãn `spam` hoặc `not spam`.
- Trả về xác suất/score spam trong khoảng 0-1 nếu model hỗ trợ.
- Demo 3-5 email mẫu, dự đoán từ file .txt và chế độ nhập tay.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import MODELS_DIR
from src.text_preprocess import clean_email_text


DEFAULT_MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"

DEMO_EMAILS = [
    "Congratulations! You won a free iPhone. Click this URL now to claim your prize.",
    "Hi team, please find attached the meeting notes from today and confirm your availability.",
    "Urgent account warning: verify your password immediately or your mailbox will be suspended.",
    "Can we move tomorrow's project discussion to 3 PM? I have another class in the morning.",
    "Cheap software offer, limited time discount, buy now and save 90 percent.",
]


def load_pipeline(model_path: Path = DEFAULT_MODEL_PATH):
    """Đọc pipeline/model `.joblib` đã lưu từ thư mục models/."""
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Thiếu thư viện joblib. Chạy: pip install joblib") from exc

    if not model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {model_path}. "
            "Hãy train model trước và lưu file .joblib vào thư mục models/."
        )
    return joblib.load(model_path)


def _score_from_decision_function(raw_score: float) -> float:
    """Đổi decision_function score về khoảng 0-1 bằng sigmoid."""
    return 1 / (1 + math.exp(-raw_score))


def get_spam_score(model, cleaned_text: str, predicted_label: int) -> float:
    """Lấy xác suất/score spam. Ưu tiên predict_proba, sau đó decision_function."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned_text])[0]
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if 1 in classes:
                return float(probabilities[classes.index(1)])
            if "spam" in classes:
                return float(probabilities[classes.index("spam")])
        return float(probabilities[-1])

    if hasattr(model, "decision_function"):
        raw_score = model.decision_function([cleaned_text])
        if hasattr(raw_score, "__len__"):
            raw_score = raw_score[0]
        return float(_score_from_decision_function(float(raw_score)))

    return float(predicted_label)


def normalize_prediction(value: Any) -> int:
    """Chuẩn hóa output model về 0 hoặc 1."""
    text = str(value).strip().lower()
    if text in {"1", "spam", "true", "malicious", "phish", "phishing"}:
        return 1
    if text in {"0", "ham", "not spam", "not_spam", "false", "normal"}:
        return 0
    raise ValueError(f"Không nhận diện được nhãn dự đoán: {value!r}")


def predict_email(text: str, model=None, model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Dự đoán một email text và trả về nhãn + spam_score."""
    if model is None:
        model = load_pipeline(model_path)

    cleaned_text = clean_email_text(text)
    predicted_raw = model.predict([cleaned_text])[0]
    predicted_label = normalize_prediction(predicted_raw)
    spam_score = get_spam_score(model, cleaned_text, predicted_label)

    return {
        "label": predicted_label,
        "prediction": "spam" if predicted_label == 1 else "not spam",
        "spam_score": round(float(spam_score), 6),
        "cleaned_text": cleaned_text,
    }


def predict_from_file(file_path: Path, model=None, model_path: Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Đọc email từ file .txt rồi dự đoán."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    result = predict_email(text, model=model, model_path=model_path)
    result["file_path"] = str(file_path)
    return result


def run_demo(model_path: Path = DEFAULT_MODEL_PATH) -> list[dict[str, Any]]:
    """Chạy demo 3-5 email mẫu và in kết quả."""
    model = load_pipeline(model_path)
    results = []
    for index, email_text in enumerate(DEMO_EMAILS, start=1):
        result = predict_email(email_text, model=model)
        result["demo_id"] = index
        result["input_text"] = email_text
        results.append(result)
        print(
            f"[Demo {index}] {result['prediction']} "
            f"(spam_score={result['spam_score']}) - {email_text[:80]}"
        )
    return results


def interactive_predict(model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """Chế độ nhập email từ bàn phím, nhập exit để thoát."""
    model = load_pipeline(model_path)
    print("Nhập nội dung email cần dự đoán. Gõ 'exit' để thoát.")
    while True:
        text = input("Email> ").strip()
        if text.lower() in {"exit", "quit", "q"}:
            break
        if not text:
            continue
        result = predict_email(text, model=model)
        print(f"Kết quả: {result['prediction']} | spam_score={result['spam_score']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dự đoán email spam/not spam bằng pipeline .joblib.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Đường dẫn file model .joblib.")
    parser.add_argument("--text", help="Nội dung email cần dự đoán.")
    parser.add_argument("--file", type=Path, help="Đường dẫn file .txt chứa email cần dự đoán.")
    parser.add_argument("--demo", action="store_true", help="Chạy demo với 5 email mẫu.")
    parser.add_argument("--interactive", action="store_true", help="Nhập email từ bàn phím để dự đoán.")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.model)
        return

    if args.interactive:
        interactive_predict(args.model)
        return

    model = load_pipeline(args.model)
    if args.file:
        result = predict_from_file(args.file, model=model)
    elif args.text:
        result = predict_email(args.text, model=model)
    else:
        parser.print_help()
        return

    print(f"Kết quả: {result['prediction']}")
    print(f"Spam score: {result['spam_score']}")


if __name__ == "__main__":
    main()
