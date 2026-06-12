"""Tiền xử lý văn bản email.

Người thực hiện: Sinh Trường.

Module này gom các bước làm sạch text trước khi đưa email vào model:
- Loại HTML/script/style.
- Chuẩn hóa URL và địa chỉ email thành token chung.
- Chuyển chữ thường, bỏ ký tự đặc biệt, gom khoảng trắng.
- Loại stopword tiếng Anh nếu tài nguyên NLTK có sẵn.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - requirements.txt đã có beautifulsoup4.
    BeautifulSoup = None

try:
    from nltk.corpus import stopwords
except ImportError:  # pragma: no cover - requirements.txt đã có nltk.
    stopwords = None


FALLBACK_ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
    "your",
}


def remove_html_tags(text: str) -> str:
    """Loại bỏ HTML tags, script/style và decode HTML entities."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    if BeautifulSoup is not None:
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    else:
        text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def normalize_entities(text: str) -> str:
    """Thay URL, email và số bằng token chung để giảm nhiễu."""
    if not isinstance(text, str):
        return ""

    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text, flags=re.IGNORECASE)
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", " emailtoken ", text)
    text = re.sub(r"\d+", " numbertoken ", text)
    return text


def clean_special_chars(text: str) -> str:
    """Chuyển lowercase, chỉ giữ chữ cái a-z và khoảng trắng."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_stopwords(language: str = "english") -> set[str]:
    """Lấy stopword từ NLTK, fallback sang bộ nhỏ nếu corpus chưa tải."""
    if stopwords is None:
        return set(FALLBACK_ENGLISH_STOPWORDS)

    try:
        return set(stopwords.words(language))
    except LookupError:
        try:
            import nltk

            nltk.download("stopwords", quiet=True)
            return set(stopwords.words(language))
        except Exception:
            return set(FALLBACK_ENGLISH_STOPWORDS)


def remove_stopwords(text: str, language: str = "english", extra_stopwords: Iterable[str] | None = None) -> str:
    """Loại bỏ stopwords để giảm các từ ít giá trị phân loại."""
    if not isinstance(text, str):
        return ""

    stop_words = get_stopwords(language)
    if extra_stopwords:
        stop_words.update(word.lower() for word in extra_stopwords)

    tokens = [token for token in text.split() if token and token not in stop_words]
    return " ".join(tokens)


def process_text(text: str, remove_common_words: bool = True) -> str:
    """Áp dụng toàn bộ pipeline tiền xử lý cho một email."""
    if not isinstance(text, str):
        return ""

    text = remove_html_tags(text)
    text = normalize_entities(text)
    text = clean_special_chars(text)
    if remove_common_words:
        text = remove_stopwords(text)
    return text


def clean_email_text(text: str) -> str:
    """Entry point dùng trong `src/predict.py` khi dự đoán email mới."""
    return process_text(text)


def process_dataframe(
    frame: pd.DataFrame,
    text_column: str = "text",
    output_column: str = "clean_text",
    save_path: str | Path | None = None,
    min_text_length: int = 0,
) -> pd.DataFrame:
    """Làm sạch một cột text trong DataFrame và tùy chọn lưu ra CSV."""
    if text_column not in frame.columns:
        raise ValueError(f"Không tìm thấy cột text: {text_column}")

    processed = frame.copy()
    processed[output_column] = processed[text_column].fillna("").astype(str).map(clean_email_text)

    if min_text_length > 0:
        processed = processed[processed[output_column].str.len() >= min_text_length].reset_index(drop=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        processed.to_csv(save_path, index=False, encoding="utf-8")

    return processed


def clean_data(frame: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Alias tương thích nếu module khác gọi `clean_data()`."""
    return process_dataframe(frame, text_column=text_column, output_column="clean_text")


def example_unit_test() -> None:
    """Demo nhanh khi chạy trực tiếp file này."""
    samples = [
        "<html><body>Hi team! Visit https://example.com or email test@example.com</body></html>",
        "FREE money!!! Click http://spam.example now and claim prize 1000",
    ]
    frame = pd.DataFrame({"text": samples})
    print(process_dataframe(frame))


__all__ = [
    "remove_html_tags",
    "normalize_entities",
    "clean_special_chars",
    "get_stopwords",
    "remove_stopwords",
    "process_text",
    "clean_email_text",
    "process_dataframe",
    "clean_data",
]


if __name__ == "__main__":
    example_unit_test()
