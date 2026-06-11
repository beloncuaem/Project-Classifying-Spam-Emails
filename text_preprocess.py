import re
from typing import Optional

from bs4 import BeautifulSoup
import pandas as pd

import nltk
from nltk.corpus import stopwords
import sys
import os
# Thêm thư mục cha (nơi chứa config.py) vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR, COMBINED_LABELED_PATH, MIN_TEXT_LENGTH


def _ensure_nltk_resources():
	try:
		stopwords.words("english")
	except LookupError:
		nltk.download("stopwords")


def remove_html_tags(text: str) -> str:
	"""
	Description: Loại bỏ HTML tags khỏi văn bản email.
	Input: Chuỗi (str)
	Output: Chuỗi đã được làm sạch
	"""
	if not isinstance(text, str):
		return ""
	return BeautifulSoup(text, "html.parser").get_text()


def normalize_entities(text: str) -> str:
	"""
	Description: Thay thế URL và email bằng token đặc biệt.
	Input: Chuỗi (str)
	Output: Chuỗi đã được thay thế
	"""
	if not isinstance(text, str):
		return ""
	# Replace URLs
	text = re.sub(r"http\S+|www\.\S+", "http_addr", text)
	# Replace emails (simple heuristic)
	text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "email_addr", text)
	return text


def clean_special_chars(text: str) -> str:
	"""
	Description: Chuyển về lowercase và loại bỏ số, ký tự đặc biệt.
	Input: Chuỗi (str)
	Output: Chuỗi đã được làm sạch (lowercase, chỉ giữ a-z và khoảng trắng)
	"""
	if not isinstance(text, str):
		return ""
	text = text.lower()
	# keep only lowercase letters and spaces
	text = re.sub(r"[^a-z\s]", " ", text)
	# collapse multiple spaces
	text = re.sub(r"\s+", " ", text).strip()
	return text


def remove_stopwords(text: str, lang: str = "english") -> str:
	"""
	Description: Loại bỏ stopwords bằng thư viện nltk.
	Input: Chuỗi (str), ngôn ngữ (str)
	Output: Chuỗi đã loại bỏ stopwords
	"""
	if not isinstance(text, str):
		return ""
	_ensure_nltk_resources()
	stop_words = set(stopwords.words(lang))
	tokens = [tok for tok in text.split() if tok and tok not in stop_words]
	return " ".join(tokens)


def process_text(text: str) -> str:
	"""
	Description: Áp dụng pipeline tiền xử lý lên một chuỗi văn bản.
	Input: Chuỗi (str)
	Output: Chuỗi đã được làm sạch và chuẩn hoá
	"""
	if not isinstance(text, str):
		return ""
	text = remove_html_tags(text)
	text = normalize_entities(text)
	text = clean_special_chars(text)
	text = remove_stopwords(text)
	return text


def process_dataframe(
	df: pd.DataFrame,
	text_column: str = "text",
	out_column: str = "clean_text",
	save_path: Optional[str] = None,
) -> pd.DataFrame:
	"""
	Description: Áp dụng pipeline lên cột văn bản của DataFrame và lưu kết quả.
	Input: DataFrame, tên cột văn bản, tên cột kết quả, đường dẫn lưu (optional)
	Output: DataFrame mới có cột kết quả đã được làm sạch
	"""
	if text_column not in df.columns:
		raise ValueError(f"Column '{text_column}' not found in DataFrame")

	df = df.copy()
	df[out_column] = df[text_column].astype(object).apply(process_text)

	# Filter short texts based on config
	try:
		min_len = int(MIN_TEXT_LENGTH)
	except Exception:
		min_len = 0

	df = df[df[out_column].str.len() >= min_len]

	# Save if requested
	if save_path is None:
		save_path = COMBINED_LABELED_PATH

	# Ensure directory exists
	try:
		PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
		df.to_csv(save_path, index=False)
	except Exception:
		# If saving fails, raise to make caller aware
		raise

	return df


def example_unit_test() -> None:
	"""
	Description: Một ví dụ nhỏ để kiểm tra nhanh các hàm tiền xử lý.
	Input: None
	Output: In ra kết quả mẫu
	"""
	sample = [
		"<html><body>Hi there! Visit https://example.com or mail me at test@example.com</body></html>",
		"FREE money!!! $$$ Click http://spam.io now",
	]
	df = pd.DataFrame({"text": sample})
	cleaned = process_dataframe(df, text_column="text", out_column="clean_text", save_path=PROCESSED_DATA_DIR / "example_processed.csv")
	print(cleaned)


__all__ = [
	"remove_html_tags",
	"normalize_entities",
	"clean_special_chars",
	"remove_stopwords",
	"process_text",
	"process_dataframe",
]


if __name__ == "__main__":
	# Quick interactive demo when module run directly
	print("Running example unit test for src/text_preprocess.py...")
	example_unit_test()

