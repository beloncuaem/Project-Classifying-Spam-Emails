"""Kiểm tra và lọc lỗi dataset spam email.

File này tạo ra:
- data/processed/combined_balanced_clean.csv
- reports/data_quality_report.json
- reports/data_quality_issues.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config


REQUIRED_COLUMNS = ["label", "text"]
STANDARD_COLUMNS = ["source", "file_name", "label", "label_name", "subject", "text"]


def normalize_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa schema tối thiểu trước khi kiểm tra lỗi."""
    for column in STANDARD_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[STANDARD_COLUMNS].copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["label_name"] = frame["label"].map({0: "ham", 1: "spam"}).fillna(frame["label_name"])
    return frame


def find_data_issues(frame: pd.DataFrame) -> pd.DataFrame:
    """Trả về bảng các dòng lỗi để dễ audit."""
    issues = []
    text_lengths = frame["text"].fillna("").astype(str).str.strip().str.len()
    invalid_label_mask = ~frame["label"].isin([0, 1])
    empty_text_mask = text_lengths == 0
    short_text_mask = text_lengths < config.MIN_TEXT_LENGTH
    duplicate_mask = frame.duplicated(subset=["label", "text"], keep="first")

    checks = {
        "invalid_label": invalid_label_mask,
        "empty_text": empty_text_mask,
        "short_text": short_text_mask,
        "duplicate_label_text": duplicate_mask,
    }
    for issue_name, mask in checks.items():
        bad_rows = frame.loc[mask, ["source", "file_name", "label", "label_name", "text"]].copy()
        bad_rows.insert(0, "issue", issue_name)
        bad_rows.insert(1, "row_index", bad_rows.index)
        issues.append(bad_rows)

    return pd.concat(issues, ignore_index=True) if issues else pd.DataFrame()


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Lọc dataset để giữ lại các dòng đủ điều kiện train."""
    clean = frame.copy()
    clean = clean[clean["label"].isin([0, 1])]
    clean = clean[clean["text"].str.len() >= config.MIN_TEXT_LENGTH]
    clean = clean.drop_duplicates(subset=["label", "text"], keep="first")
    clean["label"] = clean["label"].astype(int)
    clean["label_name"] = clean["label"].map({0: "ham", 1: "spam"})
    return clean.reset_index(drop=True)


def balance_clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Cân bằng lại sau khi lọc lỗi."""
    target_size = int(frame["label"].value_counts().min())
    balanced_parts = [
        group.sample(n=target_size, random_state=config.RANDOM_STATE)
        for _, group in frame.groupby("label")
    ]
    return (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=config.RANDOM_STATE)
        .reset_index(drop=True)
    )


def build_summary(raw: pd.DataFrame, clean: pd.DataFrame, issues: pd.DataFrame) -> dict:
    """Tổng hợp thống kê trước/sau lọc."""
    return {
        "input_path": str(config.COMBINED_BALANCED_PATH),
        "clean_output_path": str(config.COMBINED_BALANCED_CLEAN_PATH),
        "input_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "removed_rows": int(len(raw) - len(clean)),
        "input_label_counts": {str(k): int(v) for k, v in raw["label"].value_counts().sort_index().to_dict().items()},
        "clean_label_counts": {str(k): int(v) for k, v in clean["label"].value_counts().sort_index().to_dict().items()},
        "issue_counts": {str(k): int(v) for k, v in issues["issue"].value_counts().to_dict().items()} if not issues.empty else {},
        "required_columns_present": all(column in raw.columns for column in REQUIRED_COLUMNS),
    }


def clean_balanced_dataset(
    input_path: Path = config.COMBINED_BALANCED_PATH,
    output_path: Path = config.COMBINED_BALANCED_CLEAN_PATH,
) -> dict:
    """Đọc file balanced, lọc lỗi, cân bằng lại và lưu report."""
    frame = normalize_dataset(pd.read_csv(input_path))
    issues = find_data_issues(frame)
    clean = balance_clean_dataset(clean_dataset(frame))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False, encoding="utf-8")
    issues.to_csv(config.DATA_QUALITY_ISSUES_PATH, index=False, encoding="utf-8")

    summary = build_summary(frame, clean, issues)
    config.DATA_QUALITY_REPORT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    summary = clean_balanced_dataset()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
