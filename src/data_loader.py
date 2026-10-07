import argparse
import html
import json
import re
import sys
import tarfile
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (
    COMBINED_BALANCED_PATH,
    COMBINED_LABELED_PATH,
    DATASET_REPORT_PATH,
    HUGGINGFACE_CSV_SOURCES,
    HUGGINGFACE_PARQUET_DATASETS,
    MIN_TEXT_LENGTH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    SPAMASSASSIN_ARCHIVES,
    SPAMASSASSIN_BALANCED_PATH,
    SPAMASSASSIN_BASE_URL,
    SPAMASSASSIN_LABELED_PATH,
)


STANDARD_COLUMNS = ["source", "file_name", "label", "label_name", "subject", "text"]


def ensure_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def download_file(url: str, output_path: Path, force: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        print(f"Already exists: {output_path}")
        return output_path
    print(f"Downloading {url}")
    urlretrieve(url, output_path)
    return output_path


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        member_path = (destination / member.name).resolve()
        if not str(member_path).startswith(str(destination)):
            raise RuntimeError(f"Blocked unsafe archive member: {member.name}")
    tar.extractall(destination)


def extract_archive(archive_path: Path, force: bool = False) -> Path:
    extract_dir = RAW_DATA_DIR / archive_path.stem.replace(".tar", "")
    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:bz2") as tar:
        safe_extract(tar, extract_dir)
    return extract_dir


def decode_payload(part) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def parse_email_file(path: Path) -> tuple[str, str]:
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    subject = str(message.get("subject", "")).strip()

    body_parts = []
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            disposition = str(part.get_content_disposition() or "")
            if disposition == "attachment":
                continue
            if content_type in {"text/plain", "text/html"}:
                body_parts.append(decode_payload(part))
    else:
        body_parts.append(decode_payload(message))

    return subject, clean_text(" ".join([subject, *body_parts]))


def iter_email_paths(extract_dir: Path):
    for path in extract_dir.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            yield path


def load_spamassassin(force_download: bool = False, force_extract: bool = False) -> pd.DataFrame:
    rows = []
    for source in SPAMASSASSIN_ARCHIVES:
        archive_path = download_file(
            f"{SPAMASSASSIN_BASE_URL}/{source['file_name']}",
            RAW_DATA_DIR / source["file_name"],
            force=force_download,
        )
        extract_dir = extract_archive(archive_path, force=force_extract)

        for email_path in iter_email_paths(extract_dir):
            try:
                subject, text = parse_email_file(email_path)
            except Exception as exc:
                print(f"Skipping unreadable file {email_path}: {exc}")
                continue
            rows.append(
                {
                    "source": source["name"],
                    "file_name": email_path.name,
                    "label": source["label"],
                    "label_name": "spam" if source["label"] == 1 else "ham",
                    "subject": subject,
                    "text": text,
                }
            )

    frame = pd.DataFrame(rows)
    frame = normalize_frame(frame).drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
    frame.to_csv(SPAMASSASSIN_LABELED_PATH, index=False, encoding="utf-8")
    return frame


def normalize_label(value) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "spam", "spams", "junk", "true"}:
        return 1
    if text in {"0", "ham", "not spam", "not_spam", "non-spam", "nonspam", "legit", "false"}:
        return 0
    return None


def normalize_column_name(column: str) -> str:
    return str(column).strip().lower().replace(" ", "_")


def pick_column(columns, candidates):
    lower_map = {normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def load_huggingface_csv_sources(force_download: bool = False) -> pd.DataFrame:
    frames = []
    manual_dir = RAW_DATA_DIR / "manual_csv"
    for source in HUGGINGFACE_CSV_SOURCES:
        path = download_file(source["url"], manual_dir / source["file_name"], force=force_download)
        dataset = pd.read_csv(path)
        label_col = pick_column(dataset.columns, ["label", "label_name", "category", "class", "target", "spam/ham"])
        text_col = pick_column(dataset.columns, ["text", "email_text", "message", "body", "origin", "email", "content"])
        subject_col = pick_column(dataset.columns, ["subject", "title"])
        if label_col is None or text_col is None:
            raise ValueError(f"Cannot detect label/text columns in {path}")

        labels = dataset[label_col].map(normalize_label)
        text = dataset[text_col].fillna("").astype(str).str.strip()
        subject = ""
        if subject_col is not None:
            subject = dataset[subject_col].fillna("").astype(str).str.strip()
            text = (subject + " " + text).str.strip()
        frames.append(
            pd.DataFrame(
                {
                    "source": source["source"],
                    "file_name": path.name,
                    "label": labels,
                    "label_name": labels.map({0: "ham", 1: "spam"}),
                    "subject": subject if subject_col is not None else "",
                    "text": text,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STANDARD_COLUMNS)


def download_huggingface_parquet_sources(force_download: bool = False) -> list[tuple[str, Path]]:
    downloaded = []
    parquet_root = RAW_DATA_DIR / "huggingface_parquet"
    parquet_root.mkdir(parents=True, exist_ok=True)

    for source in HUGGINGFACE_PARQUET_DATASETS:
        api_url = f"https://datasets-server.huggingface.co/parquet?dataset={source['dataset']}"
        print(f"Reading parquet file list: {api_url}")
        with urlopen(api_url, timeout=60) as response:
            payload = json.load(response)

        output_dir = parquet_root / source["source"]
        for parquet_file in payload.get("parquet_files", []):
            path = download_file(
                parquet_file["url"],
                output_dir / parquet_file["filename"],
                force=force_download,
            )
            downloaded.append((source["source"], path))
    return downloaded


def load_huggingface_parquet_sources(force_download: bool = False) -> pd.DataFrame:
    frames = []
    for source_name, path in download_huggingface_parquet_sources(force_download=force_download):
        print(f"Importing Hugging Face parquet: {path}")
        dataset = pd.read_parquet(path, columns=["label", "text"])
        original_labels = pd.to_numeric(dataset["label"], errors="coerce")

        # locuoco source labels: 0 = ham, 1 = phish, 2 = spam.
        # Binary target for this project: ham = 0; phish/spam = 1.
        labels = original_labels.map(lambda value: 0 if value == 0 else 1 if value in {1, 2} else None)
        original_label_name = original_labels.map({0: "ham", 1: "phish", 2: "spam"})
        frames.append(
            pd.DataFrame(
                {
                    "source": source_name,
                    "file_name": path.name,
                    "label": labels,
                    "label_name": labels.map({0: "ham", 1: "spam"}),
                    "subject": original_label_name.fillna(""),
                    "text": dataset["text"].fillna("").astype(str).str.strip(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STANDARD_COLUMNS)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in STANDARD_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[STANDARD_COLUMNS].copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["label"])
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame["label"].isin([0, 1])]
    frame["label_name"] = frame["label"].map({0: "ham", 1: "spam"})
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame = frame[frame["text"] != ""]
    return frame


def balance_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    label_counts = frame["label"].value_counts()
    if len(label_counts) != 2:
        raise ValueError("Dataset must contain both labels: 0 = ham and 1 = spam.")

    target_size = int(label_counts.min())
    balanced_parts = [
        group.sample(n=target_size, random_state=RANDOM_STATE).reset_index(drop=True)
        for _, group in frame.groupby("label")
    ]
    return (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def validate_dataset(frame: pd.DataFrame) -> dict:
    text = frame["text"].fillna("").astype(str).str.strip()
    labels = pd.to_numeric(frame["label"], errors="coerce")
    return {
        "row_count": int(len(frame)),
        "label_counts": {str(k): int(v) for k, v in labels.value_counts().sort_index().to_dict().items()},
        "invalid_label_count": int((~labels.isin([0, 1])).sum()),
        "empty_text_count": int((text == "").sum()),
        "short_text_count": int((text.str.len() < MIN_TEXT_LENGTH).sum()),
        "duplicate_text_count": int(text.duplicated().sum()),
        "is_valid": bool((~labels.isin([0, 1])).sum() == 0 and (text == "").sum() == 0),
    }


def run_data_pipeline(force_download: bool = False) -> None:
    ensure_directories()

    spamassassin = load_spamassassin(force_download=force_download)
    spamassassin_balanced = balance_dataset(spamassassin)
    spamassassin_balanced.to_csv(SPAMASSASSIN_BALANCED_PATH, index=False, encoding="utf-8")

    extra_csv = load_huggingface_csv_sources(force_download=force_download)
    extra_parquet = load_huggingface_parquet_sources(force_download=force_download)
    combined = pd.concat(
        [normalize_frame(frame) for frame in [spamassassin, extra_csv, extra_parquet]],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
    combined.to_csv(COMBINED_LABELED_PATH, index=False, encoding="utf-8")

    combined_balanced = balance_dataset(combined)
    combined_balanced.to_csv(COMBINED_BALANCED_PATH, index=False, encoding="utf-8")

    report = {
        "spamassassin_labeled": validate_dataset(spamassassin),
        "spamassassin_balanced": validate_dataset(spamassassin_balanced),
        "combined_labeled": validate_dataset(combined),
        "combined_balanced": validate_dataset(combined_balanced),
        "source_counts": {str(k): int(v) for k, v in combined["source"].value_counts().to_dict().items()},
        "paths": {
            "spamassassin_labeled": str(SPAMASSASSIN_LABELED_PATH),
            "spamassassin_balanced": str(SPAMASSASSIN_BALANCED_PATH),
            "combined_labeled": str(COMBINED_LABELED_PATH),
            "combined_balanced": str(COMBINED_BALANCED_PATH),
        },
        "label_note": "For locuoco dataset, original labels 1=phish and 2=spam are mapped to binary label 1.",
    }
    DATASET_REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Combined labeled:")
    print(combined["label_name"].value_counts().to_string())
    print("Combined balanced:")
    print(combined_balanced["label_name"].value_counts().to_string())
    print(f"Report: {DATASET_REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, label, validate, combine, and balance email data.")
    parser.add_argument("--force-download", action="store_true", help="Download source files again.")
    args = parser.parse_args()
    run_data_pipeline(force_download=args.force_download)


if __name__ == "__main__":
    main()
