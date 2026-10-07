"""Train các model spam email theo format chung của project.

Model được lưu dưới dạng sklearn Pipeline `.joblib`, gồm:
- TfidfVectorizer
- Bộ phân loại NB/LR/SVM

File `models/spam_classifier.joblib` là model tốt nhất để `src/predict.py` load trực tiếp.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.append(str(Path(__file__).resolve().parents[1]))

import config


MODEL_DEFINITIONS = {
    "naive_bayes": MultinomialNB(alpha=0.5),
    "logistic_regression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=config.RANDOM_STATE,
    ),
    "linear_svm": LinearSVC(C=1.0, random_state=config.RANDOM_STATE),
}


def load_training_data(data_path: Path = config.COMBINED_BALANCED_CLEAN_PATH) -> pd.DataFrame:
    """Đọc dataset clean để train model."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file train: {data_path}. "
            "Hãy chạy: python src/data_quality.py"
        )
    dataset = pd.read_csv(data_path)
    dataset = dataset[dataset["label"].isin([0, 1])]
    dataset["text"] = dataset["text"].fillna("").astype(str)
    dataset["label"] = dataset["label"].astype(int)
    return dataset


def make_pipeline(model) -> Pipeline:
    """Tạo pipeline TF-IDF + model."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.MAX_FEATURES,
                    ngram_range=config.NGRAM_RANGE,
                    stop_words="english",
                    lowercase=True,
                ),
            ),
            ("model", model),
        ]
    )


def calculate_model_metrics(y_true, y_pred, model_name: str) -> dict[str, Any]:
    """Tính metrics chính cho class spam."""
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_spam": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_spam": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_spam": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_models(dataset: pd.DataFrame | None = None, sample_size: int | None = None):
    """Train NB, LR, SVM và trả về model tốt nhất cùng bảng metrics."""
    if dataset is None:
        dataset = load_training_data()

    label_counts = dataset["label"].value_counts()
    if len(label_counts) < 2:
        raise ValueError("Dataset cần có đủ 2 class label: 0 = not spam, 1 = spam.")

    if sample_size and len(dataset) > sample_size:
        samples_per_class = max(1, sample_size // len(label_counts))
        sampled_parts = []
        for _, group in dataset.groupby("label"):
            n_samples = min(len(group), samples_per_class)
            sampled_parts.append(group.sample(n=n_samples, random_state=config.RANDOM_STATE))
        dataset = pd.concat(sampled_parts, ignore_index=True)
        dataset = dataset.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    x_train, x_test, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=dataset["label"],
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    trained_models = {}
    metrics = []
    prediction_frame = pd.DataFrame({"label": y_test.reset_index(drop=True)})

    for model_name, estimator in MODEL_DEFINITIONS.items():
        print(f"[INFO] Training {model_name}...")
        pipeline = make_pipeline(estimator)
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        trained_models[model_name] = pipeline
        metrics.append(calculate_model_metrics(y_test, y_pred, model_name))
        prediction_frame[model_name] = y_pred
        joblib.dump(pipeline, config.MODELS_DIR / f"{model_name}_pipeline.joblib")

    metrics_table = pd.DataFrame(metrics).sort_values(["f1_spam", "recall_spam"], ascending=False)
    best_model_name = str(metrics_table.iloc[0]["model"])
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, config.MODELS_DIR / "spam_classifier.joblib")

    metrics_path = config.REPORTS_DIR / "model_metrics.csv"
    predictions_path = config.REPORTS_DIR / "model_predictions.csv"
    report_path = config.REPORTS_DIR / "model_train_report.json"
    metrics_table.to_csv(metrics_path, index=False, encoding="utf-8")
    prediction_frame.to_csv(predictions_path, index=False, encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {
                "best_model": best_model_name,
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
                "metrics_path": str(metrics_path),
                "predictions_path": str(predictions_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return best_model_name, best_model, metrics_table, prediction_frame


def run_modeling_pipeline(sample_size: int | None = None) -> pd.DataFrame:
    """Entry point dùng cho main.py hoặc notebook."""
    best_model_name, _, metrics_table, _ = train_models(sample_size=sample_size)
    print("\n=== Model comparison ===")
    print(metrics_table.to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    return metrics_table


if __name__ == "__main__":
    run_modeling_pipeline()
