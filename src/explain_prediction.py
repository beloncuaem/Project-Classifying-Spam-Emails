from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import config


def load_training_report(report_path: Path = config.REPORTS_DIR / "model_train_report.json") -> dict[str, Any]:
    if not report_path.exists():
        return {}
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_model_metrics(
    model_name: str | None = None,
    metrics_path: Path = config.REPORTS_DIR / "model_metrics.csv",
) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}

    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return {}

    row = None
    if model_name and "model" in metrics.columns:
        matched = metrics[metrics["model"] == model_name]
        if not matched.empty:
            row = matched.iloc[0]

    if row is None:
        sort_columns = [column for column in ["f1_spam", "recall_spam", "precision_spam"] if column in metrics.columns]
        row = metrics.sort_values(sort_columns, ascending=False).iloc[0] if sort_columns else metrics.iloc[0]

    result = row.to_dict()
    for key, value in list(result.items()):
        if hasattr(value, "item"):
            result[key] = value.item()
    return result


def _get_pipeline_parts(model) -> tuple[Any | None, Any | None]:
    if not hasattr(model, "named_steps"):
        return None, None
    vectorizer = model.named_steps.get("tfidf")
    estimator = model.named_steps.get("model")
    return vectorizer, estimator


def _feature_names(vectorizer) -> list[str]:
    try:
        return list(vectorizer.get_feature_names_out())
    except Exception:
        return []


def explain_terms(model, cleaned_text: str, predicted_label: int, limit: int = 8) -> list[dict[str, Any]]:
    vectorizer, estimator = _get_pipeline_parts(model)
    if vectorizer is None or estimator is None:
        return []

    feature_names = _feature_names(vectorizer)
    if not feature_names:
        return []

    matrix = vectorizer.transform([cleaned_text])
    indices = matrix.nonzero()[1]
    if len(indices) == 0:
        return []

    if hasattr(estimator, "coef_"):
        coefficients = estimator.coef_[0]
        contributions = []
        for index in indices:
            score = float(matrix[0, index] * coefficients[index])
            if predicted_label == 1 and score > 0:
                direction = "spam"
            elif predicted_label == 0 and score < 0:
                direction = "not spam"
            else:
                direction = "opposite"
            contributions.append((feature_names[index], score, direction))

        if predicted_label == 1:
            contributions.sort(key=lambda item: item[1], reverse=True)
        else:
            contributions.sort(key=lambda item: item[1])

    elif hasattr(estimator, "feature_log_prob_"):
        spam_delta = estimator.feature_log_prob_[1] - estimator.feature_log_prob_[0]
        contributions = []
        for index in indices:
            score = float(matrix[0, index] * spam_delta[index])
            if predicted_label == 1 and score > 0:
                direction = "spam"
            elif predicted_label == 0 and score < 0:
                direction = "not spam"
            else:
                direction = "opposite"
            contributions.append((feature_names[index], score, direction))

        if predicted_label == 1:
            contributions.sort(key=lambda item: item[1], reverse=True)
        else:
            contributions.sort(key=lambda item: item[1])
    else:
        return []

    selected = [item for item in contributions if item[2] != "opposite"][:limit]
    if not selected:
        selected = contributions[:limit]

    return [
        {
            "term": term,
            "weight": round(score, 6),
            "supports": direction,
        }
        for term, score, direction in selected
    ]


def build_prediction_evidence(
    model,
    prediction_result: dict[str, Any],
    model_path: Path = config.MODELS_DIR / "spam_classifier.joblib",
) -> dict[str, Any]:
    train_report = load_training_report()
    model_name = str(train_report.get("best_model") or prediction_result.get("model") or "spam_classifier")
    metrics = load_model_metrics(model_name=model_name)
    predicted_label = int(prediction_result["label"])
    cleaned_text = str(prediction_result.get("cleaned_text", ""))

    return {
        "model_path": str(model_path),
        "model_name": model_name,
        "model_metric": metrics,
        "source_files": [
            "models/spam_classifier.joblib",
            "src/predict.py",
            "reports/model_metrics.csv",
        ],
        "decision_basis": (
            "Kết quả được lấy từ pipeline TF-IDF + model đã train. "
            "Các term bên dưới là feature xuất hiện trong email và có trọng số ảnh hưởng đến class dự đoán."
        ),
        "top_terms": explain_terms(model, cleaned_text, predicted_label),
    }
