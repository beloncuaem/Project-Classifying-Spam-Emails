from __future__ import annotations

from typing import Any


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
) -> dict[str, Any]:
    predicted_label = int(prediction_result["label"])
    cleaned_text = str(prediction_result.get("cleaned_text", ""))
    top_terms = explain_terms(model, cleaned_text, predicted_label)
    prediction = "spam" if predicted_label == 1 else "not spam"
    term_text = ", ".join(item["term"] for item in top_terms[:5])

    if top_terms:
        summary = (
            f"Email được dự đoán là {prediction} vì có các từ/cụm từ nổi bật "
            f"nghiêng về nhóm {prediction}: {term_text}."
        )
    else:
        summary = (
            f"Email được dự đoán là {prediction} dựa trên toàn bộ nội dung sau tiền xử lý. "
            "Không có từ/cụm từ đơn lẻ nào đủ nổi bật để hiển thị riêng."
        )

    return {
        "summary": summary,
        "top_terms": top_terms,
    }
