"""Đánh giá model phân loại email spam/not spam.

File này thuộc nhiệm vụ của Tuấn Tú:
- Tính accuracy, precision, recall, F1-score.
- Vẽ confusion matrix 2x2 theo thứ tự TN, FP, FN, TP.
- Xuất bảng so sánh nhiều model như NB, LR, SVM.
- Tập trung vào spam class và tỷ lệ false positive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import REPORTS_DIR


LABEL_NAMES = {0: "not spam", 1: "spam"}


def _to_int_labels(values) -> list[int]:
    """Chuẩn hóa nhãn đầu vào về dạng số: 0 = not spam, 1 = spam."""
    normalized = []
    for value in values:
        text = str(value).strip().lower()
        if text in {"1", "spam", "true", "malicious", "phish", "phishing"}:
            normalized.append(1)
        elif text in {"0", "ham", "not spam", "not_spam", "false", "normal"}:
            normalized.append(0)
        else:
            raise ValueError(f"Không nhận diện được label: {value!r}")
    return normalized


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Tính 4 ô confusion matrix: TN, FP, FN, TP."""
    true_labels = _to_int_labels(y_true)
    pred_labels = _to_int_labels(y_pred)
    if len(true_labels) != len(pred_labels):
        raise ValueError("y_true và y_pred phải có cùng số phần tử.")

    tn = fp = fn = tp = 0
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label == 0 and pred_label == 0:
            tn += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1
        elif true_label == 1 and pred_label == 1:
            tp += 1

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def safe_divide(numerator: float, denominator: float) -> float:
    """Chia an toàn để tránh lỗi chia cho 0 khi một class không xuất hiện."""
    return 0.0 if denominator == 0 else numerator / denominator


def calculate_metrics(y_true, y_pred, model_name: str = "model") -> dict[str, float | int | str]:
    """Tính các metric chính, ưu tiên precision/recall/F1 của class spam."""
    counts = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = counts["tn"], counts["fp"], counts["fn"], counts["tp"]
    total = tn + fp + fn + tp

    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    false_positive_rate = safe_divide(fp, fp + tn)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision_spam": precision,
        "recall_spam": recall,
        "f1_spam": f1_score,
        "false_positive_rate": false_positive_rate,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": total,
    }


def compare_models(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Tạo bảng so sánh model, sắp xếp theo F1 spam rồi recall spam."""
    table = pd.DataFrame(results)
    if table.empty:
        return table

    metric_columns = ["accuracy", "precision_spam", "recall_spam", "f1_spam", "false_positive_rate"]
    for column in metric_columns:
        if column in table.columns:
            table[column] = table[column].astype(float)

    sort_columns = [column for column in ["f1_spam", "recall_spam", "precision_spam"] if column in table.columns]
    return table.sort_values(sort_columns, ascending=False).reset_index(drop=True)


def save_confusion_matrix_heatmap(
    y_true,
    y_pred,
    output_path: Path,
    model_name: str = "model",
) -> Path:
    """Vẽ heatmap confusion matrix 2x2 và lưu thành ảnh PNG."""
    import matplotlib.pyplot as plt

    counts = confusion_counts(y_true, y_pred)
    matrix = [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["not spam", "spam"])
    ax.set_yticks([0, 1], labels=["not spam", "spam"])

    labels = [["TN", "FP"], ["FN", "TP"]]
    max_value = max(max(row) for row in matrix) or 1
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if value > max_value / 2 else "black"
            ax.text(
                col_index,
                row_index,
                f"{labels[row_index][col_index]}\n{value}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def evaluate_predictions(
    y_true,
    y_pred,
    model_name: str = "model",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Tính metric và tùy chọn lưu confusion matrix cho một model."""
    metrics = calculate_metrics(y_true, y_pred, model_name=model_name)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = output_dir / f"{model_name}_confusion_matrix.png"
        save_confusion_matrix_heatmap(y_true, y_pred, heatmap_path, model_name=model_name)
        metrics["confusion_matrix_path"] = str(heatmap_path)
    return metrics


def evaluate_model(
    model,
    x_test,
    y_test,
    model_name: str = "model",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Dự đoán bằng model đã train rồi tính metric."""
    y_pred = model.predict(x_test)
    return evaluate_predictions(y_test, y_pred, model_name=model_name, output_dir=output_dir)


def evaluate_from_predictions_csv(
    predictions_csv: Path,
    true_col: str,
    pred_cols: list[str],
    output_dir: Path,
) -> pd.DataFrame:
    """Đánh giá nhiều model từ file CSV có cột label thật và các cột dự đoán."""
    dataset = pd.read_csv(predictions_csv)
    if true_col not in dataset.columns:
        raise ValueError(f"Không tìm thấy cột label thật: {true_col}")

    results = []
    for pred_col in pred_cols:
        if pred_col not in dataset.columns:
            raise ValueError(f"Không tìm thấy cột dự đoán: {pred_col}")
        results.append(
            evaluate_predictions(
                dataset[true_col],
                dataset[pred_col],
                model_name=pred_col,
                output_dir=output_dir,
            )
        )

    comparison = compare_models(results)
    comparison_path = output_dir / "model_comparison.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False, encoding="utf-8")
    return comparison


def save_metrics_json(metrics: dict[str, Any] | list[dict[str, Any]], output_path: Path) -> Path:
    """Lưu metric ra JSON để đưa vào report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá model spam/not spam.")
    parser.add_argument("--predictions-csv", type=Path, help="CSV có label thật và cột dự đoán.")
    parser.add_argument("--true-col", default="label", help="Tên cột label thật trong CSV.")
    parser.add_argument(
        "--pred-col",
        action="append",
        dest="pred_cols",
        help="Tên cột dự đoán. Có thể truyền nhiều lần, ví dụ --pred-col nb --pred-col svm.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Thư mục lưu confusion matrix và bảng so sánh.",
    )
    args = parser.parse_args()

    if not args.predictions_csv:
        parser.print_help()
        return
    if not args.pred_cols:
        raise SystemExit("Cần truyền ít nhất một --pred-col.")

    comparison = evaluate_from_predictions_csv(
        predictions_csv=args.predictions_csv,
        true_col=args.true_col,
        pred_cols=args.pred_cols,
        output_dir=args.output_dir,
    )
    print(comparison.to_string(index=False))
    print(f"Đã lưu bảng so sánh tại: {args.output_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
