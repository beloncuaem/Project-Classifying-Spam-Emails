"""Test nhanh các phần đã sửa trong project spam email.

Chạy toàn bộ test:
    python -m unittest discover -s tests -v

Chạy riêng file này:
    python -m unittest tests.test_project_pipeline -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

import config
from src.data_quality import balance_clean_dataset, clean_dataset, find_data_issues, normalize_dataset
from src.model_evaluate import calculate_metrics, confusion_counts
from src.model_train import MODEL_DEFINITIONS, train_models
from src.predict import predict_email


def build_small_training_dataset(repeat: int = 24) -> pd.DataFrame:
    """Tạo dataset nhỏ nhưng tách biệt rõ spam/ham để test train nhanh."""
    ham_texts = [
        f"team meeting schedule project update report calendar discussion normal work email {index}"
        for index in range(repeat)
    ]
    spam_texts = [
        f"winner prize free money urgent click claim bonus discount lottery offer {index}"
        for index in range(repeat)
    ]
    return pd.DataFrame(
        {
            "text": ham_texts + spam_texts,
            "label": [0] * repeat + [1] * repeat,
        }
    )


class DataQualityTest(unittest.TestCase):
    def test_clean_dataset_filters_invalid_short_and_duplicate_rows(self) -> None:
        raw = pd.DataFrame(
            {
                "label": [0, 1, 1, 2, 0, 1],
                "text": [
                    "normal team email about schedule and project planning",
                    "urgent free prize click now to claim your money",
                    "urgent free prize click now to claim your money",
                    "invalid label but long enough text for checking",
                    "",
                    "short",
                ],
            }
        )

        normalized = normalize_dataset(raw)
        issues = find_data_issues(normalized)
        clean = clean_dataset(normalized)

        self.assertIn("invalid_label", set(issues["issue"]))
        self.assertIn("short_text", set(issues["issue"]))
        self.assertIn("duplicate_label_text", set(issues["issue"]))
        self.assertEqual(len(clean), 2)
        self.assertEqual(set(clean["label"]), {0, 1})

    def test_balance_clean_dataset_keeps_two_classes_equal(self) -> None:
        frame = pd.DataFrame(
            {
                "label": [0, 0, 0, 1, 1],
                "text": [
                    "ham email one has enough words for testing",
                    "ham email two has enough words for testing",
                    "ham email three has enough words for testing",
                    "spam email one has enough words for testing",
                    "spam email two has enough words for testing",
                ],
            }
        )

        balanced = balance_clean_dataset(frame)
        self.assertEqual(balanced["label"].value_counts().to_dict(), {0: 2, 1: 2})


class ModelEvaluateTest(unittest.TestCase):
    def test_confusion_counts_and_metrics_focus_on_spam_class(self) -> None:
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 1, 1, 0, 1]

        counts = confusion_counts(y_true, y_pred)
        metrics = calculate_metrics(y_true, y_pred, model_name="unit_test_model")

        self.assertEqual(counts, {"tn": 1, "fp": 1, "fn": 1, "tp": 2})
        self.assertAlmostEqual(metrics["accuracy"], 0.6)
        self.assertAlmostEqual(metrics["precision_spam"], 2 / 3)
        self.assertAlmostEqual(metrics["recall_spam"], 2 / 3)
        self.assertAlmostEqual(metrics["f1_spam"], 2 / 3)


class ModelTrainTest(unittest.TestCase):
    def test_train_models_saves_pipeline_and_returns_valid_metrics(self) -> None:
        dataset = build_small_training_dataset()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            with patch.object(config, "MODELS_DIR", temp_path / "models"), patch.object(
                config, "REPORTS_DIR", temp_path / "reports"
            ):
                best_model_name, best_model, metrics_table, predictions = train_models(dataset=dataset)

            self.assertIn(best_model_name, MODEL_DEFINITIONS)
            self.assertTrue((temp_path / "models" / "spam_classifier.joblib").exists())
            self.assertTrue(hasattr(best_model, "predict"))
            self.assertEqual(set(metrics_table["model"]), set(MODEL_DEFINITIONS))
            self.assertEqual(set(predictions.columns), {"label", *MODEL_DEFINITIONS.keys()})

            metric_columns = ["accuracy", "precision_spam", "recall_spam", "f1_spam"]
            for column in metric_columns:
                self.assertTrue(metrics_table[column].between(0, 1).all(), f"Metric ngoài khoảng 0-1: {column}")


class TrainedArtifactQualityTest(unittest.TestCase):
    MIN_F1_SPAM = 0.90
    MIN_RECALL_SPAM = 0.90

    def test_trained_reports_show_model_quality_above_threshold(self) -> None:
        metrics_path = config.REPORTS_DIR / "model_metrics.csv"
        self.assertTrue(metrics_path.exists(), "Thiếu reports/model_metrics.csv. Hãy chạy train trước.")

        metrics = pd.read_csv(metrics_path)
        required_columns = {"model", "accuracy", "precision_spam", "recall_spam", "f1_spam"}
        self.assertTrue(required_columns.issubset(metrics.columns))

        best_row = metrics.sort_values(["f1_spam", "recall_spam"], ascending=False).iloc[0]
        self.assertGreaterEqual(best_row["f1_spam"], self.MIN_F1_SPAM)
        self.assertGreaterEqual(best_row["recall_spam"], self.MIN_RECALL_SPAM)

    def test_saved_model_can_predict_spam_and_not_spam_examples(self) -> None:
        model_path = config.MODELS_DIR / "spam_classifier.joblib"
        self.assertTrue(model_path.exists(), "Thiếu models/spam_classifier.joblib. Hãy chạy train trước.")

        model = joblib.load(model_path)
        spam_result = predict_email(
            "Congratulations winner, claim your free lottery prize money now by clicking this urgent link.",
            model=model,
        )
        ham_result = predict_email(
            "Hi team, please confirm tomorrow meeting agenda and send the project report when ready.",
            model=model,
        )

        self.assertEqual(spam_result["prediction"], "spam")
        self.assertEqual(ham_result["prediction"], "not spam")
        self.assertGreaterEqual(spam_result["spam_score"], 0.0)
        self.assertLessEqual(spam_result["spam_score"], 1.0)
        self.assertGreaterEqual(ham_result["spam_score"], 0.0)
        self.assertLessEqual(ham_result["spam_score"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
