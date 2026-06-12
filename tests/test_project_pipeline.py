"""Test nhanh các phần đã sửa trong project spam email.

Chạy toàn bộ test:
    python -m unittest discover -s tests -v

Chạy riêng file này để xem báo cáo ngắn:
    python tests/test_project_pipeline.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_quality import balance_clean_dataset, clean_dataset, clean_balanced_dataset, find_data_issues, normalize_dataset
from src.model_evaluate import calculate_metrics, confusion_counts
from src.model_train import MODEL_DEFINITIONS, train_models
from src.predict import predict_email
from src.text_preprocess import clean_email_text, process_dataframe


LABEL_NAMES = {0: "not spam", 1: "spam"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_clean_dataset_report() -> dict:
    """Tạo lại report nếu thiếu, rồi trả về thống kê dataset clean."""
    if not config.DATA_QUALITY_REPORT_PATH.exists() and config.COMBINED_BALANCED_PATH.exists():
        clean_balanced_dataset()
    if not config.DATA_QUALITY_REPORT_PATH.exists():
        raise FileNotFoundError("Thiếu reports/data_quality_report.json. Hãy chạy: python src/data_quality.py")
    return read_json(config.DATA_QUALITY_REPORT_PATH)


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


class DatasetArtifactTest(unittest.TestCase):
    def test_data_quality_report_has_spam_and_not_spam_counts(self) -> None:
        report = ensure_clean_dataset_report()
        counts = {int(label): int(count) for label, count in report["clean_label_counts"].items()}

        self.assertEqual(set(counts), {0, 1})
        self.assertGreater(counts[0], 100_000)
        self.assertGreater(counts[1], 100_000)
        self.assertEqual(counts[0], counts[1])

    def test_clean_dataset_has_required_columns_and_valid_labels(self) -> None:
        if not config.COMBINED_BALANCED_CLEAN_PATH.exists():
            clean_balanced_dataset()

        sample = pd.read_csv(config.COMBINED_BALANCED_CLEAN_PATH, nrows=1000)
        required_columns = {"source", "file_name", "label", "label_name", "subject", "text"}

        self.assertTrue(required_columns.issubset(sample.columns))
        self.assertTrue(set(sample["label"].unique()).issubset({0, 1}))
        self.assertFalse(sample["text"].fillna("").eq("").any())

    def test_data_source_links_are_documented(self) -> None:
        links_csv = PROJECT_ROOT / "data_sources" / "data_links.csv"
        review_md = PROJECT_ROOT / "data_sources" / "OFFICIAL_SOURCE_REVIEW.md"

        self.assertTrue(links_csv.exists(), "Thiếu data_sources/data_links.csv")
        self.assertTrue(review_md.exists(), "Thiếu data_sources/OFFICIAL_SOURCE_REVIEW.md")

        links_text = links_csv.read_text(encoding="utf-8")
        review_text = review_md.read_text(encoding="utf-8")
        for keyword in ["SpamAssassin", "Hugging Face", "TREC", "CMU"]:
            self.assertIn(keyword, links_text + review_text)


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


class TextPreprocessTest(unittest.TestCase):
    def test_clean_email_text_removes_html_and_normalizes_entities(self) -> None:
        text = (
            "<html><body>Hello! Visit https://spamassassin.apache.org/old/publiccorpus/ "
            "or email trec@nist.gov for 100% FREE.</body></html>"
        )
        cleaned = clean_email_text(text)

        self.assertNotIn("<html>", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("@", cleaned)
        self.assertIn("urltoken", cleaned)
        self.assertIn("emailtoken", cleaned)
        self.assertIn("numbertoken", cleaned)
        self.assertIn("free", cleaned)

    def test_process_dataframe_adds_clean_text_column(self) -> None:
        frame = pd.DataFrame(
            {
                "text": [
                    "Hi team, meeting at 10.",
                    "Review official spam corpus at https://trec.nist.gov/data/spam.html",
                ]
            }
        )
        processed = process_dataframe(frame)

        self.assertIn("clean_text", processed.columns)
        self.assertEqual(len(processed), 2)
        self.assertIn("urltoken", processed.loc[1, "clean_text"])


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


class NotebookAndFigureTest(unittest.TestCase):
    def test_training_notebook_has_required_sections_and_no_encoding_errors(self) -> None:
        notebook_path = PROJECT_ROOT / "notebooks" / "02_training_and_evaluation.ipynb"
        notebook = read_json(notebook_path)
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        required_sections = [
            "Kết quả lọc dữ liệu hiện tại",
            "Metric sau train gần nhất",
            "Learning curve",
            "Confusion matrix",
            "Demo predict",
            "Kết luận nhanh",
        ]
        question = chr(63)
        bad_markers = [
            f"n{question}y",
            f"d{question} li{question}u",
            f"K{question}t",
            f"Kh{question}ng",
            f"Ch{question}a",
            question * 3,
            chr(0x00C3),
            chr(0x00E1) + chr(0x00BA),
        ]

        for section in required_sections:
            self.assertIn(section, text)
        for marker in bad_markers:
            self.assertNotIn(marker, text)

    def test_required_figure_files_exist(self) -> None:
        figures = [
            "linear_svm_confusion_matrix.png",
            "logistic_regression_confusion_matrix.png",
            "naive_bayes_confusion_matrix.png",
            "learning_curve_linear_svm.png",
            "learning_curve_logistic_regression.png",
            "learning_curve_naive_bayes.png",
        ]

        for figure_name in figures:
            figure_path = config.FIGURES_DIR / figure_name
            self.assertTrue(figure_path.exists(), f"Thiếu hình: {figure_path}")
            self.assertGreater(figure_path.stat().st_size, 1000, f"Hình quá nhỏ hoặc lỗi: {figure_path}")


class RequirementsTest(unittest.TestCase):
    def test_requirements_include_core_modules(self) -> None:
        requirements_path = PROJECT_ROOT / "requirements.txt"
        text = requirements_path.read_text(encoding="utf-8")

        for package_name in ["pandas", "pyarrow", "beautifulsoup4", "nltk", "scikit-learn", "seaborn", "jupyter"]:
            self.assertIn(package_name, text)


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


def build_test_report() -> str:
    lines = ["=== Báo cáo kiểm tra project spam email ==="]

    try:
        report = ensure_clean_dataset_report()
        clean_counts = {int(label): int(count) for label, count in report["clean_label_counts"].items()}
        input_counts = {int(label): int(count) for label, count in report["input_label_counts"].items()}
        lines.extend(
            [
                f"Tổng email trước lọc: {report['input_rows']:,}",
                f"  - Not spam trước lọc: {input_counts.get(0, 0):,}",
                f"  - Spam trước lọc: {input_counts.get(1, 0):,}",
                f"Tổng email sau lọc/cân bằng: {report['clean_rows']:,}",
                f"  - Not spam sau lọc: {clean_counts.get(0, 0):,}",
                f"  - Spam sau lọc: {clean_counts.get(1, 0):,}",
                f"Số dòng đã loại: {report['removed_rows']:,}",
                f"Lỗi chính trong data: {report.get('issue_counts', {})}",
            ]
        )
    except Exception as exc:
        lines.append(f"Không đọc được báo cáo data: {exc}")

    metrics_path = config.REPORTS_DIR / "model_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path).sort_values(["f1_spam", "recall_spam"], ascending=False)
        best = metrics.iloc[0]
        lines.extend(
            [
                f"Model tốt nhất theo F1 spam: {best['model']}",
                f"  - Accuracy: {best['accuracy']:.4f}",
                f"  - Precision spam: {best['precision_spam']:.4f}",
                f"  - Recall spam: {best['recall_spam']:.4f}",
                f"  - F1 spam: {best['f1_spam']:.4f}",
            ]
        )
    else:
        lines.append("Chưa có reports/model_metrics.csv.")

    return "\n".join(lines)


def run_direct_report() -> int:
    print(build_test_report())
    print("\n=== Đang chạy bộ test ===")

    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    hidden_output = io.StringIO()
    with contextlib.redirect_stdout(hidden_output):
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)

    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    print(f"Số test đã chạy: {total}")
    print(f"Số test đạt: {passed}")
    print(f"Số test lỗi: {failed}")

    if failed:
        print("\nChi tiết lỗi:")
        for test_case, traceback_text in result.failures + result.errors:
            print(f"- {test_case}")
            print(traceback_text)
        return 1

    print("Tất cả kiểm tra đều đạt yêu cầu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_direct_report())
