from __future__ import annotations

import io
import unittest

from app import create_app


SPAM_EMAIL = "Congratulations winner, claim your free lottery prize money now by clicking this urgent link."
HAM_EMAIL = "Hi team, please confirm tomorrow meeting agenda and send the project report when ready."


class SpamCheckerUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def test_home_page_loads(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Spam Email Checker", response.data)
        self.assertIn("Nhập email".encode("utf-8"), response.data)

    def test_health_endpoint_reports_model(self) -> None:
        response = self.client.get("/health")
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["model_exists"])
        self.assertNotIn("gmail", payload)

    def test_api_predicts_spam_email(self) -> None:
        response = self.client.post("/api/check", json={"email_text": SPAM_EMAIL})
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["prediction"], "spam")
        self.assertGreaterEqual(payload["spam_score"], 0.0)
        self.assertLessEqual(payload["spam_score"], 1.0)
        self.assertIn("evidence", payload)
        self.assertIn("summary", payload["evidence"])
        self.assertNotIn("model_path", payload["evidence"])
        self.assertTrue(payload["evidence"]["top_terms"])

    def test_api_predicts_not_spam_email(self) -> None:
        response = self.client.post("/api/check", json={"email_text": HAM_EMAIL})
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["prediction"], "not spam")
        self.assertGreaterEqual(payload["spam_score"], 0.0)
        self.assertLessEqual(payload["spam_score"], 1.0)

    def test_api_rejects_empty_input(self) -> None:
        response = self.client.post("/api/check", json={"email_text": ""})
        payload = response.get_json()

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", payload)

    def test_upload_txt_file_predicts(self) -> None:
        response = self.client.post(
            "/api/check",
            data={
                "email_file": (
                    io.BytesIO(SPAM_EMAIL.encode("utf-8")),
                    "sample_email.txt",
                )
            },
            content_type="multipart/form-data",
        )
        payload = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["prediction"], "spam")
        self.assertTrue(payload["input_source"].startswith("upload:"))

    def test_upload_rejects_wrong_extension(self) -> None:
        response = self.client.post(
            "/api/check",
            data={"email_file": (io.BytesIO(SPAM_EMAIL.encode("utf-8")), "sample.pdf")},
            content_type="multipart/form-data",
        )
        payload = response.get_json()

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", payload)

if __name__ == "__main__":
    unittest.main(verbosity=2)
