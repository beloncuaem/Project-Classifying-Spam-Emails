from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
MIN_TEXT_LENGTH = 20

SPAMASSASSIN_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus"
SPAMASSASSIN_ARCHIVES = [
    {"name": "easy_ham", "file_name": "20030228_easy_ham.tar.bz2", "label": 0},
    {"name": "hard_ham", "file_name": "20030228_hard_ham.tar.bz2", "label": 0},
    {"name": "spam", "file_name": "20030228_spam.tar.bz2", "label": 1},
    {"name": "easy_ham_2", "file_name": "20030228_easy_ham_2.tar.bz2", "label": 0},
    {"name": "spam_2", "file_name": "20030228_spam_2.tar.bz2", "label": 1},
]

HUGGINGFACE_CSV_SOURCES = [
    {
        "source": "setfit_enron_spam",
        "url": "https://huggingface.co/datasets/SetFit/enron_spam/resolve/main/enron_spam_data.csv",
        "file_name": "setfit_enron_spam.csv",
    }
]

HUGGINGFACE_PARQUET_DATASETS = [
    {
        "source": "locuoco_big_spam_ham_phish_300k",
        "dataset": "locuoco/the-biggest-spam-ham-phish-email-dataset-300000",
    }
]

SPAMASSASSIN_LABELED_PATH = PROCESSED_DATA_DIR / "spamassassin_labeled.csv"
SPAMASSASSIN_BALANCED_PATH = PROCESSED_DATA_DIR / "spamassassin_balanced.csv"
COMBINED_LABELED_PATH = PROCESSED_DATA_DIR / "combined_labeled.csv"
COMBINED_BALANCED_PATH = PROCESSED_DATA_DIR / "combined_balanced.csv"
DATASET_REPORT_PATH = REPORTS_DIR / "data_loader_report.json"

