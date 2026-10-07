"""Entry point chính cho project Classifying Spam Emails.

Luồng chạy gọn:
1. Tải/gộp/cân bằng dữ liệu.
2. Lọc lỗi dữ liệu và tạo file clean.
3. Train + tune model bằng module `src/model_train.py`.

Nếu chỉ muốn chạy notebook, có thể mở:
`notebooks/02_training_and_evaluation.ipynb`
"""

from __future__ import annotations

import argparse
import logging
import sys

import config
from src import data_loader
from src.data_quality import clean_balanced_dataset
from src.model_train import run_modeling_pipeline


def setup_logging() -> logging.Logger:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.REPORTS_DIR / "pipeline_run.log", mode="w", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


def run_pipeline(force_download: bool = False, train_model: bool = False) -> None:
    logger = setup_logging()
    logger.info("Bắt đầu pipeline Classifying Spam Emails")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Bước 1/3: tải, gộp và cân bằng dữ liệu")
    data_loader.run_data_pipeline(force_download=force_download)

    logger.info("Bước 2/3: kiểm tra và lọc lỗi dữ liệu")
    summary = clean_balanced_dataset()
    logger.info("Data clean summary: %s", summary)

    if train_model:
        logger.info("Bước 3/3: train model")
        run_modeling_pipeline()
    else:
        logger.info("Bỏ qua train model trong main.py. Notebook training đã có trong notebooks/.")

    logger.info("Hoàn thành pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy pipeline spam email.")
    parser.add_argument("--force-download", action="store_true", help="Tải lại toàn bộ dữ liệu nguồn.")
    parser.add_argument("--train-model", action="store_true", help="Chạy thêm bước train model.")
    args = parser.parse_args()
    run_pipeline(force_download=args.force_download, train_model=args.train_model)


if __name__ == "__main__":
    main()
