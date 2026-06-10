"""
Main Entry Point cho Dự án Classifying Spam Emails - Nhóm 6.
Tác giả/Lead: Bùi Quang
Chức năng: Tích hợp và chạy toàn bộ pipeline từ Raw Data đến Model Evaluation.
"""

import logging
import sys
import config

# Import các module từ thư mục src (Các thành viên khác sẽ viết code trong này)
# Lưu ý: Đảm bảo thư mục src có file __init__.py trống để Python nhận diện như một package
try:
    from src import data_loader
    from src import text_preprocess
    from src import feature_engineering
    from src import model_train
    from src import model_evaluate
except ImportError as e:
    print(f"[LỖI CẤU TRÚC] Không thể import các module trong thư mục 'src'. Chi tiết: {e}")
    sys.exit(1)

# ==============================================================================
# 1. CẤU HÌNH GHI NHẬT KÝ (LOGGING CONFIGURATION)
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.REPORTS_DIR / "pipeline_run.log", mode='w') # Lưu log ra file
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 2. HÀM CHẠY PIPELINE CHÍNH (MAIN PIPELINE)
# ==============================================================================
def run_pipeline():
    logger.info("=== BẮT ĐẦU CHẠY PIPELINE PHÂN LOẠI SPAM EMAIL ===")
    
    try:
        # ---------------------------------------------------------
        # BƯỚC 1: LOAD DỮ LIỆU (Trách nhiệm: Khắc Trường)
        # ---------------------------------------------------------
        logger.info("[1/5] Đang tải dữ liệu thô...")
        raw_data_path = config.DATA_RAW_DIR / "spam_dataset.csv" # Thay đổi tên file cho đúng thực tế
        df_raw = data_loader.load_data(raw_data_path)
        logger.info(f"Đã tải thành công tập dữ liệu với kích thước: {df_raw.shape}")

        # ---------------------------------------------------------
        # BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN (Trách nhiệm: Sinh Trường)
        # ---------------------------------------------------------
        logger.info("[2/5] Đang tiến hành làm sạch văn bản...")
        df_clean = text_preprocess.clean_data(df_raw)
        
        # Lưu lại dữ liệu đã xử lý
        processed_data_path = config.DATA_PROCESSED_DIR / "cleaned_data.csv"
        df_clean.to_csv(processed_data_path, index=False)
        logger.info("Đã lưu dữ liệu sạch vào data/processed/")

        # ---------------------------------------------------------
        # BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG (Trách nhiệm: Nhật Trung)
        # ---------------------------------------------------------
        logger.info("[3/5] Đang trích xuất đặc trưng (TF-IDF)...")
        # Hàm này trả về ma trận đặc trưng X và nhãn y
        X, y, vectorizer = feature_engineering.extract_features(
            df_clean, 
            max_features=config.MAX_FEATURES, 
            ngram_range=config.NGRAM_RANGE
        )
        logger.info(f"Đã vector hóa thành công. Kích thước ma trận X: {X.shape}")

        # ---------------------------------------------------------
        # BƯỚC 4: HUẤN LUYỆN MÔ HÌNH (Trách nhiệm: Hữu Trọng)
        # ---------------------------------------------------------
        logger.info("[4/5] Đang chia dữ liệu và huấn luyện các mô hình...")
        # Hàm train_models sẽ tự chia train/test dựa vào config, sau đó train nhiều model
        trained_models, X_test, y_test = model_train.train_models(
            X, y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE
        )
        logger.info(f"Đã huấn luyện xong {len(trained_models)} mô hình.")

        # ---------------------------------------------------------
        # BƯỚC 5: ĐÁNH GIÁ VÀ LƯU KẾT QUẢ (Trách nhiệm: Tuấn Tú)
        # ---------------------------------------------------------
        logger.info("[5/5] Đang đánh giá hiệu suất và xuất báo cáo...")
        # Hàm đánh giá sẽ nhận dictionary các models, tập test và xuất ra metrics, biểu đồ
        best_model_name = model_evaluate.evaluate_and_save(
            trained_models, 
            X_test, 
            y_test, 
            models_dir=config.MODELS_DIR,
            figures_dir=config.FIGURES_DIR
        )
        
        # Lưu lại Vectorizer (Rất quan trọng để predict.py có thể dùng lại)
        model_evaluate.save_vectorizer(vectorizer, config.MODELS_DIR / "tfidf_vectorizer.joblib")

        logger.info(f"=== PIPELINE HOÀN THÀNH. Mô hình tốt nhất: {best_model_name} ===")

    except Exception as e:
        logger.error(f"PIPELINE THẤT BẠI TẠI LỖI: {str(e)}", exc_info=True)
        sys.exit(1)

# ==============================================================================
# 3. KÍCH HOẠT CHƯƠNG TRÌNH
# ==============================================================================
if __name__ == "__main__":
    # Đảm bảo các thư mục hệ thống (models, reports, figures, data/processed) tồn tại
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Chạy pipeline
    run_pipeline()
