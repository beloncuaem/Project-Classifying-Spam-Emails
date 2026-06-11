import os
import sys
import pickle
from pathlib import Path

import pandas as pd
import numpy as np # <-- THÊM MỚI
import matplotlib.pyplot as plt # <-- THÊM MỚI
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve # <-- THÊM learning_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to sys.path so config.py can be imported when running src/model_train.py
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def load_processed_data(file_path):
    """
    Ý nghĩa xử lý: Đọc dữ liệu đã được làm sạch sau bước tiền xử lý.
    
    Input:
        - file_path (str/Path): Đường dẫn tới file dữ liệu csv.
        
    Output:
        - df (pd.DataFrame): DataFrame chứa dữ liệu email sạch.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {file_path}")
    return pd.read_csv(file_path)


def extract_tfidf_features(X_train, X_test, save_vectorizer_path=None):
    """
    Ý nghĩa xử lý: Biến đổi dữ liệu văn bản thành ma trận đặc trưng TF-IDF.
    
    Input:
        - X_train (pd.Series): Dữ liệu text huấn luyện.
        - X_test (pd.Series): Dữ liệu text kiểm thử.
        - save_vectorizer_path (Path): Đường dẫn để lưu trữ bộ vectorizer sau khi fit.
        
    Output:
        - X_train_tfidf: Ma trận đặc trưng tập train.
        - X_test_tfidf: Ma trận đặc trưng tập test.
    """
    # Khởi tạo vectorizer với các hằng số cấu hình từ config.py
    tfidf_vectorizer = TfidfVectorizer(
        max_features=config.MAX_FEATURES, 
        ngram_range=config.NGRAM_RANGE
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Lưu bộ Vectorizer phục vụ cho việc Predict email mới sau này (Tuấn Tú cần dùng)
    if save_vectorizer_path:
        save_vectorizer_path = Path(save_vectorizer_path)
        save_vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_vectorizer_path, 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
            
    return X_train_tfidf, X_test_tfidf


def train_and_tune_models(X_train_tfidf, y_train):
    """
    Ý nghĩa xử lý: Định nghĩa lưới tham số và dùng GridSearchCV để tìm 
                  tham số tối ưu cho từng mô hình (Tuning hệ số phạt C và Alpha).
                  
    Input:
        - X_train_tfidf: Ma trận đặc trưng TF-IDF tập huấn luyện.
        - y_train (pd.Series): Nhãn của tập huấn luyện.
        
    Output:
        - best_estimators (dict): Từ điển chứa các mô hình đã được chọn lựa tham số tốt nhất.
    """
    print("[INFO] Đang tiến hành Tuning tham số bằng GridSearchCV (Tối ưu theo F1-Score)...")
    
    # Cấu hình các mô hình và không gian tham số cần quét (Tuning)
    model_blueprints = {
        "naive_bayes": {
            "model": MultinomialNB(),
            "params": {
                "alpha": [0.1, 0.5, 1.0]
            }
        },
        "logistic_regression": {
            "model": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            "params": {
                "C": [0.1, 1.0, 10.0]
            }
        },
        "linear_svm": {
            "model": LinearSVC(max_iter=2000, random_state=config.RANDOM_STATE),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0]
            }
        }
    }
    
    best_estimators = {}
    
    for model_name, blueprint in model_blueprints.items():
        # Sử dụng scoring='f1' và thực hiện Cross-Validation K-Fold với K=5
        grid_search = GridSearchCV(
            estimator=blueprint["model"],
            param_grid=blueprint["params"],
            scoring="f1",
            cv=5,
            n_jobs=-1
        )
        grid_search.fit(X_train_tfidf, y_train)
        
        # Ghi nhận mô hình tốt nhất của thuật toán đó
        best_estimators[model_name] = grid_search.best_estimator_
        print(f"  -> Mô hình [{model_name}] tối ưu nhất với tham số: {grid_search.best_params_}")
        
    return best_estimators


def evaluate_predictions(models, X_test_tfidf, y_test, save_models_dir=None):
    """
    Ý nghĩa xử lý: Đánh giá hiệu năng của các mô hình đã được Tuning trên tập kiểm thử,
                  xuất ra bảng chỉ số và tìm ra Best Model ứng viên xuất sắc nhất.
                  
    Input:
        - models (dict): Từ điển chứa các mô hình đã huấn luyện xong.
        - X_test_tfidf: Ma trận đặc trưng TF-IDF tập kiểm thử.
        - y_test (pd.Series): Nhãn thực tế tập kiểm thử.
        - save_models_dir (Path): Thư mục dạng Path để lưu các file model (.pkl).
        
    Output:
        - df_metrics (pd.DataFrame): Bảng chứa Accuracy, Precision, Recall, F1-Score.
    """
    evaluation_results = []
    best_f1 = -1
    best_model_name = ""
    best_model_obj = None
    
    # Ánh xạ hiển thị đẹp mắt phục vụ Report của Tuấn Tú
    name_mapping = {
        "naive_bayes": "Naive Bayes (Tuned)",
        "logistic_regression": "Logistic Regression (Tuned)",
        "linear_svm": "Linear SVM (Tuned)"
    }
    
    for model_name, model_obj in models.items():
        # Dự đoán nhãn
        y_pred = model_obj.predict(X_test_tfidf)
        
        # Tính toán các chỉ số metric
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        display_name = name_mapping.get(model_name, model_name)
        
        evaluation_results.append({
            "Model": display_name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4)
        })
        
        # Thuật toán tìm Best Model dựa trên F1-Score cao nhất
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = display_name
            best_model_obj = model_obj
        
        # Lưu từng mô hình thành file .pkl độc lập
        if save_models_dir:
            save_models_dir = Path(save_models_dir)
            model_path = save_models_dir / f"{model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model_obj, f)
                
    # Lưu riêng một file "best_model.pkl" đại diện cho mô hình chiến thắng
    if save_models_dir and best_model_obj:
        with open(save_models_dir / "best_model.pkl", 'wb') as f:
            pickle.dump(best_model_obj, f)
            
    # Tạo DataFrame kết quả dạng bảng theo đúng yêu cầu đề bài
    df_metrics = pd.DataFrame(evaluation_results)
    
    print("\n" + "="*23 + " BẢNG KẾT QUẢ SO SÁNH SAU TUNING " + "="*23)
    print(df_metrics.to_string(index=False))
    print("="*79)
    
    print(f"\n [BEST MODEL FOUND]: {best_model_name}")
    print(f" Chi tiết thông số tối ưu: {best_model_obj}\n")
    
    return df_metrics

def plot_learning_curves(models, X, y, save_dir=None):
    """
    Ý nghĩa xử lý: Vẽ biểu đồ Learning Curve để đánh giá hiệu suất mô hình 
                  khi lượng dữ liệu huấn luyện tăng dần.
                  
    Input:
        - models (dict): Từ điển chứa các mô hình đã Tuning (best_estimators).
        - X: Ma trận đặc trưng TF-IDF.
        - y: Nhãn dữ liệu.
        - save_dir (Path): Thư mục lưu hình ảnh biểu đồ.
    """
    print("[INFO] Đang tính toán và vẽ biểu đồ Learning Curve...")
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for name, model_obj in models.items():
        plt.figure(figsize=(10, 6))
        plt.title(f"Learning Curve - {name.upper()}")
        plt.xlabel("Số lượng mẫu huấn luyện (Training examples)")
        plt.ylabel("F1-Score")

        # Tính toán learning curve chia tập train từ 10% đến 100%
        train_sizes, train_scores, test_scores = learning_curve(
            estimator=model_obj, 
            X=X, 
            y=y, 
            cv=5, # Cross-validation 5 fold
            scoring='f1', 
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5) 
        )

        # Tính trung bình và độ lệch chuẩn của các fold
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.grid()
        
        # Vẽ dải màu thể hiện khoảng biến thiên (variance)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
                         
        # Vẽ đường trung bình
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training F1-Score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation F1-Score")

        plt.legend(loc="lower right")

        # Lưu biểu đồ ra file
        if save_dir:
            file_path = save_dir / f"learning_curve_{name}.png"
            plt.savefig(file_path, bbox_inches='tight')
            print(f"  -> Đã lưu biểu đồ tại: {file_path}")
        
        plt.close() # Đóng plot để giải phóng bộ nhớ

def run_modeling_pipeline():
    """
    Ý nghĩa xử lý: Hàm điều khiển chính (pipeline) kết nối toàn bộ quá trình 
                  đọc dữ liệu, trích xuất đặc trưng, huấn luyện và trả về bảng kết quả.
    """
    print("[INFO] Bắt đầu quá trình huấn luyện và tối ưu mô hình...")
    
    # 1. Định nghĩa các đường dẫn từ cấu hình config (sử dụng định dạng Path mới)
    processed_data_path = config.COMBINED_BALANCED_PATH
    vectorizer_save_path = config.MODELS_DIR / 'tfidf_vectorizer.pkl'
    models_save_directory = config.MODELS_DIR
    models_save_directory.mkdir(parents=True, exist_ok=True)
    
    # 2. Tải dữ liệu
    df_clean = load_processed_data(processed_data_path)
    
    # Đọc đúng cột text và label từ dataset đã xử lý
    X = df_clean['text'].fillna('').astype(str)
    y = df_clean['label']
    
    # 3. Chia tập dữ liệu Train/Test theo tỷ lệ và mã random đồng nhất toàn nhóm
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=y
    )
    
    # 4. Trích xuất đặc trưng TF-IDF
    X_train_tfidf, X_test_tfidf = extract_tfidf_features(X_train, X_test, vectorizer_save_path)
    print("[INFO] Trích xuất đặc trưng TF-IDF thành công.")
    
    # 5. Huấn luyện + Tuning tham số bằng GridSearchCV (Gồm Naive Bayes, Logistic Regression, Linear SVM)
    tuned_models = train_and_tune_models(X_train_tfidf, y_train)
    print("[INFO] Hoàn thành huấn luyện và tối ưu tất cả các mô hình.")
    
    # ================= THÊM MỚI Ở ĐÂY =================
    # Thư mục để lưu biểu đồ
    plots_save_directory = config.MODELS_DIR / 'plots'
    
    # Gọi hàm vẽ biểu đồ cho tập huấn luyện
    plot_learning_curves(
        models=tuned_models, 
        X=X_train_tfidf, 
        y=y_train, 
        save_dir=plots_save_directory
    )
    # ==================================================
    
    # 6. Đánh giá, xuất bảng kết quả metric và lưu trữ mô hình tốt nhất
    df_metrics = evaluate_predictions(
        tuned_models, 
        X_test_tfidf, 
        y_test, 
        save_models_dir=models_save_directory
    )
    
    return df_metrics


if __name__ == "__main__":
    run_modeling_pipeline()
