# 📧 Dự án Machine Learning: Classifying Spam Emails - Nhóm 2

## 📖 1. Tổng quan dự án (Project Overview)
Dự án **Classifying Spam Emails** là một hệ thống 파i-pline (pipeline) Học máy toàn diện được xây dựng bởi Nhóm 2. Dự án tập trung vào việc áp dụng các kỹ thuật Xử lý Ngôn ngữ Tự nhiên (NLP) và các thuật toán Machine Learning truyền thống để tự động phân loại email.

### ❓ Vấn đề (The Problem)
Sự gia tăng không ngừng của thư rác (spam), lừa đảo (phishing) và quảng cáo độc hại qua email gây lãng phí tài nguyên lưu trữ và tiềm ẩn rủi ro bảo mật lớn cho người dùng. Khối lượng email quá lớn khiến việc lọc thủ công trở nên bất khả thi, đòi hỏi một hệ thống tự động phân tích ngữ nghĩa và các đặc trưng của văn bản để chặn email rác hiệu quả.

### 🎯 Mục tiêu (The Goal)
1. **Mục tiêu kỹ thuật:** Xây dựng một pipeline phân loại nhị phân chính xác để dán nhãn email thành **Spam (1)** hoặc **Ham/Not Spam (0)**.
2. **Mục tiêu hiệu suất:** Tối ưu hóa chỉ số **F1-Score** và **Recall** đối với lớp Spam (để không lọt lưới email rác), đồng thời phải kiểm soát nghiêm ngặt tỷ lệ False Positive (đảm bảo không chặn nhầm email công việc quan trọng của người dùng).
3. **Mục tiêu dự án:** Cung cấp bộ mã nguồn sạch, module hóa tốt, có thể cấu hình linh hoạt và dễ dàng tái triển khai (deploy) với email thực tế.

---

## 👥 2. Đội ngũ phát triển (Team & Roles)

| Thành viên | Trách nhiệm cốt lõi (Roles) |
| :--- | :--- |
| **Bùi Quang** | **Lead tích hợp & Quản lý Repo:** Xây dựng kiến trúc dự án, config, ghép nối pipeline `main.py` và kiểm soát chất lượng code. |
| **Khắc Trường** | **Data Owner:** Thu thập tập dữ liệu, EDA (khai phá dữ liệu), thống kê nhãn và xử lý các vấn đề đạo đức dữ liệu. |
| **Sinh Trường** | **Preprocessing Owner:** Làm sạch văn bản (xóa HTML tags, lọc Stopwords, chuẩn hóa chữ thường). |
| **Nhật Trung** | **Feature Engineering Owner:** Chuyển hóa văn bản thành vector (TF-IDF), trích xuất đặc trưng (n-gram, tỷ lệ viết hoa, độ dài). |
| **Hữu Trọng** | **Modeling Owner:** Chia tập dữ liệu, huấn luyện các mô hình (Naive Bayes, Logistic Regression, SVM) và tinh chỉnh tham số. |
| **Tuấn Tú** | **Evaluation & Report Owner:** Đánh giá mô hình bằng metrics, vẽ Confusion Matrix, làm script dự đoán email mới. |

---

## 📥 3. Đầu vào và Đầu ra (Input & Output)

* **Đầu vào (Input):** Tập dữ liệu văn bản thô dạng bảng (CSV/TXT) chứa nội dung các email (Text Body, Subject Line) và siêu dữ liệu (metadata) nếu có. Dữ liệu đầu vào khi dự đoán (`predict.py`) là một chuỗi văn bản (string) đại diện cho nội dung 1 email mới.
* **Đầu ra (Output):** * Trong quá trình huấn luyện: Các mô hình đã lưu (`.joblib`), biểu đồ đánh giá (Confusion Matrix) và bảng báo cáo các chỉ số (Accuracy, Precision, Recall, F1-Score).
  * Trong quá trình dự đoán: Nhãn phân loại cuối cùng — `1` (Spam) hoặc `0` (Ham).

---

## 🛠️ 4. Các bước thực hiện (Implementation Pipeline)
Hệ thống vận hành theo một luồng (workflow) 5 bước nghiêm ngặt, được tự động hóa trong `main.py`:

1. **Thu thập & Khám phá (Data Loading & EDA):** Đọc dữ liệu từ `data/raw/`, kiểm tra mất cân bằng dữ liệu (imbalanced data) và trực quan hóa phân phối nhãn.
2. **Tiền xử lý văn bản (Text Preprocessing):** * Loại bỏ nhiễu: HTML tags, URLs, ký tự đặc biệt, số.
   * Chuyển về chữ thường (lowercase) và loại bỏ từ dừng (stopwords) bằng thư viện `nltk`.
3. **Trích xuất đặc trưng (Feature Engineering):** * Số hóa văn bản sử dụng kỹ thuật **TF-IDF Vectorizer** (giới hạn `MAX_FEATURES=25000` và `NGRAM_RANGE=(1, 2)`).
   * Khai thác độ dài email và tỷ lệ ký tự in hoa.
4. **Huấn luyện mô hình (Model Training):**
   * Chia dữ liệu Train/Test theo tỷ lệ 80/20 (Stratified Split).
   * Huấn luyện song song các mô hình: Multinomial Naive Bayes (Baseline), Logistic Regression, và Linear SVM.
5. **Đánh giá & Triển khai (Evaluation & Deployment):**
   * So sánh chéo các mô hình. Chọn mô hình tối ưu nhất lưu vào thư mục `models/`.
   * Cung cấp module `predict.py` để demo với email bất kỳ do người dùng nhập vào.

---

## 📂 5. Cấu trúc thư mục (Project Structure)

```text
spam_email_classifier/
├── data/
│   ├── raw/                        # Chứa dữ liệu gốc (CSV, txt) do Khắc Trường tải về
│   └── processed/                  # Chứa dữ liệu sạch do Sinh Trường xử lý
├── models/                         # Lưu các mô hình tốt nhất (.joblib) của Hữu Trọng
├── notebooks/                      # Chứa Jupyter Notebooks cho EDA và thử nghiệm nháp
├── reports/                        # Nơi xuất kết quả đánh giá của Tuấn Tú
│   ├── figures/                    # Lưu các biểu đồ, Confusion Matrix
│   └── final_report.docx
├── src/                            # Thư mục mã nguồn chính (Source code)
│   ├── data_loader.py              # Hàm đọc dữ liệu an toàn
│   ├── text_preprocess.py          # Module làm sạch văn bản
│   ├── feature_engineering.py      # Module trích xuất TF-IDF và features
│   ├── model_train.py              # Module huấn luyện các thuật toán
│   ├── model_evaluate.py           # Module tính toán metrics đánh giá
│   ├── predict.py                  # Script demo dự đoán email mới
│   └── utils.py                    # Các hàm hỗ trợ dùng chung
├── config.py                       # File cấu hình trung tâm (Paths, Hyperparams) - Bùi Quang
├── main.py                         # Entry point chạy toàn bộ pipeline - Bùi Quang
├── requirements.txt                # Danh sách thư viện môi trường
└── README.md                       # Tài liệu mô tả dự án này
