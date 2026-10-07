# Code and Notebook Rules

File này mô tả chuẩn đặt tên, cấu trúc source và chuẩn notebook cho project.

## 1. Chuẩn đặt tên

- File Python: `snake_case.py`
- Hàm: `snake_case`
- Biến: `snake_case`
- Class: `PascalCase`
- Hằng số: `UPPER_SNAKE_CASE`
- Tên model: `naive_bayes`, `logistic_regression`, `linear_svm`
- Tên label: `0 = not spam`, `1 = spam`

Không dùng tên mơ hồ như:

```text
data1
abc
temp2
result_final_final
```

## 2. Chuẩn encoding

- Tất cả file `.py`, `.md`, `.html`, `.ipynb`, `.csv` phải lưu UTF-8.
- Không để text mojibake hoặc ký tự thay thế do sai encoding.
- Khi sửa notebook, dùng script đọc/ghi JSON với `encoding='utf-8'`.
- Sau khi sửa notebook, kiểm tra:

```powershell
python -m json.tool notebooks/99_final_project_notebook.ipynb > $null
```

## 3. Trách nhiệm từng file chính

| File | Nhiệm vụ |
| --- | --- |
| `config.py` | Định nghĩa path, random state, source data, hyperparameter |
| `main.py` | Entry point chạy data pipeline, data quality và tùy chọn train |
| `src/data_loader.py` | Tải, parse, chuẩn hóa, gộp và cân bằng dữ liệu |
| `src/data_quality.py` | Kiểm tra label/text, lọc lỗi, cân bằng lại dataset |
| `src/text_preprocess.py` | Làm sạch text email |
| `src/feature_engineering.py` | TF-IDF và manual features |
| `src/model_train.py` | Train NB/LR/SVM, lưu model và metrics |
| `src/model_evaluate.py` | Tính metrics, confusion matrix, model comparison |
| `src/predict.py` | Load model và dự đoán email mới |
| `src/explain_prediction.py` | Lấy từ/cụm từ ảnh hưởng để giải thích dự đoán |
| `app.py` | Giao diện Flask kiểm tra spam email |
| `tests/` | Unit/integration tests cho pipeline và UI |

## 4. Chuẩn giao diện

Giao diện chỉ cần:

- Textarea để dán email.
- Upload file `.txt` hoặc `.eml`.
- Kết quả `SPAM` hoặc `NOT SPAM`.
- `spam_score`.
- Dẫn chứng bằng các từ/cụm từ ảnh hưởng.
- Text sau tiền xử lý.

Không hiển thị:

- Phần Gmail tùy chọn.
- Đường dẫn model.
- Metric model trong phần dẫn chứng.
- Mô tả kỹ thuật dài trong giao diện.

## 5. Chuẩn notebook

Notebook cuối bắt buộc:

- Có tiêu đề rõ.
- Có phần problem definition.
- Có phần data collection.
- Có phần data quality.
- Có phần preprocessing.
- Có phần model training/evaluation.
- Có confusion matrix.
- Có learning curve.
- Có demo predict.
- Có hướng dẫn chạy web UI.

Notebook cuối hiện tại:

```text
notebooks/99_final_project_notebook.ipynb
```

Các hình learning curve và confusion matrix phải được gọi là kết quả đã train/evaluation, không gọi là ảnh mẫu.

## 6. Chuẩn test

Test tối thiểu cần pass:

- Data quality.
- Text preprocessing.
- Model evaluation.
- Model training với dataset nhỏ.
- Saved model prediction.
- Web UI home page.
- Web UI API `/api/check`.
- Upload `.txt`.
- Reject input rỗng và file sai đuôi.

Lệnh chuẩn:

```powershell
python -m unittest discover -s tests -v
```
