# AI Runbook Rules - Classifying Spam Emails

File này là checklist để một AI coding agent hoặc thành viên nhóm có thể chạy, kiểm tra và tiếp tục phát triển toàn bộ chương trình.

## 1. Luôn chạy từ project root

Trước khi chạy lệnh, kiểm tra đang đứng tại thư mục root của repo:

```powershell
Get-Location
```

Thư mục đúng phải chứa các file/thư mục:

```text
config.py
main.py
app.py
src/
tests/
notebooks/
reports/
models/
```

Không chạy trực tiếp trong `src/`, `tests/` hoặc `notebooks/` nếu lệnh không yêu cầu.

## 2. Cài môi trường

```powershell
python -m pip install -r requirements.txt
```

Nếu thiếu Flask khi chạy giao diện:

```powershell
python -m pip install Flask
```

Nếu thiếu thư viện train/test:

```powershell
python -m pip install pandas numpy scikit-learn joblib matplotlib seaborn beautifulsoup4 nltk pyarrow
```

## 3. Luồng chạy dữ liệu

Chạy tải/gộp/cân bằng dữ liệu:

```powershell
python main.py --force-download
```

Chạy lọc lỗi dữ liệu:

```powershell
python src/data_quality.py
```

Output mong đợi:

```text
data/processed/combined_balanced_clean.csv
reports/data_quality_report.json
reports/data_quality_issues.csv
```

## 4. Train model

```powershell
python src/model_train.py
```

Output mong đợi:

```text
models/naive_bayes_pipeline.joblib
models/logistic_regression_pipeline.joblib
models/linear_svm_pipeline.joblib
models/spam_classifier.joblib
reports/model_metrics.csv
reports/model_predictions.csv
reports/model_train_report.json
```

## 5. Đánh giá model

```powershell
python src/model_evaluate.py --predictions reports/model_predictions.csv --true-col label --pred-cols naive_bayes logistic_regression linear_svm
```

Output mong đợi:

```text
reports/figures/naive_bayes_confusion_matrix.png
reports/figures/logistic_regression_confusion_matrix.png
reports/figures/linear_svm_confusion_matrix.png
reports/figures/model_comparison.csv
```

## 6. Dự đoán bằng CLI

```powershell
python src/predict.py --demo
python src/predict.py --text "Congratulations winner, claim your free lottery prize money now"
python src/predict.py --file sample_emails/spam_01_prize_claim.txt
```

Kết quả phải trả về:

- `spam` hoặc `not spam`
- `spam_score` trong khoảng `0-1`

## 7. Chạy giao diện web

```powershell
python app.py
```

Mở đúng URL:

```text
http://127.0.0.1:5000
```

Không mở trực tiếp file:

```text
templates/spam_checker.html
```

Lý do: file template cần Flask render cú pháp Jinja như `{% ... %}` và `{{ ... }}`. Nếu mở bằng Live Server port `5500`, trình duyệt sẽ hiển thị nguyên code template.

## 8. Chạy notebook cuối

Notebook tổng hợp cuối:

```text
notebooks/99_final_project_notebook.ipynb
```

Notebook này dùng các report/model đã có sẵn. Nếu thiếu file report hoặc model, chạy lại các bước data/train/evaluate ở trên.

## 9. Chạy test

Chạy toàn bộ test:

```powershell
python -m unittest discover -s tests -v
```

Chạy test giao diện:

```powershell
python -m unittest tests.test_spam_checker_ui -v
```

Chạy kiểm tra cú pháp Python:

```powershell
$files = @('app.py','config.py','main.py') + (Get-ChildItem src,tests -Filter *.py | ForEach-Object { $_.FullName })
python -m py_compile $files
```

## 10. Quy tắc khi AI sửa code

- Không đổi mapping label: `0 = not spam`, `1 = spam`.
- Không hard-code đường dẫn tuyệt đối của máy cá nhân.
- Không đưa dữ liệu raw lớn vào commit nếu không được yêu cầu.
- Không mở trực tiếp template HTML để test Flask.
- Không thêm lại phần Gmail nếu người dùng không yêu cầu.
- Dẫn chứng trên giao diện chỉ cần giải thích vì sao spam/not spam bằng từ/cụm từ ảnh hưởng.
- Sau khi sửa, luôn chạy `python -m unittest discover -s tests -v`.
