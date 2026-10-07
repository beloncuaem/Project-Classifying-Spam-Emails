# Kế hoạch giao diện kiểm tra email spam

## 1. Mục tiêu

Tạo một giao diện web chạy local để người dùng kiểm tra một email có phải spam hay không.

Giao diện cần trả về:

- Kết luận: `spam` hoặc `not spam`.
- Điểm spam: `spam_score` trong khoảng `0-1`.
- Dẫn chứng vì sao là spam hoặc not spam dựa trên các từ/cụm từ ảnh hưởng trong email.
- Nội dung email sau khi tiền xử lý để nhóm giải thích khi thuyết trình.
- Cách nhập email bằng paste text hoặc upload file `.txt/.eml`.

## 2. Công nghệ chọn

Chọn Flask vì:

- Nhẹ, dễ chạy local.
- Có giao diện HTML thật trong trình duyệt.
- Có thể test bằng Flask test client.
- Dễ thêm API JSON cho kiểm thử tự động.

Các file thêm mới:

```text
app.py
src/explain_prediction.py
templates/spam_checker.html
static/spam_checker.css
tests/test_spam_checker_ui.py
docs/SPAM_CHECKER_UI_PLAN.md
```

## 3. Luồng giao diện

### Bước 1: Người dùng nhập dữ liệu

Người dùng có 3 cách:

- Paste nội dung email vào textarea.
- Upload file `.txt` hoặc `.eml`.

### Bước 2: Backend xử lý

Backend gọi:

```python
src.predict.predict_email(text)
```

Sau đó gọi:

```python
src.explain_prediction.build_prediction_evidence(...)
```

để gom dẫn chứng.

### Bước 3: Giao diện trả kết quả

Giao diện hiển thị:

- Badge `SPAM` hoặc `NOT SPAM`.
- `spam_score`.
- Model đang dùng.
- Top token/term có ảnh hưởng.
- Text đã clean.

## 4. Dẫn chứng từ đâu?

Kết quả không phải do giao diện tự đoán. Phần dẫn chứng lấy từ:

- File model đã train: `models/spam_classifier.joblib`.
- Hàm dự đoán: `src/predict.py`.
- Feature/term trong pipeline TF-IDF của model:
  - Với Linear SVM hoặc Logistic Regression: lấy hệ số `coef_` của model.
  - Với Naive Bayes: lấy chênh lệch log probability giữa spam và not spam.
  - Chỉ hiển thị những token thật sự xuất hiện trong email sau khi vectorize.

## 6. Test cần có

Test trong `tests/test_spam_checker_ui.py`:

- Trang chủ load được.
- API `/api/check` trả JSON có `prediction`, `spam_score`, `evidence`.
- Spam example trả về `spam`.
- Ham example trả về `not spam`.
- Upload file `.txt` dự đoán được.
- API reject input rỗng.
- Evidence có câu giải thích và token ảnh hưởng.

## 7. Lệnh chạy

Cài thư viện:

```powershell
python -m pip install -r requirements.txt
```

Chạy giao diện:

```powershell
python app.py
```

Mở trình duyệt:

```text
http://127.0.0.1:5000
```

Chạy test:

```powershell
python -m unittest tests.test_spam_checker_ui -v
python -m unittest discover -s tests -v
```
