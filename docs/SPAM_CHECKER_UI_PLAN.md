# Kế hoạch giao diện kiểm tra email spam

## 1. Mục tiêu

Tạo một giao diện web chạy local để người dùng kiểm tra một email có phải spam hay không.

Giao diện cần trả về:

- Kết luận: `spam` hoặc `not spam`.
- Điểm spam: `spam_score` trong khoảng `0-1`.
- Dẫn chứng model: model đang dùng, metric đã train, các token/term trong email ảnh hưởng đến dự đoán.
- Nội dung email sau khi tiền xử lý để nhóm giải thích khi thuyết trình.
- Cách nhập email bằng paste text, upload file `.txt/.eml`, và module Gmail OAuth tùy chọn.

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
src/gmail_reader.py
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
- Nhập Gmail message id/link để thử module Gmail OAuth nếu đã có `credentials.json`.

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
- Metric từ `reports/model_metrics.csv`.
- Top token/term có ảnh hưởng.
- Text đã clean.

## 4. Dẫn chứng từ đâu?

Kết quả không phải do giao diện tự đoán. Dẫn chứng lấy từ:

- File model đã train: `models/spam_classifier.joblib`.
- Hàm dự đoán: `src/predict.py`.
- Metric đã train: `reports/model_metrics.csv`.
- Feature/term trong pipeline TF-IDF của model:
  - Với Linear SVM hoặc Logistic Regression: lấy hệ số `coef_` của model.
  - Với Naive Bayes: lấy chênh lệch log probability giữa spam và not spam.
  - Chỉ hiển thị những token thật sự xuất hiện trong email sau khi vectorize.

## 5. Gmail integration

Không đọc được link Gmail cá nhân trực tiếp nếu chưa có quyền OAuth.

Lý do:

- Gmail web link chỉ mở được trong trình duyệt đã đăng nhập.
- Backend local không có quyền đọc nội dung thư từ link đó.
- Google yêu cầu OAuth cho app truy cập dữ liệu Gmail.

Thiết kế:

- `src/gmail_reader.py` cung cấp hàm đọc Gmail bằng Gmail API khi người dùng đặt `credentials.json` trong thư mục root.
- Nếu chưa có credentials hoặc thiếu thư viện Google, giao diện hiển thị hướng dẫn thay vì crash.
- Trước mắt người dùng có thể copy nội dung mail từ Gmail rồi paste vào giao diện, hoặc tải mail dạng `.eml` nếu có.

## 6. Test cần có

Test trong `tests/test_spam_checker_ui.py`:

- Trang chủ load được.
- API `/api/check` trả JSON có `prediction`, `spam_score`, `evidence`.
- Spam example trả về `spam`.
- Ham example trả về `not spam`.
- Upload file `.txt` dự đoán được.
- API reject input rỗng.
- Evidence có metric model và token giải thích.
- Gmail route không crash khi chưa có OAuth credentials.

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
