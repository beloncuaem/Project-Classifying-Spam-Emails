# Final Code Review Report

Ngày kiểm tra: 2026-06-19

Repo được clone sparse từ:

```text
https://github.com/beloncuaem/Project-Classifying-Spam-Emails.git
```

Branch local:

```text
final-review
```

## 1. Việc đã làm

- Xem lại toàn bộ notebook hiện có trong `notebooks/`.
- Tạo notebook tổng hợp cuối: `notebooks/99_final_project_notebook.ipynb`.
- Thêm rule chạy project: `docs/AI_RUNBOOK_RULES.md`.
- Thêm rule chuẩn code/notebook: `docs/CODE_AND_NOTEBOOK_RULES.md`.
- Kiểm tra cú pháp Python.
- Chạy toàn bộ test.
- Kiểm tra notebook cuối là JSON hợp lệ.
- Kiểm tra toàn bộ 8 notebook parse được.
- Kiểm tra các file chính không có dấu hiệu mojibake.
- Kiểm tra naming convention cơ bản cho file/function/class Python.

## 2. Kết quả kiểm tra

### Notebook JSON

Lệnh:

```powershell
python -m json.tool notebooks/99_final_project_notebook.ipynb > $null
```

Kết quả: pass.

Kiểm tra toàn bộ notebook:

```text
notebooks_checked = 8
bad = []
```

### Python syntax

Lệnh:

```powershell
$files = @('app.py','config.py','main.py') + (Get-ChildItem src,tests -Filter *.py | ForEach-Object { $_.FullName })
python -m py_compile $files
```

Kết quả: pass.

### Unit tests

Lệnh:

```powershell
python -m unittest discover -s tests -v
```

Kết quả: 15/15 tests pass.

### Naming và encoding

Kết quả scan:

```text
naming_issues = [('tests/test_spam_checker_ui.py', 'function', 'setUpClass')]
encoding_hits = []
```

Ghi chú: `setUpClass` là method name chuẩn của `unittest`, nên không tính là lỗi naming của project.

## 3. Đánh giá code theo chuẩn

| Hạng mục | Trạng thái | Ghi chú |
| --- | --- | --- |
| Cấu trúc thư mục | Đạt | Source, notebooks, reports, models, tests, UI tách rõ |
| Naming convention | Đạt | File/function/variable chính dùng snake_case |
| Label mapping | Đạt | `0 = not spam`, `1 = spam` |
| Preprocessing | Đạt | Có clean HTML, URL, email, number, stopwords fallback |
| Train model | Đạt | Có NB, LR, Linear SVM và lưu best model |
| Evaluation | Đạt | Có metrics, confusion matrix, comparison |
| Prediction | Đạt | Có CLI và dùng model đã train |
| Web UI | Đạt | Paste/upload email, trả prediction, score, dẫn chứng |
| Gmail optional | Đã bỏ | Đúng yêu cầu mới của người dùng |
| Notebook cuối | Đạt | Đã tổng hợp thành `99_final_project_notebook.ipynb` |
| Test suite | Đạt | 15/15 tests pass |
| Encoding | Đạt | Không phát hiện mojibake trong source/docs/notebooks đã scan |
| Naming scan | Đạt | Chỉ có `setUpClass` theo chuẩn `unittest` |

## 4. Ghi chú còn lại

- Repo có dữ liệu/model/report đã được commit từ trước. Khi clone full có thể lâu vì data lớn, nên lần kiểm tra này dùng sparse checkout để tránh kéo `data/raw` nặng.
- Nếu muốn chạy lại toàn bộ data pipeline từ đầu, cần clone đủ dữ liệu hoặc cho phép script tải lại source data.
- Giao diện Flask phải mở bằng `http://127.0.0.1:5000`, không mở trực tiếp file template qua Live Server.

## 5. Kết luận

Code hiện tại đạt chuẩn để chạy demo, test và thuyết trình. Notebook cuối đã gom các phần quan trọng của project thành một luồng báo cáo thống nhất.
