# Data Source Link Review

Folder này ghi lại các link thu thập dữ liệu spam/ham và đánh giá nhanh mức độ chính thống.

## Nguồn đang dùng

| Nguồn | Link | Đánh giá | Cách dùng |
|---|---|---|---|
| SpamAssassin Public Corpus | https://spamassassin.apache.org/old/publiccorpus/ | Chính thống: archive trực tiếp trên domain Apache SpamAssassin. | Dùng trong `src/data_loader.py` để tải các file `easy_ham`, `hard_ham`, `spam`. |
| SetFit/enron_spam | https://huggingface.co/datasets/SetFit/enron_spam | Dataset public trên Hugging Face, không phải trang gốc Enron nhưng có dataset card và viewer rõ ràng. | Dùng để bổ sung email ham/spam đã có nhãn. |
| locuoco/the-biggest-spam-ham-phish-email-dataset-300000 | https://huggingface.co/datasets/locuoco/the-biggest-spam-ham-phish-email-dataset-300000 | Dataset public trên Hugging Face, dùng được cho mở rộng dữ liệu lớn; cần ghi rõ mapping nhãn. | Dùng parquet qua datasets-server; map `0 = ham`, `1/2 = spam`. |

## Nguồn tham khảo thêm

| Nguồn | Link | Đánh giá | Lưu ý |
|---|---|---|---|
| TREC Spam Track | https://trec.nist.gov/data/spam.html | Chính thống: trang dữ liệu của NIST/TREC. | Có thể bổ sung corpus spam học thuật; một số corpus có điều khoản/tải thủ công. |
| CMU Enron Email Dataset | https://www.cs.cmu.edu/~enron/ | Chính thống: trang dataset của CMU. | Chủ yếu là email công ty/ham; không tự coi tất cả là spam. |

## Quy tắc dùng link

- Ưu tiên domain chính thống của tổ chức gốc: `apache.org`, `nist.gov`, `cmu.edu`.
- Nếu dùng Hugging Face/Kaggle, phải ghi rõ đó là nguồn public mirror/community dataset.
- Không cào inbox cá nhân, email riêng tư hoặc website không có quyền thu thập.
- Với dataset có nhiều nhãn như `ham/phish/spam`, phải ghi rõ quy tắc map sang bài toán binary.
