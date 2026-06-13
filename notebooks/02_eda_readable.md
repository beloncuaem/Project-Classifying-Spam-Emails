# Giải thích EDA

Mục đích: giải thích dễ hiểu từng bước trong notebook `notebooks/01_eda.ipynb` (phiên bản code hiện tại) để Dế biết mỗi dòng code làm gì và cho thông tin gì.

---

## Cell 1 (code) — Tổng quan các bước

1. import pandas as pd / import matplotlib.pyplot as plt / import seaborn as sns
   - Nạp thư viện: `pandas` xử lý dữ liệu bảng, `matplotlib`/`seaborn` vẽ biểu đồ.

2. # Đọc dữ liệu
   df = pd.read_csv('../data/raw/combined_data.csv')
   - Mở file CSV vào biến `df`. Nếu đường dẫn sai hoặc file không tồn tại, sẽ lỗi. `df` là bảng dữ liệu.

3. # Xem kích thước dữ liệu
   print(f"Tổng số dòng: {len(df)}")
   print(f"Số cột: {len(df.columns)}")
   - Hiển thị số lượng mẫu (hàng) và số cột (thuộc tính). Biết quy mô dữ liệu.

4. # Xem 5 dòng đầu
   print("\n5 dòng đầu tiên:")
   print(df.head())
   - Cho ví dụ mẫu của dữ liệu: tên cột, ví dụ text và label; phát hiện nhanh lỗi encoding.

5. # Kiểm tra thông tin
   print("\nThông tin dữ liệu:")
   print(df.info())
   - Hiển thị kiểu dữ liệu từng cột, số giá trị không rỗng. Dùng để phát hiện cột thiếu giá trị hoặc cột không phải kiểu mong đợi.

6. # Kiểm tra giá trị null
   print("\nSố lượng giá trị null:")
   print(df.isnull().sum())
   - Đếm giá trị trống theo cột. Nếu `text` có null thì cần xử lý trước khi tính độ dài.

7. # Phân bố nhãn (spam=1, ham=0)
   print("\nPhân bố nhãn:")
   print(df['label'].value_counts())
   - Đếm số email từng nhãn. Quan trọng để biết dữ liệu có cân bằng không.

8. # Vẽ biểu đồ phân bố spam/ham
   plt.figure(figsize=(6,4))
   df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
   plt.title('Phân bố email Spam vs Ham')
   plt.xlabel('Nhãn (0=Ham, 1=Spam)')
   plt.ylabel('Số lượng')
   plt.xticks(rotation=0)
   plt.show()
   - Vẽ cột thể hiện số lượng ham và spam để nhìn trực quan mức độ mất cân bằng.

9. # Thêm cột độ dài email
   df['length'] = df['text'].apply(len)
   - Tạo cột `length` bằng độ dài chuỗi ở cột `text`. Phải chắc `text` không rỗng.

10. # Thống kê độ dài
    print("\nThống kê độ dài email:")
    print(df['length'].describe())
    - Hiển thị min, max, trung vị, trung bình, và các percentiles của độ dài văn bản.

11. # Vẽ histogram độ dài email
    plt.figure(figsize=(10,5))
    plt.hist(df[df['label']==0]['length'], bins=50, alpha=0.7, label='Ham', color='green')
    plt.hist(df[df['label']==1]['length'], bins=50, alpha=0.7, label='Spam', color='red')
    plt.title('Phân bố độ dài email theo loại')
    plt.xlabel('Độ dài')
    plt.ylabel('Số lượng')
    plt.legend()
    plt.show()
    - So sánh phân bố độ dài giữa ham và spam; giúp biết liệu độ dài có phân biệt hai lớp hay không.

12. print(" EDA hoàn thành!")
    - Thông báo kết thúc.

---

## Ghi chú quan trọng (vì Dế dễ quên)
- Nếu cột chứa nội dung tên khác (`body`, `message`) thì thay `df['text']` bằng tên đúng.
- Trước khi chạy `len`, nên kiểm tra `df['text'].isnull().sum()` để tránh lỗi với giá trị rỗng.
- Đường dẫn file phải đúng; trong repo hiện có nhiều file CSV ở `data/processed/` — nếu `combined_data.csv` không ở `data/raw/`, hãy sửa đường dẫn.

## Muốn tôi làm gì tiếp?
- Chạy notebook và đưa kết quả (nếu muốn tôi chạy, tôi sẽ sửa path và thực thi các bước). 
- Hoặc tôi có thể tạo phiên bản script `scripts/run_eda.py` để Dế chạy nhanh trong PowerShell.


---

Tệp này đã được lưu tại: `notebooks/01_eda_readable_for_de.md` (mở file này trong VS Code để xem).