# Đại Học Lạc Hồng - Xử Lý Ngôn Ngữ Tự Nhiên - Cơ Sở Ngôn Ngữ Học
Cơ Sở Ngôn Ngữ Học – Hỗ Trợ Xử Lý Ngôn Ngữ Tiếng Việt

**Thông tin sinh viên:**

| STT | Mã Số Sinh Viên | Họ và Tên |
|-----|-----------------|-----------|
| 1   | 123000007       | Trần Văn Hội |
| 2   |                 |           |

## Mô tả dự án

Ứng dụng web Streamlit hỗ trợ xử lý ngôn ngữ tiếng Việt với các chức năng: tokenize (tách từ), POS tagging (gán nhãn từ loại), hiển thị kết quả trực quan với highlight màu, và export kết quả ra CSV.

## Các nhiệm vụ cần hoàn thành trong `app.py`

### Nhiệm vụ 1: Xử lý tokenize và hiển thị kết quả ở col1
- Sử dụng thư viện `underthesea` hoặc `pyvi` để tách từ tiếng Việt
- Hiển thị danh sách các token trong cột bên trái của giao diện Streamlit

### Nhiệm vụ 2: Xử lý POS tagging và hiển thị kết quả ở col2
- Gán nhãn từ loại (danh từ, động từ, tính từ,...) cho từng token
- Hiển thị kết quả POS tagging trong cột bên phải

### Nhiệm vụ 3: Bảng giải thích các nhãn từ loại (POS tags)
- Tạo bảng tra cứu các nhãn POS phổ biến trong tiếng Việt:
  - `N` - Danh từ
  - `V` - Động từ
  - `A` - Tính từ
  - `P` - Đại từ
  - `R` - Phó từ
  - `E` - Giới từ
  - `C` - Liên từ
  - `M` - Số từ
  - `Np` - Danh từ riêng
  - v.v.

### Nhiệm vụ 4: Xử lý lỗi khi input rỗng
- Kiểm tra input trước khi xử lý
- Hiển thị thông báo cảnh báo nếu người dùng chưa nhập văn bản

### Nhiệm vụ 5: Export kết quả ra file CSV
- Cho phép người dùng tải xuống kết quả tokenize và POS tagging dưới dạng file CSV
- Sử dụng `st.download_button` của Streamlit

### Nhiệm vụ 6: Highlight màu cho từng loại từ loại khác nhau
- Gán màu khác nhau cho mỗi nhãn POS (ví dụ: danh từ - xanh dương, động từ - đỏ, tính từ - xanh lá,...)
- Hiển thị văn bản đã gán nhãn với màu sắc tương ứng bằng HTML/CSS trong Streamlit

## Công nghệ sử dụng
- Python
- Streamlit
- underthesea (xử lý NLP tiếng Việt)
- pandas (xử lý dữ liệu, export CSV)
# baitap
# baitap
# baitap
# baitap
