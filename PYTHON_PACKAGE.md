# Giải thích các thư viện Python

## streamlit

Framework xây dựng ứng dụng web tương tác cho dữ liệu và machine learning. Cho phép tạo giao diện web chỉ bằng Python, không cần HTML/CSS/JS. Thường dùng để tạo dashboard, demo model AI, hoặc công cụ phân tích dữ liệu.

- Trang chủ: https://streamlit.io
- Cài đặt: `pip install streamlit`
- Chạy ứng dụng: `streamlit run app.py`

## underthesea

Thư viện xử lý ngôn ngữ tự nhiên (NLP) dành riêng cho tiếng Việt. Hỗ trợ các tác vụ:

- Tách từ (word segmentation): `word_tokenize("Tôi yêu Việt Nam")`
- Gán nhãn từ loại (POS tagging)
- Nhận diện thực thể (NER)
- Phân loại văn bản (text classification)
- Phân tích cảm xúc (sentiment analysis)

Trang chủ: https://github.com/undertheseanlp/underthesea

## pandas

Thư viện phân tích và xử lý dữ liệu phổ biến nhất trong Python. Cung cấp cấu trúc `DataFrame` để làm việc với dữ liệu dạng bảng (tương tự Excel, SQL). Hỗ trợ:

- Đọc/ghi dữ liệu từ CSV, Excel, JSON, SQL
- Lọc, sắp xếp, nhóm dữ liệu
- Thống kê, tổng hợp dữ liệu
- Xử lý dữ liệu thiếu (missing data)

Trang chủ: https://pandas.pydata.org

## watchdog

Thư viện giám sát thay đổi file trên hệ thống (file system events). Streamlit sử dụng `watchdog` để tự động reload ứng dụng khi file mã nguồn thay đổi, giúp quá trình phát triển nhanh hơn.

- Trên macOS, cần cài `xcode-select --install` để watchdog hoạt động tối ưu.

Trang chủ: https://github.com/gorakhargosh/watchdog
