# Hướng dẫn cài đặt

## Yêu cầu

- Python 3.x
- macOS / Linux / Windows

## Cài đặt

1. Tạo môi trường ảo:

```bash
python3 -m venv venv
```

2. Kích hoạt môi trường ảo:

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. Cài đặt các thư viện cần thiết:

```bash
pip install streamlit underthesea pandas
```

## Quản lý thư viện với requirements.txt

- Export danh sách thư viện hiện tại:

```bash
pip freeze > requirements.txt
```

- Cài đặt từ file requirements.txt:

```bash
pip install -r requirements.txt
```

## Cài đặt Xcode Command Line Tools (macOS)

Để `watchdog` hoạt động tốt trên macOS, cần cài Xcode Command Line Tools:

```bash
xcode-select --install
```

## Ghi chú

- Mỗi lần mở terminal mới, cần chạy lại lệnh `source venv/bin/activate` để kích hoạt môi trường ảo.
- Nếu gặp lỗi `externally-managed-environment`, hãy đảm bảo đã kích hoạt môi trường ảo trước khi chạy `pip install`.
