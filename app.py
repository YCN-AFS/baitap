# app.py
import streamlit as st
from underthesea import word_tokenize, pos_tag

st.set_page_config(page_title="Demo POS Tagging Tiếng Việt", layout="wide")

st.title("Demo POS Tagging Tiếng Việt với Streamlit")
st.write("Nhập một câu tiếng Việt, ứng dụng sẽ tách từ và gán nhãn từ loại.")

# Input
text = st.text_area(
    "Nhập câu tiếng Việt ở đây:",
    "Hệ thống phân loại bình luận tiếng Việt rất chính xác.",
    height=100
)

analyze_clicked = st.button("🔍 Phân tích", type="primary", width="stretch")

col1, col2 = st.columns(2)

import pandas as pd
import base64

# Bảng giải thích nhãn từ loại
POS_TAGS_EXPLANATION = {
    "N": "Danh từ",
    "Np": "Danh từ riêng",
    "Nc": "Danh từ chỉ loại",
    "Nu": "Danh từ đơn vị",
    "V": "Động từ",
    "A": "Tính từ",
    "P": "Đại từ",
    "R": "Phó từ",
    "L": "Định từ",
    "M": "Số từ",
    "E": "Giới từ",
    "C": "Liên từ",
    "I": "Thán từ",
    "T": "Trợ từ, tiểu từ",
    "B": "Từ gốc Hán-Việt",
    "Y": "Từ viết tắt",
    "S": "Từ ngoại lai",
    "X": "Từ không phân loại",
    "CH": "Dấu câu",
}

# Màu cho từng loại từ loại
POS_COLORS = {
    "N": "#FF6B6B",
    "Np": "#FF4444",
    "Nc": "#FF8888",
    "Nu": "#FFAAAA",
    "V": "#4ECDC4",
    "A": "#FFE66D",
    "P": "#A8E6CF",
    "R": "#95E1D3",
    "L": "#DDA0DD",
    "M": "#87CEEB",
    "E": "#FFA07A",
    "C": "#98D8C8",
    "I": "#F7DC6F",
    "T": "#BB8FCE",
    "B": "#F0B27A",
    "Y": "#AED6F1",
    "S": "#F5B7B1",
    "X": "#D5DBDB",
    "CH": "#BDC3C7",
}

if analyze_clicked:
    if not text.strip():
        # Nhiệm vụ 4: Xử lý lỗi khi input rỗng
        st.warning("Vui lòng nhập văn bản trước khi phân tích.")
    else:
        # Nhiệm vụ 1: Xử lý tokenize và hiển thị kết quả ở col1
        with col1:
            st.subheader("1. Kết quả Tách từ (Tokenize)")
            tokens = word_tokenize(text)
            st.write(tokens)
            
        # Nhiệm vụ 2 + 6: Xử lý POS tagging, highlight màu và hiển thị ở col2
        with col2:
            st.subheader("2. Kết quả Gán nhãn Từ loại (POS Tagging)")
            pos_tags = pos_tag(text)
            
            # Highlight màu cho từng loại từ loại khác nhau
            html_content = "<div style='line-height: 2.5;'>"
            for word, tag in pos_tags:
                color = POS_COLORS.get(tag, "#E5E7EB") # Màu mặc định
                html_content += f"<span style='background-color: {color}; color: #000; padding: 0.2rem 0.6rem; border-radius: 0.3rem; margin-right: 0.4rem; font-weight: 500;'>{word} <span style='font-size: 0.8em; opacity: 0.8;'>[{tag}]</span></span> "
            html_content += "</div>"
            
            st.markdown(html_content, unsafe_allow_html=True)
            
        st.divider()
        # Nhiệm vụ 5: Export kết quả ra file CSV
        st.subheader("3. Xuất dữ liệu CSV")
        df_export = pd.DataFrame(pos_tags, columns=["Từ (Word)", "Nhãn Từ Loại (POS Tag)"])
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="⬇️ Tải file CSV",
            data=csv_data,
            file_name='pos_tag_result.csv',
            mime='text/csv',
        )

# Nhiệm vụ 3: Bảng giải thích các nhãn từ loại
st.divider()
st.subheader("Bảng giải thích các nhãn từ loại phổ biến")

df_explanation = pd.DataFrame({
    "Nhãn (Tag)": list(POS_TAGS_EXPLANATION.keys()),
    "Giải thích": list(POS_TAGS_EXPLANATION.values())
})

st.table(df_explanation)