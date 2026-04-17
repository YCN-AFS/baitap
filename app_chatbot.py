# ============================================================
# app_chatbot.py – Chatbot Phân tích Phản hồi Sinh viên
# Đầy đủ tất cả TODO 1–15
# Yêu cầu: pip install streamlit pandas underthesea wordcloud matplotlib langdetect openpyxl
# ============================================================

# ============================================================
# IMPORTS
# ============================================================
import json
import os
import io
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── underthesea (TODO 2: cache resource) ──────────────────────
try:
    from underthesea import sentiment as uts_sentiment
    from underthesea import word_tokenize as uts_tokenize
    from underthesea import langdetect as uts_langdetect
    UNDERTHESEA_OK = True
except ImportError:
    UNDERTHESEA_OK = False

# ── WordCloud ──────────────────────────────────────────────────
try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except ImportError:
    WORDCLOUD_OK = False

# ── langdetect fallback ────────────────────────────────────────
try:
    from langdetect import detect as ld_detect
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False


# ============================================================
# CONSTANTS
# ============================================================
HISTORY_PATH = "chat_history.json"
STOPWORDS_PATH = "stopwords_vi.txt"
EMOJI_MAP = {"positive": "😊", "negative": "😟", "neutral": "😐"}

# Stopwords nội tuyến mặc định (dùng khi không có file ngoài) – TODO 1
_DEFAULT_STOPWORDS = (
    "và,của,là,trong,có,được,cho,với,từ,đến,những,các,một,này,đó,như,hay,hoặc,"
    "không,thì,về,ra,đã,sẽ,bị,mà,cũng,theo,vào,rất,khi,nhiều,nên,hơn,lên,"
    "tôi,em,anh,chị,thầy,cô,bạn,họ,chúng,mình,ta,bởi,vì,vậy,nhưng,nếu,"
    "còn,hết,lại,đang,đây,đó,nào,nên,thêm,tất,cả,nhau,qua,dù,tuy,luôn,đây,"
    "được,rằng,đã,cần,giúp,để,làm,sau,trước,hơn,nữa,thật,quá,ít,nhiều,lắm"
)


# ============================================================
# TODO 1 – STOPWORDS
# ============================================================
@st.cache_data(show_spinner=False)
def load_stopwords(path: str = STOPWORDS_PATH) -> set:
    """Đọc stopwords từ file ngoài; fallback về danh sách nội tuyến."""
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            words = {w.strip().lower() for w in f if w.strip()}
        return words
    return {w.strip() for w in _DEFAULT_STOPWORDS.split(",")}


# ============================================================
# TODO 2 – CACHE MODEL  (underthesea tự cache nội bộ; ta cache wrapper)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_nlp_pipeline():
    """Trả về dict chứa các hàm NLP để tránh import lại mỗi lần rerun."""
    return {
        "sentiment": uts_sentiment if UNDERTHESEA_OK else None,
        "tokenize": uts_tokenize if UNDERTHESEA_OK else None,
    }


# ============================================================
# TODO 8, 13 – ANALYZE FEEDBACK (cảm xúc + từ khóa + confidence)
# ============================================================
def analyze_feedback(text: str, stopwords: set | None = None) -> dict:
    """
    Phân tích cảm xúc và trích xuất từ khóa.
    Trả về dict: sentiment, confidence, keywords, token_count, language, warning.
    """
    if stopwords is None:
        stopwords = load_stopwords()

    result = {
        "text": text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sentiment": "neutral",
        "confidence": 0.0,
        "keywords": [],
        "token_count": 0,
        "language": "vi",
        "warning": None,
    }

    # ── TODO 13: Edge cases ──────────────────────────────────
    cleaned = text.strip()
    if not cleaned:
        result["warning"] = "Văn bản rỗng."
        return result

    # Emoji-only
    no_emoji = re.sub(r"[^\w\s]", "", cleaned, flags=re.UNICODE).strip()
    if not no_emoji:
        result["warning"] = "Phản hồi chỉ chứa emoji/ký tự đặc biệt."
        # Heuristic cảm xúc từ emoji
        positive_emojis = set("😊👍❤️😍🥰😄😃🎉✅🌟💯")
        negative_emojis = set("😢😞👎😡🤬😖😭❌💔")
        emojis_in = set(cleaned)
        if emojis_in & positive_emojis:
            result["sentiment"] = "positive"
            result["confidence"] = 0.6
        elif emojis_in & negative_emojis:
            result["sentiment"] = "negative"
            result["confidence"] = 0.6
        return result

    # Quá ngắn (≤ 2 token)
    rough_tokens = cleaned.split()
    if len(rough_tokens) <= 2:
        result["warning"] = "Phản hồi quá ngắn, kết quả có thể kém chính xác."

    # ── TODO 9: Detect ngôn ngữ ──────────────────────────────
    result["language"] = detect_language(cleaned)

    # ── Phân tích cảm xúc ────────────────────────────────────
    nlp = get_nlp_pipeline()
    if nlp["sentiment"] and result["language"] == "vi":
        try:
            raw = nlp["sentiment"](cleaned)
            # underthesea trả về label hoặc (label, score)
            if isinstance(raw, (list, tuple)) and len(raw) == 2:
                label, score = raw
                result["sentiment"] = str(label).lower()
                result["confidence"] = round(float(score), 3)
            else:
                result["sentiment"] = str(raw).lower()
                result["confidence"] = 0.75  # default khi không có score
        except Exception:
            result["sentiment"] = _rule_based_sentiment(cleaned)
            result["confidence"] = 0.5
    else:
        result["sentiment"] = _rule_based_sentiment(cleaned)
        result["confidence"] = 0.5

    # Chuẩn hoá nhãn
    label = result["sentiment"]
    if "pos" in label or label == "1":
        result["sentiment"] = "positive"
    elif "neg" in label or label == "-1":
        result["sentiment"] = "negative"
    else:
        result["sentiment"] = "neutral"

    # ── Trích xuất từ khóa ────────────────────────────────────
    if nlp["tokenize"] and result["language"] == "vi":
        try:
            tokens = nlp["tokenize"](cleaned, format="text").split()
        except Exception:
            tokens = cleaned.split()
    else:
        tokens = cleaned.split()

    result["token_count"] = len(tokens)
    keywords = [
        t.lower() for t in tokens
        if len(t) >= 2
        and t.lower() not in stopwords
        and re.search(r"\w", t)
    ]
    # Top 10 từ khoá theo tần suất
    from collections import Counter
    freq = Counter(keywords)
    result["keywords"] = [w for w, _ in freq.most_common(10)]

    return result


def _rule_based_sentiment(text: str) -> str:
    """Phân tích cảm xúc đơn giản dựa trên từ điển khi không có model."""
    pos = {"tốt", "hay", "xuất sắc", "tuyệt", "thích", "hài lòng", "ổn",
           "nhanh", "nhiệt tình", "rõ ràng", "dễ hiểu", "hữu ích", "tốt lắm"}
    neg = {"tệ", "kém", "chán", "dở", "thất vọng", "không hiểu", "khó",
           "chậm", "khó hiểu", "nhàm", "buồn", "mệt", "không ổn", "quá khó"}
    tl = text.lower()
    pos_score = sum(1 for w in pos if w in tl)
    neg_score = sum(1 for w in neg if w in tl)
    if pos_score > neg_score:
        return "positive"
    if neg_score > pos_score:
        return "negative"
    return "neutral"


# ============================================================
# TODO 10 – RENDER ANALYSIS (markdown đẹp cho chat bubble)
# ============================================================
def render_analysis(result: dict) -> str:
    emoji = EMOJI_MAP.get(result["sentiment"], "😐")
    label_vi = {"positive": "Tích cực", "negative": "Tiêu cực", "neutral": "Trung lập"}
    conf_pct = int(result["confidence"] * 100)
    conf_bar = "█" * (conf_pct // 10) + "░" * (10 - conf_pct // 10)
    kw_str = ", ".join(f"`{k}`" for k in result["keywords"]) if result["keywords"] else "_Không có_"
    lang_flag = {"vi": "🇻🇳", "en": "🇬🇧"}.get(result["language"], "🌐")
    warn_str = f"\n> ⚠️ **Lưu ý:** {result['warning']}" if result.get("warning") else ""

    md = f"""
{warn_str}
| Chỉ số | Giá trị |
|--------|---------|
| **Cảm xúc** | {emoji} {label_vi.get(result['sentiment'], result['sentiment'])} |
| **Độ tin cậy** | `{conf_bar}` {conf_pct}% |
| **Số token** | {result['token_count']} |
| **Ngôn ngữ** | {lang_flag} `{result['language']}` |

**🔑 Từ khóa nổi bật:** {kw_str}
""".strip()
    return md


# ============================================================
# TODO 3 – FILE UPLOAD (CSV / Excel)
# ============================================================
def handle_file_upload(uploaded_file) -> list:
    """Đọc file CSV/Excel upload, trả về list phản hồi (str)."""
    if uploaded_file is None:
        return []
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return []

    # Ưu tiên cột có tên gợi ý
    for col in df.columns:
        if any(k in col.lower() for k in ("phan_hoi", "phản hồi", "feedback", "noi_dung", "comment", "text")):
            return df[col].dropna().astype(str).tolist()
    # Fallback: lấy cột đầu tiên
    return df.iloc[:, 0].dropna().astype(str).tolist()


# ============================================================
# TODO 4 – EXPORT HISTORY
# ============================================================
def export_history(history: list) -> bytes:
    """Chuyển lịch sử phân tích thành Excel bytes để download."""
    if not history:
        return b""
    rows = []
    for item in history:
        rows.append({
            "Thời gian": item.get("timestamp", ""),
            "Phản hồi": item.get("text", ""),
            "Cảm xúc": item.get("sentiment", ""),
            "Độ tin cậy (%)": int(item.get("confidence", 0) * 100),
            "Ngôn ngữ": item.get("language", ""),
            "Số token": item.get("token_count", 0),
            "Từ khóa": ", ".join(item.get("keywords", [])),
            "Cảnh báo": item.get("warning") or "",
        })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Lịch sử")
    return buf.getvalue()


# ============================================================
# TODO 5 – WORD CLOUD
# ============================================================
def render_wordcloud(keywords_all: list):
    """Vẽ word cloud từ toàn bộ từ khóa trong lịch sử."""
    if not WORDCLOUD_OK:
        st.info("Cài `wordcloud` để hiển thị word cloud: `pip install wordcloud`")
        return
    if not keywords_all:
        st.caption("Chưa có từ khóa.")
        return
    text = " ".join(keywords_all)
    wc = WordCloud(
        width=600, height=280,
        background_color=None, mode="RGBA",
        colormap="plasma",
        max_words=80,
        prefer_horizontal=0.85,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(6, 2.8), facecolor="none")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# TODO 6 – SENTIMENT TIMELINE
# ============================================================
def render_sentiment_timeline(history: list):
    """Vẽ biểu đồ xu hướng cảm xúc theo thời gian."""
    if len(history) < 2:
        st.caption("Cần ít nhất 2 phản hồi để vẽ timeline.")
        return
    score_map = {"positive": 1, "neutral": 0, "negative": -1}
    times = [h.get("timestamp", "")[-8:] for h in history]
    scores = [score_map.get(h.get("sentiment", "neutral"), 0) for h in history]
    colors = ["#4ade80" if s == 1 else "#f87171" if s == -1 else "#94a3b8" for s in scores]
    fig, ax = plt.subplots(figsize=(6, 2.2), facecolor="none")
    ax.set_facecolor("none")
    ax.plot(range(len(scores)), scores, color="#60a5fa", linewidth=1.5, zorder=1)
    ax.scatter(range(len(scores)), scores, c=colors, s=60, zorder=2)
    ax.axhline(0, color="#475569", linewidth=0.8, linestyle="--")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Tiêu cực", "Trung lập", "Tích cực"], fontsize=8, color="#94a3b8")
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45, fontsize=7, color="#94a3b8")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#94a3b8", length=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# TODO 5, 12 – SIDEBAR STATS
# ============================================================
def render_sidebar_stats(history: list):
    """Hiển thị thống kê tổng hợp trên sidebar."""
    st.markdown("### 📊 Thống kê tổng hợp")
    if not history:
        st.caption("Chưa có phản hồi nào.")
        return

    total = len(history)
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for h in history:
        counts[h.get("sentiment", "neutral")] = counts.get(h.get("sentiment", "neutral"), 0) + 1

    col1, col2, col3 = st.columns(3)
    col1.metric("😊 Tích cực", counts["positive"])
    col2.metric("😟 Tiêu cực", counts["negative"])
    col3.metric("😐 Trung lập", counts["neutral"])

    # Mini pie
    labels = [k for k, v in counts.items() if v > 0]
    values = [counts[k] for k in labels]
    label_vi = {"positive": "Tích cực", "negative": "Tiêu cực", "neutral": "Trung lập"}
    clr = {"positive": "#4ade80", "negative": "#f87171", "neutral": "#94a3b8"}
    fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor="none")
    ax.pie(values, labels=[label_vi[l] for l in labels],
           colors=[clr[l] for l in labels],
           autopct="%1.0f%%", textprops={"color": "#e2e8f0", "fontsize": 9},
           wedgeprops={"linewidth": 0})
    ax.set_facecolor("none")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Timeline
    st.markdown("**📈 Xu hướng cảm xúc**")
    render_sentiment_timeline(history)

    # Word cloud
    st.markdown("**☁️ Word Cloud từ khóa**")
    all_kw = [kw for h in history for kw in h.get("keywords", [])]
    render_wordcloud(all_kw)

    # TODO 12: So sánh 2 nhóm
    st.markdown("---")
    st.markdown("**⚖️ So sánh 2 nhóm phản hồi**")
    _render_comparison_mode(history)


def _render_comparison_mode(history: list):
    """TODO 12: Chia lịch sử thành 2 nhóm theo mốc thời gian và so sánh."""
    if len(history) < 4:
        st.caption("Cần ít nhất 4 phản hồi để so sánh.")
        return
    mid = len(history) // 2
    group_a = history[:mid]
    group_b = history[mid:]

    def avg_conf(g):
        return round(sum(x.get("confidence", 0) for x in g) / len(g) * 100, 1)

    def dominant(g):
        from collections import Counter
        c = Counter(x.get("sentiment", "neutral") for x in g)
        return c.most_common(1)[0][0]

    label_vi = {"positive": "Tích cực 😊", "negative": "Tiêu cực 😟", "neutral": "Trung lập 😐"}
    df_cmp = pd.DataFrame({
        "Nhóm": ["Nhóm A (đầu)", "Nhóm B (sau)"],
        "Số phản hồi": [len(group_a), len(group_b)],
        "Cảm xúc chủ đạo": [label_vi[dominant(group_a)], label_vi[dominant(group_b)]],
        "Độ tin cậy TB (%)": [avg_conf(group_a), avg_conf(group_b)],
    })
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)


# ============================================================
# TODO 9 – DETECT LANGUAGE
# ============================================================
def detect_language(text: str) -> str:
    """Detect ngôn ngữ, trả về mã ('vi', 'en', ...)."""
    # Nhanh: kiểm tra tỉ lệ ký tự có dấu tiếng Việt
    vi_chars = sum(1 for c in text if unicodedata.category(c) == "Mn"
                   or c in "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
    if vi_chars / max(len(text), 1) > 0.05:
        return "vi"
    if LANGDETECT_OK:
        try:
            return ld_detect(text)
        except Exception:
            pass
    return "vi"


# ============================================================
# TODO 10 – HELP PAGE
# ============================================================
def render_help_page():
    st.markdown("""
## 📖 Hướng dẫn sử dụng

### 1. Nhập phản hồi trực tiếp
Gõ phản hồi vào ô chat ở cuối trang. Có thể nhập nhiều dòng cùng lúc (mỗi dòng là một phản hồi độc lập).

### 2. Upload file CSV/Excel
Dùng nút **📂 Upload CSV/Excel** ở sidebar để phân tích hàng loạt phản hồi từ file.  
File cần có một cột tên chứa từ: `phản hồi`, `feedback`, `comment`, `text`, hoặc `noi_dung`.

### 3. Xuất kết quả
Nhấn **⬇️ Tải xuống Excel** ở sidebar để lưu toàn bộ lịch sử phân tích ra file.

### 4. Ý nghĩa các chỉ số

| Chỉ số | Ý nghĩa |
|--------|---------|
| **Cảm xúc** | Tích cực / Tiêu cực / Trung lập |
| **Độ tin cậy** | Mức độ chắc chắn của model (0–100%) |
| **Số token** | Số từ/cụm từ sau khi tách |
| **Ngôn ngữ** | Ngôn ngữ phát hiện được |
| **Từ khóa** | 10 từ/cụm từ xuất hiện nhiều nhất |

### 5. Xóa phản hồi
Mỗi phản hồi có nút 🗑️ để xóa khỏi lịch sử.

### 6. Lịch sử tự động lưu
Lịch sử được lưu vào `chat_history.json` và khôi phục khi reload trang.
""")


# ============================================================
# TODO 11 – PERSIST HISTORY
# ============================================================
def save_history(history: list, path: str = HISTORY_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # Silently fail (e.g. read-only filesystem)


def load_history(path: str = HISTORY_PATH) -> list:
    if not Path(path).exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


# ============================================================
# TODO 7 – DELETE FEEDBACK
# ============================================================
def delete_feedback(index: int):
    """Xóa một phản hồi theo index khỏi history + messages."""
    if index < 0 or index >= len(st.session_state.history):
        return
    st.session_state.history.pop(index)
    # Xây lại messages từ history
    msgs = [{"role": "assistant",
             "content": "👋 Xin chào! Nhập phản hồi sinh viên để tôi phân tích."}]
    for h in st.session_state.history:
        msgs.append({"role": "user", "content": h["text"]})
        msgs.append({"role": "assistant", "content": render_analysis(h)})
    st.session_state.messages = msgs
    save_history(st.session_state.history)


# ============================================================
# INIT SESSION STATE
# ============================================================
def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = load_history()
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant",
             "content": "👋 Xin chào! Nhập phản hồi sinh viên để tôi phân tích cảm xúc và trích xuất từ khóa."}
        ]
        # Khôi phục messages từ history
        for h in st.session_state.history:
            st.session_state.messages.append({"role": "user", "content": h["text"]})
            st.session_state.messages.append({"role": "assistant", "content": render_analysis(h)})
    if "page" not in st.session_state:
        st.session_state.page = "chat"
    if "stopwords" not in st.session_state:
        st.session_state.stopwords = load_stopwords()


# ============================================================
# MAIN
# ============================================================
def main():
    st.set_page_config(
        page_title="Chatbot Phân tích Phản hồi SV",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Global CSS ────────────────────────────────────────────
    st.markdown("""
<style>
[data-testid="stChatMessage"] { border-radius: 12px; padding: 8px; }
[data-testid="stSidebar"] { background: #0f172a; }
.stDataFrame { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

    init_session_state()
    stopwords = st.session_state.stopwords

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🎓 Phân tích Phản hồi SV")
        page = st.radio("Trang", ["💬 Chat phân tích", "📖 Hướng dẫn"], label_visibility="collapsed")
        st.session_state.page = page

        st.markdown("---")

        # TODO 3: Upload file
        st.markdown("#### 📂 Upload CSV/Excel")
        uploaded = st.file_uploader("Chọn file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
        if uploaded and st.button("▶️ Phân tích file"):
            feedbacks = handle_file_upload(uploaded)
            if feedbacks:
                with st.spinner(f"Đang phân tích {len(feedbacks)} phản hồi…"):
                    for fb in feedbacks:
                        result = analyze_feedback(fb, stopwords)
                        st.session_state.history.append(result)
                        st.session_state.messages.append({"role": "user", "content": fb})
                        st.session_state.messages.append(
                            {"role": "assistant", "content": render_analysis(result)}
                        )
                save_history(st.session_state.history)
                st.success(f"✅ Đã phân tích {len(feedbacks)} phản hồi!")
                st.rerun()

        # TODO 4: Export
        st.markdown("---")
        if st.session_state.history:
            xlsx_bytes = export_history(st.session_state.history)
            st.download_button(
                "⬇️ Tải xuống Excel",
                data=xlsx_bytes,
                file_name=f"lich_su_phan_tich_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        # Xóa toàn bộ
        if st.button("🗑️ Xóa toàn bộ lịch sử"):
            st.session_state.history = []
            st.session_state.messages = [
                {"role": "assistant",
                 "content": "👋 Xin chào! Nhập phản hồi sinh viên để tôi phân tích."}
            ]
            save_history([])
            st.rerun()

        st.markdown("---")
        render_sidebar_stats(st.session_state.history)

    # ── MAIN AREA ─────────────────────────────────────────────
    if "Hướng dẫn" in st.session_state.page:
        render_help_page()
        return

    st.title("🤖 Chatbot Phân tích Phản hồi Sinh viên")

    # TODO 7: Bảng lịch sử với nút xóa
    if st.session_state.history:
        with st.expander(f"📋 Lịch sử phân tích ({len(st.session_state.history)} phản hồi)", expanded=False):
            for i, h in enumerate(st.session_state.history):
                cols = st.columns([5, 1])
                cols[0].markdown(
                    f"**{i+1}.** `{h.get('timestamp','')}`  "
                    f"{EMOJI_MAP.get(h.get('sentiment','neutral'), '😐')} "
                    f"{h.get('text','')[:80]}…" if len(h.get('text','')) > 80 else
                    f"**{i+1}.** `{h.get('timestamp','')}`  "
                    f"{EMOJI_MAP.get(h.get('sentiment','neutral'), '😐')} {h.get('text','')}"
                )
                if cols[1].button("🗑️", key=f"del_{i}"):
                    delete_feedback(i)
                    st.rerun()

    st.markdown("---")

    # Hiển thị chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Ô nhập chat
    if prompt := st.chat_input("Nhập phản hồi sinh viên tại đây… (nhiều dòng được)"):
        lines = [l.strip() for l in prompt.splitlines() if l.strip()]
        if not lines:
            lines = [prompt.strip()]

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        for line in lines:
            if not line:
                continue
            with st.spinner("Đang phân tích…"):
                result = analyze_feedback(line, stopwords)
            md = render_analysis(result)
            with st.chat_message("assistant"):
                if len(lines) > 1:
                    st.markdown(f"**Phản hồi:** _{line}_")
                st.markdown(md)
            st.session_state.messages.append({"role": "assistant", "content": md})
            st.session_state.history.append(result)

        save_history(st.session_state.history)
        st.rerun()


if __name__ == "__main__":
    main()
