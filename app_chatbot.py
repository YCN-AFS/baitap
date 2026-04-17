# ============================================================
# app_chatbot.py – Chatbot Phân tích Phản hồi Sinh viên
# Version: 2.0 | Full Implementation
# ============================================================

import streamlit as st
import pandas as pd
import json
import re
import io
import os
from datetime import datetime
from collections import Counter

# ── Optional heavy deps ──────────────────────────────────────
try:
    from underthesea import sentiment as vn_sentiment, word_tokenize as vn_tokenize
    UNDERTHESEA_OK = True
except ImportError:
    UNDERTHESEA_OK = False

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_OK = True
except ImportError:
    WORDCLOUD_OK = False

try:
    from langdetect import detect as langdetect_detect
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="EduSense – Phân tích Phản hồi SV",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS – Dark academic / editorial aesthetic
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0f0f13;
    --bg-card: #16161e;
    --bg-card2: #1c1c28;
    --accent-gold: #e8b84b;
    --accent-teal: #4ecdc4;
    --accent-rose: #ff6b6b;
    --accent-purple: #a78bfa;
    --text-primary: #f0ede8;
    --text-muted: #8b8996;
    --border: rgba(255,255,255,0.07);
    --radius: 12px;
}

/* ── Global resets ── */
.stApp { background: var(--bg-primary); }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1200px; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Header ── */
.edusense-header {
    display: flex; align-items: center; gap: 16px;
    padding: 24px 0 8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.edusense-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; font-weight: 400;
    color: var(--text-primary); margin: 0;
    letter-spacing: -0.02em;
}
.edusense-header .tagline {
    font-size: 0.8rem; color: var(--text-muted);
    letter-spacing: 0.1em; text-transform: uppercase;
    margin: 0;
}
.header-badge {
    background: linear-gradient(135deg, var(--accent-gold), #c9962a);
    color: #0f0f13; font-size: 0.7rem; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 0.08em; text-transform: uppercase;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem; color: var(--accent-gold);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px; margin-bottom: 16px;
}

/* ── Metric cards ── */
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0; }
.metric-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px; text-align: center;
}
.metric-card .val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem; color: var(--accent-gold);
    line-height: 1;
}
.metric-card .lbl {
    font-size: 0.72rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Sentiment badges ── */
.badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-positive { background: rgba(78,205,196,0.15); color: var(--accent-teal); border: 1px solid rgba(78,205,196,0.3); }
.badge-negative { background: rgba(255,107,107,0.15); color: var(--accent-rose); border: 1px solid rgba(255,107,107,0.3); }
.badge-neutral  { background: rgba(167,139,250,0.15); color: var(--accent-purple); border: 1px solid rgba(167,139,250,0.3); }
.badge-mixed    { background: rgba(232,184,75,0.15); color: var(--accent-gold); border: 1px solid rgba(232,184,75,0.3); }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}
.analysis-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-gold);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 0.9rem;
}
.analysis-card .card-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem; color: var(--accent-gold);
    margin-bottom: 12px;
}
.kw-chip {
    display: inline-block;
    background: rgba(232,184,75,0.1);
    border: 1px solid rgba(232,184,75,0.25);
    color: var(--accent-gold);
    padding: 2px 8px; border-radius: 4px;
    font-size: 0.75rem; margin: 2px;
    font-family: 'JetBrains Mono', monospace;
}
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 4px; height: 6px; margin-top: 4px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-teal), var(--accent-gold));
    transition: width 0.5s ease;
}
.stat-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.85rem; }
.stat-row .key { color: var(--text-muted); }
.stat-row .val { color: var(--text-primary); font-weight: 500; }

/* ── Help page ── */
.help-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px; margin-bottom: 16px;
}
.help-section h4 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem; color: var(--accent-gold);
    margin-bottom: 12px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card2) !important;
    border: 1px dashed rgba(232,184,75,0.3) !important;
    border-radius: var(--radius) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent;
    border: 1px solid var(--accent-gold);
    color: var(--accent-gold);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem; font-weight: 500;
    border-radius: 6px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: var(--accent-gold);
    color: #0f0f13;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid var(--border);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem; color: var(--text-muted);
    background: transparent !important;
    border: none !important;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    color: var(--accent-gold) !important;
    border-bottom: 2px solid var(--accent-gold) !important;
}

/* ── Comparison mode ── */
.compare-box {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px; margin-bottom: 12px;
}
.compare-box h5 {
    font-family: 'DM Serif Display', serif;
    font-size: 0.95rem; color: var(--accent-teal);
    margin-bottom: 10px;
}

/* ── Edge case warning ── */
.edge-warn {
    background: rgba(255,107,107,0.1);
    border: 1px solid rgba(255,107,107,0.25);
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.82rem; color: var(--accent-rose);
    margin: 6px 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONSTANTS
# ============================================================
HISTORY_PATH = "chat_history.json"

STOPWORDS_VI_DEFAULT = set([
    "và","của","là","có","trong","đã","được","cho","với","về","một","các","những",
    "này","đó","từ","tôi","bạn","mình","họ","chúng","cũng","khi","thì","mà","nhưng",
    "nếu","vì","để","hay","hoặc","như","lại","rất","hơn","quá","thật","thế","vậy",
    "bởi","do","nên","tuy","dù","dù","ở","tại","trên","dưới","sau","trước","theo",
    "giữa","qua","ra","vào","lên","xuống","đi","lại","đây","đấy","thôi","nhé","ạ",
    "à","ơi","ừ","uh","ok","okay","à","ừ","nhỉ","nhỉ","ấy","vậy","thôi","đây",
    "đó","kia","nọ","này","đây","đó","kia","nọ","thật","thực","sự","cả","hết",
    "chỉ","mới","đã","sẽ","đang","chưa","không","chẳng","chả","nào","gì","ai",
    "sao","làm","nói","biết","thấy","muốn","cần","phải","được","bị","của","này"
])

POSITIVE_WORDS = set([
    "tốt","hay","giỏi","xuất sắc","tuyệt","vời","hài lòng","thích","yêu","thú vị",
    "bổ ích","hiệu quả","rõ ràng","dễ hiểu","nhiệt tình","tận tâm","chu đáo","nhanh",
    "kịp thời","chuyên nghiệp","sáng tạo","đổi mới","phong phú","sinh động","vui",
    "thú vị","tích cực","cải thiện","tiến bộ","phát triển","ủng hộ","khuyến khích",
    "hỗ trợ","giúp đỡ","quan tâm","cảm ơn","trân trọng","hữu ích","dễ","nhanh chóng"
])

NEGATIVE_WORDS = set([
    "kém","tệ","chán","nhàm","khó","khó hiểu","mơ hồ","không rõ","lẫn lộn","thiếu",
    "không đủ","chậm","trễ","quá tải","áp lực","không công bằng","không hỗ trợ","bỏ qua",
    "không giải đáp","không phản hồi","thất vọng","buồn","lo lắng","căng thẳng",
    "không hài lòng","phàn nàn","vấn đề","khiếu nại","bất cập","hạn chế","yếu","không tốt"
])

EMOJI_MAP = {
    "positive": "✦ Tích cực",
    "negative": "✦ Tiêu cực",
    "neutral": "✦ Trung tính",
    "mixed": "✦ Hỗn hợp",
}


# ============================================================
# TODO 1: STOPWORDS – load từ file ngoài
# ============================================================
@st.cache_data(show_spinner=False)
def load_stopwords(path: str = "stopwords_vi.txt") -> set[str]:
    """Đọc stopwords từ file ngoài; fallback về default set."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            words = {line.strip().lower() for line in f if line.strip()}
        return words | STOPWORDS_VI_DEFAULT
    return STOPWORDS_VI_DEFAULT


STOPWORDS = load_stopwords()


# ============================================================
# TODO 9: LANGUAGE DETECTION
# ============================================================
def detect_language(text: str) -> str:
    """Detect ngôn ngữ, trả về mã ('vi', 'en', ...)."""
    if LANGDETECT_OK:
        try:
            return langdetect_detect(text)
        except Exception:
            pass
    # Fallback: heuristic đơn giản
    vi_chars = set("àáâãäåæăắặằẳẵấậầẩẫđèéêëếệềểễìíîïòóôõöơớợờởỡùúûüứựừửữỳýỷỹ")
    ratio = sum(1 for c in text.lower() if c in vi_chars) / max(len(text), 1)
    return "vi" if ratio > 0.05 else "en"


# ============================================================
# TODO 2 + 8 + 13: ANALYZE FEEDBACK (cached, với confidence + edge cases)
# ============================================================
@st.cache_data(show_spinner=False)
def analyze_feedback(text: str) -> dict:
    """Phân tích cảm xúc + từ khóa, xử lý edge case, trả confidence score."""
    result = {
        "text": text,
        "sentiment": "neutral",
        "confidence": 0.5,
        "keywords": [],
        "word_count": 0,
        "lang": "vi",
        "edge_case": None,
        "timestamp": datetime.now().isoformat(),
    }

    # ── TODO 13: Edge cases ──────────────────────────────────
    stripped = text.strip()

    # Emoji-only
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F9FF\U00002600-\U000027BF]+", flags=re.UNICODE
    )
    if emoji_pattern.fullmatch(stripped):
        result["edge_case"] = "emoji_only"
        pos_emojis = set("😊😄🥰👍✅💯🎉😍🤩😃😁👏🙌")
        neg_emojis = set("😞😢😡👎😤😠😰😰💔😒😔")
        if any(e in stripped for e in pos_emojis):
            result["sentiment"] = "positive"; result["confidence"] = 0.7
        elif any(e in stripped for e in neg_emojis):
            result["sentiment"] = "negative"; result["confidence"] = 0.7
        return result

    # Quá ngắn
    tokens_raw = stripped.split()
    if len(tokens_raw) <= 2:
        result["edge_case"] = "too_short"
        result["word_count"] = len(tokens_raw)
        result["keywords"] = [t.lower() for t in tokens_raw if t.lower() not in STOPWORDS]
        return result

    # Ký tự đặc biệt / số thuần
    if re.fullmatch(r"[\d\s\W]+", stripped):
        result["edge_case"] = "special_chars"
        return result

    # ── TODO 9: Language detect ──────────────────────────────
    lang = detect_language(stripped)
    result["lang"] = lang

    # ── TODO 2: Tokenize (underthesea nếu có) ────────────────
    if UNDERTHESEA_OK and lang == "vi":
        tokens = vn_tokenize(stripped)
    else:
        tokens = stripped.lower().split()

    # ── Keywords ─────────────────────────────────────────────
    keywords = [
        t.lower() for t in tokens
        if t.lower() not in STOPWORDS and len(t) > 1
        and not re.fullmatch(r"\W+", t)
    ]
    result["keywords"] = keywords[:20]
    result["word_count"] = len(tokens)

    # ── Sentiment ────────────────────────────────────────────
    if UNDERTHESEA_OK and lang == "vi":
        try:
            raw_sent = vn_sentiment(stripped)
            # underthesea trả về "positive"/"negative"/"neutral" hoặc tuple
            if isinstance(raw_sent, (list, tuple)):
                raw_sent = raw_sent[0] if raw_sent else "neutral"
            sentiment_label = str(raw_sent).lower()
            if sentiment_label not in ("positive", "negative", "neutral"):
                sentiment_label = "neutral"
            result["sentiment"] = sentiment_label
            result["confidence"] = 0.82
        except Exception:
            pass
    else:
        # Fallback từ điển
        text_lower = stripped.lower()
        pos_score = sum(1 for w in POSITIVE_WORDS if w in text_lower)
        neg_score = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
        total = pos_score + neg_score
        if total == 0:
            result["sentiment"] = "neutral"
            result["confidence"] = 0.55
        elif pos_score > neg_score:
            result["sentiment"] = "positive"
            result["confidence"] = round(0.5 + 0.4 * (pos_score / total), 2)
        elif neg_score > pos_score:
            result["sentiment"] = "negative"
            result["confidence"] = round(0.5 + 0.4 * (neg_score / total), 2)
        else:
            result["sentiment"] = "mixed"
            result["confidence"] = 0.5

    return result


# ============================================================
# RENDER ANALYSIS (markdown card)
# ============================================================
def render_analysis(result: dict) -> str:
    """Tạo HTML card hiển thị kết quả phân tích."""
    sent = result.get("sentiment", "neutral")
    conf = result.get("confidence", 0.5)
    keywords = result.get("keywords", [])[:10]
    wc = result.get("word_count", 0)
    lang = result.get("lang", "vi")
    edge = result.get("edge_case")
    ts = result.get("timestamp", "")[:19].replace("T", " ")

    badge_class = {
        "positive": "badge-positive",
        "negative": "badge-negative",
        "neutral": "badge-neutral",
        "mixed": "badge-mixed",
    }.get(sent, "badge-neutral")

    sent_label = {
        "positive": "😊 Tích cực",
        "negative": "😟 Tiêu cực",
        "neutral": "😐 Trung tính",
        "mixed": "🔀 Hỗn hợp",
    }.get(sent, "😐 Trung tính")

    lang_label = {"vi": "🇻🇳 Tiếng Việt", "en": "🇬🇧 English"}.get(lang, f"🌐 {lang}")

    kw_html = "".join(f'<span class="kw-chip">{kw}</span>' for kw in keywords) if keywords else "<span style='color:var(--text-muted);font-size:0.8rem'>Không trích xuất được</span>"

    conf_pct = int(conf * 100)
    conf_color = "#4ecdc4" if conf >= 0.7 else ("#e8b84b" if conf >= 0.5 else "#ff6b6b")

    edge_html = ""
    if edge == "too_short":
        edge_html = '<div class="edge-warn">⚠️ Phản hồi quá ngắn – độ chính xác phân tích thấp hơn.</div>'
    elif edge == "emoji_only":
        edge_html = '<div class="edge-warn">ℹ️ Phản hồi chỉ chứa emoji – phân tích dựa trên biểu tượng cảm xúc.</div>'
    elif edge == "special_chars":
        edge_html = '<div class="edge-warn">⚠️ Văn bản chứa ký tự đặc biệt / số – không thể phân tích cảm xúc.</div>'

    return f"""
<div class="analysis-card">
  <div class="card-header">📊 Kết quả phân tích</div>
  {edge_html}
  <div class="stat-row"><span class="key">Cảm xúc</span><span><span class="badge {badge_class}">{sent_label}</span></span></div>
  <div class="stat-row" style="margin-top:8px"><span class="key">Độ tin cậy (TODO 8)</span>
    <span style="color:{conf_color};font-weight:600;font-family:'JetBrains Mono',monospace">{conf_pct}%</span>
  </div>
  <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf_pct}%;background:linear-gradient(90deg,{conf_color},{conf_color}99)"></div></div>
  <div class="stat-row" style="margin-top:10px"><span class="key">Số từ</span><span class="val">{wc}</span></div>
  <div class="stat-row"><span class="key">Ngôn ngữ</span><span class="val">{lang_label}</span></div>
  <div class="stat-row"><span class="key">Thời điểm</span><span class="val" style="font-family:'JetBrains Mono',monospace;font-size:0.78rem">{ts}</span></div>
  <div style="margin-top:12px"><span class="key" style="font-size:0.78rem;text-transform:uppercase;letter-spacing:.08em">Từ khóa</span><div style="margin-top:6px">{kw_html}</div></div>
</div>
"""


# ============================================================
# TODO 3: FILE UPLOAD
# ============================================================
def handle_file_upload(uploaded_file) -> list[str]:
    """Đọc file CSV/Excel upload, trả về list phản hồi."""
    if uploaded_file is None:
        return []
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        else:
            df = pd.read_excel(uploaded_file)

        # Tìm cột phản hồi
        text_cols = [c for c in df.columns if any(
            k in c.lower() for k in ["phản hồi", "noi dung", "content", "feedback", "text", "response", "ý kiến"]
        )]
        col = text_cols[0] if text_cols else df.columns[0]
        texts = df[col].dropna().astype(str).tolist()
        return texts
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return []


# ============================================================
# TODO 4: EXPORT HISTORY
# ============================================================
def export_history(history: list[dict]) -> bytes:
    """Chuyển lịch sử phân tích thành CSV bytes để download."""
    if not history:
        return b""
    rows = []
    for item in history:
        rows.append({
            "Thời điểm": item.get("timestamp", "")[:19].replace("T", " "),
            "Phản hồi": item.get("text", ""),
            "Cảm xúc": item.get("sentiment", ""),
            "Độ tin cậy (%)": int(item.get("confidence", 0) * 100),
            "Số từ": item.get("word_count", 0),
            "Ngôn ngữ": item.get("lang", ""),
            "Từ khóa": ", ".join(item.get("keywords", [])[:10]),
        })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


# ============================================================
# TODO 5: WORD CLOUD
# ============================================================
def render_wordcloud(keywords: list[str]):
    """Vẽ word cloud từ danh sách từ khóa."""
    if not keywords:
        st.info("Chưa có đủ từ khóa để vẽ word cloud.")
        return
    if not WORDCLOUD_OK:
        # Fallback: bảng tần suất đẹp
        freq = Counter(keywords).most_common(20)
        df = pd.DataFrame(freq, columns=["Từ khóa", "Tần suất"])
        st.dataframe(df, hide_index=True, use_container_width=True)
        return
    freq = Counter(keywords)
    wc = WordCloud(
        width=700, height=300,
        background_color=None, mode="RGBA",
        colormap="YlOrBr",
        prefer_horizontal=0.8,
        max_words=60,
        font_path=None,  # dùng font mặc định
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(7, 3), facecolor="none")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# TODO 6: SENTIMENT TIMELINE
# ============================================================
def render_sentiment_timeline(history: list[dict]):
    """Vẽ biểu đồ xu hướng cảm xúc theo thời gian."""
    if len(history) < 2:
        st.info("Cần ít nhất 2 phản hồi để hiển thị xu hướng.")
        return

    df = pd.DataFrame([{
        "idx": i + 1,
        "sentiment": h.get("sentiment", "neutral"),
        "confidence": h.get("confidence", 0.5),
        "text": h.get("text", "")[:40] + "…",
        "ts": h.get("timestamp", "")[:16].replace("T", " "),
    } for i, h in enumerate(history)])

    sent_num = {"positive": 1, "mixed": 0.5, "neutral": 0, "negative": -1}
    sent_color = {"positive": "#4ecdc4", "mixed": "#e8b84b", "neutral": "#a78bfa", "negative": "#ff6b6b"}
    df["score"] = df["sentiment"].map(sent_num)
    df["color"] = df["sentiment"].map(sent_color)

    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["idx"], y=df["score"],
            mode="lines+markers",
            line=dict(color="#e8b84b", width=2, dash="solid"),
            marker=dict(color=df["color"].tolist(), size=10, line=dict(color="#0f0f13", width=2)),
            hovertemplate="<b>%{text}</b><br>Điểm: %{y}<extra></extra>",
            text=df["text"],
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f0ede8", family="DM Sans"),
            xaxis=dict(showgrid=False, zeroline=False, title="Phản hồi thứ"),
            yaxis=dict(
                showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                zeroline=True, zerolinecolor="rgba(255,255,255,0.1)",
                tickvals=[-1, 0, 0.5, 1],
                ticktext=["Tiêu cực", "Trung tính", "Hỗn hợp", "Tích cực"],
                range=[-1.4, 1.4],
            ),
            margin=dict(l=10, r=10, t=20, b=20),
            height=220,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df.set_index("idx")["score"])


# ============================================================
# TODO 11: PERSIST HISTORY
# ============================================================
def save_history(history: list[dict], path: str = HISTORY_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_history(path: str = HISTORY_PATH) -> list[dict]:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


# ============================================================
# TODO 7: DELETE FEEDBACK
# ============================================================
def delete_feedback(index: int):
    """Xóa một phản hồi theo index khỏi history + messages."""
    if 0 <= index < len(st.session_state.history):
        st.session_state.history.pop(index)
        save_history(st.session_state.history)
        # Xóa cặp (user + assistant) tương ứng trong messages
        # Mỗi phản hồi tạo ra 2 messages: user + assistant
        msg_idx = index * 2
        if msg_idx + 1 < len(st.session_state.messages):
            del st.session_state.messages[msg_idx:msg_idx + 2]


# ============================================================
# TODO 10: HELP PAGE
# ============================================================
def render_help_page():
    st.markdown("""
<div class="help-section">
<h4>📖 Giới thiệu</h4>
<p style="color:var(--text-muted);font-size:0.9rem;line-height:1.6">
<b>EduSense</b> là công cụ phân tích phản hồi sinh viên tự động. 
Hệ thống nhận diện cảm xúc, trích xuất từ khóa và tổng hợp thống kê để 
hỗ trợ giảng viên nắm bắt ý kiến lớp học hiệu quả.
</p>
</div>

<div class="help-section">
<h4>🎯 Các chỉ số phân tích</h4>
<table style="width:100%;border-collapse:collapse;font-size:0.85rem">
<tr style="border-bottom:1px solid var(--border)">
  <td style="padding:8px 0;color:var(--accent-gold);width:30%"><b>Cảm xúc</b></td>
  <td style="color:var(--text-muted)">Phân loại: Tích cực / Tiêu cực / Trung tính / Hỗn hợp</td>
</tr>
<tr style="border-bottom:1px solid var(--border)">
  <td style="padding:8px 0;color:var(--accent-gold)"><b>Độ tin cậy</b></td>
  <td style="color:var(--text-muted)">Mức độ chắc chắn của mô hình (0–100%). Cao hơn = đáng tin hơn.</td>
</tr>
<tr style="border-bottom:1px solid var(--border)">
  <td style="padding:8px 0;color:var(--accent-gold)"><b>Từ khóa</b></td>
  <td style="color:var(--text-muted)">Các từ quan trọng sau khi loại stopwords. Dùng để xây word cloud.</td>
</tr>
<tr>
  <td style="padding:8px 0;color:var(--accent-gold)"><b>Xu hướng</b></td>
  <td style="color:var(--text-muted)">Biểu đồ cảm xúc theo thứ tự nhập – cho thấy chiều hướng thay đổi.</td>
</tr>
</table>
</div>

<div class="help-section">
<h4>📁 Upload file</h4>
<p style="color:var(--text-muted);font-size:0.85rem;line-height:1.6">
Hỗ trợ file <b>.csv</b> và <b>.xlsx</b>. Cột chứa phản hồi cần có tên như: 
<code style="background:rgba(255,255,255,0.07);padding:1px 5px;border-radius:3px">phản hồi</code>, 
<code style="background:rgba(255,255,255,0.07);padding:1px 5px;border-radius:3px">feedback</code>, 
<code style="background:rgba(255,255,255,0.07);padding:1px 5px;border-radius:3px">content</code>, 
<code style="background:rgba(255,255,255,0.07);padding:1px 5px;border-radius:3px">text</code>, 
<code style="background:rgba(255,255,255,0.07);padding:1px 5px;border-radius:3px">ý kiến</code>.
Nếu không tìm thấy, cột đầu tiên sẽ được dùng.
</p>
</div>

<div class="help-section">
<h4>🌍 Hỗ trợ đa ngôn ngữ</h4>
<p style="color:var(--text-muted);font-size:0.85rem;line-height:1.6">
Hệ thống tự nhận diện tiếng Việt và tiếng Anh. 
Nếu cài <code style="background:rgba(255,255,255,0.07);padding:1px 5px;border-radius:3px">underthesea</code>, 
phân tích tiếng Việt sẽ chính xác hơn. Ngôn ngữ khác dùng phân tích từ điển.
</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR STATS (TODO 5, 6, 12)
# ============================================================
def render_sidebar_stats(history: list[dict]):
    st.markdown("### 📈 Tổng quan")

    if not history:
        st.markdown('<p style="color:var(--text-muted);font-size:0.85rem">Chưa có dữ liệu. Nhập phản hồi để bắt đầu.</p>', unsafe_allow_html=True)
        return

    total = len(history)
    pos = sum(1 for h in history if h.get("sentiment") == "positive")
    neg = sum(1 for h in history if h.get("sentiment") == "negative")
    avg_conf = sum(h.get("confidence", 0) for h in history) / total

    st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card"><div class="val">{total}</div><div class="lbl">Tổng phản hồi</div></div>
  <div class="metric-card"><div class="val" style="color:var(--accent-teal)">{pos}</div><div class="lbl">Tích cực</div></div>
  <div class="metric-card"><div class="val" style="color:var(--accent-rose)">{neg}</div><div class="lbl">Tiêu cực</div></div>
  <div class="metric-card"><div class="val" style="color:var(--accent-purple)">{int(avg_conf*100)}%</div><div class="lbl">Tin cậy TB</div></div>
</div>
""", unsafe_allow_html=True)

    # Phân bố cảm xúc
    if PLOTLY_OK and total > 0:
        sent_counts = Counter(h.get("sentiment", "neutral") for h in history)
        colors_map = {"positive": "#4ecdc4", "negative": "#ff6b6b", "neutral": "#a78bfa", "mixed": "#e8b84b"}
        labels = list(sent_counts.keys())
        values = list(sent_counts.values())
        colors = [colors_map.get(l, "#888") for l in labels]
        label_vi = {"positive": "Tích cực", "negative": "Tiêu cực", "neutral": "Trung tính", "mixed": "Hỗn hợp"}
        fig = go.Figure(go.Pie(
            labels=[label_vi.get(l, l) for l in labels],
            values=values,
            hole=0.55,
            marker=dict(colors=colors, line=dict(color="#0f0f13", width=2)),
            textinfo="percent",
            textfont=dict(color="#f0ede8", size=11),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(font=dict(color="#f0ede8", size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=5, r=5, t=5, b=5), height=200,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Timeline
    st.markdown("**Xu hướng cảm xúc**")
    render_sentiment_timeline(history)

    # Word cloud / top keywords
    all_kw = [kw for h in history for kw in h.get("keywords", [])]
    if all_kw:
        st.markdown("**Từ khóa nổi bật**")
        render_wordcloud(all_kw)


# ============================================================
# TODO 12: COMPARISON MODE
# ============================================================
def render_comparison_mode():
    st.markdown("""
<div style="font-family:'DM Serif Display',serif;font-size:1.3rem;margin-bottom:16px;color:var(--accent-gold)">
🔀 So sánh 2 nhóm phản hồi
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="compare-box"><h5>Nhóm A – Trước cải tiến</h5>', unsafe_allow_html=True)
        group_a_text = st.text_area("Nhập phản hồi (mỗi dòng 1 phản hồi)", height=150, key="cmp_a",
                                     placeholder="Giảng viên giải thích chưa rõ...\nBài tập quá nhiều...")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="compare-box"><h5>Nhóm B – Sau cải tiến</h5>', unsafe_allow_html=True)
        group_b_text = st.text_area("Nhập phản hồi (mỗi dòng 1 phản hồi)", height=150, key="cmp_b",
                                     placeholder="Giảng viên rất nhiệt tình...\nBài học dễ hiểu hơn...")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡ Phân tích & So sánh", use_container_width=True):
        def analyze_group(text: str) -> list[dict]:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            return [analyze_feedback(l) for l in lines]

        results_a = analyze_group(group_a_text)
        results_b = analyze_group(group_b_text)

        if not results_a or not results_b:
            st.warning("Vui lòng nhập đủ dữ liệu cho cả 2 nhóm.")
            return

        def group_stats(results):
            total = len(results)
            pos = sum(1 for r in results if r["sentiment"] == "positive")
            neg = sum(1 for r in results if r["sentiment"] == "negative")
            avg_conf = sum(r["confidence"] for r in results) / total
            return {"total": total, "pos": pos, "neg": neg,
                    "pos_pct": pos/total*100, "neg_pct": neg/total*100,
                    "avg_conf": avg_conf}

        sa, sb = group_stats(results_a), group_stats(results_b)

        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown(f"""
<div class="compare-box">
<h5>Nhóm A</h5>
<div class="stat-row"><span class="key">Tích cực</span><span class="val" style="color:var(--accent-teal)">{sa['pos_pct']:.0f}%</span></div>
<div class="stat-row"><span class="key">Tiêu cực</span><span class="val" style="color:var(--accent-rose)">{sa['neg_pct']:.0f}%</span></div>
<div class="stat-row"><span class="key">Tin cậy TB</span><span class="val">{int(sa['avg_conf']*100)}%</span></div>
</div>""", unsafe_allow_html=True)
        with col2:
            delta_pos = sb["pos_pct"] - sa["pos_pct"]
            arrow = "↑" if delta_pos > 0 else ("↓" if delta_pos < 0 else "→")
            color = "var(--accent-teal)" if delta_pos > 0 else ("var(--accent-rose)" if delta_pos < 0 else "var(--text-muted)")
            st.markdown(f"""
<div style="text-align:center;padding:30px 0">
<div style="font-size:2.5rem;color:{color}">{arrow}</div>
<div style="color:{color};font-weight:600;font-size:1rem">{abs(delta_pos):.0f}%</div>
<div style="color:var(--text-muted);font-size:0.75rem">Tích cực</div>
</div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
<div class="compare-box">
<h5>Nhóm B</h5>
<div class="stat-row"><span class="key">Tích cực</span><span class="val" style="color:var(--accent-teal)">{sb['pos_pct']:.0f}%</span></div>
<div class="stat-row"><span class="key">Tiêu cực</span><span class="val" style="color:var(--accent-rose)">{sb['neg_pct']:.0f}%</span></div>
<div class="stat-row"><span class="key">Tin cậy TB</span><span class="val">{int(sb['avg_conf']*100)}%</span></div>
</div>""", unsafe_allow_html=True)


# ============================================================
# TODO 11: INIT SESSION STATE
# ============================================================
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = load_history()
    if "page" not in st.session_state:
        st.session_state.page = "chat"


# ============================================================
# MAIN
# ============================================================
def main():
    init_session_state()

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
<div style="padding:16px 0 8px;border-bottom:1px solid var(--border);margin-bottom:16px">
  <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;color:var(--accent-gold)">EduSense</div>
  <div style="font-size:0.72rem;color:var(--text-muted);letter-spacing:.1em;text-transform:uppercase">Phân tích Phản hồi SV</div>
</div>
""", unsafe_allow_html=True)

        # Navigation
        pages = {"💬 Chat phân tích": "chat", "🔀 So sánh nhóm": "compare", "📖 Hướng dẫn": "help"}
        for label, key in pages.items():
            if st.button(label, use_container_width=True, key=f"nav_{key}"):
                st.session_state.page = key

        st.markdown("---")
        render_sidebar_stats(st.session_state.history)

        st.markdown("---")

        # TODO 3: File upload
        st.markdown("### 📁 Nhập từ file")
        uploaded = st.file_uploader("CSV hoặc Excel", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
        if uploaded:
            feedbacks = handle_file_upload(uploaded)
            if feedbacks:
                st.success(f"Đọc được {len(feedbacks)} phản hồi")
                if st.button("⚡ Phân tích tất cả", use_container_width=True):
                    with st.spinner("Đang phân tích..."):
                        for fb in feedbacks:
                            result = analyze_feedback(fb)
                            st.session_state.history.append(result)
                            st.session_state.messages.append({"role": "user", "content": fb})
                            st.session_state.messages.append({"role": "assistant", "content": render_analysis(result)})
                        save_history(st.session_state.history)
                    st.rerun()

        st.markdown("---")

        # TODO 4: Export
        st.markdown("### 💾 Xuất dữ liệu")
        if st.session_state.history:
            csv_bytes = export_history(st.session_state.history)
            st.download_button(
                label="📥 Tải CSV",
                data=csv_bytes,
                file_name=f"edusense_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if st.button("🗑️ Xóa toàn bộ lịch sử", use_container_width=True):
                st.session_state.history = []
                st.session_state.messages = []
                save_history([])
                st.rerun()

    # ── Main area ────────────────────────────────────────────
    # Header
    st.markdown("""
<div class="edusense-header">
  <div>
    <div style="display:flex;align-items:center;gap:12px">
      <h1>🎓 EduSense</h1>
      <span class="header-badge">Beta</span>
    </div>
    <p class="tagline">Hệ thống phân tích phản hồi sinh viên thông minh</p>
  </div>
</div>
""", unsafe_allow_html=True)

    page = st.session_state.get("page", "chat")

    # ── PAGE: HELP ───────────────────────────────────────────
    if page == "help":
        render_help_page()
        return

    # ── PAGE: COMPARE ────────────────────────────────────────
    if page == "compare":
        render_comparison_mode()
        return

    # ── PAGE: CHAT ───────────────────────────────────────────
    tabs = st.tabs(["💬 Nhập thủ công", "📝 Quản lý lịch sử"])

    with tabs[0]:
        # Hiển thị lịch sử chat
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
<div style="text-align:center;padding:60px 20px;color:var(--text-muted)">
  <div style="font-size:3rem;margin-bottom:16px">💬</div>
  <div style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:var(--text-primary);margin-bottom:8px">
    Bắt đầu phân tích
  </div>
  <div style="font-size:0.85rem">
    Nhập một hoặc nhiều phản hồi sinh viên vào ô bên dưới.<br>
    Mỗi dòng sẽ được phân tích riêng biệt.
  </div>
</div>
""", unsafe_allow_html=True)

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"], unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Nhập phản hồi sinh viên (Enter để gửi, nhiều dòng = nhiều phản hồi)..."):
            lines = [l.strip() for l in prompt.splitlines() if l.strip()]

            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Đang phân tích..."):
                    all_html = ""
                    for line in lines:
                        result = analyze_feedback(line)
                        st.session_state.history.append(result)
                        all_html += f"<div style='margin-bottom:4px;color:var(--text-muted);font-size:0.8rem'>#{len(st.session_state.history)} · {line[:60]}{'…' if len(line)>60 else ''}</div>"
                        all_html += render_analysis(result)

                    st.markdown(all_html, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": all_html})
                    save_history(st.session_state.history)
            st.rerun()

    with tabs[1]:
        # TODO 7: Quản lý + xóa từng phản hồi
        if not st.session_state.history:
            st.info("Chưa có lịch sử phân tích.")
        else:
            st.markdown(f'<p style="color:var(--text-muted);font-size:0.85rem">Tổng cộng <b style="color:var(--accent-gold)">{len(st.session_state.history)}</b> phản hồi đã phân tích</p>', unsafe_allow_html=True)

            sent_filter = st.selectbox(
                "Lọc theo cảm xúc",
                ["Tất cả", "Tích cực", "Tiêu cực", "Trung tính", "Hỗn hợp"],
                label_visibility="visible",
            )
            filter_map = {"Tất cả": None, "Tích cực": "positive", "Tiêu cực": "negative",
                          "Trung tính": "neutral", "Hỗn hợp": "mixed"}
            f_val = filter_map[sent_filter]

            for i, item in enumerate(st.session_state.history):
                if f_val and item.get("sentiment") != f_val:
                    continue

                sent = item.get("sentiment", "neutral")
                badge_cls = {"positive": "badge-positive", "negative": "badge-negative",
                             "neutral": "badge-neutral", "mixed": "badge-mixed"}.get(sent, "badge-neutral")
                sent_vi = {"positive": "😊 Tích cực", "negative": "😟 Tiêu cực",
                           "neutral": "😐 Trung tính", "mixed": "🔀 Hỗn hợp"}.get(sent, sent)

                with st.expander(f"#{i+1} · {item.get('text','')[:70]}…"):
                    col_a, col_b = st.columns([5, 1])
                    with col_a:
                        st.markdown(f"""
<div style="font-size:0.85rem;line-height:1.7">
<b>Văn bản:</b> {item.get('text','')}<br>
<b>Cảm xúc:</b> <span class="badge {badge_cls}">{sent_vi}</span>
&nbsp;&nbsp;<b>Tin cậy:</b> {int(item.get('confidence',0)*100)}%
&nbsp;&nbsp;<b>Từ:</b> {item.get('word_count',0)}
&nbsp;&nbsp;<b>Ngôn ngữ:</b> {item.get('lang','vi')}<br>
<b>Từ khóa:</b> {', '.join(item.get('keywords',[])[:8]) or '—'}<br>
<b>Thời điểm:</b> <span style="font-family:'JetBrains Mono',monospace">{item.get('timestamp','')[:19].replace('T',' ')}</span>
</div>
""", unsafe_allow_html=True)
                    with col_b:
                        if st.button("🗑️ Xóa", key=f"del_{i}"):
                            delete_feedback(i)
                            st.rerun()


if __name__ == "__main__":
    main()
