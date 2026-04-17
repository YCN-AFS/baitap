# ============================================================
# app_chatbot.py – Chatbot Phân tích Phản hồi Sinh viên
# Tất cả TODO 1–15 – phân tích chính xác, không phụ thuộc underthesea model
# pip install streamlit pandas wordcloud matplotlib langdetect openpyxl pyvi
# ============================================================

import json
import io
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

try:
    from wordcloud import WordCloud
    WORDCLOUD_OK = True
except ImportError:
    WORDCLOUD_OK = False

try:
    from langdetect import detect as ld_detect
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

try:
    from pyvi import ViTokenizer
    PYVI_OK = True
except ImportError:
    PYVI_OK = False

try:
    from underthesea import word_tokenize as _uts_tokenize
    _uts_tokenize("test")
    UNDERTHESEA_TOKENIZE_OK = True
except Exception:
    UNDERTHESEA_TOKENIZE_OK = False

# ============================================================
# CONSTANTS
# ============================================================
HISTORY_PATH = "chat_history.json"
STOPWORDS_PATH = "stopwords_vi.txt"
EMOJI_MAP = {"positive": "😊", "negative": "😟", "neutral": "😐"}

_DEFAULT_STOPWORDS_RAW = """
và của là trong có được cho với từ đến những các một này đó như hay hoặc
không thì về ra đã sẽ bị mà cũng theo vào rất khi nhiều nên hơn lên
tôi em anh chị thầy cô bạn họ chúng mình ta bởi vì vậy nhưng nếu
còn hết lại đang đây nào thêm tất cả nhau qua dù tuy luôn
rằng cần giúp để làm sau trước nữa thật quá ít lắm ở so mỗi tại vẫn
tuy nhiên mặc dù cho nên vì vậy bởi do nhé ạ ơi nhỉ thôi thế nha
đây đó này kia ấy đâu sao gì ai bao giờ bao nhiêu thế nào
"""

# ── Từ điển cảm xúc ─────────────────────────────────────────
_POS_PHRASES = [
    "đánh giá cao","rất tốt","rất hay","xuất sắc","tuyệt vời",
    "hài lòng","hiệu quả","rõ ràng","hệ thống","dễ theo dõi",
    "dễ hiểu","dễ nắm bắt","dễ tiếp thu","chủ động","sâu hơn",
    "giải đáp","trao đổi","sinh động","thú vị","nhiệt tình",
    "liên hệ thực tế","liên hệ lý thuyết","đầy đủ","chi tiết",
    "phù hợp","cuốn hút","bổ ích","hữu ích","tích cực",
    "nắm bắt kiến thức","tiếp thu tốt","hiểu bài","học tốt",
    "tốt","hay","ổn",
]
_NEG_PHRASES = [
    "chưa hiệu quả","chưa đạt kỳ vọng","chưa thực sự","chưa hợp lý",
    "chưa rõ ràng","chưa biết áp dụng","chưa sát thực tế",
    "đọc slide","đọc lại slide","đọc chép",
    "thiếu giải thích","thiếu ví dụ","thiếu tương tác","thiếu liên kết",
    "khó theo kịp","khó hiểu","khó theo dõi","khó nắm bắt",
    "không tương tác","không nhấn mạnh","không động lực","không liên kết",
    "giảng một chiều","giảng nhanh","tốc độ nhanh",
    "thụ động","mất tập trung","rời rạc","đối phó",
    "mơ hồ","bị trôi kiến thức","khô khan","nhàm chán",
    "tệ","kém","chán","dở","thất vọng",
    "giảm hiệu quả","hiệu quả thấp","kém hiệu quả",
    "không biết áp dụng","không cải thiện","tiếp tục thấp",
    "chưa đạt","chưa đủ","chưa ổn","chưa có",
]
_NEU_PHRASES = [
    "mong thầy","mong cô","mong thầy/cô","hy vọng",
    "đề nghị","góp ý","nếu có thể","nếu thầy","nếu cô",
    "đôi lúc","đôi khi","có thể cải thiện","cần cải thiện",
    "bên cạnh đó","ngoài ra","tuy nhiên","mặc dù",
]


# ============================================================
# TODO 1 – STOPWORDS
# ============================================================
@st.cache_data(show_spinner=False)
def load_stopwords(path: str = STOPWORDS_PATH) -> set:
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return {w.strip().lower() for w in f if w.strip()}
    return set(_DEFAULT_STOPWORDS_RAW.split())


# ============================================================
# TODO 2 – CACHE TOKENIZER
# ============================================================
@st.cache_resource(show_spinner=False)
def get_tokenizer():
    if UNDERTHESEA_TOKENIZE_OK:
        def _tok(text):
            from underthesea import word_tokenize as wt
            return wt(text, format="text").split()
        return _tok
    if PYVI_OK:
        def _tok(text):
            return ViTokenizer.tokenize(text).split()
        return _tok
    return lambda text: text.split()


# ============================================================
# TODO 8 + 13 – PHÂN TÍCH CẢM XÚC CHÍNH XÁC
# ============================================================
def _lexicon_sentiment(text: str) -> tuple:
    tl = text.lower()
    pos = sum(1 for w in _POS_PHRASES if w in tl)
    neg = sum(1 for w in _NEG_PHRASES if w in tl)
    neu = sum(1 for w in _NEU_PHRASES if w in tl)

    if neu >= 2:
        pos = max(0, pos - 1)
        neg = max(0, neg - 1)

    total = pos + neg
    if total == 0:
        simple_pos = {"tốt","hay","ổn","tuyệt","đỉnh","thích"}
        simple_neg = {"tệ","kém","chán","dở","thất vọng","buồn","khó"}
        words = set(tl.split())
        sp = len(words & simple_pos)
        sn = len(words & simple_neg)
        if sp > sn:
            return "positive", 0.55
        if sn > sp:
            return "negative", 0.55
        return "neutral", 0.45

    diff = abs(pos - neg)
    conf = min(0.55 + 0.08 * diff + 0.03 * total, 0.97)

    if pos > neg:
        return "positive", round(conf, 2)
    if neg > pos:
        return "negative", round(conf, 2)
    return "neutral", round(0.40 + 0.02 * total, 2)


def _clean_token(t: str) -> str:
    return re.sub(r"^[^\wÀ-ỹà-ỹ]+|[^\wÀ-ỹà-ỹ]+$", "", t, flags=re.UNICODE)


def analyze_feedback(text: str, stopwords=None) -> dict:
    if stopwords is None:
        stopwords = load_stopwords()

    result = {
        "text": text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sentiment": "neutral",
        "confidence": 0.45,
        "keywords": [],
        "token_count": 0,
        "language": "vi",
        "warning": None,
    }

    # TODO 13: Edge cases
    cleaned = text.strip()
    if not cleaned:
        result["warning"] = "Văn bản rỗng."
        return result

    no_space = re.sub(r"[\s\W]", "", cleaned, flags=re.UNICODE)
    if not no_space:
        result["warning"] = "Phản hồi chỉ chứa emoji / ký tự đặc biệt."
        pos_e = set("😊👍❤️😍🥰😄😃🎉✅🌟💯")
        neg_e = set("😢😞👎😡🤬😖😭❌💔")
        chars = set(cleaned)
        if chars & pos_e:
            result["sentiment"], result["confidence"] = "positive", 0.60
        elif chars & neg_e:
            result["sentiment"], result["confidence"] = "negative", 0.60
        return result

    if len(cleaned.split()) <= 2:
        result["warning"] = "Phản hồi quá ngắn, kết quả có thể kém chính xác."

    # TODO 9
    result["language"] = detect_language(cleaned)

    # Cảm xúc
    label, conf = _lexicon_sentiment(cleaned)
    result["sentiment"] = label
    result["confidence"] = conf

    # TODO 2: Tokenize
    tokenizer = get_tokenizer()
    try:
        raw_tokens = tokenizer(cleaned)
    except Exception:
        raw_tokens = cleaned.split()

    result["token_count"] = len(raw_tokens)

    # Lọc token sạch
    tokens = []
    for t in raw_tokens:
        ct = _clean_token(t)
        if len(ct) < 2:
            continue
        if ct.lower() in stopwords:
            continue
        if re.fullmatch(r"\d+", ct):
            continue
        tokens.append(ct)

    freq = Counter(t.lower() for t in tokens)
    result["keywords"] = [w for w, _ in freq.most_common(10)]
    return result


# ============================================================
# TODO 10 – RENDER ANALYSIS
# ============================================================
def render_analysis(result: dict) -> str:
    emoji = EMOJI_MAP.get(result["sentiment"], "😐")
    label_vi = {"positive": "Tích cực", "negative": "Tiêu cực", "neutral": "Trung lập"}
    conf_pct = int(result["confidence"] * 100)
    bar = "█" * (conf_pct // 10) + "░" * (10 - conf_pct // 10)
    kw_str = ", ".join(f"`{k}`" for k in result["keywords"]) if result["keywords"] else "_Không tìm thấy_"
    lang_flag = {"vi": "🇻🇳", "en": "🇬🇧"}.get(result["language"], "🌐")
    warn_str = f"\n> ⚠️ **Lưu ý:** {result['warning']}\n" if result.get("warning") else ""

    return f"""{warn_str}
| Chỉ số | Giá trị |
|--------|---------|
| **Cảm xúc** | {emoji} {label_vi.get(result['sentiment'], result['sentiment'])} |
| **Độ tin cậy** | `{bar}` {conf_pct}% |
| **Số token** | {result['token_count']} |
| **Ngôn ngữ** | {lang_flag} `{result['language']}` |

**🔑 Từ khóa nổi bật:** {kw_str}
""".strip()


# ============================================================
# TODO 3 – FILE UPLOAD
# ============================================================
def handle_file_upload(uploaded_file) -> list:
    if uploaded_file is None:
        return []
    name = uploaded_file.name.lower()
    try:
        df = pd.read_csv(uploaded_file) if name.endswith(".csv") else pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return []
    for col in df.columns:
        if any(k in col.lower() for k in ("phan_hoi","phản hồi","feedback","noi_dung","comment","text")):
            return df[col].dropna().astype(str).tolist()
    return df.iloc[:, 0].dropna().astype(str).tolist()


# ============================================================
# TODO 4 – EXPORT EXCEL
# ============================================================
def export_history(history: list) -> bytes:
    if not history:
        return b""
    rows = [{
        "Thời gian": h.get("timestamp",""),
        "Phản hồi": h.get("text",""),
        "Cảm xúc": h.get("sentiment",""),
        "Độ tin cậy (%)": int(h.get("confidence",0)*100),
        "Ngôn ngữ": h.get("language",""),
        "Số token": h.get("token_count",0),
        "Từ khóa": ", ".join(h.get("keywords",[])),
        "Cảnh báo": h.get("warning") or "",
    } for h in history]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame(rows).to_excel(w, index=False, sheet_name="Lịch sử")
    return buf.getvalue()


# ============================================================
# TODO 5 – WORD CLOUD
# ============================================================
def render_wordcloud(keywords_all: list):
    if not WORDCLOUD_OK:
        st.info("Cài `wordcloud` để hiển thị: `pip install wordcloud`")
        return
    if not keywords_all:
        st.caption("Chưa có từ khóa.")
        return
    wc = WordCloud(width=600, height=260, background_color=None,
                   mode="RGBA", colormap="plasma", max_words=80,
                   prefer_horizontal=0.85).generate(" ".join(keywords_all))
    fig, ax = plt.subplots(figsize=(6, 2.6), facecolor="none")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# TODO 6 – SENTIMENT TIMELINE
# ============================================================
def render_sentiment_timeline(history: list):
    if len(history) < 2:
        st.caption("Cần ít nhất 2 phản hồi để vẽ timeline.")
        return
    score_map = {"positive": 1, "neutral": 0, "negative": -1}
    clr_map = {"positive": "#4ade80", "neutral": "#94a3b8", "negative": "#f87171"}
    scores = [score_map.get(h.get("sentiment","neutral"), 0) for h in history]
    colors = [clr_map.get(h.get("sentiment","neutral"), "#94a3b8") for h in history]
    labels = [h.get("timestamp","")[-8:] for h in history]
    fig, ax = plt.subplots(figsize=(6, 2.2), facecolor="none")
    ax.set_facecolor("none")
    ax.plot(range(len(scores)), scores, color="#60a5fa", lw=1.5, zorder=1)
    ax.scatter(range(len(scores)), scores, c=colors, s=55, zorder=2)
    ax.axhline(0, color="#475569", lw=0.8, ls="--")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Tiêu cực","Trung lập","Tích cực"], fontsize=8, color="#94a3b8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, fontsize=7, color="#94a3b8")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors="#94a3b8", length=0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# TODO 5 + 12 – SIDEBAR STATS
# ============================================================
def _render_comparison(history: list):
    if len(history) < 4:
        st.caption("Cần ít nhất 4 phản hồi.")
        return
    mid = len(history) // 2
    ga, gb = history[:mid], history[mid:]
    def dominant(g):
        return Counter(x.get("sentiment","neutral") for x in g).most_common(1)[0][0]
    def avg_conf(g):
        return round(sum(x.get("confidence",0) for x in g) / len(g) * 100, 1)
    lv = {"positive":"Tích cực 😊","negative":"Tiêu cực 😟","neutral":"Trung lập 😐"}
    st.dataframe(pd.DataFrame({
        "Nhóm": ["Nhóm A (đầu)","Nhóm B (sau)"],
        "Số PH": [len(ga), len(gb)],
        "Cảm xúc chủ đạo": [lv[dominant(ga)], lv[dominant(gb)]],
        "Tin cậy TB (%)": [avg_conf(ga), avg_conf(gb)],
    }), use_container_width=True, hide_index=True)


def render_sidebar_stats(history: list):
    st.markdown("### 📊 Thống kê tổng hợp")
    if not history:
        st.caption("Chưa có phản hồi nào.")
        return
    counts = Counter(h.get("sentiment","neutral") for h in history)
    c1, c2, c3 = st.columns(3)
    c1.metric("😊", counts.get("positive",0))
    c2.metric("😟", counts.get("negative",0))
    c3.metric("😐", counts.get("neutral",0))

    label_vi = {"positive":"Tích cực","negative":"Tiêu cực","neutral":"Trung lập"}
    clr = {"positive":"#4ade80","negative":"#f87171","neutral":"#94a3b8"}
    present = {k: v for k, v in counts.items() if v > 0}
    fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor="none")
    ax.pie(list(present.values()),
           labels=[label_vi[k] for k in present],
           colors=[clr[k] for k in present],
           autopct="%1.0f%%",
           textprops={"color":"#e2e8f0","fontsize":9},
           wedgeprops={"linewidth":0})
    ax.set_facecolor("none")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("**📈 Xu hướng cảm xúc**")
    render_sentiment_timeline(history)
    st.markdown("**☁️ Word Cloud**")
    render_wordcloud([kw for h in history for kw in h.get("keywords",[])])
    st.markdown("---")
    st.markdown("**⚖️ So sánh 2 nhóm**")
    _render_comparison(history)


# ============================================================
# TODO 9 – DETECT LANGUAGE
# ============================================================
def detect_language(text: str) -> str:
    vi_chars = sum(1 for c in text
                   if unicodedata.category(c) == "Mn"
                   or c in "àáâãèéêìíòóôõùúýăđơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")
    if vi_chars / max(len(text), 1) > 0.04:
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
Gõ vào ô chat cuối trang. Nhiều dòng = nhiều phản hồi độc lập.

### 2. Upload file CSV / Excel
Sidebar → **📂 Upload CSV/Excel**.  
Cột dữ liệu nên đặt tên chứa: `phản hồi`, `feedback`, `comment`, `text`, hoặc `noi_dung`.

### 3. Xuất kết quả
Sidebar → **⬇️ Tải xuống Excel**.

### 4. Ý nghĩa các chỉ số

| Chỉ số | Ý nghĩa |
|--------|---------|
| **Cảm xúc** | Tích cực / Tiêu cực / Trung lập |
| **Độ tin cậy** | Mức chắc chắn của phân tích (0–100%) |
| **Số token** | Số từ/cụm từ sau tách từ |
| **Ngôn ngữ** | Ngôn ngữ phát hiện |
| **Từ khóa** | Top 10 từ/cụm nổi bật (đã lọc stopword & dấu câu) |

### 5. Xóa phản hồi
Mở **📋 Lịch sử** → nhấn 🗑️ trên từng dòng.

### 6. So sánh 2 nhóm
Cần ≥ 4 phản hồi → bảng so sánh tự động xuất hiện trong sidebar.

### 7. Lịch sử tự lưu
Tự động ghi vào `chat_history.json`, khôi phục khi reload trang.
""")


# ============================================================
# TODO 11 – PERSIST HISTORY
# ============================================================
def save_history(history: list, path: str = HISTORY_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


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
    if index < 0 or index >= len(st.session_state.history):
        return
    st.session_state.history.pop(index)
    msgs = [{"role":"assistant",
             "content":"👋 Xin chào! Nhập phản hồi sinh viên để tôi phân tích."}]
    for h in st.session_state.history:
        msgs.append({"role":"user","content":h["text"]})
        msgs.append({"role":"assistant","content":render_analysis(h)})
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
            {"role":"assistant",
             "content":"👋 Xin chào! Nhập phản hồi sinh viên để tôi phân tích cảm xúc và trích xuất từ khóa."}
        ]
        for h in st.session_state.history:
            st.session_state.messages.append({"role":"user","content":h["text"]})
            st.session_state.messages.append({"role":"assistant","content":render_analysis(h)})
    if "page" not in st.session_state:
        st.session_state.page = "chat"
    if "stopwords" not in st.session_state:
        st.session_state.stopwords = load_stopwords()


# ============================================================
# TODO 14 – UNIT TESTS  (python app_chatbot.py --test)
# ============================================================
def _run_tests():
    import sys
    sw = load_stopwords()
    cases = [
        ("Em đánh giá cao chất lượng giảng dạy. Nội dung rõ ràng, hệ thống. Thầy cô nhiệt tình.", "positive"),
        ("Tốc độ giảng đôi lúc nhanh, chưa nhiều ví dụ. Em mong thầy cô cải thiện.", "neutral"),
        ("Chưa hiệu quả. Đọc slide, thiếu giải thích. Khó theo kịp bài. Không tương tác.", "negative"),
        ("Rời rạc, chưa đạt kỳ vọng, thiếu ví dụ minh họa, thụ động, mất tập trung.", "negative"),
        ("", None),
        ("👍👍👍", "positive"),
        ("Tốt", "positive"),
    ]
    passed = failed = 0
    for text, expected in cases:
        r = analyze_feedback(text, sw)
        ok = bool(r["warning"]) if expected is None else r["sentiment"] == expected
        status = "✓ PASS" if ok else "✗ FAIL"
        if ok: passed += 1
        else: failed += 1
        print(f"{status} | expected={expected} got={r['sentiment']} conf={r['confidence']:.0%} | {text[:60]!r}")
    dummy = {"text":"test","timestamp":"2024-01-01T00:00:00","sentiment":"positive",
             "confidence":0.85,"keywords":["giảng","dạy"],"token_count":10,
             "language":"vi","warning":None}
    md = render_analysis(dummy)
    assert "Tích cực" in md and "85%" in md, "render_analysis broken"
    print(f"\nKết quả: {passed}/{passed+failed} PASS")
    sys.exit(0 if failed == 0 else 1)


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
    st.markdown("""
<style>
[data-testid="stChatMessage"] { border-radius: 12px; padding: 8px; }
</style>
""", unsafe_allow_html=True)

    init_session_state()
    stopwords = st.session_state.stopwords

    # ── SIDEBAR ──
    with st.sidebar:
        st.markdown("## 🎓 Phân tích Phản hồi SV")
        page = st.radio("Trang", ["💬 Chat phân tích","📖 Hướng dẫn"],
                        label_visibility="collapsed")
        st.session_state.page = page
        st.markdown("---")

        st.markdown("#### 📂 Upload CSV / Excel")
        uploaded = st.file_uploader("Chọn file", type=["csv","xlsx","xls"],
                                    label_visibility="collapsed")
        if uploaded and st.button("▶️ Phân tích file"):
            feedbacks = handle_file_upload(uploaded)
            if feedbacks:
                with st.spinner(f"Đang phân tích {len(feedbacks)} phản hồi…"):
                    for fb in feedbacks:
                        r = analyze_feedback(fb, stopwords)
                        st.session_state.history.append(r)
                        st.session_state.messages += [
                            {"role":"user","content":fb},
                            {"role":"assistant","content":render_analysis(r)},
                        ]
                save_history(st.session_state.history)
                st.success(f"✅ Đã phân tích {len(feedbacks)} phản hồi!")
                st.rerun()

        st.markdown("---")
        if st.session_state.get("history"):
            xlsx = export_history(st.session_state.history)
            st.download_button(
                "⬇️ Tải xuống Excel",
                data=xlsx,
                file_name=f"ket_qua_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        if st.button("🗑️ Xóa toàn bộ lịch sử"):
            st.session_state.history = []
            st.session_state.messages = [
                {"role":"assistant","content":"👋 Xin chào! Nhập phản hồi sinh viên để tôi phân tích."}
            ]
            save_history([])
            st.rerun()

        st.markdown("---")
        render_sidebar_stats(st.session_state.history)

    # ── MAIN AREA ──
    if "Hướng dẫn" in st.session_state.page:
        render_help_page()
        return

    st.title("🤖 Chatbot Phân tích Phản hồi Sinh viên")

    if st.session_state.history:
        with st.expander(f"📋 Lịch sử phân tích ({len(st.session_state.history)} phản hồi)",
                         expanded=False):
            for i, h in enumerate(st.session_state.history):
                c1, c2 = st.columns([8, 1])
                preview = h.get("text","")[:90]
                if len(h.get("text","")) > 90: preview += "…"
                c1.markdown(
                    f"**{i+1}.** `{h.get('timestamp','')}` "
                    f"{EMOJI_MAP.get(h.get('sentiment','neutral'),'😐')} "
                    f"**{h.get('sentiment','')}** ({int(h.get('confidence',0)*100)}%)  \n"
                    f"{preview}"
                )
                if c2.button("🗑️", key=f"del_{i}"):
                    delete_feedback(i)
                    st.rerun()

    st.markdown("---")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Nhập phản hồi sinh viên… (nhiều dòng cũng được)"):
        lines = [l.strip() for l in prompt.splitlines() if l.strip()] or [prompt.strip()]
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role":"user","content":prompt})
        for line in lines:
            if not line:
                continue
            with st.spinner("Đang phân tích…"):
                result = analyze_feedback(line, stopwords)
            md = render_analysis(result)
            with st.chat_message("assistant"):
                if len(lines) > 1:
                    st.markdown(f"**Phản hồi:** _{line[:80]}{'…' if len(line)>80 else ''}_")
                st.markdown(md)
            st.session_state.messages.append({"role":"assistant","content":md})
            st.session_state.history.append(result)
        save_history(st.session_state.history)
        st.rerun()


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_tests()
    else:
        main()
