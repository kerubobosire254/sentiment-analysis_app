import streamlit as st
import pickle
import re
import time
import pandas as pd
from collections import Counter
import math

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ · Review Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS

# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Fira+Code:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
  --bg:        #f7f8fa;
  --surface:   #ffffff;
  --surface2:  #f0f2f5;
  --border:    #e2e5ea;
  --accent:    #2563eb;
  --accent2:   #059669;
  --neg:       #dc2626;
  --pos:       #059669;
  --text:      #1f2937;
  --muted:     #9ca3af;
  --font-head: 'Nunito', sans-serif;
  --font-body: 'Nunito', sans-serif;
  --font-mono: 'Fira Code', monospace;
}

/* ── Global reset ── */
.stApp { background: var(--bg) !important; }
.stApp > header { display: none; }
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1300px; }
h1,h2,h3,h4 { font-family: var(--font-head) !important; color: var(--text) !important; }
p, li, label, div, span { font-family: var(--font-body) !important; color: var(--text); }
* { box-sizing: border-box; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #eff6ff 0%, #ffffff 50%, #f0fdf4 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(37,99,235,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(5,150,105,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-label {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: var(--font-head) !important;
    font-size: 3.2rem;
    font-weight: 300;
    line-height: 1.1;
    color: var(--text) !important;
    margin-bottom: 0.75rem;
    letter-spacing: -0.01em;
}
.hero-title em { color: var(--accent); font-style: italic; }
.hero-sub {
    font-family: var(--font-body) !important;
    font-size: 1rem;
    color: var(--muted);
    max-width: 520px;
    line-height: 1.6;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
}
.card-title {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── Result pill ── */
.result-box {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1.25rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.result-pos {
    background: rgba(5,150,105,0.06);
    border: 1px solid rgba(5,150,105,0.25);
}
.result-neg {
    background: rgba(220,38,38,0.05);
    border: 1px solid rgba(220,38,38,0.2);
}
.result-emoji { font-size: 2rem; }
.result-label {
    font-family: var(--font-head) !important;
    font-size: 1.8rem;
    font-weight: 300;
    line-height: 1;
}
.result-pos .result-label { color: var(--pos); }
.result-neg .result-label { color: var(--neg); }
.result-conf {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* ── Confidence bar ── */
.conf-track {
    height: 6px;
    background: var(--surface2);
    border-radius: 99px;
    margin: 1rem 0;
    overflow: hidden;
}
.conf-fill-pos { height: 100%; background: var(--pos); border-radius: 99px; transition: width 0.8s ease; }
.conf-fill-neg { height: 100%; background: var(--neg); border-radius: 99px; transition: width 0.8s ease; }

/* ── Word chips ── */
.word-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.chip {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem;
    padding: 0.25rem 0.6rem;
    border-radius: 99px;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--muted);
}
.chip-hot { border-color: rgba(37,99,235,0.3); color: var(--accent); background: rgba(37,99,235,0.05); }

/* ── Stat tiles ── */
.stat-tile {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.25rem;
    text-align: center;
}
.stat-num {
    font-family: var(--font-head) !important;
    font-size: 1.9rem;
    font-weight: 300;
    color: var(--accent);
    line-height: 1;
}
.stat-lbl {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 0.3rem;
}

/* ── Example buttons ── */
.stButton button {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s;
}
.stButton button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ── Textarea ── */
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(232,197,71,0.12) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* ── Sidebar items ── */
[data-testid="stSidebar"] .stMarkdown, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: var(--text) !important; }

/* ── History row ── */
.hist-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.hist-badge {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem;
    padding: 0.2rem 0.5rem;
    border-radius: 99px;
    flex-shrink: 0;
}
.hist-pos { background: rgba(76,240,176,0.12); color: var(--pos); }
.hist-neg { background: rgba(255,87,87,0.12); color: var(--neg); }
.hist-text { color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.hist-conf { font-family: var(--font-mono) !important; font-size: 0.7rem; color: var(--muted); flex-shrink: 0; }

/* ── Section label ── */
.section-label {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
}

/* ── Divider ── */
hr { border: none; border-top: 1px solid var(--border) !important; margin: 2rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

/* ── DataFrame ── */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("model.pkl", "rb"))
    except FileNotFoundError:
        return None

model = load_model()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "he","him","his","she","her","hers","it","its","they","them","their",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","to","from","in","out","on","off","then",
    "so","just","very","get","got","also","would","could","should","will","can",
}

POSITIVE_WORDS = {
    "excellent","amazing","love","great","fantastic","perfect","wonderful",
    "awesome","outstanding","best","superb","beautiful","brilliant","good",
    "happy","satisfied","recommend","pleased","quality","fast","easy",
    "durable","reliable","sturdy","value","affordable","impressed",
}

NEGATIVE_WORDS = {
    "terrible","awful","poor","worst","bad","horrible","disappointed",
    "broken","cheap","waste","useless","disappointing","defective","slow",
    "difficult","frustrating","never","refund","return","damaged","misleading",
    "fake","overpriced","fragile","flimsy","stopped","failed","issues",
}

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def get_text_stats(text: str) -> dict:
    words = text.split()
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
    return {
        "words": len(words),
        "sentences": max(len(sentences), 1),
        "avg_word_len": round(avg_word_len, 1),
        "unique_words": len(set(words)),
    }

def get_keyword_signals(text: str) -> tuple:
    words_clean = set(re.findall(r'\b[a-z]+\b', text.lower()))
    pos_hits = sorted(words_clean & POSITIVE_WORDS)
    neg_hits = sorted(words_clean & NEGATIVE_WORDS)
    return pos_hits, neg_hits

def get_top_words(text: str, n: int = 12) -> list:
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    return [w for w, _ in Counter(filtered).most_common(n)]

def predict_sentiment(text: str):
    """Returns (label, confidence) or raises if no model."""
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([cleaned])[0]
        conf = float(max(proba)) * 100
    else:
        conf = None
    return pred, conf

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">⚙ Settings</div>', unsafe_allow_html=True)
    show_keywords = st.toggle("Keyword signal analysis", value=True)
    show_stats    = st.toggle("Text statistics", value=True)
    show_top_words = st.toggle("Top word cloud", value=True)

    st.markdown("---")
    st.markdown('<div class="section-label">📋 Analysis History</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown('<p style="color:var(--muted);font-size:0.82rem;">No analyses yet.</p>', unsafe_allow_html=True)
    else:
        for item in reversed(st.session_state.history[-10:]):
            badge_cls = "hist-pos" if item["label"] == 1 else "hist-neg"
            badge_txt = "POS" if item["label"] == 1 else "NEG"
            conf_txt  = f"{item['conf']:.0f}%" if item["conf"] else "—"
            short_txt = item["text"][:55] + "…" if len(item["text"]) > 55 else item["text"]
            st.markdown(
                f'<div class="hist-row">'
                f'<span class="hist-badge {badge_cls}">{badge_txt}</span>'
                f'<span class="hist-text">{short_txt}</span>'
                f'<span class="hist-conf">{conf_txt}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<p style="color:var(--muted);font-size:0.75rem;font-family:var(--font-mono);line-height:1.6;">'
        'Model: Logistic Regression<br>Vectorizer: TF-IDF<br>Dataset: Amazon Reviews</p>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-label">🧠 NLP-Powered · Real-time Analysis</div>
  <div class="hero-title">Sentiment<em>IQ</em></div>
  <div class="hero-sub">
    Decode the emotion behind any product review. Built on Logistic Regression + TF-IDF —
    fast, interpretable, and production-ready.
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main layout: input | result
# ─────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">📝 Input Review</div>', unsafe_allow_html=True)

    # Quick-fill example buttons
    ex_cols = st.columns(3)
    examples = [
        ("✅ Strong Positive", "Absolutely love this product! Build quality is superb, arrives fast, and it exceeded every expectation. Would 100% recommend."),
        ("⚠️ Mixed", "The product quality is excellent, but delivery was painfully slow and packaging was damaged. Great value though."),
        ("❌ Negative", "Terrible. Stopped working after two days. Cheap materials, misleading description. Complete waste of money."),
    ]
    for i, (label, sample) in enumerate(examples):
        with ex_cols[i]:
            if st.button(label, use_container_width=True):
                st.session_state.review_text = sample

    review = st.text_area(
        "Review text",
        value=st.session_state.review_text,
        height=180,
        placeholder="Paste or type a product review here…",
        label_visibility="collapsed",
    )

    analyze_btn = st.button("→  Analyze Sentiment", type="primary", use_container_width=True)

    # ── Batch CSV upload ──
    st.markdown("---")
    st.markdown('<div class="section-label">📂 Batch Analysis (CSV)</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a CSV with a 'review' column",
        type=["csv"],
        label_visibility="collapsed",
    )

with right:
    st.markdown('<div class="section-label">📊 Result</div>', unsafe_allow_html=True)

    result_placeholder = st.empty()

    if not analyze_btn and not st.session_state.history:
        result_placeholder.markdown(
            '<div class="card" style="text-align:center;padding:3rem 1.5rem;">'
            '<div style="font-size:2.5rem;margin-bottom:1rem;">🧠</div>'
            '<div style="color:var(--muted);font-size:0.9rem;line-height:1.6;">Your analysis results will<br>appear here.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if analyze_btn:
        if not review.strip():
            result_placeholder.warning("Please enter a review first.")
        elif not model:
            result_placeholder.error("model.pkl not found. Place your trained model in the same directory.")
        else:
            with st.spinner("Analysing…"):
                time.sleep(0.3)  # slight delay for UX feel
                label, conf = predict_sentiment(review)
                st.session_state.history.append({"text": review, "label": label, "conf": conf})

            is_pos = label == 1
            emoji  = "😊" if is_pos else "😠"
            word   = "Positive" if is_pos else "Negative"
            fill_cls = "conf-fill-pos" if is_pos else "conf-fill-neg"
            box_cls  = "result-pos" if is_pos else "result-neg"
            conf_val = conf if conf else 0

            result_placeholder.markdown(
                f'<div class="result-box {box_cls}">'
                f'<span class="result-emoji">{emoji}</span>'
                f'<div>'
                f'<div class="result-label">{word}</div>'
                f'<div class="result-conf">{"Confidence: " + f"{conf:.1f}%" if conf else "Probability unavailable"}</div>'
                f'</div>'
                f'</div>'
                f'<div class="conf-track"><div class="{fill_cls}" style="width:{conf_val:.0f}%"></div></div>',
                unsafe_allow_html=True,
            )

            stats = get_text_stats(review)

            if show_stats:
                st.markdown('<div class="section-label" style="margin-top:1.5rem;">📐 Text Stats</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                for col, num, lbl in zip(
                    [c1, c2, c3, c4],
                    [stats["words"], stats["sentences"], stats["unique_words"], stats["avg_word_len"]],
                    ["Words", "Sentences", "Unique", "Avg len"],
                ):
                    col.markdown(
                        f'<div class="stat-tile"><div class="stat-num">{num}</div>'
                        f'<div class="stat-lbl">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            if show_keywords:
                pos_hits, neg_hits = get_keyword_signals(review)
                st.markdown('<div class="section-label" style="margin-top:1.5rem;">🔑 Keyword Signals</div>', unsafe_allow_html=True)
                kc1, kc2 = st.columns(2)
                with kc1:
                    st.markdown('<span style="color:var(--pos);font-size:0.72rem;font-family:var(--font-mono);">POSITIVE SIGNALS</span>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="chip chip-hot">{w}</span>' for w in pos_hits) or '<span style="color:var(--muted);font-size:0.8rem;">None detected</span>'
                    st.markdown(f'<div class="word-chips">{chips}</div>', unsafe_allow_html=True)
                with kc2:
                    st.markdown('<span style="color:var(--neg);font-size:0.72rem;font-family:var(--font-mono);">NEGATIVE SIGNALS</span>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="chip" style="border-color:rgba(255,87,87,0.4);color:var(--neg);background:rgba(255,87,87,0.06);">{w}</span>' for w in neg_hits) or '<span style="color:var(--muted);font-size:0.8rem;">None detected</span>'
                    st.markdown(f'<div class="word-chips">{chips}</div>', unsafe_allow_html=True)

            if show_top_words:
                top = get_top_words(review)
                if top:
                    st.markdown('<div class="section-label" style="margin-top:1.5rem;">📌 Top Keywords</div>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="chip">{w}</span>' for w in top)
                    st.markdown(f'<div class="word-chips">{chips}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Batch CSV section (full width below)
# ─────────────────────────────────────────────
if uploaded:
    st.markdown("---")
    st.markdown('<div class="section-label">📂 Batch Results</div>', unsafe_allow_html=True)

    if not model:
        st.error("model.pkl not found.")
    else:
        try:
            df = pd.read_csv(uploaded)
            col_candidates = [c for c in df.columns if "review" in c.lower() or "text" in c.lower() or "comment" in c.lower()]
            review_col = col_candidates[0] if col_candidates else df.columns[0]

            with st.spinner(f"Analysing {len(df)} reviews…"):
                labels, confs = [], []
                for txt in df[review_col].fillna("").astype(str):
                    try:
                        lbl, conf = predict_sentiment(txt)
                        labels.append("Positive" if lbl == 1 else "Negative")
                        confs.append(f"{conf:.1f}%" if conf else "—")
                    except Exception:
                        labels.append("Error"); confs.append("—")

            df["Sentiment"]  = labels
            df["Confidence"] = confs

            pos_count = labels.count("Positive")
            neg_count = labels.count("Negative")
            total     = len(labels)

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f'<div class="stat-tile"><div class="stat-num">{total}</div><div class="stat-lbl">Total Reviews</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="stat-tile"><div class="stat-num" style="color:var(--pos)">{pos_count}</div><div class="stat-lbl">Positive</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="stat-tile"><div class="stat-num" style="color:var(--neg)">{neg_count}</div><div class="stat-lbl">Negative</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="stat-tile"><div class="stat-num">{pos_count/total*100:.0f}%</div><div class="stat-lbl">Positive Rate</div></div>', unsafe_allow_html=True)

            st.dataframe(df, use_container_width=True, height=350)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇  Download Results CSV",
                data=csv_out,
                file_name="sentiment_results.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:var(--muted);font-family:var(--font-mono);font-size:0.72rem;letter-spacing:0.08em;">'
    'SENTIMENTIQ · Built by Kerubo Bosire · Logistic Regression + TF-IDF · Streamlit'
    '</p>',
    unsafe_allow_html=True,
)