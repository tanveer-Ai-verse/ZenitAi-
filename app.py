import subprocess
import sys
import streamlit as st
import spacy
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import re
import os
from textblob import TextBlob
from langdetect import detect, DetectorFactory
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import random

DetectorFactory.seed = 0

st.set_page_config(
    page_title="ZenitAi",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*='css'] {
    font-family: 'DM Sans', sans-serif;
    background-color: #06181C;
    color: #E8F4F8;
}

.stApp { background: linear-gradient(135deg, #06181C 0%, #08262C 50%, #0D3540 100%); }

[data-testid='stSidebar'] {
    background: linear-gradient(180deg, #08262C 0%, #041014 100%) !important;
    border-right: 1px solid rgba(30,123,140,0.3) !important;
}

[data-testid='stSidebar'] * { color: #E8F4F8 !important; }

.elite-card {
    background: linear-gradient(145deg, #0D3540, #08262C);
    border: 1px solid rgba(30,123,140,0.3);
    border-radius: 16px;
    padding: 24px;
    margin: 10px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(77,217,240,0.1);
}

.title-hero {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #4DD9F0, #1E7B8C);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
    line-height: 1.1;
}

.sub-hero {
    color: rgba(77,217,240,0.3);
    font-size: 0.9rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-weight: 300;
}

.pos-tag {
    display: inline-block;
    padding: 5px 13px;
    margin: 3px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
    border: 1px solid rgba(255,255,255,0.15);
}

.err-card {
    background: linear-gradient(135deg, rgba(26,10,10,0.8), rgba(44,16,16,0.8));
    border-left: 4px solid #FF4D6D;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.9rem;
}

.ok-card {
    background: linear-gradient(135deg, rgba(10,26,14,0.8), rgba(16,44,26,0.8));
    border-left: 4px solid #2DDB7F;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
}

.metric-box {
    background: linear-gradient(145deg, #0D3540, #08262C);
    border: 1px solid rgba(30,123,140,0.3);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    margin: 4px 0;
}

.metric-val { font-family: 'DM Serif Display', serif; font-size: 2rem; color: #4DD9F0; }
.metric-lbl { font-size: 0.75rem; color: #7AB8C4; text-transform: uppercase; letter-spacing: 1px; }

.stTextInput input, .stTextArea textarea {
    background: #08262C !important;
    border: 1px solid rgba(30,123,140,0.5) !important;
    border-radius: 10px !important;
    color: #E8F4F8 !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #4DD9F0 !important;
    box-shadow: 0 0 0 2px rgba(77,217,240,0.15) !important;
}

.stButton button {
    background: linear-gradient(135deg, #1E7B8C, #124A59) !important;
    color: #E8F4F8 !important;
    border: 1px solid rgba(77,217,240,0.2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stButton button:hover {
    background: linear-gradient(135deg, #4DD9F0, #1E7B8C) !important;
    color: #06181C !important;
}

.stTabs [data-baseweb='tab-list'] { background: #08262C; border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb='tab'] { background: transparent; color: #7AB8C4; border-radius: 8px; font-weight: 500; }
.stTabs [aria-selected='true'] { background: linear-gradient(135deg, #1E7B8C, #124A59) !important; color: #4DD9F0 !important; }

.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #4DD9F0, transparent);
    margin: 20px 0;
    opacity: 0.4;
}
</style>
""", unsafe_allow_html=True)


# ── spaCy model auto-download ─────────────────────────────────────────────────
# ── spaCy model loader ────────────────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    # Define a local path in your project folder
    model_path = os.path.join(os.getcwd(), "spacy_models", "en_core_web_sm")
    
    try:
        return spacy.load(model_path)
    except OSError:
        # If not found, download it to the local folder
        print("Downloading spaCy model to local directory...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm", 
            "--target", os.path.join(os.getcwd(), "spacy_models")
        ])
        # Add the download path to the system path so spaCy can find it
        sys.path.append(os.path.join(os.getcwd(), "spacy_models"))
        return spacy.load("en_core_web_sm")

nlp = load_nlp()

# ── NLTK data auto-download ───────────────────────────────────────────────────
@st.cache_resource
def ensure_nltk():
    nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    nltk.data.path.append(nltk_data_path)
    
    for pkg in ['stopwords', 'punkt', 'wordnet']:
        try:
            nltk.download(pkg, download_dir=nltk_data_path, quiet=True)
        except Exception:
            pass
    return True

nlp = load_nlp()
ensure_nltk()


# ── Helpers ───────────────────────────────────────────────────────────────────
def base_layout(title_text="", height=420):
    return dict(
        title=dict(text=title_text, font=dict(color="#4DD9F0", size=15, family="DM Serif Display")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8F4F8", family="DM Sans"),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor="rgba(30,123,140,0.15)", linecolor="rgba(30,123,140,0.3)"),
        yaxis=dict(gridcolor="rgba(30,123,140,0.15)", linecolor="rgba(30,123,140,0.3)"),
    )


def get_stats(text):
    words = re.findall(r'\b\w+\b', text)
    sentences = [s for s in re.split(r'[.!?]+', text.strip()) if s.strip()]
    unique = len(set(w.lower() for w in words))
    avg_word_len = round(sum(len(w) for w in words) / max(len(words), 1), 1)
    reading_time = round(len(words) / 200, 1)
    return {
        "Words": len(words),
        "Sentences": len(sentences),
        "Chars": len(text),
        "Chars (no space)": len(text.replace(" ", "")),
        "Unique Words": unique,
        "Avg Word Len": avg_word_len,
        "Read Time (min)": reading_time,
    }


def grammar_check(text):
    doc = nlp(text)
    issues = []
    uncountable = ['information', 'water', 'knowledge', 'advice', 'furniture', 'equipment', 'luggage', 'news']

    if not text.strip():
        return []
    if not text[0].isupper():
        issues.append('❌ Capitalization: First letter must be uppercase.')
    if text.strip() and text.strip()[-1] not in ['.', '!', '?']:
        issues.append('❌ Punctuation: Missing terminal punctuation (. ! ?).')

    for token in doc:
        if token.text.lower() in uncountable:
            if any(c.text.lower() in ['a', 'an'] for c in token.children):
                issues.append(f'❌ Countability: "{token.text}" is uncountable — remove a/an.')

    return issues


TENSE_MAP = {
    'Simple Present': lambda l: l + 's',
    'Simple Past': lambda l: l + 'ed',
    'Simple Future': lambda l: 'will ' + l,
    'Present Continuous': lambda l: 'is ' + (l[:-1] if l.endswith('e') else l) + 'ing',
    'Past Continuous': lambda l: 'was ' + (l[:-1] if l.endswith('e') else l) + 'ing',
    'Future Continuous': lambda l: 'will be ' + (l[:-1] if l.endswith('e') else l) + 'ing',
    'Present Perfect': lambda l: 'has ' + l + 'ed',
    'Past Perfect': lambda l: 'had ' + l + 'ed',
    'Future Perfect': lambda l: 'will have ' + l + 'ed',
    'Present Perfect Continuous': lambda l: 'has been ' + (l[:-1] if l.endswith('e') else l) + 'ing',
    'Past Perfect Continuous': lambda l: 'had been ' + (l[:-1] if l.endswith('e') else l) + 'ing',
    'Future Perfect Continuous': lambda l: 'will have been ' + (l[:-1] if l.endswith('e') else l) + 'ing',
}


def transform_tense(text, tense):
    if not text.strip():
        return 'Please enter a sentence.'
    doc = nlp(text)
    fn = TENSE_MAP.get(tense)
    if not fn:
        return text
    out = []
    for t in doc:
        if t.pos_ == 'VERB':
            out.append(fn(t.lemma_))
        else:
            out.append(t.text)
    return ' '.join(out)


def get_sentiment(text):
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    sub = blob.sentiment.subjectivity
    if pol > 0.3:
        label, emoji = 'Positive', '😊'
    elif pol < -0.3:
        label, emoji = 'Negative', '😔'
    else:
        label, emoji = 'Neutral', '😐'
    return pol, sub, label, emoji


def get_word_freq(text, top_n=15):
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    filtered = [w for w in words if w not in stop_words]
    return Counter(filtered).most_common(top_n)


def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return 'en'


def ai_unavailable(feature_name="This AI feature"):
    """Consistent placeholder shown when Groq is not configured."""
    return (
        f"⚠️ {feature_name} requires an LLM backend. "
        "The Groq integration has been removed from this deployment. "
        "Re-add the groq library and API key to re-enable AI features."
    )


def translate_text(text, target_lang):
    return ai_unavailable("Translation")


POS_COLORS = {
    'NOUN': ('#1E7B8C', '#4DD9F0'),
    'VERB': ('#7C2D8C', '#D87BE0'),
    'ADJ': ('#1A6B3A', '#4DD990'),
    'ADV': ('#8C5A00', '#F0C44D'),
    'PRON': ('#1A3A7C', '#4D90F0'),
    'DET': ('#5A1A1A', '#F07070'),
    'ADP': ('#2A2A6C', '#9090F0'),
    'CONJ': ('#4A2A00', '#D4A030'),
    'PUNCT': ('#2A2A2A', '#888888'),
    'NUM': ('#1A4A3A', '#60C8A0'),
}
DEFAULT_POS_COLOR = ('#1A2A2A', '#6AA0A8')


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family: DM Serif Display, serif; font-size:1.8rem; color:#4DD9F0; margin:0;">🌟 ZenitAi</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#7AB8C4; font-size:0.75rem; letter-spacing:2px; text-transform:uppercase;">Pro Edition v1.0</p>', unsafe_allow_html=True)
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("**📊 Session Analytics**")
    if 'analyses_done' not in st.session_state:
        st.session_state.analyses_done = 0
        st.session_state.ai_calls = 0
        st.session_state.history = []
        st.session_state.score = 0

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Analyses", st.session_state.analyses_done)
    with col_b:
        st.metric("AI Calls", st.session_state.ai_calls)
    with col_c:
        st.metric("Score", st.session_state.score)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown("**🧬 Engine Status**")
    st.progress(95, text="spaCy NLP")
    st.progress(88, text="TextBlob Sentiment")


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<p class="sub-hero">🌍 MULTILINGUAL · 🎓 EDUCATIONAL · 🤖 AI-POWERED</p>', unsafe_allow_html=True)
st.markdown('<p class="title-hero">ZenitAi — AI English Teacher</p>', unsafe_allow_html=True)
st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

col_inp, col_btn = st.columns([5, 1])
with col_inp:
    txt = st.text_area(
        "📝 Enter text to analyze:",
        value="He gives me information about the new equipment in the laboratory.",
        height=100,
        help="Type or paste any English text for comprehensive NLP analysis"
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔬 Analyze", use_container_width=True)

if txt.strip():
    st.session_state.analyses_done += 1
    if txt not in st.session_state.history:
        st.session_state.history.append(txt)

if txt.strip():
    stats = get_stats(txt)
    lang = detect_language(txt)
    pol, sub, sent_label, sent_emoji = get_sentiment(txt)

    sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
    quick_stats = [
        (sc1, stats['Words'], "Words"),
        (sc2, stats['Sentences'], "Sentences"),
        (sc3, stats['Unique Words'], "Unique Words"),
        (sc4, f"{stats['Read Time (min)']}m", "Read Time"),
        (sc5, f"{sent_emoji} {sent_label}", "Sentiment"),
        (sc6, lang.upper(), "Language"),
    ]
    for col, val, lbl in quick_stats:
        with col:
            st.markdown(
                f"""<div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
                </div>""",
                unsafe_allow_html=True
            )

st.markdown("<br>", unsafe_allow_html=True)

tabs = st.tabs([
    "🛡️ Grammar",
    "⏳ 12-Tenses",
    "🎨 POS Tags",
    "📊 Frequency",
    "🌍 Translate",
    "💬 Paraphrase",
    "💡 AI Tutor",
    "🔭 Deep Analysis",
    "✨ Style",
    "📚 Practice",
])

with tabs[0]:
    st.markdown("### 🛡️ Grammar Guard")
    if txt.strip():
        errors = grammar_check(txt)
        col_g1, col_g2 = st.columns([2, 1])
        with col_g1:
            if errors:
                st.markdown(f'<div class="err-card">Found <b>{len(errors)}</b> issue(s):</div>', unsafe_allow_html=True)
                for e in errors:
                    st.markdown(f'<div class="err-card">{e}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ok-card">✅ Perfect grammar! No errors detected.</div>', unsafe_allow_html=True)
        with col_g2:
            score = max(0, 100 - len(errors) * 20)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "%"},
                gauge={
                    'axis': {"range": [0, 100]},
                    'bar': {"color": "#4DD9F0"},
                    'bgcolor': "rgba(8,38,44,0.5)",
                }
            ))
            fig_gauge.update_layout(**base_layout("Grammar Score", height=250))
            st.plotly_chart(fig_gauge, use_container_width=True)

        if errors and st.button("🤖 AI Grammar Fix", key="ai_fix"):
            st.session_state.ai_calls += 1
            st.markdown(f'<div class="elite-card">{ai_unavailable("AI Grammar Fix")}</div>', unsafe_allow_html=True)
    else:
        st.info("Enter text above to check grammar.")

with tabs[1]:
    st.markdown("### ⏳ 12-Tense Matrix")
    if txt.strip():
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            selected_tense = st.selectbox("Select Tense", list(TENSE_MAP.keys()))
            result = transform_tense(txt, selected_tense)
            st.markdown(f"""<div class="elite-card">
                <div style="color:#7AB8C4; font-size:0.75rem;">OUTPUT</div>
                <div style="font-size:1.05rem; color:#4DD9F0; margin-top:10px;">{result}</div>
            </div>""", unsafe_allow_html=True)
        with col_t2:
            st.markdown("**📋 All Transformations**")
            all_results = []
            for tense_name in TENSE_MAP:
                transformed = transform_tense(txt, tense_name)
                all_results.append({"Tense": tense_name, "Result": transformed})
            st.dataframe(pd.DataFrame(all_results), hide_index=True, use_container_width=True)
    else:
        st.info("Enter text to use tense matrix.")

with tabs[2]:
    st.markdown("### 🎨 Parts-of-Speech Painter")
    if txt.strip():
        doc = nlp(txt)
        tokens_html = ''
        for t in doc:
            bg, fg = POS_COLORS.get(t.pos_, DEFAULT_POS_COLOR)
            tokens_html += f'<span class="pos-tag" style="background:{bg}; color:{fg};" title="{t.pos_}">{t.text}</span>'
        st.markdown(f'<div class="elite-card" style="line-height:2.5;">{tokens_html}</div>', unsafe_allow_html=True)

        pos_counts = Counter(t.pos_ for t in doc if t.pos_ not in ['SPACE', 'PUNCT'])
        if pos_counts:
            labels_pos = list(pos_counts.keys())
            values_pos = list(pos_counts.values())
            colors_pos = [POS_COLORS.get(p, DEFAULT_POS_COLOR)[1] for p in labels_pos]
            fig_pos = go.Figure(go.Bar(
                x=labels_pos, y=values_pos,
                marker=dict(color=colors_pos),
                text=values_pos, textposition='outside',
            ))
            fig_pos.update_layout(**base_layout("POS Distribution", height=320))
            st.plotly_chart(fig_pos, use_container_width=True)
    else:
        st.info("Enter text to paint POS tags.")

with tabs[3]:
    st.markdown("### 📊 Word Frequency Lab")
    if txt.strip():
        freq_data = get_word_freq(txt, top_n=20)
        if freq_data:
            words_freq = [w for w, c in freq_data]
            counts_freq = [c for w, c in freq_data]
            col_f1, col_f2 = st.columns(2)

            with col_f1:
                fig_hbar = go.Figure(go.Bar(
                    x=counts_freq[::-1], y=words_freq[::-1],
                    orientation='h',
                    marker=dict(color='#4DD9F0'),
                ))
                fig_hbar.update_layout(**base_layout("Top Words", height=400))
                st.plotly_chart(fig_hbar, use_container_width=True)

            with col_f2:
                fig_donut = go.Figure(go.Pie(
                    labels=words_freq[:8], values=counts_freq[:8],
                    hole=0.55,
                ))
                fig_donut.update_layout(**base_layout("Top 8 Share", height=400))
                st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("Not enough content.")
    else:
        st.info("Enter text to explore frequencies.")

with tabs[4]:
    st.markdown("### 🌍 AI Translator")
    if txt.strip():
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            target_lang = st.selectbox("Translate to:", ['Urdu', 'Pashto', 'Hindi', 'Arabic'])
        with col_t2:
            if st.button("🌐 Translate Now", key="translate_btn"):
                st.session_state.ai_calls += 1
                with st.spinner(f"Translating to {target_lang}..."):
                    translation = translate_text(txt, target_lang)
                    st.markdown(f'<div class="elite-card"><b>Translation:</b><br>{translation}</div>', unsafe_allow_html=True)
    else:
        st.info("Enter text to translate.")

with tabs[5]:
    st.markdown("### 💬 Smart Paraphraser")
    if txt.strip():
        formality = st.selectbox("Formality Level:", ['Formal', 'Neutral', 'Informal'])
        if st.button("✍️ Paraphrase", key="paraphrase_btn"):
            st.session_state.ai_calls += 1
            with st.spinner("Rephrasing..."):
                st.markdown(f'<div class="elite-card">{ai_unavailable("Smart Paraphraser")}</div>', unsafe_allow_html=True)
    else:
        st.info("Enter text to paraphrase.")

with tabs[6]:
    st.markdown("### 💡 AI Tutor")
    if txt.strip():
        mode = st.radio(
            "Choose Mode:",
            ['📖 Explain', '🎓 Teach', '💬 Simplify', '🔍 Breakdown'],
            horizontal=True
        )
        if st.button("🚀 Ask AI", key="ai_ask"):
            st.session_state.ai_calls += 1
            with st.spinner("Thinking..."):
                st.markdown(f'<div class="elite-card">{ai_unavailable("AI Tutor")}</div>', unsafe_allow_html=True)
    else:
        st.info("Enter text to use AI Tutor.")

with tabs[7]:
    st.markdown("### 🔭 Deep Analysis")
    if txt.strip():
        doc = nlp(txt)
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("**🎭 Sentiment**")
            pol, sub, sent_label, sent_emoji = get_sentiment(txt)
            fig_sent = go.Figure(go.Indicator(
                mode="gauge+number",
                value=(pol + 1) * 50,
                number={"suffix": "%"},
                title=sent_label,
            ))
            fig_sent.update_layout(**base_layout(f"Polarity: {pol:.2f}", height=250))
            st.plotly_chart(fig_sent, use_container_width=True)

        with col_d2:
            st.markdown("**🏷️ Named Entities**")
            ents = [(e.text, e.label_) for e in doc.ents]
            if ents:
                for ent_text, ent_label in ents:
                    st.markdown(f"• **{ent_text}** ({ent_label})")
            else:
                st.markdown("No entities found.")

        st.markdown("**Dependencies**")
        dep_data = [{"Token": t.text, "POS": t.pos_, "Dependency": t.dep_, "Head": t.head.text} for t in doc]
        st.dataframe(pd.DataFrame(dep_data), hide_index=True, use_container_width=True)
    else:
        st.info("Enter text for deep analysis.")

with tabs[8]:
    st.markdown("### ✨ Style Transformer")
    if txt.strip():
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            style = st.selectbox("Style:", ['Academic', 'Casual', 'Business', 'Creative'])
        with col_s2:
            tone = st.selectbox("Tone:", ['Formal', 'Friendly', 'Professional', 'Persuasive'])

        if st.button("🎨 Transform", key="transform"):
            st.session_state.ai_calls += 1
            with st.spinner("Transforming..."):
                st.markdown(f'<div class="elite-card">{ai_unavailable("Style Transformer")}</div>', unsafe_allow_html=True)
    else:
        st.info("Enter text to transform.")

with tabs[9]:
    st.markdown("### 📚 Practice Mode")
    st.markdown("**🎯 Quiz Generator**")
    if txt.strip():
        if st.button("Generate MCQ", key="generate_mcq"):
            st.session_state.ai_calls += 1
            with st.spinner("Creating quiz..."):
                st.markdown(f'<div class="elite-card">{ai_unavailable("Quiz Generator")}</div>', unsafe_allow_html=True)
    else:
        st.info("Enter text to generate practice questions.")

st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center; color:#4DD9F044; font-size:0.75rem;">'
    'ZenitAi Pro v1.0 | Powered by spaCy NLP + TextBlob'
    '</p>',
    unsafe_allow_html=True
)
