"""Composants UI réutilisables."""
import streamlit as st
from config import ACTION_COLORS, COLOR_SEQ


# ── CSS global ────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:       #0f1117;
    --surface:  #1a1f2e;
    --surface2: #242938;
    --border:   #2d3348;
    --accent:   #00c896;
    --danger:   #ff4b4b;
    --warning:  #ff8c42;
    --info:     #4a90d9;
    --text:     #e8eaf0;
    --muted:    #8892a4;
    --radius:   10px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background-color: var(--bg) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metrics Streamlit overrides */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'JetBrains Mono' !important; }

/* Cards */
.fw-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 22px;
    margin-bottom: 14px;
}
.fw-card-accent  { border-left: 3px solid var(--accent); }
.fw-card-danger  { border-left: 3px solid var(--danger); }
.fw-card-info    { border-left: 3px solid var(--info); }
.fw-card-warning { border-left: 3px solid var(--warning); }

/* Section title */
.fw-section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    border-bottom: 2px solid var(--accent);
    padding-bottom: 5px;
    margin: 22px 0 14px 0;
    letter-spacing: 0.01em;
}

/* Page header */
.fw-page-header {
    padding: 10px 0 18px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 22px;
}
.fw-page-header h1 {
    font-size: 1.7rem;
    font-weight: 700;
    margin: 0 0 4px 0;
}
.fw-page-header p {
    color: var(--muted);
    margin: 0;
    font-size: 0.9rem;
}

/* LLM response */
.fw-llm {
    background: linear-gradient(135deg, #0d2137 0%, #0d1f2d 100%);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    padding: 18px 22px;
    color: var(--text);
    line-height: 1.8;
    font-size: 0.88rem;
    margin-top: 12px;
}
.fw-user-msg {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
}

/* Badge */
.fw-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.03em;
}
.fw-badge-green  { background: #0d4429; color: #3fb950; }
.fw-badge-red    { background: #3d1215; color: #f85149; }
.fw-badge-blue   { background: #0c2d6b; color: #79c0ff; }
.fw-badge-orange { background: #3d2200; color: #ffa657; }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Tabs */
[data-testid="stTab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}

/* Inputs */
.stTextInput > div > input,
.stTextArea textarea,
.stSelectbox > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
"""

PLOTLY_DARK = dict(
    paper_bgcolor="#1a1f2e",
    plot_bgcolor="#0f1117",
    font=dict(family="Inter", color="#e8eaf0", size=11),
    xaxis=dict(gridcolor="#2d3348", linecolor="#2d3348", zerolinecolor="#2d3348"),
    yaxis=dict(gridcolor="#2d3348", linecolor="#2d3348", zerolinecolor="#2d3348"),
    legend=dict(bgcolor="#1a1f2e", bordercolor="#2d3348", borderwidth=1),
    margin=dict(t=48, b=36, l=40, r=20),
    colorway=["#00c896","#4a90d9","#ff4b4b","#ff8c42","#a78bfa","#34d399","#f59e0b"],
)


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = ""):
    full_title = f"{icon} {title}" if icon else title
    sub_html   = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"<div class='fw-page-header'><h1>{full_title}</h1>{sub_html}</div>",
        unsafe_allow_html=True
    )


def section_title(text: str):
    st.markdown(f"<div class='fw-section-title'>{text}</div>", unsafe_allow_html=True)


def card(content: str, variant: str = ""):
    cls = f"fw-card fw-card-{variant}" if variant else "fw-card"
    st.markdown(f"<div class='{cls}'>{content}</div>", unsafe_allow_html=True)


def badge(text: str, color: str = "blue"):
    st.markdown(f"<span class='fw-badge fw-badge-{color}'>{text}</span>",
                unsafe_allow_html=True)


def llm_response(text: str):
    text_html = text.replace("\n", "<br>")
    st.markdown(
        f"<div class='fw-llm'>🤖 <strong>Mistral :</strong><br><br>{text_html}</div>",
        unsafe_allow_html=True
    )


def user_message(text: str):
    st.markdown(
        f"<div class='fw-user-msg'>👤 <strong>Vous :</strong> {text}</div>",
        unsafe_allow_html=True
    )


def kpi_row(items: list[tuple]):
    """items = [(value, label), ...]"""
    cols = st.columns(len(items))
    for col, (val, label) in zip(cols, items):
        col.metric(label, val)


def llm_button(api_key: str, btn_key: str, context: str,
               label: str = "🤖 Analyser avec Mistral", help_txt: str = ""):
    """Bouton LLM autonome — affiche la réponse en dessous."""
    from modules.llm import analyze
    if not api_key:
        st.info("🔑 Configurez votre clé API Mistral dans la sidebar pour activer l'analyse IA.")
        return
    if st.button(label, key=btn_key, help=help_txt):
        with st.spinner("Mistral analyse en cours..."):
            resp = analyze(api_key, context)
        llm_response(resp)
