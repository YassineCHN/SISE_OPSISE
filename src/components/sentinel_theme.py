import streamlit as st


def apply_sentinel_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
        :root {
          --bg:#07090f; --bg2:#0d1117; --bg3:#12181f; --border:#1e2a38;
          --accent:#00d4ff; --accent2:#ff3c6e; --accent3:#a259ff; --accent4:#00ff9d;
          --text:#c8d8e8; --text-dim:#4a6072; --text-hi:#e8f4ff;
        }
        html, body, [class*="css"] {
          font-family:'Space Mono', monospace !important;
          background-color:var(--bg) !important;
          color:var(--text) !important;
        }
        .main .block-container {
          background:var(--bg) !important;
          max-width:1600px;
        }
        [data-testid="stSidebar"] {
          background:var(--bg2) !important;
          border-right:1px solid var(--border) !important;
        }
        [data-testid="stSidebar"] * { color:var(--text) !important; }
        [data-testid="metric-container"] {
          background:var(--bg2) !important;
          border:1px solid var(--border) !important;
          border-radius:6px !important;
        }
        [data-testid="metric-container"] [data-testid="stMetricValue"] {
          font-family:'Syne', sans-serif !important;
          color:var(--text-hi) !important;
          font-weight:800 !important;
        }
        [data-testid="metric-container"] [data-testid="stMetricLabel"] {
          color:var(--text-dim) !important;
          letter-spacing:1px !important;
          text-transform:uppercase !important;
          font-size:0.68rem !important;
        }
        .stButton > button {
          background:transparent !important;
          color:var(--accent) !important;
          border:1px solid var(--accent) !important;
          border-radius:4px !important;
          text-transform:uppercase !important;
          letter-spacing:1px !important;
        }
        .stButton > button:hover {
          background:rgba(0,212,255,0.08) !important;
        }
        [data-testid="stDataFrame"] {
          border:1px solid var(--border) !important;
          border-radius:6px !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
          background:var(--bg2) !important;
          border-bottom:1px solid var(--border) !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
          color:var(--text-dim) !important;
          text-transform:uppercase !important;
          letter-spacing:1px !important;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
          color:var(--accent) !important;
          border-bottom:2px solid var(--accent) !important;
        }
        hr { border-color:var(--border) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
