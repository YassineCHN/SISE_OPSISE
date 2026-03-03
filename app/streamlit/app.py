"""
NetFlow Sentinel v2 — Threat Intelligence Platform
LLM intégré dans chaque onglet pour interprétation et génération de rapports.
"""

import math, time, json, os, re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from utils import (
    MISTRAL_API_KEY_ENV, MISTRAL_MODEL_ENV,
    port_label, is_public, geolocate_ips, arrow_angle,
)
from llm_analyst import generate_analysis

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NetFlow Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');
:root {
  --bg:#07090f; --bg2:#0d1117; --bg3:#12181f; --border:#1e2a38;
  --accent:#00d4ff; --accent2:#ff3c6e; --accent3:#a259ff; --accent4:#00ff9d;
  --text:#c8d8e8; --text-dim:#4a6072; --text-hi:#e8f4ff;
  --card-glow:0 0 20px rgba(0,212,255,0.08);
}
html,body,[class*="css"]{font-family:'Space Mono',monospace!important;background-color:var(--bg)!important;color:var(--text)!important}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:var(--bg2)}::-webkit-scrollbar-thumb{background:var(--accent);border-radius:4px}
.main .block-container{background:var(--bg)!important;padding:1.2rem 2rem!important;max-width:1600px}
[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border)!important}
[data-testid="stSidebar"] *{color:var(--text)!important}
[data-testid="stSidebar"] .stMarkdown h3{color:var(--accent)!important;font-family:'Syne',sans-serif!important;letter-spacing:2px;text-transform:uppercase;font-size:0.7rem}
[data-testid="stSidebar"] hr{border-color:var(--border)!important}
[data-testid="metric-container"]{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:6px!important;box-shadow:var(--card-glow)!important;padding:1rem 1.2rem!important;position:relative;overflow:hidden}
[data-testid="metric-container"]::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),var(--accent3))}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Syne',sans-serif!important;font-size:2rem!important;font-weight:800!important;color:var(--text-hi)!important}
[data-testid="metric-container"] [data-testid="stMetricLabel"]{font-size:0.62rem!important;color:var(--text-dim)!important;letter-spacing:2px!important;text-transform:uppercase!important}
.stButton>button{background:rgba(0,0,0,0)!important;color:var(--accent)!important;border:1px solid var(--accent)!important;border-radius:4px!important;font-family:'Space Mono',monospace!important;font-size:0.75rem!important;letter-spacing:1px!important;padding:10px 24px!important;text-transform:uppercase!important;transition:all .2s!important}
.stButton>button:hover{background:rgba(0,212,255,0.08)!important;box-shadow:0 0 20px rgba(0,212,255,0.25)!important;transform:translateY(-1px)!important}
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:6px!important}
.stProgress>div>div{background:linear-gradient(90deg,var(--accent),var(--accent3))!important}
[data-testid="stFileUploader"]{background:var(--bg2)!important;border:1px dashed var(--border)!important;border-radius:6px!important}
[data-testid="stTabs"] [data-baseweb="tab-list"]{background:var(--bg2)!important;border-bottom:1px solid var(--border)!important;gap:0!important}
[data-testid="stTabs"] [data-baseweb="tab"]{background:rgba(0,0,0,0)!important;color:var(--text-dim)!important;font-family:'Space Mono',monospace!important;font-size:0.72rem!important;letter-spacing:1px!important;text-transform:uppercase!important;padding:12px 24px!important;border:none!important;border-bottom:2px solid rgba(0,0,0,0)!important;transition:all .2s!important}
[data-testid="stTabs"] [aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important}
[data-testid="stAlert"]{border-radius:4px!important;border-left:3px solid var(--accent)!important}
label{color:var(--text-dim)!important;font-size:0.7rem!important;letter-spacing:1px!important;text-transform:uppercase}
hr{border-color:var(--border)!important}

.section-hd{font-family:'Syne',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);border-bottom:1px solid var(--border);padding-bottom:8px;margin:24px 0 16px 0;display:flex;align-items:center;gap:8px}
.section-hd::before{content:'';display:inline-block;width:16px;height:2px;background:var(--accent)}
.kpi-row{display:flex;gap:12px;margin:12px 0;flex-wrap:wrap}
.kpi-chip{background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:6px 14px;font-size:0.68rem;letter-spacing:1px;display:inline-flex;align-items:center;gap:8px}
.kpi-chip.deny{border-color:var(--accent2);color:var(--accent2)}
.kpi-chip.ok{border-color:var(--accent4);color:var(--accent4)}
.kpi-chip.info{border-color:var(--accent);color:var(--accent)}
.kpi-chip.warn{border-color:#ffb800;color:#ffb800}
.feed-card{background:var(--bg2);border:1px solid var(--border);border-radius:4px;padding:8px 12px;margin:4px 0;font-size:0.65rem;line-height:1.8;animation:fadeIn .3s ease}
.feed-card.deny{border-left:2px solid var(--accent2)}.feed-card.ok{border-left:2px solid var(--accent4)}
@keyframes fadeIn{from{opacity:0;transform:translateX(-4px)}to{opacity:1;transform:translateX(0)}}
.story-banner{background:linear-gradient(135deg,var(--bg2),var(--bg3));border:1px solid var(--border);border-left:3px solid var(--accent3);border-radius:6px;padding:20px 24px;margin:12px 0 24px 0;font-size:0.82rem;line-height:1.8;color:var(--text)}
.story-banner .highlight{color:var(--accent);font-weight:700}
.story-banner .danger{color:var(--accent2);font-weight:700}
.profile-badge{display:inline-block;padding:3px 10px;border-radius:3px;font-size:0.62rem;letter-spacing:1px;font-weight:700;text-transform:uppercase}
.pb-normal{background:rgba(0,255,157,0.1);color:var(--accent4);border:1px solid rgba(0,255,157,0.3)}
.pb-scan{background:rgba(0,212,255,0.1);color:var(--accent);border:1px solid rgba(0,212,255,0.3)}
.pb-ddos{background:rgba(255,60,110,0.1);color:var(--accent2);border:1px solid rgba(255,60,110,0.3)}
.pb-nocturne{background:rgba(162,89,255,0.1);color:var(--accent3);border:1px solid rgba(162,89,255,0.3)}
.pb-blocked{background:rgba(255,184,0,0.1);color:#ffb800;border:1px solid rgba(255,184,0,0.3)}
.pb-targeted{background:rgba(255,60,110,0.15);color:#ff6b6b;border:1px solid rgba(255,100,100,0.4)}
.report-box{background:var(--bg2);border:1px solid var(--border);border-left:3px solid var(--accent3);border-radius:6px;padding:24px 28px;font-size:0.83rem;line-height:1.9;color:var(--text)}
.report-box h2{color:var(--accent);font-family:'Syne',sans-serif;font-size:0.95rem;margin-top:1.2em;letter-spacing:1px}
.report-box strong{color:var(--text-hi)}
.ai-panel{background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--accent3);border-radius:6px;padding:20px 24px;margin-top:24px}
.ai-panel-hd{font-family:'Syne',sans-serif;font-size:0.7rem;font-weight:700;color:var(--accent3);letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.map-wait{background:var(--bg2);border:1px dashed var(--border);border-radius:6px;text-align:center;padding:80px 20px}
.stat-block{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:18px 20px;text-align:center}
.stat-block .val{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:var(--accent2);line-height:1}
.stat-block .lbl{font-size:0.6rem;color:var(--text-dim);letter-spacing:2px;text-transform:uppercase;margin-top:4px}
.ip-selector{background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:12px 16px;margin:8px 0;cursor:pointer;font-size:0.72rem;transition:border-color .15s}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HELPERS UI
# ═══════════════════════════════════════════════════════════════
def render_ai_panel(key: str, label: str, generate_fn, requires_key=True):
    """
    Composant réutilisable : panneau IA avec bouton + zone de rapport streamé.
    key         : clé unique session_state pour stocker le rapport
    label       : texte du bouton
    generate_fn : callable(api_key, model) → generator de chunks str
    """
    sk = f"llm_{key}"
    if sk not in st.session_state:
        st.session_state[sk] = None

    st.markdown(f"""<div class='ai-panel'>
      <div class='ai-panel-hd'> Interprétation IA — Mistral</div>""", unsafe_allow_html=True)

    if requires_key and not mistral_key:
        st.markdown("<span style='color:#4a6072;font-size:0.72rem;'>💡 Ajoutez votre clé Mistral dans la sidebar pour activer l'analyse IA.</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    btn_col, info_col = st.columns([1, 3])
    with btn_col:
        run = st.button(label, key=f"btn_{key}")
    with info_col:
        if st.session_state[sk]:
            if st.button("🔄 Régénérer", key=f"regen_{key}"):
                st.session_state[sk] = None
                st.rerun()
        else:
            st.caption(f"Modèle : **{mistral_model}**")

    if run:
        st.session_state[sk] = None
        report_box = st.empty()
        full_text  = ""
        try:
            for chunk in generate_fn(mistral_key, mistral_model):
                full_text += chunk
                report_box.markdown(f"<div class='report-box'>{full_text}▌</div>", unsafe_allow_html=True)
            report_box.markdown(f"<div class='report-box'>{full_text}</div>", unsafe_allow_html=True)
            st.session_state[sk] = full_text
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
    elif st.session_state[sk]:
        st.markdown(f"<div class='report-box'>{st.session_state[sk]}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
defaults = {
    "arc_df": None, "arrow_df": None, "scatter_df": None,
    "detail_df": None, "top_src_df": None,
    "geo_count": 0, "flow_log": [], "geo_cache": {},
    "country_src": None, "country_dst": None,
    "ip_features": None,
    "rf_report": None, "rf_cm": None, "rf_importance": None,
    "rf_roc": None, "rf_cv": None,
    "ts_data": None, "ts_pics": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
now = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
st.markdown(f"""
<div style="background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--accent);
            border-radius:6px;padding:24px 32px;margin-bottom:20px;
            display:flex;align-items:center;justify-content:space-between;">
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:0.6rem;letter-spacing:4px;color:var(--text-dim);text-transform:uppercase;margin-bottom:6px;">
      ▌ Threat Intelligence Platform
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:var(--text-hi);letter-spacing:-1px;line-height:1;">
      NetFlow <span style="color:var(--accent);">Sentinel</span>
    </div>
    <div style="color:var(--text-dim);font-size:0.68rem;margin-top:6px;letter-spacing:1px;">
      Détection · Classification · Analyse temporelle · Cartographie · IA
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:0.62rem;color:var(--text-dim);letter-spacing:2px;">SYSTEM TIME</div>
    <div style="font-family:'Syne',sans-serif;color:var(--accent);font-size:0.85rem;margin-top:2px;">{now}</div>
    <div style="margin-top:8px;">
      <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent4);margin-right:6px;box-shadow:0 0 6px var(--accent4);"></span>
      <span style="font-size:0.6rem;color:var(--accent4);letter-spacing:1px;">ONLINE</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("###  Configuration")
    st.markdown("---")
    uploaded = st.file_uploader(" Charger un CSV", type=["csv"])
    st.caption("Par défaut : **df_1000.csv**")
    st.markdown("---")
    filter_action   = st.multiselect("Action", ["DENY", "PERMIT"], default=["DENY", "PERMIT"])
    filter_protocol = st.multiselect("Protocole", ["TCP", "UDP", "ICMP"], default=["TCP", "UDP", "ICMP"])
    max_rows        = st.slider("Flux à analyser", 50, 1000, 200, step=50)
    st.markdown("---")
    show_arrows = st.checkbox("Flèches directionnelles", value=True)
    show_trips  = st.checkbox("Particules animées", value=False)
    map_style   = st.selectbox("Style de carte", ["Dark Matter (sombre)", "Voyager (colorée)", "Positron (claire)"])
    map_pitch   = st.slider("Inclinaison carte", 0, 55, 30)
    arc_width   = st.slider("Épaisseur arcs", 1, 6, 2)
    arrow_size  = st.slider("Taille flèches", 10, 35, 18)
    st.markdown("---")
    if st.button(" Réinitialiser tout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    st.markdown("---")
    # st.markdown("###  Mistral AI")
    # if MISTRAL_API_KEY_ENV:
    #     st.success(" Clé .env détectée")
    # else:
    #     st.info("💡 Sans clé : rapports de secours activés")
    # mistral_key   = st.text_input("Clé API Mistral", value=MISTRAL_API_KEY_ENV, type="password",
    #                                help="Laissez vide pour utiliser les rapports de secours intégrés")
    # _models = ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]
    # _def    = _models.index(MISTRAL_MODEL_ENV) if MISTRAL_MODEL_ENV in _models else 0
    # mistral_model = st.selectbox("Modèle", _models, index=_def)
    # st.caption("Sans clé : fallback templates activés automatiquement")

# ═══════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    local = Path(__file__).parent / "df_1000.csv"
    if not local.exists():
        st.error("❌ df_1000.csv introuvable.")
        st.stop()
    df_raw = load_csv(str(local))

if "datetime" in df_raw.columns:
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], errors="coerce")

df_filt = df_raw.copy()
if filter_action:
    df_filt = df_filt[df_filt["action"].isin(filter_action)]
if filter_protocol and "protocol_clean" in df_filt.columns:
    df_filt = df_filt[df_filt["protocol_clean"].isin(filter_protocol)]
df = df_filt.head(max_rows).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
# KPI STRIP
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# KPI STRIP — cards custom
# ═══════════════════════════════════════════════════════════════
_total   = len(df_raw)
_deny    = int((df_raw['action'] == 'DENY').sum())
_permit  = int((df_raw['action'] == 'PERMIT').sum())
_src     = df_raw['ip_src'].nunique()
_dst     = df_raw['ip_dst'].nunique()
_pct_deny   = (_deny   / _total * 100) if _total else 0
_pct_permit = (_permit / _total * 100) if _total else 0

st.markdown(f"""
<style>
.kpi-strip{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:0 0 20px 0}}
.kpi-card{{
  background:var(--bg2);
  border:1px solid var(--border);
  border-radius:8px;
  padding:18px 20px 14px 20px;
  position:relative;
  overflow:hidden;
  transition:border-color .2s, box-shadow .2s;
}}
.kpi-card:hover{{border-color:var(--kpi-color);box-shadow:0 0 24px color-mix(in srgb,var(--kpi-color) 18%,transparent)}}
.kpi-card::before{{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--kpi-color);
}}
.kpi-card::after{{
  content:'';position:absolute;top:0;right:0;width:60px;height:60px;
  background:radial-gradient(circle at top right,color-mix(in srgb,var(--kpi-color) 12%,transparent),transparent 70%);
}}
.kpi-icon{{font-size:1.1rem;margin-bottom:10px;opacity:.85}}
.kpi-label{{
  font-size:0.58rem;letter-spacing:2.5px;text-transform:uppercase;
  color:var(--text-dim);margin-bottom:8px;font-family:'Space Mono',monospace;
}}
.kpi-value{{
  font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;
  color:var(--text-hi);line-height:1;margin-bottom:10px;
}}
.kpi-sub{{
  display:flex;align-items:center;justify-content:space-between;
  font-size:0.6rem;color:var(--text-dim);letter-spacing:1px;margin-bottom:7px;
}}
.kpi-sub .kpi-pct{{
  color:var(--kpi-color);font-weight:700;font-size:0.68rem;
}}
.kpi-bar-track{{
  height:3px;background:var(--border);border-radius:2px;overflow:hidden;
}}
.kpi-bar-fill{{
  height:100%;border-radius:2px;
  background:linear-gradient(90deg,var(--kpi-color),color-mix(in srgb,var(--kpi-color) 60%,var(--accent3)));
  transition:width .6s ease;
}}
</style>

<div class="kpi-strip">

  <div class="kpi-card" style="--kpi-color:#00d4ff">
    <div class="kpi-icon"></div>
    <div class="kpi-label">Flux total</div>
    <div class="kpi-value">{_total:,}</div>
    <div class="kpi-sub">
      <span>Toutes actions</span>
      <span class="kpi-pct">100 %</span>
    </div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
  </div>

  <div class="kpi-card" style="--kpi-color:#ff3c6e">
    <div class="kpi-icon"></div>
    <div class="kpi-label">DENY</div>
    <div class="kpi-value">{_deny:,}</div>
    <div class="kpi-sub">
      <span>Connexions bloquées</span>
      <span class="kpi-pct">{_pct_deny:.1f} %</span>
    </div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:{_pct_deny:.1f}%"></div></div>
  </div>

  <div class="kpi-card" style="--kpi-color:#00ff9d">
    <div class="kpi-icon"></div>
    <div class="kpi-label">PERMIT</div>
    <div class="kpi-value">{_permit:,}</div>
    <div class="kpi-sub">
      <span>Connexions autorisées</span>
      <span class="kpi-pct">{_pct_permit:.1f} %</span>
    </div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:{_pct_permit:.1f}%"></div></div>
  </div>

  <div class="kpi-card" style="--kpi-color:#a259ff">
    <div class="kpi-icon"></div>
    <div class="kpi-label">Sources uniques</div>
    <div class="kpi-value">{_src:,}</div>
    <div class="kpi-sub">
      <span>IPs source distinctes</span>
      <span class="kpi-pct">—</span>
    </div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
  </div>

  <div class="kpi-card" style="--kpi-color:#ffb800">
    <div class="kpi-icon"></div>
    <div class="kpi-label">Destinations</div>
    <div class="kpi-value">{_dst:,}</div>
    <div class="kpi-sub">
      <span>IPs destination distinctes</span>
      <span class="kpi-pct">—</span>
    </div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
  </div>

</div>
""", unsafe_allow_html=True)

# k1,k2,k3,k4,k5 = st.columns(5)
# k1.metric("📦 Flux total",      f"{len(df_raw):,}")
# k2.metric("🚫 DENY",            f"{int((df_raw['action']=='DENY').sum()):,}")
# k3.metric("✅ PERMIT",          f"{int((df_raw['action']=='PERMIT').sum()):,}")
# k4.metric("🖥 Sources uniques", f"{df_raw['ip_src'].nunique():,}")
# k5.metric("🎯 Destinations",    f"{df_raw['ip_dst'].nunique():,}")
# st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_home, tab_map, tab_anomaly, tab_classif, tab_temporal, tab_behavior, tab_ia = st.tabs([
    "  Accueil",
    "🗺  Carte des flux",
    "  Détection d'anomalies",
    "  Classification ML",
    "  Analyse temporelle",
    "  Comportement des attaques",
    "  Threat Analyst IA",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 0 — ACCUEIL (point d'entrée)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_home:

    # ── Métriques de base ─────────────────────────────────────
    _total   = len(df_raw)
    _deny    = int((df_raw["action"] == "DENY").sum())
    _permit  = int((df_raw["action"] == "PERMIT").sum())
    _src     = df_raw["ip_src"].nunique()
    _dst     = df_raw["ip_dst"].nunique()
    _pct_deny   = (_deny   / _total * 100) if _total else 0
    _pct_permit = (_permit / _total * 100) if _total else 0

    _pcol       = "protocol_clean" if "protocol_clean" in df_raw.columns else "protocol"
    _top_port   = int(df_raw["port_dst"].mode()[0]) if "port_dst" in df_raw.columns else "N/A"
    _top_src_ip = df_raw["ip_src"].value_counts().idxmax() if "ip_src" in df_raw.columns else "N/A"
    _rules      = df_raw["rule_id"].nunique() if "rule_id" in df_raw.columns else "N/A"

    _date_start = df_raw["datetime"].min().strftime("%d %b %Y") if "datetime" in df_raw.columns else "—"
    _date_end   = df_raw["datetime"].max().strftime("%d %b %Y") if "datetime" in df_raw.columns else "—"
    _days       = (df_raw["datetime"].max() - df_raw["datetime"].min()).days if "datetime" in df_raw.columns else 0

    # ── CSS spécifique à cette page ───────────────────────────
    st.markdown("""
    <style>
    /* ─── Hero banner ─────────────────────────────────────── */
    .hero-wrap{
      background:linear-gradient(135deg,#07090f 0%,#0d1117 40%,#0f1a24 100%);
      border:1px solid #1e2a38;border-radius:10px;
      padding:40px 44px 36px 44px;margin-bottom:28px;
      position:relative;overflow:hidden;
    }
    .hero-wrap::before{
      content:'';position:absolute;top:-60px;right:-60px;
      width:340px;height:340px;border-radius:50%;
      background:radial-gradient(circle,rgba(0,212,255,0.07) 0%,transparent 65%);
      pointer-events:none;
    }
    .hero-wrap::after{
      content:'';position:absolute;bottom:-80px;left:30%;
      width:280px;height:280px;border-radius:50%;
      background:radial-gradient(circle,rgba(162,89,255,0.06) 0%,transparent 65%);
      pointer-events:none;
    }
    .hero-tag{
      font-size:0.58rem;letter-spacing:4px;text-transform:uppercase;
      color:#4a6072;margin-bottom:14px;display:flex;align-items:center;gap:10px;
    }
    .hero-tag::before{content:'';display:inline-block;width:24px;height:1px;background:#00d4ff;}
    .hero-title{
      font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;
      color:#e8f4ff;line-height:1.1;margin-bottom:12px;letter-spacing:-1px;
    }
    .hero-title span{color:#00d4ff;}
    .hero-desc{
      font-size:0.82rem;color:#6a8a9a;line-height:1.8;max-width:680px;
      margin-bottom:28px;
    }
    .hero-badges{display:flex;gap:10px;flex-wrap:wrap;}
    .hbadge{
      background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);
      border-radius:20px;padding:5px 14px;
      font-size:0.62rem;letter-spacing:1.5px;color:#00d4ff;text-transform:uppercase;
    }
    .hbadge.red{background:rgba(255,60,110,0.06);border-color:rgba(255,60,110,0.2);color:#ff3c6e;}
    .hbadge.purple{background:rgba(162,89,255,0.06);border-color:rgba(162,89,255,0.2);color:#a259ff;}
    .hbadge.green{background:rgba(0,255,157,0.06);border-color:rgba(0,255,157,0.2);color:#00ff9d;}

    /* ─── Timeline storytelling ───────────────────────────── */
    .timeline{position:relative;padding-left:32px;margin:8px 0 24px 0;}
    .timeline::before{
      content:'';position:absolute;left:8px;top:0;bottom:0;
      width:1px;background:linear-gradient(to bottom,#00d4ff,#a259ff,rgba(255,60,110,0.3));
    }
    .tl-item{
      position:relative;margin-bottom:22px;
      animation:fadeInLeft .4s ease both;
    }
    .tl-item:nth-child(1){animation-delay:.05s}
    .tl-item:nth-child(2){animation-delay:.12s}
    .tl-item:nth-child(3){animation-delay:.19s}
    .tl-item:nth-child(4){animation-delay:.26s}
    .tl-item:nth-child(5){animation-delay:.33s}
    .tl-item:nth-child(6){animation-delay:.40s}
    @keyframes fadeInLeft{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}
    .tl-dot{
      position:absolute;left:-28px;top:4px;
      width:12px;height:12px;border-radius:50%;
      border:2px solid var(--dot-color);
      background:var(--dot-bg);
      box-shadow:0 0 8px var(--dot-color);
    }
    .tl-head{
      font-family:'Syne',sans-serif;font-size:0.7rem;font-weight:700;
      letter-spacing:2px;text-transform:uppercase;
      color:var(--dot-color);margin-bottom:5px;
    }
    .tl-body{
      background:#0d1117;border:1px solid #1e2a38;
      border-left:2px solid var(--dot-color);
      border-radius:0 6px 6px 0;
      padding:12px 16px;font-size:0.78rem;line-height:1.7;color:#c8d8e8;
    }
    .tl-body b{color:#e8f4ff;}
    .tl-body .hl{color:var(--dot-color);font-weight:700;}

    /* ─── Insight cards ───────────────────────────────────── */
    .insight-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0;}
    .ins-card{
      background:#0d1117;border:1px solid #1e2a38;border-radius:8px;
      padding:16px 18px;position:relative;overflow:hidden;
    }
    .ins-card::before{
      content:'';position:absolute;top:0;left:0;right:0;height:2px;
      background:var(--ins-color);
    }
    .ins-num{
      font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;
      color:var(--ins-color);line-height:1;margin-bottom:4px;
    }
    .ins-label{font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#4a6072;}
    .ins-desc{font-size:0.72rem;color:#6a8a9a;margin-top:6px;line-height:1.5;}

    /* ─── Nav cards ─────────────────────────────────────────── */
    .nav-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:20px;}
    .nav-card{
      background:#0d1117;border:1px solid #1e2a38;border-radius:8px;
      padding:18px 20px;cursor:default;
      transition:border-color .2s,box-shadow .2s;
      position:relative;overflow:hidden;
    }
    .nav-card:hover{
      border-color:var(--nav-color);
      box-shadow:0 0 20px color-mix(in srgb,var(--nav-color) 12%,transparent);
    }
    .nav-card::after{
      content:'';position:absolute;top:0;right:0;
      width:50px;height:50px;
      background:radial-gradient(circle at top right,color-mix(in srgb,var(--nav-color) 10%,transparent),transparent 70%);
    }
    .nav-icon{font-size:1.4rem;margin-bottom:10px;}
    .nav-title{
      font-family:'Syne',sans-serif;font-size:0.75rem;font-weight:700;
      letter-spacing:1px;text-transform:uppercase;color:var(--nav-color);margin-bottom:6px;
    }
    .nav-desc{font-size:0.68rem;color:#4a6072;line-height:1.5;}

    /* ─── Datatable filters ───────────────────────────────── */
    .filter-bar{
      background:#0d1117;border:1px solid #1e2a38;border-radius:8px;
      padding:14px 18px;margin-bottom:14px;
      display:flex;align-items:center;gap:8px;flex-wrap:wrap;
    }
    </style>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # HERO BANNER
    # ══════════════════════════════════════════════════════════
    st.markdown(f"""
    <div class="hero-wrap">
      <div class="hero-tag">Threat Intelligence Platform — SISE / OPSIE 2026</div>
      <div class="hero-title">
        Un réseau sous <span>surveillance.</span><br>Des menaces sous analyse.
      </div>
      <div class="hero-desc">
        <b style="color:#e8f4ff;">{_total:,} connexions réseau</b> observées du
        <b style="color:#00d4ff;">{_date_start}</b> au <b style="color:#00d4ff;">{_date_end}</b>
        sur un firewall Iptables cloud.
        Chaque paquet raconte une histoire : tentative d'intrusion, scan furtif, flood applicatif,
        ou simple navigation légitime. NetFlow Sentinel les écoute toutes.
      </div>
      <div class="hero-badges">
        <span class="hbadge">🛡 Isolation Forest</span>
        <span class="hbadge purple">⬡ Random Forest</span>
        <span class="hbadge red">⚡ {_deny:,} menaces bloquées</span>
        <span class="hbadge green">✅ {_permit:,} accès légitimes</span>
        <span class="hbadge">{_days} jours d'observation</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # KPI STRIP
    # ══════════════════════════════════════════════════════════
    st.markdown(f"""
    <style>
    .kpi-strip{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:0 0 28px 0}}
    .kpi-card{{background:#0d1117;border:1px solid #1e2a38;border-radius:8px;
      padding:18px 20px 14px 20px;position:relative;overflow:hidden;
      transition:border-color .2s,box-shadow .2s;}}
    .kpi-card:hover{{border-color:var(--kpi-color);
      box-shadow:0 0 22px color-mix(in srgb,var(--kpi-color) 16%,transparent);}}
    .kpi-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--kpi-color);}}
    .kpi-card::after{{content:'';position:absolute;top:0;right:0;width:60px;height:60px;
      background:radial-gradient(circle at top right,color-mix(in srgb,var(--kpi-color) 10%,transparent),transparent 70%);}}
    .kpi-icon{{font-size:1.05rem;margin-bottom:10px;}}
    .kpi-label{{font-size:0.57rem;letter-spacing:2.5px;text-transform:uppercase;color:#4a6072;margin-bottom:7px;}}
    .kpi-value{{font-family:'Syne',sans-serif;font-size:1.85rem;font-weight:800;color:#e8f4ff;line-height:1;margin-bottom:10px;}}
    .kpi-sub{{display:flex;align-items:center;justify-content:space-between;
      font-size:0.6rem;color:#4a6072;letter-spacing:1px;margin-bottom:7px;}}
    .kpi-pct{{color:var(--kpi-color);font-weight:700;font-size:0.67rem;}}
    .kpi-bar-track{{height:3px;background:#1e2a38;border-radius:2px;overflow:hidden;}}
    .kpi-bar-fill{{height:100%;border-radius:2px;
      background:linear-gradient(90deg,var(--kpi-color),color-mix(in srgb,var(--kpi-color) 55%,#a259ff));}}
    </style>
    <div class="kpi-strip">
      <div class="kpi-card" style="--kpi-color:#00d4ff">
        <div class="kpi-icon"></div>
        <div class="kpi-label">Flux total</div>
        <div class="kpi-value">{_total:,}</div>
        <div class="kpi-sub"><span>Toutes actions</span><span class="kpi-pct">100 %</span></div>
        <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
      </div>
      <div class="kpi-card" style="--kpi-color:#ff3c6e">
        <div class="kpi-icon"></div>
        <div class="kpi-label">DENY — Bloqués</div>
        <div class="kpi-value">{_deny:,}</div>
        <div class="kpi-sub"><span>Connexions bloquées</span><span class="kpi-pct">{_pct_deny:.1f} %</span></div>
        <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:{_pct_deny:.1f}%"></div></div>
      </div>
      <div class="kpi-card" style="--kpi-color:#00ff9d">
        <div class="kpi-icon"></div>
        <div class="kpi-label">PERMIT — Autorisés</div>
        <div class="kpi-value">{_permit:,}</div>
        <div class="kpi-sub"><span>Connexions autorisées</span><span class="kpi-pct">{_pct_permit:.1f} %</span></div>
        <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:{_pct_permit:.1f}%"></div></div>
      </div>
      <div class="kpi-card" style="--kpi-color:#a259ff">
        <div class="kpi-icon"></div>
        <div class="kpi-label">Sources uniques</div>
        <div class="kpi-value">{_src:,}</div>
        <div class="kpi-sub"><span>IPs source distinctes</span><span class="kpi-pct">—</span></div>
        <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
      </div>
      <div class="kpi-card" style="--kpi-color:#ffb800">
        <div class="kpi-icon"></div>
        <div class="kpi-label">Destinations</div>
        <div class="kpi-value">{_dst:,}</div>
        <div class="kpi-sub"><span>IPs destination distinctes</span><span class="kpi-pct">—</span></div>
        <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # STORYTELLING — TIMELINE + INSIGHTS
    # ══════════════════════════════════════════════════════════
    left_col, right_col = st.columns([5, 4], gap="large")

    with left_col:
        st.markdown("<div class='section-hd'>Chronologie de l'analyse</div>", unsafe_allow_html=True)

        # Calculs pour le storytelling
        _proto_top = df_raw[_pcol].value_counts().idxmax() if _pcol in df_raw.columns else "TCP"
        _proto_cnt = df_raw[_pcol].value_counts().max()    if _pcol in df_raw.columns else 0
        _port_name = port_label(_top_port)
        _deny_hour = ""
        if "datetime" in df_raw.columns and "action" in df_raw.columns:
            _deny_df  = df_raw[df_raw["action"] == "DENY"].copy()
            _deny_df["hour"] = _deny_df["datetime"].dt.hour
            if not _deny_df.empty:
                _peak_h   = int(_deny_df["hour"].value_counts().idxmax())
                _peak_cnt = int(_deny_df["hour"].value_counts().max())
                _deny_hour = f"Pic d'activité hostile à <b class='hl'>{_peak_h:02d}h</b> : {_peak_cnt:,} tentatives."

        st.markdown(f"""
        <div class="timeline">

          <div class="tl-item" style="--dot-color:#00d4ff;--dot-bg:#07090f">
            <div class="tl-dot"></div>
            <div class="tl-head">Collecte des données</div>
            <div class="tl-body">
              <b>{_total:,} événements firewall</b> ont été collectés du
              <b>{_date_start}</b> au <b>{_date_end}</b>.
              Les logs proviennent d'un firewall <b>Iptables on-cloud</b>
              (FW=6), couvrant <b>{_days} jours</b> d'observation continue.
              <b>{_src:,}</b> IPs sources distinctes ont été recensées.
            </div>
          </div>

          <div class="tl-item" style="--dot-color:#a259ff;--dot-bg:#07090f">
            <div class="tl-dot"></div>
            <div class="tl-head">Profil du trafic</div>
            <div class="tl-body">
              <b class="hl">{_pct_deny:.1f} %</b> du trafic a été <b>bloqué</b> —
              soit <b>{_deny:,}</b> connexions refusées.
              Le protocole dominant est <b class="hl">{_proto_top}</b>
              avec <b>{_proto_cnt:,}</b> flux.
              Le port le plus ciblé : <b class="hl">:{_top_port} ({_port_name})</b>.
              {_deny_hour}
            </div>
          </div>

          <div class="tl-item" style="--dot-color:#ff3c6e;--dot-bg:#07090f">
            <div class="tl-dot"></div>
            <div class="tl-head">Scénario 1 — Détection d'anomalies</div>
            <div class="tl-body">
              Chaque IP est encodée en un <b>vecteur à 7 dimensions</b>
              (volume, diversité des ports, ratio DENY, activité nocturne…).
              <b>Isolation Forest</b> isole les comportements statistiquement déviants.
              <b>DBSCAN</b> regroupe les signatures similaires en clusters.
              Résultat : chaque IP reçoit un <b class="hl">profil de menace</b>
              (Port Scan, DDoS, Attaque ciblée, Normal…).
            </div>
          </div>

          <div class="tl-item" style="--dot-color:#00ff9d;--dot-bg:#07090f">
            <div class="tl-dot"></div>
            <div class="tl-head">Scénario 2 — Classification ML</div>
            <div class="tl-body">
              Un <b>Random Forest (200 arbres)</b> est entraîné sur les vecteurs
              comportementaux pour prédire le profil de chaque IP.
              Les courbes ROC, matrices de confusion et feature importance
              permettent d'évaluer la précision du modèle.
              <b>Chaque IP inconnue peut être classifiée en temps réel.</b>
            </div>
          </div>

          <div class="tl-item" style="--dot-color:#ffb800;--dot-bg:#07090f">
            <div class="tl-dot"></div>
            <div class="tl-head">Scénario 3 — Analyse temporelle</div>
            <div class="tl-body">
              Les flux sont projetés sur l'axe du temps pour révéler les
              <b class="hl">patterns horaires et hebdomadaires</b>.
              La détection de pics par <b>Z-score</b> identifie automatiquement
              les sessions d'attaques massives.
              Les heatmaps PERMIT/DENY exposent les fenêtres de vulnérabilité.
            </div>
          </div>

          <div class="tl-item" style="--dot-color:#a259ff;--dot-bg:#07090f">
            <div class="tl-dot"></div>
            <div class="tl-head">Intelligence IA — Mistral</div>
            <div class="tl-body">
              Toutes les données sont synthétisées par <b>Mistral AI</b>
              en un rapport structuré : résumé exécutif, menaces identifiées,
              géographie des attaques, et
              <b class="hl">recommandations opérationnelles</b> (règles iptables,
              CIDRs à bloquer, score de risque /100).
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)

    with right_col:
        # ── Mini charts ──────────────────────────────────────
        st.markdown("<div class='section-hd'>Distribution des actions</div>", unsafe_allow_html=True)
        _act = df_raw["action"].value_counts().reset_index()
        _act.columns = ["action","count"]
        _colors = {"DENY":"#ff3c6e","PERMIT":"#00ff9d"}
        _fig_pie = px.pie(_act, names="action", values="count",
                          color="action", color_discrete_map=_colors, hole=0.6)
        _fig_pie.update_traces(textinfo="percent+label",
                               textfont=dict(family="Space Mono", size=10),
                               marker=dict(line=dict(color="#07090f", width=2)))
        _fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#c8d8e8", height=210,
                               margin=dict(t=0,b=0,l=0,r=0),
                               showlegend=True,
                               legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
        st.plotly_chart(_fig_pie, use_container_width=True)

        if _pcol in df_raw.columns:
            st.markdown("<div class='section-hd'>Protocoles</div>", unsafe_allow_html=True)
            _prot = df_raw[_pcol].value_counts().reset_index()
            _prot.columns = ["proto","count"]
            _fig_bar = px.bar(_prot, x="proto", y="count",
                              color_discrete_sequence=["#00d4ff","#a259ff","#ffb800","#ff3c6e"])
            _fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                   font_color="#c8d8e8", height=180, showlegend=False,
                                   margin=dict(t=0,b=0,l=0,r=0),
                                   xaxis=dict(gridcolor="#1e2a38", title=""),
                                   yaxis=dict(gridcolor="#1e2a38", title=""))
            _fig_bar.update_traces(texttemplate="%{y:,}", textposition="outside",
                                   textfont=dict(size=9, color="#4a6072"))
            st.plotly_chart(_fig_bar, use_container_width=True)

        # ── Insight cards ─────────────────────────────────────
        st.markdown(f"""
        <div class="insight-grid">
          <div class="ins-card" style="--ins-color:#ff3c6e">
            <div class="ins-num">{_pct_deny:.0f}%</div>
            <div class="ins-label">Taux de blocage</div>
            <div class="ins-desc">Du trafic total refusé par le firewall</div>
          </div>
          <div class="ins-card" style="--ins-color:#00d4ff">
            <div class="ins-num">{_rules}</div>
            <div class="ins-label">Règles actives</div>
            <div class="ins-desc">Politiques de filtrage distinctes</div>
          </div>
          <div class="ins-card" style="--ins-color:#a259ff">
            <div class="ins-num">:{_top_port}</div>
            <div class="ins-label">Port le + ciblé</div>
            <div class="ins-desc">{_port_name}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # NAVIGATION RAPIDE VERS LES ONGLETS
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("<div class='section-hd'>Modules d'analyse disponibles</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="nav-grid">
      <div class="nav-card" style="--nav-color:#00d4ff">
        <div class="nav-icon">🗺</div>
        <div class="nav-title">Carte des flux</div>
        <div class="nav-desc">Géolocalisation des IPs, arcs src→dst, feed en direct. Visualisez d'où viennent les attaques sur le globe.</div>
      </div>
      <div class="nav-card" style="--nav-color:#ff3c6e">
        <div class="nav-icon"></div>
        <div class="nav-title">Détection d'anomalies</div>
        <div class="nav-desc">Isolation Forest + DBSCAN. Chaque IP reçoit un score d'anomalie et un profil comportemental.</div>
      </div>
      <div class="nav-card" style="--nav-color:#a259ff">
        <div class="nav-icon"></div>
        <div class="nav-title">Classification ML</div>
        <div class="nav-desc">Random Forest 200 arbres. Feature importance, courbes ROC et prédiction en temps réel.</div>
      </div>
      <div class="nav-card" style="--nav-color:#ffb800">
        <div class="nav-icon"></div>
        <div class="nav-title">Analyse temporelle</div>
        <div class="nav-desc">Heatmaps horaires, séries temporelles, détection de pics d'activité par Z-score.</div>
      </div>
      <div class="nav-card" style="--nav-color:#00ff9d">
        <div class="nav-icon"></div>
        <div class="nav-title">Comportement des attaques</div>
        <div class="nav-desc">Radar comportemental, corrélations, top attaquants, TOP ports < 1024 autorisés.</div>
      </div>
      <div class="nav-card" style="--nav-color:#a259ff">
        <div class="nav-icon"></div>
        <div class="nav-title">Threat Analyst IA</div>
        <div class="nav-desc">Rapport Mistral AI : résumé exécutif, menaces détectées, recommandations firewall et score de risque.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # DATATABLE — filtrable + exportable
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("<div class='section-hd'>Explorateur de données brutes</div>", unsafe_allow_html=True)

    # ── Filtres ───────────────────────────────────────────────
    f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 1])
    with f1:
        _f_action = st.multiselect(
            "Action", df_raw["action"].unique().tolist(),
            default=df_raw["action"].unique().tolist(), key="home_action"
        )
    with f2:
        if _pcol in df_raw.columns:
            _f_proto = st.multiselect(
                "Protocole", df_raw[_pcol].dropna().unique().tolist(),
                default=df_raw[_pcol].dropna().unique().tolist(), key="home_proto"
            )
        else:
            _f_proto = None
    with f3:
        _f_ip = st.text_input("IP source contient", placeholder="ex: 192.168", key="home_ip")
    with f4:
        _port_min, _port_max = 0, 65535
        if "port_dst" in df_raw.columns:
            _port_min = int(df_raw["port_dst"].min())
            _port_max = int(df_raw["port_dst"].max())
        _f_port = st.slider("Port destination", _port_min, _port_max,
                            (_port_min, _port_max), key="home_port")
    with f5:
        _n_rows = st.selectbox("Lignes", [100, 250, 500, 1000, 5000], index=1, key="home_nrows")

    # Application des filtres
    _df_table = df_raw.copy()
    if _f_action:
        _df_table = _df_table[_df_table["action"].isin(_f_action)]
    if _f_proto and _pcol in _df_table.columns:
        _df_table = _df_table[_df_table[_pcol].isin(_f_proto)]
    if _f_ip.strip():
        _df_table = _df_table[_df_table["ip_src"].astype(str).str.contains(_f_ip.strip(), na=False)]
    if "port_dst" in _df_table.columns:
        _df_table = _df_table[_df_table["port_dst"].between(_f_port[0], _f_port[1])]

    # ── Compteur résultats ────────────────────────────────────
    _n_match  = len(_df_table)
    _n_deny_f = int((_df_table["action"] == "DENY").sum())
    _n_perm_f = int((_df_table["action"] == "PERMIT").sum())
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
      <span style="font-size:0.68rem;color:#4a6072;letter-spacing:1px;">
        <b style="color:#e8f4ff;">{_n_match:,}</b> résultats
      </span>
      <span class="kpi-chip deny"> {_n_deny_f:,} DENY</span>
      <span class="kpi-chip ok"> {_n_perm_f:,} PERMIT</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tableau ────────────────────────────────────────────────
    _display_cols = [c for c in ["datetime","ip_src","ip_dst",_pcol,"port_src","port_dst","rule_id","action","interface_in"]
                     if c in _df_table.columns]
    st.dataframe(
        _df_table[_display_cols].head(_n_rows).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        height=380,
        column_config={
            "datetime":      st.column_config.DatetimeColumn("Horodatage", format="DD/MM/YYYY HH:mm:ss"),
            "ip_src":        st.column_config.TextColumn("IP Source"),
            "ip_dst":        st.column_config.TextColumn("IP Dest"),
            _pcol:           st.column_config.TextColumn("Protocole"),
            "port_src":      st.column_config.NumberColumn("Port Src"),
            "port_dst":      st.column_config.NumberColumn("Port Dst"),
            "rule_id":       st.column_config.NumberColumn("Règle"),
            "action":        st.column_config.TextColumn("Action"),
            "interface_in":  st.column_config.TextColumn("Interface"),
        }
    )

    # ── Export CSV ────────────────────────────────────────────
    _csv_data = _df_table[_display_cols].head(_n_rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇ Exporter {min(_n_match, _n_rows):,} lignes en CSV",
        data=_csv_data,
        file_name="netflow_sentinel_export.csv",
        mime="text/csv",
        key="home_export",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — CARTE DES FLUX
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_map:
    map_col, feed_col = st.columns([3, 1], gap="large")
    with map_col:
        st.markdown("<div class='section-hd'>Carte des flux — Source ──▶ Destination</div>", unsafe_allow_html=True)
        if st.button("🔍 Géolocaliser & Afficher la carte"):
            all_ips = list(df["ip_src"].dropna()) + list(df["ip_dst"].dropna())
            pbar = st.progress(0, text="Interrogation de ip-api.com…")
            geo  = geolocate_ips(all_ips)
            pbar.progress(55, text="Construction des arcs…")
            arcs, arrows, scatter, flow_log = [], [], {}, []
            for _, row in df.iterrows():
                sg = geo.get(str(row["ip_src"])); dg = geo.get(str(row["ip_dst"]))
                if not sg or not dg: continue
                is_deny = row["action"] == "DENY"
                sc = [255,60,110,210] if is_deny else [0,255,157,200]
                dc = [255,120,60,210] if is_deny else [0,212,255,200]
                port = row.get("port_dst","")
                arcs.append({"src_lat":sg["lat"],"src_lon":sg["lon"],"dst_lat":dg["lat"],"dst_lon":dg["lon"],
                             "src_ip":str(row["ip_src"]),"dst_ip":str(row["ip_dst"]),"action":row["action"],
                             "src_color":sc,"dst_color":dc,"src_city":sg["city"],"src_country":sg["country"],
                             "dst_city":dg["city"],"dst_country":dg["country"],
                             "protocol":row.get("protocol_clean","TCP"),"port_dst":port})
                ax = sg["lat"]+0.65*(dg["lat"]-sg["lat"]); ay = sg["lon"]+0.65*(dg["lon"]-sg["lon"])
                arrows.append({"lat":ax,"lon":ay,"arrow":"▶",
                               "angle":arrow_angle(sg["lat"],sg["lon"],dg["lat"],dg["lon"]),
                               "color":sc[:3]+[240],"size":arrow_size})
                for ip,g,col in [(str(row["ip_src"]),sg,sc),(str(row["ip_dst"]),dg,dc)]:
                    scatter[ip]={"ip":ip,"lat":g["lat"],"lon":g["lon"],"city":g["city"],"country":g["country"],
                                 "color":col,"radius":65000 if is_deny else 48000}
                flow_log.append({"action":row["action"],"src_ip":str(row["ip_src"]),"dst_ip":str(row["ip_dst"]),
                                 "src_city":sg["city"],"src_country":sg["country"],
                                 "dst_city":dg["city"],"dst_country":dg["country"],
                                 "protocol":row.get("protocol_clean","TCP"),"port":port})
            pbar.progress(90, text="Calcul des statistiques…")
            st.session_state.arc_df     = pd.DataFrame(arcs)
            st.session_state.arrow_df   = pd.DataFrame(arrows)
            st.session_state.scatter_df = pd.DataFrame(list(scatter.values()))
            st.session_state.geo_count  = len(geo)
            st.session_state.geo_cache  = dict(st.session_state.get("geo_cache",{}))
            st.session_state.flow_log   = flow_log
            arc_df_b = st.session_state.arc_df
            st.session_state.top_src_df = (df.groupby(["ip_src","action"]).size()
                                             .reset_index(name="Nb connexions")
                                             .sort_values("Nb connexions",ascending=False).head(15))
            st.session_state.detail_df  = (arc_df_b[["src_ip","src_city","src_country","dst_ip","dst_city","dst_country","action","protocol","port_dst"]]
                                            .drop_duplicates()
                                            .rename(columns={"src_ip":"IP Source","src_city":"Ville Src","src_country":"Pays Src",
                                                             "dst_ip":"IP Dest","dst_city":"Ville Dst","dst_country":"Pays Dst",
                                                             "action":"Action","protocol":"Protocole","port_dst":"Port Dst"}))
            st.session_state.country_src = arc_df_b["src_country"].value_counts().head(10)
            st.session_state.country_dst = arc_df_b["dst_country"].value_counts().head(10)
            pbar.progress(100, text="✅ Terminé !"); time.sleep(0.4); pbar.empty()

        arc_df_s = st.session_state.arc_df
        arrow_df_s = st.session_state.arrow_df
        scatter_df_s = st.session_state.scatter_df
        MAP_STYLES = {
            "Dark Matter (sombre)":"https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            "Voyager (colorée)":"https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            "Positron (claire)":"https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        }
        if arc_df_s is not None and not arc_df_s.empty:
            layers = [
                pdk.Layer("ArcLayer",data=arc_df_s,
                          get_source_position=["src_lon","src_lat"],get_target_position=["dst_lon","dst_lat"],
                          get_source_color="src_color",get_target_color="dst_color",
                          get_width=arc_width,get_height=0.4,pickable=True,auto_highlight=True),
                pdk.Layer("ScatterplotLayer",data=scatter_df_s,
                          get_position=["lon","lat"],get_fill_color="color",get_radius="radius",
                          pickable=True,opacity=0.85,stroked=True,get_line_color=[255,255,255,40],line_width_min_pixels=1),
            ]
            if show_arrows and arrow_df_s is not None:
                layers.append(pdk.Layer("TextLayer",data=arrow_df_s,get_position=["lon","lat"],
                              get_text="arrow",get_size="size",get_color="color",get_angle="angle",
                              font_family="Arial",font_weight="bold",billboard=True,pickable=False))
            st.pydeck_chart(pdk.Deck(layers=layers,
                initial_view_state=pdk.ViewState(latitude=25,longitude=10,zoom=1.4,pitch=map_pitch),
                map_style=MAP_STYLES[map_style],
                tooltip={"html":"""<div style='font-family:monospace;font-size:11px;background:#0d1117;
                  border:1px solid #1e2a38;border-radius:6px;padding:12px 16px;color:#c8d8e8;min-width:200px;'>
                  <div style='color:#00d4ff;font-weight:bold;margin-bottom:8px;'>{action}</div>
                  <div style='color:#ff3c6e;'>{src_ip} → {dst_ip}</div>
                  <div style='color:#4a6072;font-size:10px;margin-top:4px;'>{src_city}, {src_country} ──▶ {dst_city}, {dst_country}</div>
                  <div style='color:#4a6072;font-size:10px;margin-top:4px;'>{protocol} : {port_dst}</div></div>""",
                  "style":{"padding":"0","background":"transparent","border":"none"}}),
                use_container_width=True)
            st.markdown(f"""<div class='kpi-row'>
              <span class='kpi-chip deny'>🔴 DENY</span><span class='kpi-chip ok'>🟢 PERMIT</span>
              <span class='kpi-chip info'>⚡ {st.session_state.geo_count} IPs · {len(arc_df_s)} arcs</span>
            </div>""", unsafe_allow_html=True)
        elif arc_df_s is not None:
            st.warning("⚠️ Aucun arc — IPs privées ou API indisponible.")
        else:
            st.markdown("""<div class='map-wait'>
              <div style='font-size:3rem;margin-bottom:16px;opacity:0.4;'>🌍</div>
              <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#c8d8e8;'>Prêt pour l'analyse</div>
              <div style='color:#4a6072;font-size:0.72rem;margin-top:8px;letter-spacing:1px;'>CLIQUEZ SUR "GÉOLOCALISER & AFFICHER"</div>
            </div>""", unsafe_allow_html=True)

    with feed_col:
        st.markdown("<div class='section-hd'>Flux en direct</div>", unsafe_allow_html=True)
        if st.session_state.flow_log:
            logs = st.session_state.flow_log
            deny_n   = sum(1 for f in logs if f["action"]=="DENY")
            permit_n = len(logs)-deny_n
            st.markdown(f"<div class='kpi-row'><span class='kpi-chip deny'>🚫 {deny_n}</span><span class='kpi-chip ok'>✅ {permit_n}</span></div>", unsafe_allow_html=True)
            html = "<div style='max-height:480px;overflow-y:auto;'>"
            for f in logs:
                cls="deny" if f["action"]=="DENY" else "ok"
                ico="🔴" if cls=="deny" else "🟢"
                html+=f"""<div class='feed-card {cls}'>
                  <span style='color:{"#ff3c6e" if cls=="deny" else "#00ff9d"};'>{ico} {f['action']}</span><br>
                  <span style='color:#00d4ff;'>{f['src_ip']}</span><span style='color:#4a6072;'> ──▶ </span><span style='color:#ff3c6e;'>{f['dst_ip']}</span><br>
                  <span style='color:#4a6072;font-size:0.6rem;'>📍 {f['src_city'][:14]}, {f['src_country'][:10]}</span><br>
                  <span style='color:#4a6072;font-size:0.6rem;'>🎯 {f['dst_city'][:14]}, {f['dst_country'][:10]}</span><br>
                  <span style='color:#2a3a4a;font-size:0.58rem;'>{f['protocol']} :{f['port']}</span>
                </div>"""
            st.markdown(html+"</div>", unsafe_allow_html=True)
        else:
            st.markdown("""<div style='background:var(--bg2);border:1px dashed var(--border);border-radius:6px;
              text-align:center;padding:40px 16px;'>
              <div style='font-size:1.5rem;opacity:0.3;'>📡</div>
              <div style='color:#4a6072;font-size:0.68rem;letter-spacing:2px;margin-top:8px;'>EN ATTENTE…</div>
            </div>""", unsafe_allow_html=True)

    if st.session_state.top_src_df is not None:
        st.markdown("---")
        a1,a2 = st.columns([1,2], gap="large")
        with a1:
            st.markdown("<div class='section-hd'>Top IPs sources</div>", unsafe_allow_html=True)
            st.dataframe(st.session_state.top_src_df, use_container_width=True, hide_index=True, height=300)
        with a2:
            st.markdown("<div class='section-hd'>Connexions géolocalisées</div>", unsafe_allow_html=True)
            st.dataframe(st.session_state.detail_df, use_container_width=True, hide_index=True, height=300)
        if st.session_state.country_src is not None:
            st.markdown("---")
            c1,c2 = st.columns(2)
            for col, data, color, title in [
                (c1, st.session_state.country_src, "#ff3c6e", "Pays sources — Top 10"),
                (c2, st.session_state.country_dst, "#00d4ff", "Pays destinations — Top 10"),
            ]:
                with col:
                    st.markdown(f"<div class='section-hd'>{title}</div>", unsafe_allow_html=True)
                    fig = px.bar(x=data.values, y=data.index, orientation="h",
                                 color_discrete_sequence=[color])
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                      font_color="#c8d8e8", height=250,
                                      margin=dict(l=0,r=0,t=0,b=0),
                                      xaxis=dict(gridcolor="#1e2a38"), yaxis=dict(gridcolor="#1e2a38"))
                    st.plotly_chart(fig, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — DÉTECTION D'ANOMALIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_anomaly:
    st.markdown("""<div class='story-banner'>
      <b style='font-family:Syne,sans-serif;color:#00d4ff;letter-spacing:2px;font-size:0.7rem;'>SCÉNARIO 1 — DÉTECTION D'ANOMALIES</b><br><br>
      Chaque IP est transformée en un <span class='highlight'>vecteur comportemental</span> à 7 dimensions.
      <span class='highlight'>Isolation Forest</span> isole les comportements statistiquement anormaux dans cet hyperespace.
      <span class='highlight'>DBSCAN</span> regroupe ensuite les IPs par signature comportementale similaire.
      Le résultat : chaque IP reçoit un <span class='danger'>profil de menace</span> et un score d'anomalie.
    </div>""", unsafe_allow_html=True)

    if st.button(" Lancer la détection d'anomalies"):
        with st.spinner("Feature engineering & Isolation Forest…"):
            dw = df_raw.copy()
            if "datetime" in dw.columns:
                dw["datetime"] = pd.to_datetime(dw["datetime"], errors="coerce")
                dw["hour"] = dw["datetime"].dt.hour
            else:
                dw["hour"] = 0
            ip_feat = dw.groupby("ip_src").agg(
                nb_connexions     =("ip_src",   "count"),
                nb_ports_distincts=("port_dst", "nunique"),
                nb_ips_dst        =("ip_dst",   "nunique"),
                ratio_deny        =("action",   lambda x: (x=="DENY").mean()),
                nb_ports_sensibles=("port_dst", lambda x: x.isin([21,22,23,80,443,3306]).sum()),
                activite_nuit     =("hour",     lambda x: ((x>=0)&(x<6)).mean()),
                port_dst_std      =("port_dst", "std"),
            ).reset_index()
            ip_feat["port_dst_std"] = ip_feat["port_dst_std"].fillna(0)

            FEATURES = ["nb_connexions","nb_ports_distincts","nb_ips_dst","ratio_deny",
                        "nb_ports_sensibles","activite_nuit","port_dst_std"]
            X = ip_feat[FEATURES].values
            X_s = StandardScaler().fit_transform(X)
            iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
            ip_feat["anomaly_iso"]   = iso.fit_predict(X_s)
            ip_feat["anomaly_score"] = iso.decision_function(X_s)

            q99 = ip_feat["nb_connexions"].quantile(0.99)
            def profil(row):
                if row["nb_ports_distincts"] > 100:         return "Port Scan"
                elif row["nb_connexions"] > q99:            return "DDoS / Flood"
                elif row["nb_ports_sensibles"]>10 and row["ratio_deny"]>0.8: return "Attaque ciblée"
                elif row["activite_nuit"] > 0.7:           return "Activité nocturne suspecte"
                elif row["ratio_deny"] > 0.9:              return "Comportement bloqué"
                else:                                       return "Normal"
            ip_feat["profil"] = ip_feat.apply(profil, axis=1)

            # DBSCAN (sur échantillon)
            try:
                anom = ip_feat[ip_feat["anomaly_iso"]==-1]
                norm = ip_feat[ip_feat["anomaly_iso"]==1].sample(n=min(2000,len(ip_feat[ip_feat["anomaly_iso"]==1])),random_state=42)
                samp = pd.concat([anom,norm]).reset_index(drop=True)
                Xdb  = StandardScaler().fit_transform(samp[["nb_connexions","nb_ports_distincts","ratio_deny"]])
                db   = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1)
                samp["cluster_dbscan"] = db.fit_predict(Xdb)
                ip_feat = ip_feat.merge(samp[["ip_src","cluster_dbscan"]], on="ip_src", how="left")
                ip_feat["cluster_dbscan"] = ip_feat["cluster_dbscan"].fillna(-9).astype(int)
            except:
                ip_feat["cluster_dbscan"] = 0

            st.session_state.ip_features = ip_feat
        st.success("✅ Détection terminée !")

    ipf = st.session_state.ip_features
    if ipf is not None:
        n_total    = len(ipf)
        n_anom     = (ipf["anomaly_iso"]==-1).sum()
        n_suspects = (ipf["profil"]!="Normal").sum()
        profil_counts = ipf["profil"].value_counts()

        # KPIs
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f"<div class='stat-block'><div class='val'>{n_total:,}</div><div class='lbl'>IPs analysées</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat-block'><div class='val' style='color:#ff3c6e;'>{n_anom:,}</div><div class='lbl'>Anomalies ISO Forest</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat-block'><div class='val' style='color:#ffb800;'>{n_suspects:,}</div><div class='lbl'>IPs suspectes</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='stat-block'><div class='val' style='color:#00ff9d;'>{(ipf['profil']=='Normal').sum():,}</div><div class='lbl'>Comportements normaux</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        v1,v2 = st.columns([3,2], gap="large")
        with v1:
            st.markdown("<div class='section-hd'>Isolation Forest — Connexions vs Ports distincts</div>", unsafe_allow_html=True)
            fig = px.scatter(ipf, x="nb_connexions", y="nb_ports_distincts",
                             color=ipf["anomaly_iso"].map({1:"Normal",-1:"Anomalie"}),
                             color_discrete_map={"Normal":"#00d4ff","Anomalie":"#ff3c6e"},
                             hover_data=["ip_src","profil","ratio_deny"], log_x=True, opacity=0.65)
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117", font_color="#c8d8e8",
                              height=340, legend_title="", legend=dict(bgcolor="rgba(0,0,0,0)"),
                              xaxis=dict(gridcolor="#1e2a38",title="Nb connexions (log)"),
                              yaxis=dict(gridcolor="#1e2a38",title="Ports distincts"))
            st.plotly_chart(fig, use_container_width=True)
        with v2:
            st.markdown("<div class='section-hd'>Profils comportementaux</div>", unsafe_allow_html=True)
            pal = {"Normal":"#00ff9d","Port Scan":"#00d4ff","DDoS / Flood":"#ff3c6e",
                   "Activité nocturne suspecte":"#a259ff","Comportement bloqué":"#ffb800","Attaque ciblée":"#ff6b6b"}
            fig2 = px.bar(profil_counts.reset_index(), x="count", y="profil", orientation="h",
                          color="profil", color_discrete_map=pal)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117", font_color="#c8d8e8",
                               height=340, showlegend=False,
                               xaxis=dict(gridcolor="#1e2a38"), yaxis=dict(gridcolor="#1e2a38"))
            st.plotly_chart(fig2, use_container_width=True)

        # Distribution score anomalie
        st.markdown("<div class='section-hd'>Distribution du score d'anomalie</div>", unsafe_allow_html=True)
        fig3 = go.Figure()
        for grp,color,name in [(-1,"#ff3c6e","Anomalie"),(1,"#00d4ff","Normal")]:
            sub = ipf[ipf["anomaly_iso"]==grp]["anomaly_score"]
            fig3.add_trace(go.Histogram(x=sub, name=name, marker_color=color, opacity=0.75, nbinsx=50))
        fig3.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                           font_color="#c8d8e8", height=230, legend=dict(bgcolor="rgba(0,0,0,0)"),
                           xaxis=dict(gridcolor="#1e2a38",title="Score d'anomalie"),
                           yaxis=dict(gridcolor="#1e2a38"), margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig3, use_container_width=True)

        # Top suspects table
        st.markdown("<div class='section-hd'>Top IPs suspectes</div>", unsafe_allow_html=True)
        suspects = (ipf[ipf["anomaly_iso"]==-1].sort_values("anomaly_score").head(20)
                    [["ip_src","nb_connexions","nb_ports_distincts","ratio_deny","nb_ports_sensibles","profil","anomaly_score"]])
        pb_map = {"Normal":"pb-normal","Port Scan":"pb-scan","DDoS / Flood":"pb-ddos",
                  "Activité nocturne suspecte":"pb-nocturne","Comportement bloqué":"pb-blocked","Attaque ciblée":"pb-targeted"}
        rows_html = ""
        for _,r in suspects.iterrows():
            cls = pb_map.get(r["profil"],"pb-normal")
            rows_html += f"""<tr>
              <td style='color:#00d4ff;font-size:0.68rem;padding:6px 10px;'>{r['ip_src']}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['nb_connexions']:,}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['nb_ports_distincts']}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['ratio_deny']:.2f}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['nb_ports_sensibles']}</td>
              <td style='padding:6px 10px;'><span class='profile-badge {cls}'>{r['profil']}</span></td>
              <td style='text-align:right;padding:6px 10px;color:#ff3c6e;'>{r['anomaly_score']:.4f}</td>
            </tr>"""
        st.markdown(f"""<table style='width:100%;border-collapse:collapse;font-size:0.7rem;'>
          <thead><tr style='color:#4a6072;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1e2a38;'>
            <th style='text-align:left;padding:6px 10px;'>IP Source</th><th style='text-align:right;padding:6px 10px;'>Connexions</th>
            <th style='text-align:right;padding:6px 10px;'>Ports</th><th style='text-align:right;padding:6px 10px;'>Ratio DENY</th>
            <th style='text-align:right;padding:6px 10px;'>Ports sensibles</th><th style='padding:6px 10px;'>Profil</th>
            <th style='text-align:right;padding:6px 10px;'>Score anomalie</th>
          </tr></thead><tbody style='border-top:1px solid #1e2a38;'>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

        # ── Panneau IA global anomalies
        anom_stats = {
            "n_total": int(n_total), "n_anomalies": int(n_anom), "n_suspects": int(n_suspects),
            "profil_counts": {k: int(v) for k,v in profil_counts.items()},
            "top_suspects": [{"ip":r["ip_src"],"nb_connexions":int(r["nb_connexions"]),
                               "nb_ports_distincts":int(r["nb_ports_distincts"]),
                               "ratio_deny":float(r["ratio_deny"]),"profil":r["profil"],
                               "anomaly_score":float(r["anomaly_score"])}
                              for _,r in suspects.head(8).iterrows()]
        }
        render_ai_panel(
            key="anomaly_global",
            label="🔬 Interpréter les anomalies",
            generate_fn=lambda key, model: generate_analysis("anomaly", key, model, stats=anom_stats),
            requires_key=False,
        )

        # ── Rapport d'incident par IP
        st.markdown("---")
        st.markdown("<div class='section-hd'>Rapport d'incident par IP</div>", unsafe_allow_html=True)
        suspect_ips = suspects["ip_src"].tolist()
        if suspect_ips:
            selected_ip = st.selectbox("Choisir une IP suspecte", suspect_ips, key="ip_select_anomaly")
            if selected_ip:
                ip_row = ipf[ipf["ip_src"]==selected_ip].iloc[0]
                geo_info = st.session_state.get("geo_cache",{}).get(selected_ip,{})

                # Exemples d'événements pour cette IP
                ip_events = df_raw[df_raw["ip_src"]==selected_ip].head(6)
                examples  = ip_events.apply(lambda r: f"{r.get('datetime','')} → {r.get('ip_dst','')}:{r.get('port_dst','')} [{r.get('action','')}]", axis=1).tolist()

                inc_stats = {
                    "nb_connexions":      int(ip_row["nb_connexions"]),
                    "nb_ports_distincts": int(ip_row["nb_ports_distincts"]),
                    "nb_ips_dst":         int(ip_row["nb_ips_dst"]),
                    "ratio_deny":         float(ip_row["ratio_deny"]),
                    "nb_ports_sensibles": int(ip_row["nb_ports_sensibles"]),
                    "activite_nuit":      float(ip_row["activite_nuit"]),
                    "port_dst_std":       float(ip_row["port_dst_std"]),
                    "profil":             ip_row["profil"],
                    "anomaly_score":      float(ip_row["anomaly_score"]),
                    "geo":                geo_info,
                }
                render_ai_panel(
                    key=f"incident_{selected_ip}",
                    label=f"📋 Générer le rapport d'incident — {selected_ip}",
                    generate_fn=lambda key, model, _ip=selected_ip, _stats=inc_stats, _ex=examples: generate_analysis("incident", key, model, stats=_stats, ip=_ip, examples=_ex),
                    requires_key=False,
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — CLASSIFICATION ML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_classif:
    st.markdown("""<div class='story-banner'>
      <b style='font-family:Syne,sans-serif;color:#00d4ff;letter-spacing:2px;font-size:0.7rem;'>SCÉNARIO 2 — CLASSIFICATION RANDOM FOREST</b><br><br>
      Un <span class='highlight'>Random Forest à 200 arbres</span> apprend les signatures de chaque profil d'attaque.
      Objectif : créer un <span class='danger'>classificateur déployable en temps réel</span> capable de scorer
      instantanément toute nouvelle IP source. Les courbes ROC et la matrice de confusion quantifient la fiabilité opérationnelle.
    </div>""", unsafe_allow_html=True)

    ipf = st.session_state.ip_features
    if ipf is None:
        st.info("💡 Lancez d'abord **Détection d'anomalies** pour générer les features.")
    else:
        if st.button(" Entraîner le Random Forest"):
            with st.spinner("Entraînement en cours…"):
                FEATURES = ["nb_connexions","nb_ports_distincts","nb_ips_dst","ratio_deny",
                            "nb_ports_sensibles","activite_nuit","port_dst_std"]
                X = ipf[FEATURES]; y = ipf["profil"]
                if y.nunique() < 2:
                    st.warning("Pas assez de classes."); st.stop()
                X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
                rf = RandomForestClassifier(n_estimators=200,max_depth=15,min_samples_leaf=5,
                                            class_weight="balanced",random_state=42,n_jobs=-1)
                rf.fit(X_tr,y_tr)
                y_pred   = rf.predict(X_te)
                classes  = rf.classes_
                cv_scores= cross_val_score(rf,X,y,cv=5,scoring="accuracy",n_jobs=-1)
                cm       = confusion_matrix(y_te,y_pred,labels=classes)
                imp_df   = pd.DataFrame({"feature":FEATURES,"importance":rf.feature_importances_}).sort_values("importance",ascending=False)
                rep_dict = classification_report(y_te,y_pred,output_dict=True)
                # ROC
                try:
                    yb = label_binarize(y_te,classes=classes); ys = rf.predict_proba(X_te)
                    roc_data = [(cls,*roc_curve(yb[:,i],ys[:,i])[:2],auc(*roc_curve(yb[:,i],ys[:,i])[:2])) for i,cls in enumerate(classes)]
                    st.session_state.rf_roc = roc_data
                except: st.session_state.rf_roc = None
                st.session_state.rf_report = rep_dict
                st.session_state.rf_cm     = (cm, classes)
                st.session_state.rf_importance = imp_df
                st.session_state.rf_cv     = cv_scores
            st.success(f" Entraîné — Accuracy : {rep_dict.get('accuracy',0):.1%}")

        if st.session_state.rf_report is not None:
            rep    = st.session_state.rf_report
            cm,cls = st.session_state.rf_cm
            imp_df = st.session_state.rf_importance
            cv     = st.session_state.rf_cv
            acc    = rep.get("accuracy",0)

            c1,c2,c3 = st.columns(3)
            c1.markdown(f"<div class='stat-block'><div class='val' style='color:#00ff9d;'>{acc:.1%}</div><div class='lbl'>Accuracy globale</div></div>", unsafe_allow_html=True)
            tf = imp_df.iloc[0]
            c2.markdown(f"<div class='stat-block'><div class='val' style='color:#00d4ff;font-size:1rem;'>{tf['feature']}</div><div class='lbl'>Feature discriminante</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-block'><div class='val' style='color:#a259ff;'>{cv.mean():.3f} ±{cv.std():.3f}</div><div class='lbl'>CV 5-fold</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_cm,col_feat = st.columns(2,gap="large")
            with col_cm:
                st.markdown("<div class='section-hd'>Matrice de confusion</div>", unsafe_allow_html=True)
                fig_cm = px.imshow(cm,x=list(cls),y=list(cls),
                                   color_continuous_scale=[[0,"#0d1117"],[0.5,"#1e3a5f"],[1,"#00d4ff"]],
                                   text_auto=True,aspect="auto")
                fig_cm.update_traces(textfont_size=11)
                fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0d1117",font_color="#c8d8e8",
                                     height=360,xaxis=dict(title="Prédit",tickangle=-30),yaxis=dict(title="Réel"),
                                     margin=dict(t=0,b=40,l=0,r=0),coloraxis_showscale=False)
                st.plotly_chart(fig_cm,use_container_width=True)
            with col_feat:
                st.markdown("<div class='section-hd'>Importance des features (Gini)</div>", unsafe_allow_html=True)
                colors_f = ["#ff3c6e" if i==0 else "#00d4ff" for i in range(len(imp_df))]
                fig_fi = px.bar(imp_df,x="importance",y="feature",orientation="h",
                                color="feature",color_discrete_sequence=colors_f,text="importance")
                fig_fi.update_traces(texttemplate="%{text:.4f}",textposition="outside",textfont_size=9)
                fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0d1117",font_color="#c8d8e8",
                                     height=360,showlegend=False,
                                     xaxis=dict(gridcolor="#1e2a38"),yaxis=dict(gridcolor="#1e2a38"),
                                     margin=dict(t=0,b=0,l=0,r=60))
                st.plotly_chart(fig_fi,use_container_width=True)

            # ROC
            if st.session_state.rf_roc:
                st.markdown("<div class='section-hd'>Courbes ROC (one-vs-rest)</div>", unsafe_allow_html=True)
                cr = ["#ff3c6e","#00d4ff","#00ff9d","#ffb800","#a259ff","#ff6b6b"]
                fig_r = go.Figure()
                for i,(c_name,fpr,tpr,roc_auc) in enumerate(st.session_state.rf_roc):
                    fig_r.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{c_name} (AUC={roc_auc:.3f})",
                                               line=dict(color=cr[i%len(cr)],width=2)))
                fig_r.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Aléatoire",
                                           line=dict(color="#4a6072",dash="dash",width=1)))
                fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0d1117",font_color="#c8d8e8",
                                    height=340,legend=dict(bgcolor="rgba(0,0,0,0)",font_size=10),
                                    xaxis=dict(gridcolor="#1e2a38",title="FPR"),
                                    yaxis=dict(gridcolor="#1e2a38",title="TPR"),
                                    margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_r,use_container_width=True)

            # Métriques par classe
            st.markdown("<div class='section-hd'>Rapport de classification par classe</div>", unsafe_allow_html=True)
            rows_m=""
            for c_name in cls:
                d    = rep.get(c_name,{})
                f1   = d.get("f1-score",0)
                col_f= "#00ff9d" if f1>0.8 else "#ffb800" if f1>0.5 else "#ff3c6e"
                rows_m+=f"""<tr>
                  <td style='padding:7px 12px;'>{c_name}</td>
                  <td style='text-align:right;padding:7px 12px;color:{col_f};'>{d.get('precision',0):.3f}</td>
                  <td style='text-align:right;padding:7px 12px;color:{col_f};'>{d.get('recall',0):.3f}</td>
                  <td style='text-align:right;padding:7px 12px;color:{col_f};font-weight:700;'>{f1:.3f}</td>
                  <td style='text-align:right;padding:7px 12px;color:#4a6072;'>{int(d.get('support',0))}</td>
                </tr>"""
            st.markdown(f"""<table style='width:100%;border-collapse:collapse;font-size:0.72rem;'>
              <thead><tr style='color:#4a6072;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1e2a38;'>
                <th style='text-align:left;padding:7px 12px;'>Classe</th><th style='text-align:right;padding:7px 12px;'>Précision</th>
                <th style='text-align:right;padding:7px 12px;'>Rappel</th><th style='text-align:right;padding:7px 12px;'>F1-score</th>
                <th style='text-align:right;padding:7px 12px;'>Support</th>
              </tr></thead><tbody style='border-top:1px solid #1e2a38;'>{rows_m}</tbody>
            </table>""", unsafe_allow_html=True)

            # ── Panneau IA classification
            clf_stats = {
                "accuracy": float(acc),
                "top_feature": str(imp_df.iloc[0]["feature"]),
                "top_feat_score": float(imp_df.iloc[0]["importance"]),
                "n_classes": len(cls),
                "classes": list(cls),
                "per_class": {k: {kk: float(vv) for kk,vv in v.items()} for k,v in rep.items() if isinstance(v,dict)},
                "importance": {row["feature"]: float(row["importance"]) for _,row in imp_df.iterrows()},
                "cv_mean": float(cv.mean()),
            }
            render_ai_panel(
                key="classif_global",
                label=" Interpréter les résultats ML",
                generate_fn=lambda key, model: generate_analysis("classification", key, model, stats=clf_stats),
                requires_key=False,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — ANALYSE TEMPORELLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_temporal:
    st.markdown("""<div class='story-banner'>
      <b style='font-family:Syne,sans-serif;color:#00d4ff;letter-spacing:2px;font-size:0.7rem;'>SCÉNARIO 3 — ANALYSE TEMPORELLE</b><br><br>
      Le temps révèle ce que la snapshot statique cache.
      Heatmaps <span class='highlight'>heure × jour</span>, séries temporelles <span class='highlight'>PERMIT vs DENY</span>,
      et détection de pics par <span class='highlight'>Z-score</span> exposent les patterns d'attaque récurrents
      et les <span class='danger'>fenêtres temporelles critiques</span> à surveiller.
    </div>""", unsafe_allow_html=True)

    if "datetime" not in df_raw.columns or df_raw["datetime"].isna().all():
        st.warning("⚠️ Colonne `datetime` absente ou invalide.")
    else:
        if st.button(" Lancer l'analyse temporelle"):
            with st.spinner("Extraction des composantes temporelles…"):
                dts = df_raw.copy()
                dts["datetime"]    = pd.to_datetime(dts["datetime"],errors="coerce")
                dts                = dts.dropna(subset=["datetime"])
                dts["hour"]        = dts["datetime"].dt.hour
                dts["day_of_week"] = dts["datetime"].dt.day_name()
                dts["date"]        = dts["datetime"].dt.date
                DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

                hm_vol  = (dts.groupby(["day_of_week","hour"]).size().reset_index(name="count")
                           .pivot(index="day_of_week",columns="hour",values="count").reindex(DAY_ORDER).fillna(0))
                hm_deny = (dts[dts["action"]=="DENY"].groupby(["day_of_week","hour"]).size().reset_index(name="count")
                           .pivot(index="day_of_week",columns="hour",values="count").reindex(DAY_ORDER).fillna(0))

                ts_h = (dts.groupby([pd.Grouper(key="datetime",freq="h"),"action"]).size().reset_index(name="count"))
                ts_total = (dts.groupby(pd.Grouper(key="datetime",freq="h")).size()
                            .reset_index(name="count").rename(columns={"datetime":"hour_ts"}))
                ts_total["zscore"] = scipy_stats.zscore(ts_total["count"])
                ts_total["is_pic"] = ts_total["zscore"] > 2.5
                pics = ts_total[ts_total["is_pic"]].sort_values("zscore",ascending=False)

                # Profils horaires (si disponible)
                hourly_profil = None
                if st.session_state.ip_features is not None:
                    dp = dts.merge(st.session_state.ip_features[["ip_src","profil"]],on="ip_src",how="left")
                    dp["profil"] = dp["profil"].fillna("Normal")
                    hourly_profil = dp.groupby(["hour","profil"]).size().reset_index(name="count")

                # Stats pour LLM
                deny_by_h  = dts[dts["action"]=="DENY"].groupby("hour").size().nlargest(5).to_dict()
                permit_by_h= dts[dts["action"]=="PERMIT"].groupby("hour").size().nlargest(5).to_dict() if "PERMIT" in dts["action"].values else {}
                top_day    = dts.groupby("day_of_week").size().reindex(DAY_ORDER).idxmax()
                pics_details = [{"horodatage":str(r["hour_ts"])[:16],"count":int(r["count"]),"zscore":float(r["zscore"])}
                                for _,r in pics.head(5).iterrows()]
                profil_hours = {}
                if hourly_profil is not None:
                    for p in hourly_profil["profil"].unique():
                        sub = hourly_profil[hourly_profil["profil"]==p]
                        if not sub.empty:
                            profil_hours[p] = int(sub.loc[sub["count"].idxmax(),"hour"])

                st.session_state.ts_data = {
                    "heatmap_vol": hm_vol, "heatmap_deny": hm_deny,
                    "ts_hourly": ts_h, "ts_total": ts_total, "day_order": DAY_ORDER,
                    "hourly_profil": hourly_profil, "n_days": dts["date"].nunique(),
                    "t_start": str(dts["datetime"].min())[:16], "t_end": str(dts["datetime"].max())[:16],
                    "deny_by_h": deny_by_h, "permit_by_h": permit_by_h,
                    "top_day": top_day, "pics_details": pics_details, "profil_hours": profil_hours,
                }
                st.session_state.ts_pics = pics
            st.success(" Analyse temporelle terminée !")

        ts   = st.session_state.ts_data
        pics = st.session_state.ts_pics
        if ts is not None:
            n_pics   = len(pics) if pics is not None else 0
            ts_total = ts["ts_total"]
            c1,c2,c3,c4 = st.columns(4)
            c1.markdown(f"<div class='stat-block'><div class='val' style='color:#00d4ff;'>{ts['n_days']}</div><div class='lbl'>Jours analysés</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-block'><div class='val' style='color:#ff3c6e;'>{n_pics}</div><div class='lbl'>Pics Z-score>2.5</div></div>", unsafe_allow_html=True)
            hmax = ts_total.loc[ts_total["count"].idxmax(),"hour_ts"]
            c3.markdown(f"<div class='stat-block'><div class='val' style='color:#ffb800;font-size:1rem;'>{str(hmax)[:13]}</div><div class='lbl'>Pic max d'activité</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='stat-block'><div class='val' style='color:#a259ff;font-size:0.9rem;'>{ts['top_day'][:3]}</div><div class='lbl'>Jour le + actif</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Heatmaps
            st.markdown("<div class='section-hd'>Heatmaps — Volume total & DENY par heure × jour</div>", unsafe_allow_html=True)
            hm1,hm2 = st.columns(2,gap="large")
            with hm1:
                fig_hm = go.Figure(go.Heatmap(z=ts["heatmap_vol"].values, x=list(range(24)),
                                              y=ts["heatmap_vol"].index.tolist(),
                                              colorscale=[[0,"#07090f"],[0.5,"#1e3a5f"],[1,"#00d4ff"]]))
                fig_hm.update_layout(title="Volume total", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                     font_color="#c8d8e8", height=280, margin=dict(t=30,b=0,l=0,r=0))
                st.plotly_chart(fig_hm, use_container_width=True)
            with hm2:
                fig_hm2 = go.Figure(go.Heatmap(z=ts["heatmap_deny"].values, x=list(range(24)),
                                               y=ts["heatmap_deny"].index.tolist(),
                                               colorscale=[[0,"#07090f"],[0.5,"#3a1015"],[1,"#ff3c6e"]]))
                fig_hm2.update_layout(title="Connexions DENY", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                      font_color="#c8d8e8", height=280, margin=dict(t=30,b=0,l=0,r=0))
                st.plotly_chart(fig_hm2, use_container_width=True)

            # Séries temporelles
            st.markdown("<div class='section-hd'>Séries temporelles — PERMIT vs DENY par heure</div>", unsafe_allow_html=True)
            ts_h = ts["ts_hourly"]
            permit_ts = ts_h[ts_h["action"]=="PERMIT"].set_index("datetime")["count"]
            deny_ts   = ts_h[ts_h["action"]=="DENY"].set_index("datetime")["count"]
            fig_ts = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.08)
            fig_ts.add_trace(go.Scatter(x=permit_ts.index,y=permit_ts.values,name="PERMIT",
                                        line=dict(color="#00d4ff",width=1.5),
                                        fill="tozeroy",fillcolor="rgba(0,212,255,0.08)"),row=1,col=1)
            fig_ts.add_trace(go.Scatter(x=deny_ts.index,y=deny_ts.values,name="DENY",
                                        line=dict(color="#ff3c6e",width=1.5),
                                        fill="tozeroy",fillcolor="rgba(255,60,110,0.08)"),row=2,col=1)
            fig_ts.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0d1117",font_color="#c8d8e8",
                                 height=360,legend=dict(bgcolor="rgba(0,0,0,0)"),margin=dict(t=0,b=0,l=0,r=0))
            for i in [1,2]:
                fig_ts.update_xaxes(gridcolor="#1e2a38",row=i,col=1)
                fig_ts.update_yaxes(gridcolor="#1e2a38",row=i,col=1)
            st.plotly_chart(fig_ts, use_container_width=True)

            # Z-score
            st.markdown("<div class='section-hd'>Détection de pics — Z-score (seuil = 2.5)</div>", unsafe_allow_html=True)
            seuil = ts_total["count"].mean() + 2.5*ts_total["count"].std()
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=ts_total["hour_ts"],y=ts_total["count"],name="Volume horaire",
                                       line=dict(color="#4a6072",width=1.2),fill="tozeroy",fillcolor="rgba(74,96,114,0.1)"))
            if not ts_total[ts_total["is_pic"]].empty:
                fig_z.add_trace(go.Scatter(x=ts_total.loc[ts_total["is_pic"],"hour_ts"],
                                           y=ts_total.loc[ts_total["is_pic"],"count"],
                                           mode="markers",name="Pic anormal",
                                           marker=dict(color="#ff3c6e",size=8,line=dict(color="#fff",width=1))))
            fig_z.add_hline(y=seuil,line_dash="dash",line_color="#ff3c6e",
                             annotation_text="Seuil z=2.5",annotation_font_color="#ff3c6e")
            fig_z.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0d1117",font_color="#c8d8e8",
                                 height=280,legend=dict(bgcolor="rgba(0,0,0,0)"),
                                 xaxis=dict(gridcolor="#1e2a38"),yaxis=dict(gridcolor="#1e2a38"),
                                 margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig_z, use_container_width=True)

            # Profils horaires
            if ts["hourly_profil"] is not None:
                st.markdown("<div class='section-hd'>Distribution horaire par profil comportemental</div>", unsafe_allow_html=True)
                pal = {"Normal":"#00d4ff","Activité nocturne suspecte":"#a259ff","Comportement bloqué":"#ffb800",
                       "DDoS / Flood":"#ff3c6e","Port Scan":"#00ff9d","Attaque ciblée":"#ff6b6b"}
                fig_p = px.line(ts["hourly_profil"],x="hour",y="count",color="profil",
                                color_discrete_map=pal,markers=True)
                fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#0d1117",font_color="#c8d8e8",
                                    height=300,legend=dict(bgcolor="rgba(0,0,0,0)",font_size=10),
                                    xaxis=dict(gridcolor="#1e2a38",title="Heure",tickmode="linear"),
                                    yaxis=dict(gridcolor="#1e2a38"),margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_p, use_container_width=True)

            if n_pics > 0:
                st.markdown("<div class='section-hd'>Top pics détectés</div>", unsafe_allow_html=True)
                st.dataframe(pics.head(10)[["hour_ts","count","zscore"]].rename(columns={
                    "hour_ts":"Horodatage","count":"Nb connexions","zscore":"Z-score"
                }).style.format({"Z-score":"{:.3f}","Nb connexions":"{:,}"}),
                    use_container_width=True,hide_index=True,height=260)

            # ── Panneau IA temporel
            temp_stats = {
                "n_days": ts["n_days"], "t_start": ts["t_start"], "t_end": ts["t_end"],
                "n_pics": n_pics, "peak_hour": str(hmax)[:16],
                "low_hour": str(ts_total.loc[ts_total["count"].idxmin(),"hour_ts"])[:16],
                "top_day": ts["top_day"],
                "deny_by_hour":   {str(k):int(v) for k,v in ts["deny_by_h"].items()},
                "permit_by_hour": {str(k):int(v) for k,v in ts["permit_by_h"].items()},
                "pics_details": ts["pics_details"],
                "profil_hours": {k:str(v) for k,v in ts["profil_hours"].items()},
            }
            render_ai_panel(
                key="temporal_global",
                label=" Interpréter les patterns temporels",
                generate_fn=lambda key, model: generate_analysis("temporal", key, model, stats=temp_stats),
                requires_key=False,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — COMPORTEMENT DES ATTAQUES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_behavior:

    st.markdown("""
    <div style='background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--accent4);
                border-radius:6px;padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;gap:16px;'>
      <div style='font-size:2.5rem;'></div>
      <div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:var(--text-hi);letter-spacing:1px;'>
          Comportement des Attaques
        </div>
        <div style='color:var(--text-dim);font-size:0.68rem;margin-top:4px;letter-spacing:1px;'>
          IP Sources · IP Destinations · Top Attaquants · Corrélations & Radar
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Nécessite ip_features — guider l'utilisateur si absent
    ipf_b = st.session_state.ip_features
    if ipf_b is None:
        st.info("💡 Lancez d'abord **🔬 Détection d'anomalies** pour calculer les features comportementales.")
        st.stop()

    PROFIL_COLORS = {
        "Normal":                    "#00ff9d",
        "Port Scan":                 "#00d4ff",
        "DDoS / Flood":              "#ff3c6e",
        "Activité nocturne suspecte":"#a259ff",
        "Comportement bloqué":       "#ffb800",
        "Attaque ciblée":            "#ff6b6b",
    }
    PORT_NAMES_B = {21:"FTP",22:"SSH",23:"Telnet",25:"SMTP",53:"DNS",80:"HTTP",
                    110:"POP3",143:"IMAP",443:"HTTPS",445:"SMB",1433:"MSSQL",
                    3306:"MySQL",3389:"RDP",5432:"PostgreSQL",5900:"VNC",
                    6379:"Redis",8080:"HTTP-Alt",8443:"HTTPS-Alt"}

    def is_private_ip(ip: str) -> bool:
        try:
            p = [int(x) for x in str(ip).split(".")]
            return (p[0]==10 or (p[0]==172 and 16<=p[1]<=31)
                    or (p[0]==192 and p[1]==168) or p[0]==127)
        except: return False

    bt1, bt2, bt3, bt4 = st.tabs([
        " IP Sources", " IP Destinations",
        " Top Attaquants", " Corrélations & Radar",
    ])

    # ────────────────────────────────────────────────────────────
    # BT1 — IP SOURCES
    # ────────────────────────────────────────────────────────────
    with bt1:
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#00ff9d;letter-spacing:2px;font-size:0.7rem;'>IP SOURCES — QUI ATTAQUE ?</b><br><br>
          Chaque adresse IP source laisse une <span class='highlight'>empreinte comportementale unique</span> :
          volume de connexions, diversité de ports, ratio de blocage, activité nocturne.
          Ce sont ces 7 dimensions combinées qui distinguent un utilisateur légitime d'un
          <span class='danger'>attaquant automatisé</span>.
        </div>""", unsafe_allow_html=True)

        profils_dispo = ipf_b["profil"].unique().tolist()
        profils_sel   = st.multiselect(
            "Filtrer par profil",
            profils_dispo,
            default=[p for p in profils_dispo if p != "Normal"],
            key="beh_src_filter"
        )
        ip_sel = ipf_b[ipf_b["profil"].isin(profils_sel)] if profils_sel else ipf_b

        sc1, sc2 = st.columns(2, gap="large")
        with sc1:
            st.markdown("<div class='section-hd'>Connexions vs Ports distincts — taille = ratio DENY</div>", unsafe_allow_html=True)
            fig_s1 = px.scatter(
                ip_sel, x="nb_connexions", y="nb_ports_distincts",
                color="profil", color_discrete_map=PROFIL_COLORS,
                size="ratio_deny", size_max=22,
                hover_data=["ip_src","nb_ports_sensibles","activite_nuit","ratio_deny"],
                log_x=True, opacity=0.75
            )
            fig_s1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                  font_color="#c8d8e8", height=360, legend_title="",
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
                                  xaxis=dict(gridcolor="#1e2a38", title="Nb connexions (log)"),
                                  yaxis=dict(gridcolor="#1e2a38", title="Ports distincts"))
            st.plotly_chart(fig_s1, use_container_width=True)

        with sc2:
            st.markdown("<div class='section-hd'>Activité nocturne vs Ratio DENY — taille = volume</div>", unsafe_allow_html=True)
            fig_s2 = px.scatter(
                ip_sel, x="activite_nuit", y="ratio_deny",
                color="profil", color_discrete_map=PROFIL_COLORS,
                size="nb_connexions", size_max=25,
                hover_data=["ip_src","nb_ports_distincts"],
                opacity=0.75
            )
            fig_s2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                  font_color="#c8d8e8", height=360, legend_title="",
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
                                  xaxis=dict(gridcolor="#1e2a38", title="% activité 0h–6h", tickformat=".0%"),
                                  yaxis=dict(gridcolor="#1e2a38", title="Ratio DENY", tickformat=".0%"))
            st.plotly_chart(fig_s2, use_container_width=True)

        st.markdown("<div class='section-hd'>Distribution du volume de connexions par profil (log)</div>", unsafe_allow_html=True)
        fig_box = px.box(
            ip_sel, x="profil", y="nb_connexions",
            color="profil", color_discrete_map=PROFIL_COLORS, log_y=True
        )
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                               font_color="#c8d8e8", height=300, showlegend=False,
                               xaxis=dict(gridcolor="#1e2a38", title=""),
                               yaxis=dict(gridcolor="#1e2a38", title="Nb connexions (log)"),
                               margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_box, use_container_width=True)

        # Top 5 IP sources
        st.markdown("<div class='section-hd'> Top 5 IP sources les plus émettrices</div>", unsafe_allow_html=True)
        top5_src = df_raw["ip_src"].value_counts().head(5).reset_index()
        top5_src.columns = ["ip_src", "nb_connexions"]
        ip_map = ipf_b.set_index("ip_src")
        top5_src["ratio_deny"] = top5_src["ip_src"].map(ip_map["ratio_deny"])
        top5_src["profil"]     = top5_src["ip_src"].map(ip_map["profil"])
        top5_src["score"]      = top5_src["ip_src"].map(ip_map["anomaly_score"])

        pb_map_b = {"Normal":"pb-normal","Port Scan":"pb-scan","DDoS / Flood":"pb-ddos",
                    "Activité nocturne suspecte":"pb-nocturne","Comportement bloqué":"pb-blocked","Attaque ciblée":"pb-targeted"}
        rows_t5 = ""
        for _, r in top5_src.iterrows():
            cls = pb_map_b.get(r.get("profil","Normal"), "pb-normal")
            rows_t5 += f"""<tr>
              <td style='padding:7px 12px;color:#00d4ff;'>{r['ip_src']}</td>
              <td style='text-align:right;padding:7px 12px;'>{r['nb_connexions']:,}</td>
              <td style='text-align:right;padding:7px 12px;'>{r['ratio_deny']:.1%}</td>
              <td style='padding:7px 12px;'><span class='profile-badge {cls}'>{r.get('profil','—')}</span></td>
              <td style='text-align:right;padding:7px 12px;color:#ff3c6e;'>{r['score']:.4f}</td>
            </tr>"""
        st.markdown(f"""<table style='width:100%;border-collapse:collapse;font-size:0.72rem;'>
          <thead><tr style='color:#4a6072;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1e2a38;'>
            <th style='text-align:left;padding:7px 12px;'>IP Source</th>
            <th style='text-align:right;padding:7px 12px;'>Connexions</th>
            <th style='text-align:right;padding:7px 12px;'>Ratio DENY</th>
            <th style='padding:7px 12px;'>Profil</th>
            <th style='text-align:right;padding:7px 12px;'>Score anomalie</th>
          </tr></thead>
          <tbody style='border-top:1px solid #1e2a38;'>{rows_t5}</tbody>
        </table>""", unsafe_allow_html=True)

        # LLM
        src_stats = {
            "tab": "src",
            "top5_src": [{"ip":r["ip_src"],"connexions":int(r["nb_connexions"]),
                           "ratio_deny":float(r["ratio_deny"]) if pd.notna(r["ratio_deny"]) else 0,
                           "profil":str(r.get("profil","Normal"))} for _,r in top5_src.iterrows()],
            "profil_dist": {k:int(v) for k,v in ipf_b["profil"].value_counts().items()},
            "n_suspects": int((ipf_b["profil"]!="Normal").sum()),
        }
        render_ai_panel(
            key="behavior_src",
            label="🌍 Analyser les IP sources",
            generate_fn=lambda key, model: generate_analysis("behavior", key, model, stats=src_stats),
            requires_key=False,
        )

    # ────────────────────────────────────────────────────────────
    # BT2 — IP DESTINATIONS
    # ────────────────────────────────────────────────────────────
    with bt2:
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#00ff9d;letter-spacing:2px;font-size:0.7rem;'>IP DESTINATIONS — QUI EST CIBLÉ ?</b><br><br>
          Les IP destinations révèlent les <span class='highlight'>actifs les plus exposés</span> du réseau.
          Une destination recevant beaucoup de DENY est activement attaquée mais protégée.
          Les IPs sources <span class='danger'>hors plan d'adressage RFC 1918</span> indiquent des connexions
          depuis Internet non filtrées — risque d'intrusion directe.
        </div>""", unsafe_allow_html=True)

        dst_agg = df_raw.groupby("ip_dst").agg(
            nb_connexions =("ip_dst",   "count"),
            nb_src_uniques=("ip_src",   "nunique"),
            ratio_deny    =("action",   lambda x: (x=="DENY").mean()),
            ports_uniques =("port_dst", "nunique"),
            nb_permit     =("action",   lambda x: (x=="PERMIT").sum()),
            nb_deny       =("action",   lambda x: (x=="DENY").sum()),
        ).reset_index().sort_values("nb_connexions", ascending=False)

        st.markdown("<div class='section-hd'>Top 10 IP destinations — PERMIT vs DENY</div>", unsafe_allow_html=True)
        top10_dst = dst_agg.head(10)
        fig_dst = go.Figure()
        fig_dst.add_trace(go.Bar(x=top10_dst["ip_dst"], y=top10_dst["nb_permit"],
                                  name="PERMIT", marker_color="#00d4ff", opacity=0.85))
        fig_dst.add_trace(go.Bar(x=top10_dst["ip_dst"], y=top10_dst["nb_deny"],
                                  name="DENY", marker_color="#ff3c6e", opacity=0.9))
        fig_dst.update_layout(barmode="stack", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                               font_color="#c8d8e8", height=320,
                               legend=dict(bgcolor="rgba(0,0,0,0)"),
                               xaxis=dict(gridcolor="#1e2a38", tickangle=-30, tickfont=dict(size=9)),
                               yaxis=dict(gridcolor="#1e2a38", title="Nb connexions"),
                               margin=dict(t=0,b=40,l=0,r=0))
        st.plotly_chart(fig_dst, use_container_width=True)

        dc1, dc2 = st.columns([3,2], gap="large")
        with dc1:
            st.markdown("<div class='section-hd'>Table détaillée des destinations (Top 20)</div>", unsafe_allow_html=True)
            st.dataframe(
                dst_agg.head(20)[["ip_dst","nb_connexions","nb_src_uniques","ratio_deny","ports_uniques","nb_permit","nb_deny"]]
                .rename(columns={"ip_dst":"IP Dest","nb_connexions":"Connexions","nb_src_uniques":"Sources uniques",
                                  "ratio_deny":"Ratio DENY","ports_uniques":"Ports","nb_permit":"PERMIT","nb_deny":"DENY"})
                .style.format({"Connexions":"{:,}","Sources uniques":"{:,}",
                               "Ratio DENY":"{:.2%}","PERMIT":"{:,}","DENY":"{:,}"}),
                use_container_width=True, hide_index=True, height=340
            )

        with dc2:
            st.markdown("<div class='section-hd'> Accès depuis IPs hors RFC 1918</div>", unsafe_allow_html=True)
            ext_src = df_raw[~df_raw["ip_src"].apply(is_private_ip)]["ip_src"].value_counts().head(15).reset_index()
            ext_src.columns = ["ip_ext", "nb_connexions"]
            if not ext_src.empty:
                fig_ext = go.Figure(go.Bar(
                    x=ext_src["nb_connexions"], y=ext_src["ip_ext"],
                    orientation="h", marker_color="#ff3c6e", opacity=0.85
                ))
                fig_ext.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                       font_color="#c8d8e8", height=340,
                                       xaxis=dict(gridcolor="#1e2a38", title="Nb connexions"),
                                       yaxis=dict(gridcolor="#1e2a38", tickfont=dict(size=9)),
                                       margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_ext, use_container_width=True)
                st.markdown(f"""<div class='kpi-row'>
                  <span class='kpi-chip deny'>⚠️ {len(ext_src)} IPs externes détectées</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.success("✅ Aucune IP source externe détectée.")

        # LLM
        dst_stats = {
            "tab": "dst",
            "top10_dst": [{"ip":r["ip_dst"],"connexions":int(r["nb_connexions"]),
                           "ratio_deny":float(r["ratio_deny"]),"nb_src":int(r["nb_src_uniques"])}
                          for _,r in top10_dst.iterrows()],
            "n_ext_ips": len(ext_src) if not ext_src.empty else 0,
            "top_ext": [{"ip":r["ip_ext"],"connexions":int(r["nb_connexions"])}
                        for _,r in ext_src.head(8).iterrows()] if not ext_src.empty else [],
        }
        render_ai_panel(
            key="behavior_dst",
            label=" Analyser les IP destinations",
            generate_fn=lambda key, model: generate_analysis("behavior", key, model, stats=dst_stats),
            requires_key=False,
        )

    # ────────────────────────────────────────────────────────────
    # BT3 — TOP ATTAQUANTS
    # ────────────────────────────────────────────────────────────
    with bt3:
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#00ff9d;letter-spacing:2px;font-size:0.7rem;'>TOP ATTAQUANTS — LES PLUS DANGEREUX</b><br><br>
          Le classement par <span class='highlight'>score d'anomalie Isolation Forest</span> concentre les
          menaces les plus sérieuses en tête de liste. La taille des bulles représente les ports sensibles ciblés —
          un grand bulle rouge signifie une IP combinant <span class='danger'>volume + ciblage précis + ratio DENY élevé</span>.
        </div>""", unsafe_allow_html=True)

        n_att = st.slider("Nombre d'attaquants à afficher", 5, 30, 15, key="beh_top_slider")
        suspects_b = (ipf_b[ipf_b["profil"] != "Normal"]
                      .sort_values("anomaly_score").head(n_att))

        st.markdown("<div class='section-hd'>Carte comportementale des attaquants</div>", unsafe_allow_html=True)
        fig_att = px.scatter(
            suspects_b, x="nb_connexions", y="nb_ports_distincts",
            color="profil", color_discrete_map=PROFIL_COLORS,
            size="nb_ports_sensibles", size_max=45,
            hover_data=["ip_src","ratio_deny","activite_nuit","anomaly_score"],
            text="ip_src", log_x=True, opacity=0.85
        )
        fig_att.update_traces(textposition="top center", textfont=dict(size=8, color="#c8d8e8"))
        fig_att.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                               font_color="#c8d8e8", height=420, legend_title="",
                               legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
                               xaxis=dict(gridcolor="#1e2a38", title="Nb connexions (log)"),
                               yaxis=dict(gridcolor="#1e2a38", title="Nb ports distincts"))
        st.plotly_chart(fig_att, use_container_width=True)

        st.markdown("<div class='section-hd'>Tableau comportemental détaillé</div>", unsafe_allow_html=True)
        st.dataframe(
            suspects_b[["ip_src","nb_connexions","nb_ports_distincts","ratio_deny",
                         "nb_ports_sensibles","activite_nuit","profil","anomaly_score"]]
            .rename(columns={"ip_src":"IP","nb_connexions":"Connexions","nb_ports_distincts":"Ports",
                              "ratio_deny":"Ratio DENY","nb_ports_sensibles":"Ports sensibles",
                              "activite_nuit":"Activité nuit","profil":"Profil","anomaly_score":"Score"})
            .style.format({"Connexions":"{:,.0f}","Ports":"{:,.0f}","Ratio DENY":"{:.2%}",
                           "Activité nuit":"{:.2%}","Score":"{:.4f}"}),
            use_container_width=True, hide_index=True, height=360
        )

        # Top 10 ports <1024 avec accès autorisé
        st.markdown("<div class='section-hd'> Top 10 ports sensibles (<1024) avec accès PERMIT</div>", unsafe_allow_html=True)
        df_permit_b = df_raw[df_raw["action"]=="PERMIT"] if "action" in df_raw.columns else pd.DataFrame()
        if not df_permit_b.empty and "port_dst" in df_permit_b.columns:
            top_perm_ports = (df_permit_b[df_permit_b["port_dst"] < 1024]["port_dst"]
                              .value_counts().head(10).reset_index())
            top_perm_ports.columns = ["port","count"]
            top_perm_ports["service"] = top_perm_ports["port"].map(lambda p: PORT_NAMES_B.get(int(p),"Inconnu"))
            top_perm_ports["risk"]    = top_perm_ports["port"].map(
                lambda p: "🔴 Critique" if int(p) in [22,23,3389,445,1433,3306,5900]
                else ("🟡 Modéré" if int(p) in [80,443,21,25,53] else "⚪ Faible"))

            tc1, tc2 = st.columns([2,3], gap="large")
            with tc1:
                rows_pp = ""
                for _, r in top_perm_ports.iterrows():
                    rows_pp += f"""<tr>
                      <td style='padding:6px 10px;color:#00d4ff;'>:{int(r['port'])}</td>
                      <td style='padding:6px 10px;'>{r['service']}</td>
                      <td style='padding:6px 10px;text-align:right;'>{r['count']:,}</td>
                      <td style='padding:6px 10px;text-align:center;'>{r['risk']}</td>
                    </tr>"""
                st.markdown(f"""<table style='width:100%;border-collapse:collapse;font-size:0.7rem;'>
                  <thead><tr style='color:#4a6072;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1e2a38;'>
                    <th style='text-align:left;padding:6px 10px;'>Port</th>
                    <th style='text-align:left;padding:6px 10px;'>Service</th>
                    <th style='text-align:right;padding:6px 10px;'>Accès PERMIT</th>
                    <th style='text-align:center;padding:6px 10px;'>Risque</th>
                  </tr></thead>
                  <tbody style='border-top:1px solid #1e2a38;'>{rows_pp}</tbody>
                </table>""", unsafe_allow_html=True)
            with tc2:
                colors_pp = ["#ff3c6e" if int(r["port"]) in [22,23,3389,445,1433,3306,5900]
                             else "#ffb800" if int(r["port"]) in [80,443,21,25,53] else "#00d4ff"
                             for _,r in top_perm_ports.iterrows()]
                fig_pp = go.Figure(go.Bar(
                    x=top_perm_ports["count"],
                    y=[f":{int(r['port'])} {r['service']}" for _,r in top_perm_ports.iterrows()],
                    orientation="h", marker_color=colors_pp, opacity=0.9
                ))
                fig_pp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                      font_color="#c8d8e8", height=300, showlegend=False,
                                      xaxis=dict(gridcolor="#1e2a38", title="Nb accès PERMIT"),
                                      yaxis=dict(gridcolor="#1e2a38"),
                                      margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_pp, use_container_width=True)

        # LLM
        att_stats = {
            "tab": "top_attackers", "n": n_att,
            "top_attackers": [{"ip":r["ip_src"],"score":float(r["anomaly_score"]),
                               "connexions":int(r["nb_connexions"]),"ports":int(r["nb_ports_distincts"]),
                               "ratio_deny":float(r["ratio_deny"]),"profil":r["profil"]}
                              for _,r in suspects_b.head(8).iterrows()],
            "top_perm_ports": [{"port":int(r["port"]),"service":r["service"],"count":int(r["count"])}
                               for _,r in top_perm_ports.iterrows()] if not df_permit_b.empty else [],
        }
        render_ai_panel(
            key="behavior_top_att",
            label=" Analyser les top attaquants",
            generate_fn=lambda key, model: generate_analysis("behavior", key, model, stats=att_stats),
            requires_key=False,
        )

    # ────────────────────────────────────────────────────────────
    # BT4 — CORRÉLATIONS & RADAR
    # ────────────────────────────────────────────────────────────
    with bt4:
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#00ff9d;letter-spacing:2px;font-size:0.7rem;'>CORRÉLATIONS & RADAR — LA STRUCTURE DE LA MENACE</b><br><br>
          La matrice de corrélation révèle quelles features varient <span class='highlight'>ensemble</span> —
          clé pour comprendre pourquoi Isolation Forest est plus efficace qu'un seuillage simple.
          Le radar visualise l'<span class='danger'>empreinte comportementale normalisée</span> de chaque profil d'attaque.
        </div>""", unsafe_allow_html=True)

        FEAT_COLS = ["nb_connexions","nb_ports_distincts","nb_ips_dst","ratio_deny",
                     "nb_ports_sensibles","activite_nuit","port_dst_std"]
        feat_available = [f for f in FEAT_COLS + ["anomaly_score"] if f in ipf_b.columns]
        corr = ipf_b[feat_available].corr()

        st.markdown("<div class='section-hd'>Matrice de corrélation des features comportementales</div>", unsafe_allow_html=True)
        fig_corr = px.imshow(
            corr, color_continuous_scale="RdBu_r",
            text_auto=".2f", zmin=-1, zmax=1, aspect="auto"
        )
        fig_corr.update_traces(textfont_size=10)
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                font_color="#c8d8e8", height=420,
                                xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
                                yaxis=dict(tickfont=dict(size=9)),
                                coloraxis_colorbar=dict(tickfont=dict(color="#c8d8e8")),
                                margin=dict(t=0,b=40,l=0,r=0))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Radar chart
        st.markdown("<div class='section-hd'>Radar — Empreinte comportementale normalisée par profil</div>", unsafe_allow_html=True)
        radar_cats = [f for f in ["nb_connexions","nb_ports_distincts","ratio_deny","nb_ports_sensibles","activite_nuit"] if f in ipf_b.columns]
        pm = ipf_b.groupby("profil")[radar_cats].mean()
        pn = (pm - pm.min()) / (pm.max() - pm.min() + 1e-9)

        def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return f"rgba({r},{g},{b},{alpha})"

        fig_radar = go.Figure()
        pal_r = ["#00c896","#4a90d9","#ff4b4b","#ff8c42","#a78bfa","#34d399"]
        for (pf, row), col in zip(pn.iterrows(), pal_r):
            v = row.tolist(); v.append(v[0])
            c_final = PROFIL_COLORS.get(pf, col)
            fig_radar.add_trace(go.Scatterpolar(
                r=v, theta=radar_cats + [radar_cats[0]],
                fill="toself", name=pf,
                line_color=c_final, opacity=0.75,
                fillcolor=hex_to_rgba(c_final, 0.12),
            ))
        fig_radar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8d8e8", family="Space Mono"),
            polar=dict(
                bgcolor="#0d1117",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e2a38", tickfont=dict(size=8)),
                angularaxis=dict(gridcolor="#1e2a38", tickfont=dict(size=9))
            ),
            height=480,
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2a38", font_size=11),
            margin=dict(t=20,b=20,l=20,r=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Évolution temporelle des DENY par protocole
        if "datetime" in df_raw.columns and not df_raw["datetime"].isna().all():
            st.markdown("<div class='section-hd'>Évolution temporelle des DENY par protocole (granularité 6h)</div>", unsafe_allow_html=True)
            pcol = "protocol_clean" if "protocol_clean" in df_raw.columns else "protocol"
            if pcol in df_raw.columns:
                deny_ts_b = (df_raw[df_raw["action"]=="DENY"]
                             .groupby([pd.Grouper(key="datetime", freq="6h"), pcol])
                             .size().reset_index(name="count"))
                fig_ev = px.area(
                    deny_ts_b, x="datetime", y="count", color=pcol,
                    color_discrete_map={"TCP":"#00d4ff","UDP":"#ffb800","OTHER":"#4a6072"}
                )
                fig_ev.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                      font_color="#c8d8e8", height=300,
                                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                                      xaxis=dict(gridcolor="#1e2a38"),
                                      yaxis=dict(gridcolor="#1e2a38", title="Nb connexions DENY"),
                                      margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_ev, use_container_width=True)

        # LLM
        # Extraire top corrélations significatives
        corr_flat = []
        for i, fa in enumerate(corr.columns):
            for j, fb in enumerate(corr.columns):
                if j > i and abs(corr.iloc[i,j]) > 0.2:
                    corr_flat.append({"feat_a":fa,"feat_b":fb,"corr":float(corr.iloc[i,j])})
        corr_flat.sort(key=lambda x: abs(x["corr"]), reverse=True)

        corr_stats = {
            "tab": "correlations",
            "top_correlations": corr_flat[:12],
        }
        render_ai_panel(
            key="behavior_corr",
            label=" Interpréter les corrélations",
            generate_fn=lambda key, model: generate_analysis("behavior", key, model, stats=corr_stats),
            requires_key=False,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — THREAT ANALYST IA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ia:

    # ── Header
    st.markdown("""
    <div style='background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--accent3);
                border-radius:6px;padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;gap:16px;'>
      <div style='font-size:2.5rem;'>🛡</div>
      <div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:var(--text-hi);letter-spacing:1px;'>
          Threat Analyst IA — Intelligence des menaces TCP
        </div>
        <div style='color:var(--text-dim);font-size:0.68rem;margin-top:4px;letter-spacing:1px;'>
          Règles firewall · Ports ciblés · Flux TCP · Rapport IA exécutif
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Données de base
    df_deny  = df_raw[df_raw["action"]=="DENY"]  if "action" in df_raw.columns else pd.DataFrame()
    df_permit= df_raw[df_raw["action"]=="PERMIT"] if "action" in df_raw.columns else pd.DataFrame()
    # Travailler uniquement sur TCP (pas de données UDP significatives)
    df_tcp   = df_raw[df_raw.get("protocol_clean", df_raw.get("protocol","")) == "TCP"] if not df_raw.empty else pd.DataFrame()
    if df_tcp.empty and "protocol_clean" not in df_raw.columns:
        df_tcp = df_raw.copy()   # fallback : toutes les données si colonne absente

    n_deny    = len(df_deny)
    n_ips     = df_deny["ip_src"].nunique() if not df_deny.empty else 0
    n_rules   = df_raw["rule_id"].nunique() if "rule_id" in df_raw.columns else 0
    n_ports   = df_deny["port_dst"].nunique() if not df_deny.empty else 0
    top_port_val = int(df_deny["port_dst"].value_counts().idxmax()) if not df_deny.empty else "—"
    top_rule_val = int(df_raw["rule_id"].value_counts().idxmax()) if "rule_id" in df_raw.columns else "—"

    # ── KPIs
    d1,d2,d3,d4,d5 = st.columns(5)
    d1.markdown(f"<div class='stat-block'><div class='val'>{n_deny:,}</div><div class='lbl'>Connexions DENY</div></div>", unsafe_allow_html=True)
    d2.markdown(f"<div class='stat-block'><div class='val'>{n_ips:,}</div><div class='lbl'>IPs sources uniques</div></div>", unsafe_allow_html=True)
    d3.markdown(f"<div class='stat-block'><div class='val' style='color:#a259ff;'>{n_rules}</div><div class='lbl'>Règles actives</div></div>", unsafe_allow_html=True)
    d4.markdown(f"<div class='stat-block'><div class='val' style='color:#00d4ff;'>{n_ports}</div><div class='lbl'>Ports TCP ciblés</div></div>", unsafe_allow_html=True)
    d5.markdown(f"<div class='stat-block'><div class='val' style='color:#ffb800;font-size:1.4rem;'>#{top_rule_val}</div><div class='lbl'>Règle la + active</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df_deny.empty:
        st.warning("⚠️ Aucune connexion DENY dans les données.")
    else:
        # ══════════════════════════════════════════════════════
        # ACTE 1 — QUI FRAPPE ? Top règles firewall
        # ══════════════════════════════════════════════════════
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#a259ff;letter-spacing:2px;font-size:0.7rem;'>ACTE 1 — QUI FRAPPE ET COMMENT ?</b><br><br>
          Chaque connexion activant le firewall déclenche une <span class='highlight'>règle identifiée</span>.
          Cartographier quelles règles sont les plus sollicitées révèle les
          <span class='danger'>vecteurs d'attaque dominants</span> : une règle très active signifie
          une surface exposée répétitivement ciblée.
        </div>""", unsafe_allow_html=True)

        if "rule_id" in df_raw.columns:
            rule_counts = (df_raw.groupby("rule_id").size()
                           .reset_index(name="count").sort_values("count", ascending=False))
            rule_deny   = (df_deny.groupby("rule_id").size()
                           .reset_index(name="deny_count").sort_values("deny_count", ascending=False))
            rule_merged = rule_counts.merge(rule_deny, on="rule_id", how="left").fillna(0)
            rule_merged["deny_count"] = rule_merged["deny_count"].astype(int)
            rule_merged["deny_ratio"] = rule_merged["deny_count"] / rule_merged["count"]
            rule_merged["rule_label"] = rule_merged["rule_id"].apply(lambda x: f"Règle {int(x)}" if pd.notna(x) else "N/A")

            top_rules_tcp = pd.DataFrame()
            if not df_tcp.empty and "rule_id" in df_tcp.columns:
                top_rules_tcp = (df_tcp.groupby("rule_id").size()
                                 .reset_index(name="count").sort_values("count", ascending=False).head(10))
                top_rules_tcp["rule_label"] = top_rules_tcp["rule_id"].apply(lambda x: f"Règle {int(x)}")

            col_r1, col_r2 = st.columns([3, 2], gap="large")
            with col_r1:
                st.markdown("<div class='section-hd'>Top règles firewall — Volume total vs DENY</div>", unsafe_allow_html=True)
                top10 = rule_merged.head(10)
                fig_rules = go.Figure()
                fig_rules.add_trace(go.Bar(
                    x=top10["rule_label"], y=top10["count"],
                    name="Total", marker_color="#1e3a5f", opacity=0.85))
                fig_rules.add_trace(go.Bar(
                    x=top10["rule_label"], y=top10["deny_count"],
                    name="DENY", marker_color="#ff3c6e", opacity=0.9))
                fig_rules.update_layout(
                    barmode="overlay", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                    font_color="#c8d8e8", height=320, legend=dict(bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(gridcolor="#1e2a38", tickangle=-30),
                    yaxis=dict(gridcolor="#1e2a38", title="Nb connexions"),
                    margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_rules, use_container_width=True)

            with col_r2:
                st.markdown("<div class='section-hd'>Ratio DENY par règle</div>", unsafe_allow_html=True)
                top10_ratio = rule_merged.head(10).sort_values("deny_ratio", ascending=True)
                colors_ratio = ["#ff3c6e" if r > 0.7 else "#ffb800" if r > 0.3 else "#00d4ff"
                                for r in top10_ratio["deny_ratio"]]
                fig_ratio = go.Figure(go.Bar(
                    x=top10_ratio["deny_ratio"],
                    y=top10_ratio["rule_label"],
                    orientation="h",
                    marker_color=colors_ratio,
                    text=[f"{r:.0%}" for r in top10_ratio["deny_ratio"]],
                    textposition="outside",
                    textfont=dict(size=9, color="#c8d8e8")))
                fig_ratio.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                    font_color="#c8d8e8", height=320,
                    xaxis=dict(gridcolor="#1e2a38", tickformat=".0%", range=[0,1.15]),
                    yaxis=dict(gridcolor="#1e2a38"),
                    margin=dict(t=0,b=0,l=0,r=40))
                st.plotly_chart(fig_ratio, use_container_width=True)

        # ══════════════════════════════════════════════════════
        # ACTE 2 — QUELLES PORTES SONT FORCÉES ? Ports TCP
        # ══════════════════════════════════════════════════════
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#a259ff;letter-spacing:2px;font-size:0.7rem;'>ACTE 2 — QUELLES PORTES SONT FORCÉES ?</b><br><br>
          Les ports de destination révèlent l'<span class='highlight'>intention de l'attaquant</span>.
          SSH sur :22 signifie brute-force. RDP sur :3389 signifie prise de contrôle à distance.
          Un scan large de ports signifie de la <span class='danger'>reconnaissance automatisée</span>.
          TCP concentre l'essentiel des tentatives d'intrusion ciblées.
        </div>""", unsafe_allow_html=True)

        PORT_NAMES = {21:"FTP",22:"SSH",23:"Telnet",25:"SMTP",53:"DNS",80:"HTTP",
                      110:"POP3",143:"IMAP",443:"HTTPS",445:"SMB",1433:"MSSQL",
                      3306:"MySQL",3389:"RDP",5432:"PostgreSQL",5900:"VNC",
                      6379:"Redis",8080:"HTTP-Alt",8443:"HTTPS-Alt",27017:"MongoDB"}

        col_p1, col_p2 = st.columns(2, gap="large")
        with col_p1:
            st.markdown("<div class='section-hd'>Top 15 ports TCP — DENY vs PERMIT</div>", unsafe_allow_html=True)
            top_ports_deny   = df_deny["port_dst"].value_counts().head(15)
            top_ports_permit = df_permit["port_dst"].value_counts() if not df_permit.empty else pd.Series()
            port_labels = [f":{int(p)} {PORT_NAMES.get(int(p),'')}" for p in top_ports_deny.index]
            fig_ports = go.Figure()
            fig_ports.add_trace(go.Bar(
                y=port_labels, x=top_ports_deny.values,
                name="DENY", orientation="h",
                marker_color="#ff3c6e", opacity=0.9))
            fig_ports.add_trace(go.Bar(
                y=port_labels,
                x=[top_ports_permit.get(p, 0) for p in top_ports_deny.index],
                name="PERMIT", orientation="h",
                marker_color="#00d4ff", opacity=0.6))
            fig_ports.update_layout(
                barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                font_color="#c8d8e8", height=400, legend=dict(bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#1e2a38", title="Nb connexions"),
                yaxis=dict(gridcolor="#1e2a38"),
                margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig_ports, use_container_width=True)

        with col_p2:
            st.markdown("<div class='section-hd'>Répartition DENY par protocole</div>", unsafe_allow_html=True)
            if "protocol_clean" in df_deny.columns:
                proto_counts = df_deny["protocol_clean"].value_counts()
            elif "protocol" in df_deny.columns:
                proto_counts = df_deny["protocol"].value_counts()
            else:
                proto_counts = pd.Series({"TCP": len(df_deny)})

            col_proto = ["#ff3c6e","#00d4ff","#a259ff","#ffb800","#00ff9d"]
            fig_proto = go.Figure(go.Pie(
                labels=proto_counts.index.tolist(),
                values=proto_counts.values.tolist(),
                hole=0.55,
                marker=dict(colors=col_proto[:len(proto_counts)],
                            line=dict(color="#07090f", width=2)),
                textfont=dict(size=11, color="#c8d8e8"),
                hovertemplate="%{label}<br>%{value:,} connexions<br>%{percent}<extra></extra>"))
            fig_proto.add_annotation(
                text=f"<b>{len(df_deny):,}</b><br><span style='font-size:10px'>DENY</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="#c8d8e8"))
            fig_proto.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#c8d8e8",
                height=250, showlegend=True,
                legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.1),
                margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig_proto, use_container_width=True)

            # Table des ports les plus ciblés avec nom de service
            st.markdown("<div class='section-hd'>Dictionnaire des services ciblés</div>", unsafe_allow_html=True)
            top15 = df_deny["port_dst"].value_counts().head(15)
            svc_rows = ""
            for port, cnt in top15.items():
                try:
                    p = int(port)
                    svc = PORT_NAMES.get(p, "—")
                    risk = "🔴" if p in [22,23,3389,445,1433,3306] else "🟡" if p in [80,443,8080,8443] else "⚪"
                    pct  = cnt / n_deny * 100
                    svc_rows += f"""<tr>
                      <td style='padding:5px 10px;color:#00d4ff;'>:{p}</td>
                      <td style='padding:5px 10px;'>{svc}</td>
                      <td style='padding:5px 10px;text-align:right;'>{cnt:,}</td>
                      <td style='padding:5px 10px;text-align:right;color:#4a6072;'>{pct:.1f}%</td>
                      <td style='padding:5px 10px;text-align:center;'>{risk}</td>
                    </tr>"""
                except: pass
            st.markdown(f"""<table style='width:100%;border-collapse:collapse;font-size:0.68rem;'>
              <thead><tr style='color:#4a6072;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1e2a38;'>
                <th style='text-align:left;padding:5px 10px;'>Port</th>
                <th style='text-align:left;padding:5px 10px;'>Service</th>
                <th style='text-align:right;padding:5px 10px;'>Tentatives</th>
                <th style='text-align:right;padding:5px 10px;'>% DENY</th>
                <th style='text-align:center;padding:5px 10px;'>Risque</th>
              </tr></thead>
              <tbody style='border-top:1px solid #1e2a38;'>{svc_rows}</tbody>
            </table>""", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # ACTE 3 — COMMENT LE FIREWALL FILTRE ? Règle × Port
        # ══════════════════════════════════════════════════════
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#a259ff;letter-spacing:2px;font-size:0.7rem;'>ACTE 3 — COMMENT LE FIREWALL FILTRE ?</b><br><br>
          La heatmap Règle × Port révèle la <span class='highlight'>logique de filtrage</span> :
          quelles règles bloquent quels ports, et où se concentre l'exposition.
          Une case rouge intense indique un port <span class='danger'>massivement ciblé</span> sous une règle spécifique —
          potentiellement une faille dans la politique de sécurité.
        </div>""", unsafe_allow_html=True)

        if "rule_id" in df_raw.columns:
            top_ports_idx = df_deny["port_dst"].value_counts().head(12).index
            top_rules_idx = df_deny["rule_id"].value_counts().head(10).index

            hm_deny_data = (
                df_deny[df_deny["port_dst"].isin(top_ports_idx) & df_deny["rule_id"].isin(top_rules_idx)]
                .groupby(["rule_id","port_dst"]).size()
                .unstack(fill_value=0)
            )
            if not hm_deny_data.empty:
                hm_deny_data.index = [f"Règle {int(r)}" for r in hm_deny_data.index]
                hm_deny_data.columns = [f":{int(p)} {PORT_NAMES.get(int(p),'')}" for p in hm_deny_data.columns]

                fig_hm_rules = go.Figure(go.Heatmap(
                    z=hm_deny_data.values,
                    x=hm_deny_data.columns.tolist(),
                    y=hm_deny_data.index.tolist(),
                    colorscale=[[0,"#07090f"],[0.4,"#3a1015"],[0.7,"#8b1a2e"],[1,"#ff3c6e"]],
                    text=hm_deny_data.values,
                    texttemplate="%{text:,}",
                    textfont=dict(size=9),
                    hoverongaps=False))
                fig_hm_rules.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                    font_color="#c8d8e8", height=340,
                    xaxis=dict(title="Port destination", tickangle=-30, tickfont=dict(size=9)),
                    yaxis=dict(title="Règle firewall"),
                    margin=dict(t=0,b=40,l=0,r=0),
                    coloraxis_showscale=False)
                st.plotly_chart(fig_hm_rules, use_container_width=True)

        # ══════════════════════════════════════════════════════
        # ACTE 4 — FLUX SANKEY : Règle → Port → Décision
        # ══════════════════════════════════════════════════════
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#a259ff;letter-spacing:2px;font-size:0.7rem;'>ACTE 4 — LE PARCOURS D'UNE CONNEXION</b><br><br>
          Le diagramme de Sankey matérialise le <span class='highlight'>chemin décisionnel</span> de chaque connexion TCP :
          de quelle règle elle provient, vers quel port elle se dirige, et quelle décision finale tombe.
          La largeur des flux est proportionnelle au <span class='danger'>volume de tentatives</span>.
        </div>""", unsafe_allow_html=True)

        if "rule_id" in df_raw.columns:
            # Limiter aux top règles et ports pour lisibilité
            top_r = df_raw["rule_id"].value_counts().head(6).index
            top_p = df_raw["port_dst"].value_counts().head(8).index
            sankey_df = (df_raw[df_raw["rule_id"].isin(top_r) & df_raw["port_dst"].isin(top_p)]
                         .groupby(["rule_id","port_dst","action"]).size()
                         .reset_index(name="count"))

            if not sankey_df.empty:
                # Construire les nœuds : [Règles] + [Ports] + [Actions]
                rules_nodes  = [f"Règle {int(r)}" for r in sankey_df["rule_id"].unique()]
                ports_nodes  = [f":{int(p)} {PORT_NAMES.get(int(p),'')}" for p in sankey_df["port_dst"].unique()]
                action_nodes = ["✅ PERMIT", "🚫 DENY"]
                all_nodes    = rules_nodes + ports_nodes + action_nodes

                node_idx = {n: i for i, n in enumerate(all_nodes)}

                src1, tgt1, val1 = [], [], []  # règle → port
                src2, tgt2, val2 = [], [], []  # port  → action

                # Agréger règle→port
                rp = sankey_df.groupby(["rule_id","port_dst"])["count"].sum().reset_index()
                for _, row in rp.iterrows():
                    rn = f"Règle {int(row['rule_id'])}"
                    pn = f":{int(row['port_dst'])} {PORT_NAMES.get(int(row['port_dst']),'')}"
                    if rn in node_idx and pn in node_idx:
                        src1.append(node_idx[rn]); tgt1.append(node_idx[pn]); val1.append(int(row["count"]))

                # Agréger port→action
                pa = sankey_df.groupby(["port_dst","action"])["count"].sum().reset_index()
                for _, row in pa.iterrows():
                    pn = f":{int(row['port_dst'])} {PORT_NAMES.get(int(row['port_dst']),'')}"
                    an = "✅ PERMIT" if row["action"]=="PERMIT" else "🚫 DENY"
                    if pn in node_idx and an in node_idx:
                        src2.append(node_idx[pn]); tgt2.append(node_idx[an]); val2.append(int(row["count"]))

                sources = src1 + src2
                targets = tgt1 + tgt2
                values  = val1 + val2

                node_colors = (
                    ["#a259ff"] * len(rules_nodes) +
                    ["#00d4ff"] * len(ports_nodes) +
                    ["#00ff9d", "#ff3c6e"]
                )
                link_colors = [
                    "rgba(162,89,255,0.25)" if i < len(src1) else
                    ("rgba(0,255,157,0.2)" if all_nodes[targets[i]] == "✅ PERMIT" else "rgba(255,60,110,0.2)")
                    for i in range(len(sources))
                ]

                fig_sankey = go.Figure(go.Sankey(
                    arrangement="snap",
                    node=dict(
                        pad=20, thickness=16,
                        line=dict(color="#1e2a38", width=0.5),
                        label=all_nodes,
                        color=node_colors,
                        hovertemplate="%{label}<br>%{value:,} connexions<extra></extra>"),
                    link=dict(
                        source=sources, target=targets, value=values,
                        color=link_colors,
                        hovertemplate="De %{source.label}<br>→ %{target.label}<br>%{value:,} connexions<extra></extra>")))
                fig_sankey.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#c8d8e8", font_family="Space Mono",
                    height=420, margin=dict(t=10,b=10,l=0,r=0))
                st.plotly_chart(fig_sankey, use_container_width=True)

                st.markdown(f"""<div class='kpi-row'>
                  <span class='kpi-chip' style='border-color:#a259ff;color:#a259ff;'>⬡ {len(rules_nodes)} règles</span>
                  <span class='kpi-chip info'>⬡ {len(ports_nodes)} ports</span>
                  <span class='kpi-chip ok'>✅ PERMIT</span>
                  <span class='kpi-chip deny'>🚫 DENY</span>
                </div>""", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════
        # ACTE 5 — RAPPORT IA EXÉCUTIF
        # ══════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#a259ff;letter-spacing:2px;font-size:0.7rem;'>ACTE 5 — RAPPORT IA EXÉCUTIF</b><br><br>
          Toutes les données précédentes sont synthétisées par <span class='highlight'>Mistral AI</span>
          en un rapport structuré : résumé exécutif, menaces identifiées, géographie des attaques,
          et <span class='danger'>recommandations opérationnelles concrètes</span>.
        </div>""", unsafe_allow_html=True)

        geo_cache = st.session_state.get("geo_cache",{})
        render_ai_panel(
            key="global_threat",
            label="🛡 Générer le rapport IA de menaces",
            generate_fn=lambda key, model: generate_analysis(
                "global_threat", key, model,
                stats={}, df_deny=df_deny, geo_cache=geo_cache
            ),
            requires_key=True,
        )

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#2a3a4a;font-size:0.62rem;letter-spacing:2px;padding:8px 0;'>
  NETFLOW SENTINEL v2 &nbsp;·&nbsp; Isolation Forest &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp;
  ip-api.com &nbsp;·&nbsp; Mistral AI &nbsp;·&nbsp; pydeck &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
