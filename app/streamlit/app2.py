"""
NetFlow Sentinel — Application de cybersécurité avec storytelling des analyses
Intègre : Détection d'anomalies (Scénario 1), Classification RF (Scénario 2),
          Analyse temporelle (Scénario 3) + Carte des flux + IA Mistral
"""

import math
import time
import json
import os
import re
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from utils import (
    MISTRAL_API_KEY_ENV,
    MISTRAL_MODEL_ENV,
    port_label,
    is_public,
    geolocate_ips,
    arrow_angle,
    build_threat_prompt,
    stream_mistral,
)

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
# CSS — Cyberpunk / Terminal dark aesthetic
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');

:root {
  --bg:        #07090f;
  --bg2:       #0d1117;
  --bg3:       #12181f;
  --border:    #1e2a38;
  --accent:    #00d4ff;
  --accent2:   #ff3c6e;
  --accent3:   #a259ff;
  --accent4:   #00ff9d;
  --text:      #c8d8e8;
  --text-dim:  #4a6072;
  --text-hi:   #e8f4ff;
  --card-glow: 0 0 20px rgba(0,212,255,0.08);
  --deny-glow: 0 0 20px rgba(255,60,110,0.1);
}

html, body, [class*="css"] {
  font-family: 'Space Mono', monospace !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg2); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }

/* Main block */
.main .block-container {
  background: var(--bg) !important;
  padding: 1.2rem 2rem !important;
  max-width: 1600px;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stMarkdown h3 {
  color: var(--accent) !important;
  font-family: 'Syne', sans-serif !important;
  letter-spacing: 2px;
  text-transform: uppercase;
  font-size: 0.7rem;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

/* Metric cards */
[data-testid="metric-container"] {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  box-shadow: var(--card-glow) !important;
  padding: 1rem 1.2rem !important;
  position: relative;
  overflow: hidden;
}
[data-testid="metric-container"]::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent3));
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 2rem !important;
  font-weight: 800 !important;
  color: var(--text-hi) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-size: 0.62rem !important;
  color: var(--text-dim) !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
}

/* Buttons */
.stButton > button {
  background: transparent !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  border-radius: 4px !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.75rem !important;
  letter-spacing: 1px !important;
  padding: 10px 24px !important;
  text-transform: uppercase !important;
  transition: all 0.2s !important;
  position: relative;
  overflow: hidden;
}
.stButton > button::before {
  content: '';
  position: absolute;
  inset: 0;
  background: var(--accent);
  opacity: 0;
  transition: opacity 0.2s;
}
.stButton > button:hover {
  background: rgba(0,212,255,0.08) !important;
  box-shadow: 0 0 20px rgba(0,212,255,0.25) !important;
  transform: translateY(-1px) !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
}

/* Progress */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent3)) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
  background: var(--bg2) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 6px !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: var(--bg2) !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-dim) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  padding: 12px 24px !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}

/* Alerts */
[data-testid="stAlert"] { border-radius: 4px !important; border-left: 3px solid var(--accent) !important; }

/* Select, slider labels */
label { color: var(--text-dim) !important; font-size: 0.7rem !important; letter-spacing: 1px !important; text-transform: uppercase; }

/* ── Custom components ── */
.section-hd {
  font-family: 'Syne', sans-serif;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  color: var(--text-dim);
  border-bottom: 1px solid var(--border);
  padding-bottom: 8px;
  margin: 24px 0 16px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-hd::before {
  content: '';
  display: inline-block;
  width: 16px;
  height: 2px;
  background: var(--accent);
}

.terminal-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 20px 24px;
  margin: 8px 0;
  position: relative;
  box-shadow: var(--card-glow);
}
.terminal-card::before {
  content: '●  ●  ●';
  display: block;
  color: var(--text-dim);
  font-size: 0.55rem;
  letter-spacing: 4px;
  margin-bottom: 14px;
}

.kpi-row { display: flex; gap: 12px; margin: 12px 0; flex-wrap: wrap; }

.kpi-chip {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 6px 14px;
  font-size: 0.68rem;
  letter-spacing: 1px;
  display: inline-flex;
  align-items: center;
  gap: 8px;
}
.kpi-chip.deny  { border-color: var(--accent2); color: var(--accent2); }
.kpi-chip.ok    { border-color: var(--accent4); color: var(--accent4); }
.kpi-chip.info  { border-color: var(--accent);  color: var(--accent);  }
.kpi-chip.warn  { border-color: #ffb800;         color: #ffb800;         }

.feed-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 8px 12px;
  margin: 4px 0;
  font-size: 0.65rem;
  line-height: 1.8;
  animation: fadeIn 0.3s ease;
}
.feed-card.deny  { border-left: 2px solid var(--accent2); }
.feed-card.ok    { border-left: 2px solid var(--accent4); }
@keyframes fadeIn { from { opacity:0; transform:translateX(-4px); } to { opacity:1; transform:translateX(0); } }

.story-banner {
  background: linear-gradient(135deg, var(--bg2), var(--bg3));
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent3);
  border-radius: 6px;
  padding: 20px 24px;
  margin: 12px 0 24px 0;
  font-size: 0.82rem;
  line-height: 1.8;
  color: var(--text);
}
.story-banner .highlight { color: var(--accent); font-weight: 700; }
.story-banner .danger    { color: var(--accent2); font-weight: 700; }
.story-banner .success   { color: var(--accent4); font-weight: 700; }

.profile-badge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 3px;
  font-size: 0.62rem;
  letter-spacing: 1px;
  font-weight: 700;
  text-transform: uppercase;
}
.pb-normal   { background: rgba(0,255,157,0.1);  color: var(--accent4); border: 1px solid rgba(0,255,157,0.3); }
.pb-scan     { background: rgba(0,212,255,0.1);  color: var(--accent);  border: 1px solid rgba(0,212,255,0.3); }
.pb-ddos     { background: rgba(255,60,110,0.1); color: var(--accent2); border: 1px solid rgba(255,60,110,0.3); }
.pb-nocturne { background: rgba(162,89,255,0.1); color: var(--accent3); border: 1px solid rgba(162,89,255,0.3); }
.pb-blocked  { background: rgba(255,184,0,0.1);  color: #ffb800;        border: 1px solid rgba(255,184,0,0.3); }
.pb-targeted { background: rgba(255,60,110,0.15);color: #ff6b6b;        border: 1px solid rgba(255,100,100,0.4); }

.report-box {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent3);
  border-radius: 6px;
  padding: 28px 32px;
  font-size: 0.83rem;
  line-height: 1.9;
  color: var(--text);
}
.report-box h2 { color: var(--accent); font-family: 'Syne', sans-serif; font-size: 0.95rem; margin-top: 1.2em; letter-spacing: 1px; }
.report-box strong { color: var(--text-hi); }

.map-wait {
  background: var(--bg2);
  border: 1px dashed var(--border);
  border-radius: 6px;
  text-align: center;
  padding: 80px 20px;
}

.stat-block {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 18px 20px;
  text-align: center;
}
.stat-block .val { font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; color: var(--accent2); line-height: 1; }
.stat-block .lbl { font-size: 0.6rem; color: var(--text-dim); letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
defaults = {
    "arc_df": None, "arrow_df": None, "scatter_df": None,
    "detail_df": None, "top_src_df": None,
    "geo_count": 0, "flow_log": [], "geo_cache": {},
    "country_src": None, "country_dst": None,
    "ia_report": None,
    "ip_features": None,
    "rf_model": None, "rf_classes": None,
    "rf_report": None, "rf_cm": None, "rf_importance": None,
    "rf_roc": None,
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
<div style="
  background: var(--bg2);
  border: 1px solid var(--border);
  border-top: 2px solid var(--accent);
  border-radius: 6px;
  padding: 24px 32px;
  margin-bottom: 20px;
  display: flex; align-items: center; justify-content: space-between;
">
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:0.6rem;letter-spacing:4px;color:var(--text-dim);text-transform:uppercase;margin-bottom:6px;">
      ▌ Threat Intelligence Platform
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:var(--text-hi);letter-spacing:-1px;line-height:1;">
      NetFlow <span style="color:var(--accent);">Sentinel</span>
    </div>
    <div style="color:var(--text-dim);font-size:0.68rem;margin-top:6px;letter-spacing:1px;">
      Détection · Classification · Analyse temporelle · Cartographie
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
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Charger un CSV", type=["csv"])
    st.caption("Par défaut : **df_1000.csv**")
    st.markdown("---")
    filter_action   = st.multiselect("Action", ["DENY", "PERMIT"], default=["DENY", "PERMIT"])
    filter_protocol = st.multiselect("Protocole", ["TCP", "UDP", "ICMP"], default=["TCP", "UDP", "ICMP"])
    max_rows        = st.slider("Flux à analyser", 50, 1000, 200, step=50)
    st.markdown("---")
    show_arrows = st.checkbox("Flèches directionnelles", value=True)
    show_trips  = st.checkbox("Particules animées", value=False)
    map_style   = st.selectbox("Style de carte", ["Dark Matter (sombre)", "Voyager (colorée)", "Positron (claire)"], index=0)
    map_pitch   = st.slider("Inclinaison carte", 0, 55, 30)
    arc_width   = st.slider("Épaisseur arcs", 1, 6, 2)
    arrow_size  = st.slider("Taille flèches", 10, 35, 18)
    st.markdown("---")
    if st.button("🗑 Réinitialiser"):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()
    st.markdown("---")
    st.markdown("### 🤖 Analyse IA (Mistral)")
    if MISTRAL_API_KEY_ENV:
        st.success("✅ Clé .env chargée")
    mistral_key = st.text_input("Clé API Mistral", value=MISTRAL_API_KEY_ENV, type="password")
    _models = ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]
    _def = _models.index(MISTRAL_MODEL_ENV) if MISTRAL_MODEL_ENV in _models else 0
    mistral_model = st.selectbox("Modèle", _models, index=_def)

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
        st.error("❌ df_1000.csv introuvable — chargez un fichier via la sidebar.")
        st.stop()
    df_raw = load_csv(str(local))

# Normalisation datetime
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
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📦 Flux total",        f"{len(df_raw):,}")
k2.metric("🚫 DENY",              f"{int((df_raw['action']=='DENY').sum()):,}")
k3.metric("✅ PERMIT",            f"{int((df_raw['action']=='PERMIT').sum()):,}")
k4.metric("🖥 Sources uniques",   f"{df_raw['ip_src'].nunique():,}")
k5.metric("🎯 Destinations",      f"{df_raw['ip_dst'].nunique():,}")

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_map, tab_anomaly, tab_classif, tab_temporal, tab_ia = st.tabs([
    "🗺  Carte des flux",
    "🔬  Détection d'anomalies",
    "🤖  Classification ML",
    "📈  Analyse temporelle",
    "🛡  Threat Analyst IA",
])

# ───────────────────────────────────────────────────────────────
# TAB 1 — CARTE
# ───────────────────────────────────────────────────────────────
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
                sg = geo.get(str(row["ip_src"]))
                dg = geo.get(str(row["ip_dst"]))
                if not sg or not dg:
                    continue
                is_deny = row["action"] == "DENY"
                sc = [255, 60, 110, 210] if is_deny else [0, 255, 157, 200]
                dc = [255, 120, 60, 210] if is_deny else [0, 212, 255, 200]
                port = row.get("port_dst", "")
                arcs.append({"src_lat": sg["lat"], "src_lon": sg["lon"], "dst_lat": dg["lat"], "dst_lon": dg["lon"],
                             "src_ip": str(row["ip_src"]), "dst_ip": str(row["ip_dst"]), "action": row["action"],
                             "src_color": sc, "dst_color": dc, "src_city": sg["city"], "src_country": sg["country"],
                             "dst_city": dg["city"], "dst_country": dg["country"],
                             "protocol": row.get("protocol_clean", "TCP"), "port_dst": port})
                ax = sg["lat"] + 0.65 * (dg["lat"] - sg["lat"])
                ay = sg["lon"] + 0.65 * (dg["lon"] - sg["lon"])
                arrows.append({"lat": ax, "lon": ay, "arrow": "▶",
                               "angle": arrow_angle(sg["lat"], sg["lon"], dg["lat"], dg["lon"]),
                               "color": sc[:3] + [240], "size": arrow_size})
                for ip, g, col in [(str(row["ip_src"]), sg, sc), (str(row["ip_dst"]), dg, dc)]:
                    scatter[ip] = {"ip": ip, "lat": g["lat"], "lon": g["lon"],
                                   "city": g["city"], "country": g["country"],
                                   "color": col, "radius": 65000 if is_deny else 48000}
                flow_log.append({"action": row["action"], "src_ip": str(row["ip_src"]), "dst_ip": str(row["ip_dst"]),
                                 "src_city": sg["city"], "src_country": sg["country"],
                                 "dst_city": dg["city"], "dst_country": dg["country"],
                                 "protocol": row.get("protocol_clean", "TCP"), "port": port})
            pbar.progress(90, text="Calcul des statistiques…")
            st.session_state.arc_df     = pd.DataFrame(arcs)
            st.session_state.arrow_df   = pd.DataFrame(arrows)
            st.session_state.scatter_df = pd.DataFrame(list(scatter.values()))
            st.session_state.geo_count  = len(geo)
            st.session_state.flow_log   = flow_log
            st.session_state.top_src_df = (df.groupby(["ip_src","action"]).size()
                                             .reset_index(name="Nb connexions")
                                             .sort_values("Nb connexions", ascending=False).head(15))
            arc_df_b = st.session_state.arc_df
            st.session_state.detail_df  = (arc_df_b[["src_ip","src_city","src_country","dst_ip","dst_city","dst_country","action","protocol","port_dst"]]
                                            .drop_duplicates().rename(columns={"src_ip":"IP Source","src_city":"Ville Src","src_country":"Pays Src","dst_ip":"IP Dest","dst_city":"Ville Dst","dst_country":"Pays Dst","action":"Action","protocol":"Protocole","port_dst":"Port Dst"}))
            st.session_state.country_src = arc_df_b["src_country"].value_counts().head(10)
            st.session_state.country_dst = arc_df_b["dst_country"].value_counts().head(10)
            pbar.progress(100, text="✅ Analyse terminée !")
            time.sleep(0.4)
            pbar.empty()

        # Rendu carte
        arc_df_s    = st.session_state.arc_df
        arrow_df_s  = st.session_state.arrow_df
        scatter_df_s = st.session_state.scatter_df
        map_styles  = {
            "Dark Matter (sombre)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            "Voyager (colorée)":   "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            "Positron (claire)":   "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        }
        if arc_df_s is not None and not arc_df_s.empty:
            layers = [
                pdk.Layer("ArcLayer", data=arc_df_s,
                          get_source_position=["src_lon","src_lat"], get_target_position=["dst_lon","dst_lat"],
                          get_source_color="src_color", get_target_color="dst_color",
                          get_width=arc_width, get_height=0.4, pickable=True, auto_highlight=True),
                pdk.Layer("ScatterplotLayer", data=scatter_df_s,
                          get_position=["lon","lat"], get_fill_color="color", get_radius="radius",
                          pickable=True, opacity=0.85, stroked=True, get_line_color=[255,255,255,40], line_width_min_pixels=1),
            ]
            if show_arrows and arrow_df_s is not None:
                layers.append(pdk.Layer("TextLayer", data=arrow_df_s, get_position=["lon","lat"],
                              get_text="arrow", get_size="size", get_color="color", get_angle="angle",
                              font_family="Arial", font_weight="bold", billboard=True, pickable=False))
            st.pydeck_chart(pdk.Deck(layers=layers,
                initial_view_state=pdk.ViewState(latitude=25, longitude=10, zoom=1.4, pitch=map_pitch),
                map_style=map_styles[map_style],
                tooltip={"html": """
                  <div style='font-family:monospace;font-size:11px;background:#0d1117;border:1px solid #1e2a38;
                              border-radius:6px;padding:12px 16px;color:#c8d8e8;min-width:200px;'>
                    <div style='color:#00d4ff;font-weight:bold;margin-bottom:8px;'>{action}</div>
                    <div style='color:#ff3c6e;'>{src_ip} → {dst_ip}</div>
                    <div style='color:#4a6072;font-size:10px;margin-top:4px;'>
                      {src_city}, {src_country} ──▶ {dst_city}, {dst_country}
                    </div>
                    <div style='color:#4a6072;font-size:10px;margin-top:4px;'>{protocol} : {port_dst}</div>
                  </div>""",
                  "style": {"padding":"0","background":"transparent","border":"none"}}),
                use_container_width=True)
            st.markdown(f"""<div class='kpi-row'>
              <span class='kpi-chip deny'>🔴 DENY</span>
              <span class='kpi-chip ok'>🟢 PERMIT</span>
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
            deny_n   = sum(1 for f in logs if f["action"] == "DENY")
            permit_n = len(logs) - deny_n
            st.markdown(f"""<div class='kpi-row'>
              <span class='kpi-chip deny'>🚫 {deny_n}</span>
              <span class='kpi-chip ok'>✅ {permit_n}</span>
            </div>""", unsafe_allow_html=True)
            html = "<div style='max-height:480px;overflow-y:auto;'>"
            for f in logs:
                cls = "deny" if f["action"] == "DENY" else "ok"
                ico = "🔴" if f["action"] == "DENY" else "🟢"
                html += f"""<div class='feed-card {cls}'>
                  <span style='color:{"#ff3c6e" if cls=="deny" else "#00ff9d"};'>{ico} {f['action']}</span><br>
                  <span style='color:#00d4ff;'>{f['src_ip']}</span>
                  <span style='color:#4a6072;'> ──▶ </span>
                  <span style='color:#ff3c6e;'>{f['dst_ip']}</span><br>
                  <span style='color:#4a6072;font-size:0.6rem;'>📍 {f['src_city'][:14]}, {f['src_country'][:10]}</span><br>
                  <span style='color:#4a6072;font-size:0.6rem;'>🎯 {f['dst_city'][:14]}, {f['dst_country'][:10]}</span><br>
                  <span style='color:#2a3a4a;font-size:0.58rem;'>{f['protocol']} :{f['port']}</span>
                </div>"""
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown("""<div style='background:var(--bg2);border:1px dashed var(--border);border-radius:6px;
              text-align:center;padding:40px 16px;'>
              <div style='font-size:1.5rem;margin-bottom:8px;opacity:0.3;'>📡</div>
              <div style='color:#4a6072;font-size:0.68rem;letter-spacing:2px;'>EN ATTENTE…</div>
            </div>""", unsafe_allow_html=True)

    if st.session_state.top_src_df is not None:
        st.markdown("---")
        a1, a2 = st.columns([1, 2], gap="large")
        with a1:
            st.markdown("<div class='section-hd'>Top IPs sources</div>", unsafe_allow_html=True)
            st.dataframe(st.session_state.top_src_df, use_container_width=True, hide_index=True, height=300)
        with a2:
            st.markdown("<div class='section-hd'>Connexions géolocalisées</div>", unsafe_allow_html=True)
            st.dataframe(st.session_state.detail_df, use_container_width=True, hide_index=True, height=300)
        if st.session_state.country_src is not None:
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='section-hd'>Pays sources — Top 10</div>", unsafe_allow_html=True)
                fig = px.bar(x=st.session_state.country_src.values,
                             y=st.session_state.country_src.index,
                             orientation="h",
                             color_discrete_sequence=["#ff3c6e"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                  font_color="#c8d8e8", height=250, margin=dict(l=0,r=0,t=0,b=0),
                                  xaxis=dict(gridcolor="#1e2a38"), yaxis=dict(gridcolor="#1e2a38"))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("<div class='section-hd'>Pays destinations — Top 10</div>", unsafe_allow_html=True)
                fig = px.bar(x=st.session_state.country_dst.values,
                             y=st.session_state.country_dst.index,
                             orientation="h",
                             color_discrete_sequence=["#00d4ff"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                  font_color="#c8d8e8", height=250, margin=dict(l=0,r=0,t=0,b=0),
                                  xaxis=dict(gridcolor="#1e2a38"), yaxis=dict(gridcolor="#1e2a38"))
                st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────────────────────────────────────────
# TAB 2 — DÉTECTION D'ANOMALIES (Scénario 1)
# ───────────────────────────────────────────────────────────────
with tab_anomaly:
    st.markdown("""<div class='story-banner'>
      <b style='font-family:Syne,sans-serif;color:#00d4ff;letter-spacing:2px;font-size:0.7rem;'>SCÉNARIO 1 — DÉTECTION D'ANOMALIES</b><br><br>
      Chaque adresse IP source est transformée en un <span class='highlight'>vecteur de comportement</span> :
      nombre de connexions, ports distincts contactés, ratio de blocages, activité nocturne, variance des ports…
      Deux algorithmes opèrent en tandem : <span class='highlight'>Isolation Forest</span> isole les comportements statistiquement
      anormaux dans cet hyperespace, tandis que <span class='highlight'>DBSCAN</span> regroupe les IPs par signature comportementale.
      Le résultat : chaque attaquant reçoit un <span class='danger'>profil de menace</span> précis.
    </div>""", unsafe_allow_html=True)

    if st.button("🔬 Lancer la détection d'anomalies"):
        with st.spinner("Extraction des features comportementales…"):
            df_work = df_raw.copy()
            if "datetime" in df_work.columns:
                df_work["datetime"] = pd.to_datetime(df_work["datetime"], errors="coerce")
                df_work["hour"] = df_work["datetime"].dt.hour
            else:
                df_work["hour"] = 0

            ip_features = df_work.groupby("ip_src").agg(
                nb_connexions      =("ip_src",   "count"),
                nb_ports_distincts =("port_dst", "nunique"),
                nb_ips_dst         =("ip_dst",   "nunique"),
                ratio_deny         =("action",   lambda x: (x=="DENY").mean()),
                nb_ports_sensibles =("port_dst", lambda x: x.isin([21,22,23,80,443,3306]).sum()),
                activite_nuit      =("hour",     lambda x: ((x>=0)&(x<6)).mean()),
                port_dst_std       =("port_dst", "std"),
            ).reset_index()
            ip_features["port_dst_std"] = ip_features["port_dst_std"].fillna(0)

            # Isolation Forest
            features_cols = ["nb_connexions","nb_ports_distincts","nb_ips_dst","ratio_deny",
                             "nb_ports_sensibles","activite_nuit","port_dst_std"]
            X = ip_features[features_cols].values
            scaler  = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
            ip_features["anomaly_iso"]   = iso.fit_predict(X_scaled)
            ip_features["anomaly_score"] = iso.decision_function(X_scaled)

            # Profil comportemental
            q99 = ip_features["nb_connexions"].quantile(0.99)
            def deduire_profil(row):
                if row["nb_ports_distincts"] > 100: return "Port Scan"
                elif row["nb_connexions"] > q99:    return "DDoS / Flood"
                elif row["nb_ports_sensibles"] > 10 and row["ratio_deny"] > 0.8: return "Attaque ciblée"
                elif row["activite_nuit"] > 0.7:    return "Activité nocturne suspecte"
                elif row["ratio_deny"] > 0.9:       return "Comportement bloqué"
                else:                               return "Normal"
            ip_features["profil"] = ip_features.apply(deduire_profil, axis=1)

            # DBSCAN
            try:
                anomalies = ip_features[ip_features["anomaly_iso"] == -1]
                normaux   = ip_features[ip_features["anomaly_iso"] == 1].sample(
                    n=min(2000, len(ip_features[ip_features["anomaly_iso"]==1])), random_state=42)
                ip_sample = pd.concat([anomalies, normaux]).reset_index(drop=True)
                X_db = StandardScaler().fit_transform(ip_sample[["nb_connexions","nb_ports_distincts","ratio_deny"]])
                db = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1)
                ip_sample["cluster_dbscan"] = db.fit_predict(X_db)
                ip_features = ip_features.merge(ip_sample[["ip_src","cluster_dbscan"]], on="ip_src", how="left")
                ip_features["cluster_dbscan"] = ip_features["cluster_dbscan"].fillna(-9).astype(int)
            except:
                ip_features["cluster_dbscan"] = 0

            st.session_state.ip_features = ip_features
        st.success("✅ Détection terminée !")

    ip_features = st.session_state.ip_features
    if ip_features is not None:
        n_anomalies = (ip_features["anomaly_iso"] == -1).sum()
        n_total     = len(ip_features)
        profil_counts = ip_features["profil"].value_counts()
        n_suspects   = (ip_features["profil"] != "Normal").sum()

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"<div class='stat-block'><div class='val'>{n_total:,}</div><div class='lbl'>IPs analysées</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='stat-block'><div class='val' style='color:#ff3c6e;'>{n_anomalies:,}</div><div class='lbl'>Anomalies ISO Forest</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='stat-block'><div class='val' style='color:#ffb800;'>{n_suspects:,}</div><div class='lbl'>IPs suspectes</div></div>", unsafe_allow_html=True)
        c4.markdown(f"<div class='stat-block'><div class='val' style='color:#00d4ff;'>{(ip_features['profil']=='Normal').sum():,}</div><div class='lbl'>Comportements normaux</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        v1, v2 = st.columns([3, 2], gap="large")

        with v1:
            st.markdown("<div class='section-hd'>Isolation Forest — Connexions vs Ports distincts</div>", unsafe_allow_html=True)
            fig = px.scatter(ip_features, x="nb_connexions", y="nb_ports_distincts",
                             color=ip_features["anomaly_iso"].map({1:"Normal",-1:"Anomalie"}),
                             color_discrete_map={"Normal":"#00d4ff","Anomalie":"#ff3c6e"},
                             hover_data=["ip_src","profil","ratio_deny"],
                             log_x=True, opacity=0.65, size_max=8)
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                              font_color="#c8d8e8", height=360, legend_title="",
                              xaxis=dict(gridcolor="#1e2a38", title="Nb connexions (log)"),
                              yaxis=dict(gridcolor="#1e2a38", title="Ports distincts"),
                              legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)

        with v2:
            st.markdown("<div class='section-hd'>Profils comportementaux détectés</div>", unsafe_allow_html=True)
            pal = {"Normal":"#00ff9d","Port Scan":"#00d4ff","DDoS / Flood":"#ff3c6e",
                   "Activité nocturne suspecte":"#a259ff","Comportement bloqué":"#ffb800","Attaque ciblée":"#ff6b6b"}
            fig2 = px.bar(profil_counts.reset_index(), x="count", y="profil",
                          orientation="h",
                          color="profil", color_discrete_map=pal)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                               font_color="#c8d8e8", height=360, showlegend=False,
                               xaxis=dict(gridcolor="#1e2a38"), yaxis=dict(gridcolor="#1e2a38"))
            st.plotly_chart(fig2, use_container_width=True)

        # Anomaly score distribution
        st.markdown("<div class='section-hd'>Distribution du score d'anomalie (Isolation Forest)</div>", unsafe_allow_html=True)
        fig3 = go.Figure()
        for grp, color, name in [(-1,"#ff3c6e","Anomalie"), (1,"#00d4ff","Normal")]:
            subset = ip_features[ip_features["anomaly_iso"]==grp]["anomaly_score"]
            fig3.add_trace(go.Histogram(x=subset, name=name, marker_color=color, opacity=0.75, nbinsx=50))
        fig3.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                           font_color="#c8d8e8", height=250, legend=dict(bgcolor="rgba(0,0,0,0)"),
                           xaxis=dict(gridcolor="#1e2a38", title="Score d'anomalie"),
                           yaxis=dict(gridcolor="#1e2a38", title="Nb IPs"),
                           margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig3, use_container_width=True)

        # Top suspects
        st.markdown("<div class='section-hd'>Top 20 IPs les plus suspectes</div>", unsafe_allow_html=True)
        suspects = (ip_features[ip_features["anomaly_iso"]==-1]
                    .sort_values("anomaly_score").head(20)
                    [["ip_src","nb_connexions","nb_ports_distincts","ratio_deny","nb_ports_sensibles","profil","anomaly_score"]])

        pb_map = {"Normal":"pb-normal","Port Scan":"pb-scan","DDoS / Flood":"pb-ddos",
                  "Activité nocturne suspecte":"pb-nocturne","Comportement bloqué":"pb-blocked","Attaque ciblée":"pb-targeted"}
        rows_html = ""
        for _, r in suspects.iterrows():
            cls = pb_map.get(r["profil"], "pb-normal")
            rows_html += f"""<tr>
              <td style='color:#00d4ff;font-size:0.68rem;padding:6px 10px;'>{r['ip_src']}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['nb_connexions']:,}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['nb_ports_distincts']}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['ratio_deny']:.2f}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['nb_ports_sensibles']}</td>
              <td style='padding:6px 10px;'><span class='profile-badge {cls}'>{r['profil']}</span></td>
              <td style='text-align:right;padding:6px 10px;color:#ff3c6e;'>{r['anomaly_score']:.4f}</td>
            </tr>"""
        st.markdown(f"""
        <table style='width:100%;border-collapse:collapse;font-size:0.7rem;'>
          <thead>
            <tr style='color:#4a6072;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1e2a38;'>
              <th style='text-align:left;padding:6px 10px;'>IP Source</th>
              <th style='text-align:right;padding:6px 10px;'>Connexions</th>
              <th style='text-align:right;padding:6px 10px;'>Ports</th>
              <th style='text-align:right;padding:6px 10px;'>Ratio DENY</th>
              <th style='text-align:right;padding:6px 10px;'>Ports sensibles</th>
              <th style='padding:6px 10px;'>Profil</th>
              <th style='text-align:right;padding:6px 10px;'>Score</th>
            </tr>
          </thead>
          <tbody style='border-top:1px solid #1e2a38;'>{rows_html}</tbody>
        </table>""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# TAB 3 — CLASSIFICATION ML (Scénario 2)
# ───────────────────────────────────────────────────────────────
with tab_classif:
    st.markdown("""<div class='story-banner'>
      <b style='font-family:Syne,sans-serif;color:#00d4ff;letter-spacing:2px;font-size:0.7rem;'>SCÉNARIO 2 — CLASSIFICATION RANDOM FOREST</b><br><br>
      Une fois les profils comportementaux établis par Isolation Forest, un <span class='highlight'>Random Forest à 200 arbres</span>
      apprend à reconnaître chaque signature d'attaque. L'objectif : créer un <span class='danger'>classificateur temps réel</span>
      capable de scorer immédiatement toute nouvelle IP source et de déclencher une alerte si son comportement dévie de la norme.
      Les courbes ROC et la matrice de confusion quantifient la précision du modèle.
    </div>""", unsafe_allow_html=True)

    ip_features = st.session_state.ip_features
    if ip_features is None:
        st.info("💡 Lancez d'abord la **Détection d'anomalies** (onglet précédent) pour générer les features.")
    else:
        if st.button("🤖 Entraîner le Random Forest"):
            with st.spinner("Entraînement du modèle…"):
                features_cols = ["nb_connexions","nb_ports_distincts","nb_ips_dst","ratio_deny",
                                 "nb_ports_sensibles","activite_nuit","port_dst_std"]
                X = ip_features[features_cols]
                y = ip_features["profil"]

                # S'assurer d'avoir au moins 2 classes
                if y.nunique() < 2:
                    st.warning("Pas assez de classes pour entraîner le modèle. Augmentez le nombre de flux analysés.")
                    st.stop()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=5,
                                            class_weight="balanced", random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)
                classes = rf.classes_

                cm = confusion_matrix(y_test, y_pred, labels=classes)
                importance_df = pd.DataFrame({"feature": features_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
                report_dict   = classification_report(y_test, y_pred, output_dict=True)

                # ROC
                try:
                    y_test_bin = label_binarize(y_test, classes=classes)
                    y_score    = rf.predict_proba(X_test)
                    roc_data = []
                    for i, cls in enumerate(classes):
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                        roc_auc     = auc(fpr, tpr)
                        roc_data.append((cls, fpr, tpr, roc_auc))
                    st.session_state.rf_roc = roc_data
                except Exception:
                    st.session_state.rf_roc = None

                st.session_state.rf_model      = rf
                st.session_state.rf_classes    = classes
                st.session_state.rf_report     = report_dict
                st.session_state.rf_cm         = (cm, classes)
                st.session_state.rf_importance = importance_df
            st.success(f"✅ Random Forest entraîné — Accuracy : {report_dict.get('accuracy', 0):.1%}")

        if st.session_state.rf_report is not None:
            report_dict   = st.session_state.rf_report
            cm, classes   = st.session_state.rf_cm
            importance_df = st.session_state.rf_importance

            acc = report_dict.get("accuracy", 0)
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='stat-block'><div class='val' style='color:#00ff9d;'>{acc:.1%}</div><div class='lbl'>Accuracy globale</div></div>", unsafe_allow_html=True)
            top_feat = importance_df.iloc[0]
            c2.markdown(f"<div class='stat-block'><div class='val' style='color:#00d4ff;font-size:1rem;'>{top_feat['feature']}</div><div class='lbl'>Feature la + discriminante</div></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='stat-block'><div class='val' style='color:#a259ff;'>{len(classes)}</div><div class='lbl'>Classes identifiées</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_cm, col_feat = st.columns(2, gap="large")

            with col_cm:
                st.markdown("<div class='section-hd'>Matrice de confusion</div>", unsafe_allow_html=True)
                fig_cm = px.imshow(cm, x=list(classes), y=list(classes),
                                   color_continuous_scale=[[0,"#0d1117"],[0.5,"#1e3a5f"],[1,"#00d4ff"]],
                                   text_auto=True, aspect="auto")
                fig_cm.update_traces(textfont_size=11)
                fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                     font_color="#c8d8e8", height=360,
                                     xaxis=dict(title="Prédit", tickangle=-30),
                                     yaxis=dict(title="Réel"),
                                     margin=dict(t=0,b=40,l=0,r=0),
                                     coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_feat:
                st.markdown("<div class='section-hd'>Importance des features (Gini)</div>", unsafe_allow_html=True)
                colors = ["#ff3c6e" if i == 0 else "#00d4ff" for i in range(len(importance_df))]
                fig_fi = px.bar(importance_df, x="importance", y="feature",
                                orientation="h", color="feature",
                                color_discrete_sequence=colors, text="importance")
                fig_fi.update_traces(texttemplate="%{text:.4f}", textposition="outside", textfont_size=9)
                fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                     font_color="#c8d8e8", height=360, showlegend=False,
                                     xaxis=dict(gridcolor="#1e2a38"),
                                     yaxis=dict(gridcolor="#1e2a38"),
                                     margin=dict(t=0,b=0,l=0,r=60))
                st.plotly_chart(fig_fi, use_container_width=True)

            # Courbes ROC
            if st.session_state.rf_roc:
                st.markdown("<div class='section-hd'>Courbes ROC (one-vs-rest)</div>", unsafe_allow_html=True)
                colors_roc = ["#ff3c6e","#00d4ff","#00ff9d","#ffb800","#a259ff","#ff6b6b"]
                fig_roc = go.Figure()
                for i, (cls, fpr, tpr, roc_auc) in enumerate(st.session_state.rf_roc):
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{cls} (AUC={roc_auc:.3f})",
                                                  line=dict(color=colors_roc[i % len(colors_roc)], width=2)))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Aléatoire", line=dict(color="#4a6072", dash="dash", width=1)))
                fig_roc.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                       font_color="#c8d8e8", height=360,
                                       xaxis=dict(gridcolor="#1e2a38", title="Taux faux positifs"),
                                       yaxis=dict(gridcolor="#1e2a38", title="Taux vrais positifs"),
                                       legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
                                       margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_roc, use_container_width=True)

            # Tableau métriques par classe
            st.markdown("<div class='section-hd'>Rapport de classification par classe</div>", unsafe_allow_html=True)
            rows_m = ""
            for cls in classes:
                d = report_dict.get(cls, {})
                prec  = d.get("precision", 0)
                rec   = d.get("recall", 0)
                f1    = d.get("f1-score", 0)
                sup   = int(d.get("support", 0))
                color = "#00ff9d" if f1 > 0.8 else ("#ffb800" if f1 > 0.5 else "#ff3c6e")
                rows_m += f"""<tr>
                  <td style='padding:7px 12px;'>{cls}</td>
                  <td style='text-align:right;padding:7px 12px;color:{color};'>{prec:.3f}</td>
                  <td style='text-align:right;padding:7px 12px;color:{color};'>{rec:.3f}</td>
                  <td style='text-align:right;padding:7px 12px;color:{color};font-weight:700;'>{f1:.3f}</td>
                  <td style='text-align:right;padding:7px 12px;color:#4a6072;'>{sup}</td>
                </tr>"""
            st.markdown(f"""<table style='width:100%;border-collapse:collapse;font-size:0.72rem;'>
              <thead><tr style='color:#4a6072;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1e2a38;'>
                <th style='text-align:left;padding:7px 12px;'>Classe</th>
                <th style='text-align:right;padding:7px 12px;'>Précision</th>
                <th style='text-align:right;padding:7px 12px;'>Rappel</th>
                <th style='text-align:right;padding:7px 12px;'>F1-score</th>
                <th style='text-align:right;padding:7px 12px;'>Support</th>
              </tr></thead>
              <tbody style='border-top:1px solid #1e2a38;'>{rows_m}</tbody>
            </table>""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# TAB 4 — ANALYSE TEMPORELLE (Scénario 3)
# ───────────────────────────────────────────────────────────────
with tab_temporal:
    st.markdown("""<div class='story-banner'>
      <b style='font-family:Syne,sans-serif;color:#00d4ff;letter-spacing:2px;font-size:0.7rem;'>SCÉNARIO 3 — ANALYSE TEMPORELLE</b><br><br>
      Le temps révèle ce que la snapshot statique cache. En décomposant le trafic selon l'<span class='highlight'>heure</span>,
      le <span class='highlight'>jour de la semaine</span> et la <span class='highlight'>semaine calendaire</span>,
      des patterns émergent : attaques concentrées en <span class='danger'>heures creuses</span>, pics d'activité anormaux
      détectés par <span class='highlight'>Z-score</span>, et signatures temporelles distinctes par profil comportemental.
    </div>""", unsafe_allow_html=True)

    if "datetime" not in df_raw.columns or df_raw["datetime"].isna().all():
        st.warning("⚠️ Colonne `datetime` absente ou invalide dans les données.")
    else:
        if st.button("📈 Lancer l'analyse temporelle"):
            with st.spinner("Extraction des composantes temporelles…"):
                df_ts = df_raw.copy()
                df_ts["datetime"]    = pd.to_datetime(df_ts["datetime"], errors="coerce")
                df_ts                = df_ts.dropna(subset=["datetime"])
                df_ts["hour"]        = df_ts["datetime"].dt.hour
                df_ts["day_of_week"] = df_ts["datetime"].dt.day_name()
                df_ts["date"]        = df_ts["datetime"].dt.date

                day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

                # Heatmap volume total
                heatmap_vol = (df_ts.groupby(["day_of_week","hour"]).size()
                               .reset_index(name="count")
                               .pivot(index="day_of_week", columns="hour", values="count")
                               .reindex(day_order).fillna(0))

                heatmap_deny = (df_ts[df_ts["action"]=="DENY"].groupby(["day_of_week","hour"]).size()
                                .reset_index(name="count")
                                .pivot(index="day_of_week", columns="hour", values="count")
                                .reindex(day_order).fillna(0))

                # Série temporelle horaire
                ts_hourly = (df_ts.groupby([pd.Grouper(key="datetime",freq="h"), "action"])
                             .size().reset_index(name="count"))

                # Volume horaire total + Z-score
                ts_total = (df_ts.groupby(pd.Grouper(key="datetime",freq="h"))
                            .size().reset_index(name="count")
                            .rename(columns={"datetime":"hour_ts"}))
                ts_total["zscore"] = scipy_stats.zscore(ts_total["count"])
                ts_total["is_pic"] = ts_total["zscore"] > 2.5
                pics = ts_total[ts_total["is_pic"]].sort_values("zscore", ascending=False)

                # Distribution horaire par profil (si ip_features disponible)
                hourly_profil = None
                if st.session_state.ip_features is not None:
                    df_profil = df_ts.merge(st.session_state.ip_features[["ip_src","profil"]], on="ip_src", how="left")
                    df_profil["profil"] = df_profil["profil"].fillna("Normal")
                    hourly_profil = (df_profil.groupby(["hour","profil"]).size().reset_index(name="count"))

                st.session_state.ts_data  = {
                    "heatmap_vol": heatmap_vol, "heatmap_deny": heatmap_deny,
                    "ts_hourly": ts_hourly, "ts_total": ts_total, "day_order": day_order,
                    "hourly_profil": hourly_profil, "n_days": df_ts["date"].nunique(),
                    "t_start": str(df_ts["datetime"].min()), "t_end": str(df_ts["datetime"].max()),
                }
                st.session_state.ts_pics = pics
            st.success("✅ Analyse temporelle terminée !")

        ts_data = st.session_state.ts_data
        pics    = st.session_state.ts_pics
        if ts_data is not None:
            n_pics    = len(pics) if pics is not None else 0
            ts_total  = ts_data["ts_total"]
            t_start   = ts_data["t_start"][:10]
            t_end     = ts_data["t_end"][:10]

            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='stat-block'><div class='val' style='color:#00d4ff;'>{ts_data['n_days']}</div><div class='lbl'>Jours analysés</div></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='stat-block'><div class='val' style='color:#ff3c6e;'>{n_pics}</div><div class='lbl'>Pics Z-score > 2.5</div></div>", unsafe_allow_html=True)
            hmax = ts_total.loc[ts_total["count"].idxmax(), "hour_ts"]
            c3.markdown(f"<div class='stat-block'><div class='val' style='color:#ffb800;font-size:1rem;'>{str(hmax)[:13]}</div><div class='lbl'>Pic d'activité max</div></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='stat-block'><div class='val' style='color:#a259ff;font-size:1rem;'>{t_start} → {t_end}</div><div class='lbl'>Période couverte</div></div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Heatmaps
            st.markdown("<div class='section-hd'>Heatmaps — Volume total & Connexions DENY par heure × jour</div>", unsafe_allow_html=True)
            hm1, hm2 = st.columns(2, gap="large")
            with hm1:
                fig_hm = go.Figure(go.Heatmap(z=ts_data["heatmap_vol"].values,
                                              x=list(range(24)), y=ts_data["heatmap_vol"].index.tolist(),
                                              colorscale=[[0,"#07090f"],[0.5,"#1e3a5f"],[1,"#00d4ff"]],
                                              hoverongaps=False))
                fig_hm.update_layout(title="Volume total", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                     font_color="#c8d8e8", height=280, margin=dict(t=30,b=0,l=0,r=0),
                                     xaxis=dict(title="Heure"), yaxis=dict(title=""))
                st.plotly_chart(fig_hm, use_container_width=True)
            with hm2:
                fig_hm2 = go.Figure(go.Heatmap(z=ts_data["heatmap_deny"].values,
                                               x=list(range(24)), y=ts_data["heatmap_deny"].index.tolist(),
                                               colorscale=[[0,"#07090f"],[0.5,"#3a1015"],[1,"#ff3c6e"]],
                                               hoverongaps=False))
                fig_hm2.update_layout(title="Connexions DENY", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                      font_color="#c8d8e8", height=280, margin=dict(t=30,b=0,l=0,r=0),
                                      xaxis=dict(title="Heure"), yaxis=dict(title=""))
                st.plotly_chart(fig_hm2, use_container_width=True)

            # Séries temporelles
            st.markdown("<div class='section-hd'>Séries temporelles — PERMIT vs DENY par heure</div>", unsafe_allow_html=True)
            ts_hourly = ts_data["ts_hourly"]
            permit_ts = ts_hourly[ts_hourly["action"]=="PERMIT"].set_index("datetime")["count"]
            deny_ts   = ts_hourly[ts_hourly["action"]=="DENY"].set_index("datetime")["count"]

            fig_ts = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
            fig_ts.add_trace(go.Scatter(x=permit_ts.index, y=permit_ts.values, name="PERMIT",
                                        line=dict(color="#00d4ff", width=1.5),
                                        fill="tozeroy", fillcolor="rgba(0,212,255,0.08)"), row=1, col=1)
            fig_ts.add_trace(go.Scatter(x=deny_ts.index, y=deny_ts.values, name="DENY",
                                        line=dict(color="#ff3c6e", width=1.5),
                                        fill="tozeroy", fillcolor="rgba(255,60,110,0.08)"), row=2, col=1)
            fig_ts.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117", font_color="#c8d8e8",
                                 height=380, legend=dict(bgcolor="rgba(0,0,0,0)"),
                                 margin=dict(t=0,b=0,l=0,r=0))
            for i in [1,2]:
                fig_ts.update_xaxes(gridcolor="#1e2a38", row=i, col=1)
                fig_ts.update_yaxes(gridcolor="#1e2a38", row=i, col=1)
            st.plotly_chart(fig_ts, use_container_width=True)

            # Z-score
            st.markdown("<div class='section-hd'>Détection de pics d'activité — Z-score (seuil = 2.5)</div>", unsafe_allow_html=True)
            seuil_line = ts_total["count"].mean() + 2.5 * ts_total["count"].std()
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=ts_total["hour_ts"], y=ts_total["count"],
                                       name="Volume horaire", line=dict(color="#4a6072", width=1.2),
                                       fill="tozeroy", fillcolor="rgba(74,96,114,0.1)"))
            if not ts_total[ts_total["is_pic"]].empty:
                fig_z.add_trace(go.Scatter(x=ts_total.loc[ts_total["is_pic"],"hour_ts"],
                                           y=ts_total.loc[ts_total["is_pic"],"count"],
                                           mode="markers", name="Pic anormal",
                                           marker=dict(color="#ff3c6e", size=8, symbol="circle",
                                                       line=dict(color="#fff", width=1))))
            fig_z.add_hline(y=seuil_line, line_dash="dash", line_color="#ff3c6e",
                             annotation_text="Seuil z=2.5", annotation_font_color="#ff3c6e")
            fig_z.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117", font_color="#c8d8e8",
                                 height=300, legend=dict(bgcolor="rgba(0,0,0,0)"),
                                 xaxis=dict(gridcolor="#1e2a38"), yaxis=dict(gridcolor="#1e2a38"),
                                 margin=dict(t=0,b=0,l=0,r=0))
            st.plotly_chart(fig_z, use_container_width=True)

            # Distribution par profil (si disponible)
            if ts_data["hourly_profil"] is not None:
                st.markdown("<div class='section-hd'>Distribution horaire par profil comportemental</div>", unsafe_allow_html=True)
                pal_profil = {"Normal":"#00d4ff","Activité nocturne suspecte":"#a259ff",
                              "Comportement bloqué":"#ffb800","DDoS / Flood":"#ff3c6e",
                              "Port Scan":"#00ff9d","Attaque ciblée":"#ff6b6b"}
                fig_p = px.line(ts_data["hourly_profil"], x="hour", y="count", color="profil",
                                color_discrete_map=pal_profil, markers=True)
                fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117", font_color="#c8d8e8",
                                    height=320, legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
                                    xaxis=dict(gridcolor="#1e2a38", title="Heure", tickmode="linear"),
                                    yaxis=dict(gridcolor="#1e2a38", title="Nb connexions"),
                                    margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_p, use_container_width=True)

            # Tableau top pics
            if n_pics > 0:
                st.markdown("<div class='section-hd'>Top pics détectés</div>", unsafe_allow_html=True)
                st.dataframe(pics.head(10)[["hour_ts","count","zscore"]].rename(columns={
                    "hour_ts":"Horodatage","count":"Nb connexions","zscore":"Z-score"
                }).style.format({"Z-score":"{:.3f}","Nb connexions":"{:,}"}),
                    use_container_width=True, hide_index=True, height=280)

# ───────────────────────────────────────────────────────────────
# TAB 5 — THREAT ANALYST IA
# ───────────────────────────────────────────────────────────────
with tab_ia:
    st.markdown("""
    <div style='background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--accent3);
                border-radius:6px;padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;gap:16px;'>
      <div style='font-size:2.5rem;'>🛡</div>
      <div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:var(--text-hi);letter-spacing:1px;'>
          Threat Analyst IA — Propulsé par Mistral AI
        </div>
        <div style='color:var(--text-dim);font-size:0.68rem;margin-top:4px;letter-spacing:1px;'>
          Analyse intelligente des menaces · Rapport structuré · Streaming en temps réel
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_deny = df_raw[df_raw["action"]=="DENY"] if "action" in df_raw.columns else pd.DataFrame()
    n_deny  = len(df_deny)
    n_ips   = df_deny["ip_src"].nunique() if not df_deny.empty else 0
    n_ports = df_deny["port_dst"].nunique() if not df_deny.empty else 0
    top_port_val = df_deny["port_dst"].value_counts().idxmax() if not df_deny.empty else "—"

    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(f"<div class='stat-block'><div class='val'>{n_deny:,}</div><div class='lbl'>Tentatives DENY</div></div>", unsafe_allow_html=True)
    d2.markdown(f"<div class='stat-block'><div class='val'>{n_ips:,}</div><div class='lbl'>IPs suspectes</div></div>", unsafe_allow_html=True)
    d3.markdown(f"<div class='stat-block'><div class='val'>{n_ports}</div><div class='lbl'>Ports ciblés</div></div>", unsafe_allow_html=True)
    d4.markdown(f"<div class='stat-block'><div class='val' style='font-size:1.4rem;'>:{top_port_val}</div><div class='lbl'>Port n°1 ciblé</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not mistral_key:
        st.info("🔑 Entrez votre clé API Mistral dans la sidebar.")
    elif df_deny.empty:
        st.warning("⚠️ Aucune connexion DENY.")
    else:
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            run_ia = st.button("🛡 Générer le rapport IA")
        with col_info:
            st.caption(f"Modèle : **{mistral_model}** · {n_deny} flux DENY · {n_ips} IPs · Streaming SSE")

        if run_ia or st.session_state.ia_report:
            if run_ia:
                prompt = build_threat_prompt(df_deny, st.session_state.geo_cache)
                st.markdown("<div class='section-hd'>Rapport de menaces — génération en cours…</div>", unsafe_allow_html=True)
                report_box = st.empty()
                full_text  = ""
                try:
                    for chunk in stream_mistral(mistral_key, mistral_model, prompt):
                        full_text += chunk
                        report_box.markdown(f"<div class='report-box'>{full_text}▌</div>", unsafe_allow_html=True)
                    report_box.markdown(f"<div class='report-box'>{full_text}</div>", unsafe_allow_html=True)
                    st.session_state.ia_report = full_text
                    st.success("✅ Rapport généré.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"❌ Erreur API Mistral ({e.response.status_code})")
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
            elif st.session_state.ia_report:
                st.markdown("<div class='section-hd'>Dernier rapport généré</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='report-box'>{st.session_state.ia_report}</div>", unsafe_allow_html=True)
                if st.button("🔄 Régénérer"):
                    st.session_state.ia_report = None
                    st.rerun()

    st.markdown("---")
    st.markdown("<div class='section-hd'>Données DENY transmises au LLM</div>", unsafe_allow_html=True)
    preview_cols = [c for c in ["datetime","ip_src","ip_dst","port_dst","protocol_clean","rule_id"] if c in df_deny.columns]
    if preview_cols:
        st.dataframe(df_deny[preview_cols].head(50), use_container_width=True, hide_index=True, height=280)

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#2a3a4a;font-size:0.62rem;letter-spacing:2px;padding:8px 0;'>
  NETFLOW SENTINEL &nbsp;·&nbsp; ip-api.com &nbsp;·&nbsp; Mistral AI &nbsp;·&nbsp; Isolation Forest &nbsp;·&nbsp;
  Random Forest &nbsp;·&nbsp; pydeck &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)