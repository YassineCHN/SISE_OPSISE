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

from modules.preprocessing import load_data as load_parquet_data
from sentinel_utils import (
    MISTRAL_API_KEY_ENV, MISTRAL_MODEL_ENV,
    port_label, is_public, geolocate_ips, arrow_angle,
)
from sentinel_llm_analyst import generate_analysis
from components.top_nav import render_top_nav
from components.sentinel_theme import apply_sentinel_theme

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NetFlow Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_top_nav("sentinel")
apply_sentinel_theme()

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
      <div class='ai-panel-hd'>🤖 Interprétation IA — Mistral</div>""", unsafe_allow_html=True)

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
    st.markdown("### Configuration")
    st.markdown("---")
    uploaded = st.file_uploader("Charger un CSV", type=["csv"])
    st.caption("Par defaut : df_1000.csv")
    st.markdown("---")

    filter_action = st.multiselect("Action", ["DENY", "PERMIT"], default=["DENY", "PERMIT"])
    filter_protocol = st.multiselect("Protocole", ["TCP", "UDP", "ICMP"], default=["TCP", "UDP", "ICMP"])
    max_rows = st.slider("Flux a analyser", 50, 1000, 200, step=50)
    st.markdown("---")

    show_arrows = st.checkbox("Fleches directionnelles", value=True)
    show_trips = st.checkbox("Particules animees", value=False)
    map_style = st.selectbox("Style de carte", ["Dark Matter (sombre)", "Voyager (coloree)", "Positron (claire)"])
    map_pitch = st.slider("Inclinaison carte", 0, 55, 30)
    arc_width = st.slider("Epaisseur arcs", 1, 6, 2)
    arrow_size = st.slider("Taille fleches", 10, 35, 18)
    st.markdown("---")

    if st.button("Reinitialiser tout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("---")
    st.markdown("### Mistral AI")
    if MISTRAL_API_KEY_ENV:
        st.success("Cle .env detectee")
    else:
        st.info("Sans cle: rapports de secours actives")

    mistral_key = st.text_input(
        "Cle API Mistral",
        value=MISTRAL_API_KEY_ENV,
        type="password",
        help="Laissez vide pour utiliser les rapports de secours integres",
    )
    mistral_key = mistral_key.strip().strip('"').strip("'")

    _models = ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"]
    _def = _models.index(MISTRAL_MODEL_ENV) if MISTRAL_MODEL_ENV in _models else 0
    mistral_model = st.selectbox("Modele", _models, index=_def)

    if st.button("Tester la cle API", use_container_width=True):
        if not mistral_key:
            st.warning("Aucune cle fournie.")
        else:
            try:
                resp = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {mistral_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": mistral_model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 5,
                        "temperature": 0,
                        "stream": False,
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    st.success("Cle API valide.")
                elif resp.status_code == 401:
                    st.error("Cle API invalide (401).")
                else:
                    st.error(f"Test API echoue (HTTP {resp.status_code}).")
            except requests.RequestException as e:
                st.error(f"Impossible de joindre l'API Mistral: {e}")

    st.caption("Sans cle: fallback templates actives automatiquement")

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    local = Path(__file__).parent / "df_1000.csv"
    if local.exists():
        df_raw = load_csv(str(local))
    else:
        st.info("df_1000.csv introuvable, chargement automatique du parquet src.")
        df_raw = load_parquet_data().copy()

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
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("📦 Flux total",      f"{len(df_raw):,}")
k2.metric("🚫 DENY",            f"{int((df_raw['action']=='DENY').sum()):,}")
k3.metric("✅ PERMIT",          f"{int((df_raw['action']=='PERMIT').sum()):,}")
k4.metric("🖥 Sources uniques", f"{df_raw['ip_src'].nunique():,}")
k5.metric("🎯 Destinations",    f"{df_raw['ip_dst'].nunique():,}")
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

    if st.button("🔬 Lancer la détection d'anomalies"):
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
        if st.button("🤖 Entraîner le Random Forest"):
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
            st.success(f"✅ Entraîné — Accuracy : {rep_dict.get('accuracy',0):.1%}")

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
                label="📊 Interpréter les résultats ML",
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
        if st.button("📈 Lancer l'analyse temporelle"):
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
            st.success("✅ Analyse temporelle terminée !")

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
                label="📈 Interpréter les patterns temporels",
                generate_fn=lambda key, model: generate_analysis("temporal", key, model, stats=temp_stats),
                requires_key=False,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — THREAT ANALYST IA (rapport global)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ia:
    st.markdown("""
    <div style='background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--accent3);
                border-radius:6px;padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;gap:16px;'>
      <div style='font-size:2.5rem;'>🛡</div>
      <div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:var(--text-hi);letter-spacing:1px;'>
          Threat Analyst IA — Rapport global de menaces
        </div>
        <div style='color:var(--text-dim);font-size:0.68rem;margin-top:4px;letter-spacing:1px;'>
          Synthèse exécutive · Analyse des menaces · Recommandations · Scoring de risque
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    df_deny = df_raw[df_raw["action"]=="DENY"] if "action" in df_raw.columns else pd.DataFrame()
    n_deny  = len(df_deny)
    n_ips   = df_deny["ip_src"].nunique() if not df_deny.empty else 0
    n_ports = df_deny["port_dst"].nunique() if not df_deny.empty else 0
    top_port_val = df_deny["port_dst"].value_counts().idxmax() if not df_deny.empty else "—"

    d1,d2,d3,d4 = st.columns(4)
    d1.markdown(f"<div class='stat-block'><div class='val'>{n_deny:,}</div><div class='lbl'>Tentatives DENY</div></div>", unsafe_allow_html=True)
    d2.markdown(f"<div class='stat-block'><div class='val'>{n_ips:,}</div><div class='lbl'>IPs suspectes</div></div>", unsafe_allow_html=True)
    d3.markdown(f"<div class='stat-block'><div class='val'>{n_ports}</div><div class='lbl'>Ports ciblés</div></div>", unsafe_allow_html=True)
    d4.markdown(f"<div class='stat-block'><div class='val' style='font-size:1.4rem;'>:{top_port_val}</div><div class='lbl'>Port n°1 ciblé</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if df_deny.empty:
        st.warning("⚠️ Aucune connexion DENY dans les données.")
    else:
        geo_cache = st.session_state.get("geo_cache",{})
        render_ai_panel(
            key="global_threat",
            label="🛡 Générer le rapport global de menaces",
            generate_fn=lambda key, model: generate_analysis(
                "global_threat", key, model,
                stats={}, df_deny=df_deny, geo_cache=geo_cache
            ),
            requires_key=True,
        )

        st.markdown("---")
        st.markdown("<div class='section-hd'>Aperçu des données DENY</div>", unsafe_allow_html=True)
        preview_cols = [c for c in ["datetime","ip_src","ip_dst","port_dst","protocol_clean","rule_id"] if c in df_deny.columns]
        if preview_cols:
            st.dataframe(df_deny[preview_cols].head(50), use_container_width=True, hide_index=True, height=280)

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
