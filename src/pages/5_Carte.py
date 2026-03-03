import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import requests
import pydeck as pdk
import time
from datetime import datetime

from modules.preprocessing import load_data
from components.top_nav import render_top_nav
from utils import (
    port_label,
    is_public,
    geolocate_ips,
    arrow_angle,
)

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NetFlow — IP Traffic Visualizer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_top_nav("map")

# ═══════════════════════════════════════════════════════════════
# CSS — Thème clair, coloré, premium
# ═══════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #f0f4ff !important;
    color: #1e293b !important;
}

/* ── Fond principal ── */
.main .block-container {
    background: #f0f4ff !important;
    padding: 1.5rem 2rem !important;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8faff 100%) !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 12px rgba(99,102,241,0.06) !important;
}
[data-testid="stSidebar"] * {
    color: #334155 !important;
}
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown strong {
    color: #4f46e5 !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] hr {
    border-color: #e2e8f0 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 4px rgba(99,102,241,.08), 0 4px 20px rgba(99,102,241,.06) !important;
    padding: 1.1rem 1.2rem !important;
    border-top: 3px solid #6366f1 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #1e293b !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .72rem !important;
    color: #64748b !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}

/* ── Bouton principal ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 60%, #db2777 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .85rem !important;
    font-weight: 700 !important;
    letter-spacing: .5px !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 14px rgba(79,70,229,.35) !important;
    transition: all .25s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(79,70,229,.45) !important;
    filter: brightness(1.06) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Multiselect tags ── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: #eef2ff !important;
    color: #4f46e5 !important;
    border-radius: 6px !important;
    border: 1px solid #c7d2fe !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.04) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.06) !important;
}

/* ── Divider ── */
hr { border-color: #e2e8f0 !important; }

/* ── Section header ── */
.section-hd {
    font-family: 'Inter', sans-serif;
    font-size: .72rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: #64748b;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px; margin: 20px 0 12px 0;
    display: flex; align-items: center; gap: 8px;
}

/* ── Flux feed cards ── */
.feed-card {
    background: #fff;
    border-radius: 10px;
    padding: 10px 12px;
    margin: 6px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    font-family: 'JetBrains Mono', monospace;
    font-size: .72rem;
    animation: fadeUp .35s ease-out;
    line-height: 1.7;
}
.feed-deny   { border-left: 4px solid #ef4444; }
.feed-permit { border-left: 4px solid #10b981; }
@keyframes fadeUp {
    from { opacity:0; transform:translateY(6px); }
    to   { opacity:1; transform:translateY(0); }
}

.tag-deny   { background:#fef2f2; color:#ef4444; border:1px solid #fecaca; padding:2px 8px; border-radius:20px; font-size:.65rem; font-weight:600; }
.tag-permit { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; padding:2px 8px; border-radius:20px; font-size:.65rem; font-weight:600; }

.ip-src { color:#4f46e5; font-weight:600; }
.ip-dst { color:#db2777; font-weight:600; }
.geo-txt { color:#94a3b8; font-size:.66rem; }

/* ── Stat badges ── */
.badge {
    display:inline-flex; align-items:center; gap:5px;
    padding:5px 12px; border-radius:20px;
    font-family:'Inter',sans-serif; font-size:.75rem; font-weight:600;
}
.badge-deny   { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
.badge-permit { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; }
.badge-info   { background:#eef2ff; color:#4f46e5; border:1px solid #c7d2fe; }

/* ── Carte attente ── */
.map-wait {
    background: linear-gradient(135deg,#f0f4ff,#fdf4ff);
    border: 2px dashed #c7d2fe;
    border-radius: 16px;
    text-align: center;
    padding: 70px 20px;
    margin-top: 10px;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg,#4f46e5,#7c3aed,#db2777) !important;
    border-radius: 10px !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #fff !important;
    border: 2px dashed #c7d2fe !important;
    border-radius: 10px !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-testid="stThumbValue"] { color:#4f46e5 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #e2e8f0 !important;
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: #fff !important;
    border-radius: 10px 10px 0 0 !important;
    border: 1px solid #e2e8f0 !important;
    border-bottom: none !important;
    color: #64748b !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
    padding: 10px 22px !important;
    transition: all .2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    color: #fff !important;
    border-color: transparent !important;
}

/* ── Rapport IA ── */
.report-box {
    background: #fff;
    border-radius: 16px;
    padding: 28px 32px;
    box-shadow: 0 2px 12px rgba(99,102,241,.08), 0 1px 3px rgba(0,0,0,.04);
    border-left: 5px solid #4f46e5;
    font-family: 'Inter', sans-serif;
    font-size: .9rem;
    line-height: 1.75;
    color: #1e293b;
}
.report-box h2 { color:#4f46e5; font-size:1.1rem; margin-top:1.2em; margin-bottom:.4em; }
.report-box h3 { color:#7c3aed; font-size:.95rem; margin-top:1em; margin-bottom:.3em; }
.report-box strong { color:#1e293b; }
.report-box ul { padding-left:1.2em; }
.report-box li { margin-bottom:.3em; }

.threat-stat {
    background: linear-gradient(135deg,#fef2f2,#fff7f7);
    border: 1px solid #fecaca;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.threat-stat-val { font-size:1.8rem; font-weight:800; color:#dc2626; }
.threat-stat-lbl { font-size:.7rem; font-weight:600; color:#94a3b8;
                   letter-spacing:1px; text-transform:uppercase; margin-top:2px; }

.score-ring {
    font-size:3rem; font-weight:900; text-align:center;
    padding:20px; border-radius:50%; width:100px; height:100px;
    display:flex; align-items:center; justify-content:center;
    margin:0 auto;
}
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
for k, v in {
    "arc_df": None,
    "arrow_df": None,
    "scatter_df": None,
    "detail_df": None,
    "top_src_df": None,
    "geo_count": 0,
    "flow_log": [],
    "geo_cache": {},
    "country_src": None,
    "country_dst": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
now = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
st.markdown(
    f"""
<div style="
    background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 50%,#db2777 100%);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(79,70,229,.25);
    display: flex; align-items: center; justify-content: space-between;
">
  <div>
    <div style="font-size:2rem;font-weight:900;color:#fff;letter-spacing:-0.5px;line-height:1.1;">
      🌍 NetFlow <span style="font-weight:300;">Visualizer</span>
    </div>
    <div style="color:rgba(255,255,255,.75);font-size:.85rem;margin-top:4px;letter-spacing:.3px;">
      Visualisation géographique des flux réseau &nbsp;·&nbsp; IP Source → IP Destination
    </div>
  </div>
  <div style="text-align:right;">
    <div style="color:rgba(255,255,255,.5);font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:1px;">
      {now}
    </div>
    <div style="color:rgba(255,255,255,.9);font-size:.8rem;margin-top:2px;">
      📂 <b>df_1000.csv</b>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Charger un autre CSV", type=["csv"])
    st.caption("Par défaut : **df_1000.csv** chargé automatiquement")
    st.markdown("---")

    filter_action = st.multiselect(
        "🎯 Action", ["DENY", "PERMIT"], default=["DENY", "PERMIT"]
    )
    filter_protocol = st.multiselect(
        "📡 Protocole", ["TCP", "UDP", "ICMP"], default=["TCP", "UDP", "ICMP"]
    )
    max_rows = st.slider("🔢 Flux à analyser", 50, 1000, 200, step=50)
    st.markdown("---")

    show_arrows = st.checkbox("▶ Flèches directionnelles", value=True)
    show_trips = st.checkbox("✨ Particules animées", value=False)
    map_style = st.selectbox(
        "🗺 Style de carte",
        ["Voyager (colorée)", "Positron (claire)", "Dark Matter (sombre)"],
        index=0,
    )
    map_pitch = st.slider("🌍 Inclinaison carte", 0, 55, 30)
    arc_width = st.slider("⚡ Épaisseur arcs", 1, 6, 2)
    arrow_size = st.slider("▶ Taille flèches", 10, 35, 18)
    st.markdown("---")

    if st.button("🗑 Réinitialiser la carte"):
        for k in (
            "arc_df",
            "arrow_df",
            "scatter_df",
            "detail_df",
            "top_src_df",
            "flow_log",
            "country_src",
            "country_dst",
        ):
            st.session_state[k] = None if k != "flow_log" else []
        st.session_state.geo_count = 0
        st.rerun()

    st.markdown("---")
    st.markdown("**API :** [ip-api.com](http://ip-api.com)")
    st.caption("Gratuit · 100 IP/requête · Mise en cache activée")



# ═══════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


if uploaded:
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = get_data().copy()

if filter_action:
    df_raw = df_raw[df_raw["action"].isin(filter_action)]
if filter_protocol and "protocol_clean" in df_raw.columns:
    df_raw = df_raw[df_raw["protocol_clean"].isin(filter_protocol)]
df = df_raw.head(max_rows).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
# KPI STRIP
# ═══════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📦 Flux total", f"{len(df):,}")
k2.metric("🚫 DENY", f"{int((df['action']=='DENY').sum()):,}")
k3.metric("✅ PERMIT", f"{int((df['action']=='PERMIT').sum()):,}")
k4.metric("🖥 Sources uniques", f"{df['ip_src'].nunique():,}")
k5.metric("🎯 Destinations", f"{df['ip_dst'].nunique():,}")

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CARTE + FLUX + ANALYTICS
# ═══════════════════════════════════════════════════════════════
map_col, feed_col = st.columns([3, 1], gap="large")

with map_col:
    st.markdown(
        "<div class='section-hd'>🗺 Carte des flux — Source ──▶ Destination</div>",
        unsafe_allow_html=True,
    )

    if st.button("🔍  Géolocaliser & Afficher la carte"):
        all_ips = list(df["ip_src"].dropna()) + list(df["ip_dst"].dropna())
        pbar = st.progress(0, text="🌐 Interrogation de ip-api.com…")
        with st.spinner(""):
            geo = geolocate_ips(all_ips)
        pbar.progress(55, text="🔗 Construction des arcs et flèches…")

        arcs, arrows, scatter, flow_log = [], [], {}, []
        for _, row in df.iterrows():
            sg = geo.get(str(row["ip_src"]))
            dg = geo.get(str(row["ip_dst"]))
            if not sg or not dg:
                continue
            is_deny = row["action"] == "DENY"
            sc = [239, 68, 68, 210] if is_deny else [16, 185, 129, 200]
            dc = [249, 115, 22, 210] if is_deny else [99, 102, 241, 200]
            port = row.get("port_dst", "")
            arcs.append(
                {
                    "src_lat": sg["lat"],
                    "src_lon": sg["lon"],
                    "dst_lat": dg["lat"],
                    "dst_lon": dg["lon"],
                    "src_ip": str(row["ip_src"]),
                    "dst_ip": str(row["ip_dst"]),
                    "action": row["action"],
                    "src_color": sc,
                    "dst_color": dc,
                    "src_city": sg["city"],
                    "src_country": sg["country"],
                    "dst_city": dg["city"],
                    "dst_country": dg["country"],
                    "protocol": row.get("protocol_clean", "TCP"),
                    "port_dst": port,
                }
            )
            ax = sg["lat"] + 0.65 * (dg["lat"] - sg["lat"])
            ay = sg["lon"] + 0.65 * (dg["lon"] - sg["lon"])
            arrows.append(
                {
                    "lat": ax,
                    "lon": ay,
                    "arrow": "▶",
                    "angle": arrow_angle(
                        sg["lat"], sg["lon"], dg["lat"], dg["lon"]
                    ),
                    "color": sc[:3] + [240],
                    "size": arrow_size,
                }
            )
            for ip, g, col in [
                (str(row["ip_src"]), sg, sc),
                (str(row["ip_dst"]), dg, dc),
            ]:
                scatter[ip] = {
                    "ip": ip,
                    "lat": g["lat"],
                    "lon": g["lon"],
                    "city": g["city"],
                    "country": g["country"],
                    "color": col,
                    "radius": 65000 if is_deny else 48000,
                }
            flow_log.append(
                {
                    "action": row["action"],
                    "src_ip": str(row["ip_src"]),
                    "dst_ip": str(row["ip_dst"]),
                    "src_city": sg["city"],
                    "src_country": sg["country"],
                    "dst_city": dg["city"],
                    "dst_country": dg["country"],
                    "protocol": row.get("protocol_clean", "TCP"),
                    "port": port,
                }
            )

        pbar.progress(90, text="📊 Calcul des statistiques…")
        arc_df_b = pd.DataFrame(arcs)
        arrow_df_b = pd.DataFrame(arrows)
        scat_df_b = pd.DataFrame(list(scatter.values()))
        st.session_state.arc_df = arc_df_b
        st.session_state.arrow_df = arrow_df_b
        st.session_state.scatter_df = scat_df_b
        st.session_state.geo_count = len(geo)
        st.session_state.flow_log = flow_log
        st.session_state.top_src_df = (
            df.groupby(["ip_src", "action"])
            .size()
            .reset_index(name="Nb connexions")
            .sort_values("Nb connexions", ascending=False)
            .head(15)
        )
        st.session_state.detail_df = (
            arc_df_b[
                [
                    "src_ip",
                    "src_city",
                    "src_country",
                    "dst_ip",
                    "dst_city",
                    "dst_country",
                    "action",
                    "protocol",
                    "port_dst",
                ]
            ]
            .drop_duplicates()
            .rename(
                columns={
                    "src_ip": "IP Source",
                    "src_city": "Ville Src",
                    "src_country": "Pays Src",
                    "dst_ip": "IP Dest",
                    "dst_city": "Ville Dst",
                    "dst_country": "Pays Dst",
                    "action": "Action",
                    "protocol": "Protocole",
                    "port_dst": "Port Dst",
                }
            )
        )
        st.session_state.country_src = (
            arc_df_b["src_country"].value_counts().head(10)
        )
        st.session_state.country_dst = (
            arc_df_b["dst_country"].value_counts().head(10)
        )
        pbar.progress(100, text="✅ Analyse terminée !")
        time.sleep(0.4)
        pbar.empty()
        st.success(
            f"✅ **{len(geo)} IPs géolocalisées** — **{len(arcs)} connexions** — **{len(arrows)} flèches**"
        )

    # Rendu carte
    arc_df_s = st.session_state.arc_df
    arrow_df_s = st.session_state.arrow_df
    scatter_df_s = st.session_state.scatter_df
    map_styles = {
        "Voyager (colorée)": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
        "Positron (claire)": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "Dark Matter (sombre)": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    }

    if arc_df_s is not None and not arc_df_s.empty:
        layers = [
            pdk.Layer(
                "ArcLayer",
                data=arc_df_s,
                get_source_position=["src_lon", "src_lat"],
                get_target_position=["dst_lon", "dst_lat"],
                get_source_color="src_color",
                get_target_color="dst_color",
                get_width=arc_width,
                get_height=0.4,
                pickable=True,
                auto_highlight=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=scatter_df_s,
                get_position=["lon", "lat"],
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                opacity=0.85,
                stroked=True,
                get_line_color=[255, 255, 255, 80],
                line_width_min_pixels=1,
            ),
        ]
        if show_arrows and arrow_df_s is not None and not arrow_df_s.empty:
            layers.append(
                pdk.Layer(
                    "TextLayer",
                    data=arrow_df_s,
                    get_position=["lon", "lat"],
                    get_text="arrow",
                    get_size="size",
                    get_color="color",
                    get_angle="angle",
                    font_family="Arial",
                    font_weight="bold",
                    billboard=True,
                    pickable=False,
                )
            )
        if show_trips and not arc_df_s.empty:
            trips = [
                {
                    "path": [
                        [r["src_lon"], r["src_lat"]],
                        [
                            (r["src_lon"] + r["dst_lon"]) / 2,
                            (r["src_lat"] + r["dst_lat"]) / 2 + 5,
                        ],
                        [r["dst_lon"], r["dst_lat"]],
                    ],
                    "timestamps": [0, 50, 100],
                    "color": r["src_color"][:3],
                }
                for _, r in arc_df_s.iterrows()
            ]
            layers.append(
                pdk.Layer(
                    "TripsLayer",
                    data=pd.DataFrame(trips),
                    get_path="path",
                    get_timestamps="timestamps",
                    get_color="color",
                    opacity=0.85,
                    width_min_pixels=3,
                    rounded=True,
                    trail_length=40,
                    current_time=70,
                )
            )

        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=pdk.ViewState(
                    latitude=25, longitude=10, zoom=1.4, pitch=map_pitch, bearing=0
                ),
                map_style=map_styles[map_style],
                tooltip={
                    "html": """
              <div style='font-family:Inter,sans-serif;font-size:12px;background:#fff;
                          border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;
                          box-shadow:0 4px 16px rgba(0,0,0,.12);min-width:220px;'>
                <div style='font-weight:700;font-size:13px;margin-bottom:8px;
                            padding-bottom:6px;border-bottom:1px solid #f1f5f9;
                            color:#4f46e5;'>{action}</div>
                <div style='color:#4f46e5;font-weight:600;'>⬆ {src_ip}</div>
                <div style='color:#94a3b8;font-size:10px;margin-bottom:6px;'>📍 {src_city}, {src_country}</div>
                <div style='color:#db2777;font-weight:600;'>⬇ {dst_ip}</div>
                <div style='color:#94a3b8;font-size:10px;margin-bottom:6px;'>📍 {dst_city}, {dst_country}</div>
                <div style='color:#cbd5e1;font-size:10px;border-top:1px solid #f1f5f9;padding-top:4px;'>
                  {protocol} · port {port_dst}</div>
              </div>""",
                    "style": {
                        "padding": "0",
                        "background": "transparent",
                        "border": "none",
                    },
                },
            ),
            use_container_width=True,
        )

        lg1, lg2, lg3 = st.columns(3)
        lg1.markdown(
            "<span class='badge badge-deny'>🔴 DENY</span>", unsafe_allow_html=True
        )
        lg2.markdown(
            "<span class='badge badge-permit'>🟢 PERMIT</span>",
            unsafe_allow_html=True,
        )
        lg3.markdown(
            f"<span class='badge badge-info'>⚡ {st.session_state.geo_count} IPs · {len(arc_df_s)} arcs</span>",
            unsafe_allow_html=True,
        )

    elif arc_df_s is not None:
        st.warning("⚠️ Aucun arc — IPs privées ou API indisponible.")
    else:
        st.markdown(
            """<div class='map-wait'>
          <div style='font-size:3rem;margin-bottom:12px;'>🌍</div>
          <div style='font-size:1.1rem;font-weight:700;color:#475569;'>Prêt pour l'analyse</div>
          <div style='color:#94a3b8;font-size:.85rem;margin-top:6px;'>
            Cliquez sur <b>Géolocaliser & Afficher la carte</b></div>
        </div>""",
            unsafe_allow_html=True,
        )

with feed_col:
    st.markdown(
        "<div class='section-hd'>⚡ Flux en direct</div>", unsafe_allow_html=True
    )
    if st.session_state.flow_log:
        logs = st.session_state.flow_log
        deny_n = sum(1 for f in logs if f["action"] == "DENY")
        permit_n = len(logs) - deny_n
        st.markdown(
            f"""<div style='display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;'>
          <span class='badge badge-deny'>🚫 {deny_n}</span>
          <span class='badge badge-permit'>✅ {permit_n}</span></div>""",
            unsafe_allow_html=True,
        )
        html = "<div style='max-height:500px;overflow-y:auto;padding-right:4px;'>"
        for f in logs:
            cls = "feed-deny" if f["action"] == "DENY" else "feed-permit"
            tag = "tag-deny" if f["action"] == "DENY" else "tag-permit"
            ico = "🔴" if f["action"] == "DENY" else "🟢"
            html += f"""<div class='feed-card {cls}'>
              <span class='{tag}'>{ico} {f['action']}</span><br>
              <span class='ip-src'>{f['src_ip']}</span>
              <span style='color:#cbd5e1;'> ──▶ </span>
              <span class='ip-dst'>{f['dst_ip']}</span><br>
              <span class='geo-txt'>📍 {f['src_city'][:13]}, {f['src_country'][:11]}</span><br>
              <span class='geo-txt'>🎯 {f['dst_city'][:13]}, {f['dst_country'][:11]}</span><br>
              <span style='color:#94a3b8;font-size:.62rem;'>{f['protocol']} :{f['port']}</span>
            </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(
            """<div style='background:#fff;border:2px dashed #e2e8f0;border-radius:12px;
            text-align:center;padding:40px 16px;'>
          <div style='font-size:2rem;margin-bottom:8px;'>📡</div>
          <div style='color:#94a3b8;font-size:.8rem;font-weight:500;'>En attente…</div>
        </div>""",
            unsafe_allow_html=True,
        )

# Analytics (sous la carte, à l'intérieur du tab1)
if st.session_state.top_src_df is not None:
    st.markdown("---")

    def color_action(val):
        if val == "DENY":
            return "color:#dc2626;font-weight:700;"
        if val == "PERMIT":
            return "color:#16a34a;font-weight:700;"
        return ""

    a1, a2 = st.columns([1, 2], gap="large")
    with a1:
        st.markdown(
            "<div class='section-hd'>📊 Top IPs sources</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            st.session_state.top_src_df.style.applymap(
                color_action, subset=["action"]
            ),
            use_container_width=True,
            hide_index=True,
            height=320,
        )
    with a2:
        st.markdown(
            "<div class='section-hd'>📋 Connexions géolocalisées</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            st.session_state.detail_df.style.applymap(
                color_action, subset=["Action"]
            ),
            use_container_width=True,
            hide_index=True,
            height=320,
        )

    if st.session_state.country_src is not None:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "<div class='section-hd'>🌍 Pays sources — Top 10</div>",
                unsafe_allow_html=True,
            )
            st.bar_chart(
                st.session_state.country_src, use_container_width=True, height=240
            )
        with c2:
            st.markdown(
                "<div class='section-hd'>🎯 Pays destinations — Top 10</div>",
                unsafe_allow_html=True,
            )
            st.bar_chart(
                st.session_state.country_dst, use_container_width=True, height=240
            )

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    """
<div style='text-align:center;color:#94a3b8;font-size:.72rem;padding:4px 0 12px 0;'>
  NetFlow Visualizer &nbsp;·&nbsp; ip-api.com &nbsp;·&nbsp; pydeck &nbsp;·&nbsp; Streamlit
</div>
""",
    unsafe_allow_html=True,
)
