import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
import time
from datetime import datetime

from modules.preprocessing import load_data, get_data_source_info
from modules.stats import blocked_ratio
from modules.charts import pie_chart
from components.top_nav import render_top_nav
from components.sentinel_theme import apply_sentinel_theme
from components.ui import neon_metric
from components.data_source_selector import render_motherduck_table_selector
from utils import (
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
apply_sentinel_theme()
selected_table = render_motherduck_table_selector()

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
for k, v in {
    "arc_df": None,
    "arrow_df": None,
    "scatter_df": None,
    "detail_df": None,
    "top_src_df": None,
    "ip_geo_df": None,
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
      Visualisation géographique des flux réseau &nbsp;·&nbsp;
      <span style="color:#a5b4fc;font-weight:600;">IP Source</span>
      <span style="color:rgba(255,255,255,.5);"> ──▶ </span>
      <span style="color:#fbcfe8;font-weight:600;">IP Destination</span>
    </div>
  </div>
  <div style="text-align:right;">
    <div style="color:rgba(255,255,255,.5);font-family:'JetBrains Mono',monospace;font-size:.72rem;letter-spacing:1px;">
      {now}
    </div>
    <div style="margin-top:6px;">
      <span style="background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);
                   border-radius:20px;padding:3px 10px;font-size:.7rem;color:#fff;
                   font-family:'JetBrains Mono',monospace;">ip-api.com</span>
      &nbsp;
      <span style="background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
                   border-radius:20px;padding:3px 10px;font-size:.7rem;color:rgba(255,255,255,.85);
                   font-family:'JetBrains Mono',monospace;">pydeck</span>
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
    filter_action = st.multiselect(
        "🎯 Action", ["DENY", "PERMIT"], default=["DENY", "PERMIT"]
    )
    filter_protocol = st.multiselect(
        "📡 Protocole", ["TCP", "UDP", "ICMP"], default=["TCP", "UDP", "ICMP"]
    )
    max_rows = st.slider("🔢 Flux à analyser", 50, 20000, 200, step=50)
    st.markdown("---")

    show_arrows = st.checkbox("▶ Flèches directionnelles", value=True)
    show_trips = st.checkbox("✨ Particules animées", value=False)
    st.markdown("---")

    if st.button("🗑 Réinitialiser la carte"):
        for k in (
            "arc_df",
            "arrow_df",
            "scatter_df",
            "detail_df",
            "top_src_df",
            "ip_geo_df",
            "flow_log",
            "country_src",
            "country_dst",
        ):
            st.session_state[k] = None if k != "flow_log" else []
        st.session_state.geo_count = 0
        st.session_state.geo_cache = {}
        st.rerun()

    st.markdown("---")
    st.markdown("**API :** [ip-api.com](http://ip-api.com)")
    st.caption("Gratuit · 100 IP/requête · Mise en cache activée")


# ═══════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def get_data(table: str | None) -> pd.DataFrame:
    return load_data(selected_table=table)


df_raw = get_data(table=selected_table).copy()

if filter_action:
    df_raw = df_raw[df_raw["action"].isin(filter_action)]
if filter_protocol and "protocol_clean" in df_raw.columns:
    df_raw = df_raw[df_raw["protocol_clean"].isin(filter_protocol)]
df = df_raw.head(max_rows).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
# KPI STRIP — 7 métriques
# ═══════════════════════════════════════════════════════════════
n_pairs = df[["ip_src", "ip_dst"]].drop_duplicates().shape[0]
pub_src = sum(1 for ip in df["ip_src"].dropna().unique() if is_public(str(ip)))
pub_dst = sum(1 for ip in df["ip_dst"].dropna().unique() if is_public(str(ip)))

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
with k1:
    neon_metric("📦 Flux total", f"{len(df):,}")
with k2:
    neon_metric(
        "🚫 DENY", f"{int((df['action']=='DENY').sum()):,}", color="var(--accent2)"
    )
with k3:
    neon_metric(
        "✅ PERMIT", f"{int((df['action']=='PERMIT').sum()):,}", color="var(--accent4)"
    )
with k4:
    neon_metric("🔗 Paires uniques", f"{n_pairs:,}", color="var(--accent3)")
with k5:
    neon_metric(
        "🚦 Trafic bloqué", f"{blocked_ratio(df):.1f} %", color="var(--accent2)"
    )
with k6:
    neon_metric("🌐 Sources pub.", f"{pub_src:,}", color="var(--accent)")
with k7:
    neon_metric("🌐 Dest. pub.", f"{pub_dst:,}", color="var(--accent)")

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PANNEAU GÉOLOCALISATION
# ═══════════════════════════════════════════════════════════════
st.markdown(
    "<div class='section-hd'>🌐 Géolocalisation des IPs — ip-api.com</div>",
    unsafe_allow_html=True,
)

# Fréquence des IPs publiques uniquement (les privées ne sont pas géolocalisables)
_ip_concat = pd.concat([df["ip_src"].dropna().astype(str), df["ip_dst"].dropna().astype(str)])
_pub_freq = _ip_concat[_ip_concat.map(is_public)].value_counts()
n_unique_pub = len(_pub_freq)

if n_unique_pub == 0:
    st.warning(
        "⚠️ **Aucune IP publique détectée dans ce jeu de données.** "
        "La géolocalisation nécessite des IPs publiques (non RFC1918). "
        "Le jeu `generated_data` contient uniquement des IPs privées — utilisez `original_data` pour afficher la carte."
    )

geo_col1, geo_col2, geo_col3 = st.columns([2, 1, 1])
with geo_col1:
    geo_n = st.slider(
        "Limite d'IPs publiques à géolocaliser",
        min_value=10,
        max_value=200,
        value=min(100, max(10, n_unique_pub)) if n_unique_pub > 0 else 10,
        step=10,
        disabled=(n_unique_pub == 0),
        help="Seules les IPs publiques (non RFC1918) sont géolocalisables. Les N plus fréquentes sont interrogées via ip-api.com.",
    )
    st.caption(
        f"**{n_unique_pub}** IPs publiques uniques dans la sélection · 100 IP/requête · [ip-api.com](http://ip-api.com)"
    )
with geo_col2:
    st.metric("IPs en cache", f"{len(st.session_state.geo_cache):,}")
with geo_col3:
    st.markdown("<br>", unsafe_allow_html=True)
    run_geo = st.button("🔍 Géolocaliser & Afficher la carte", disabled=(n_unique_pub == 0))

# ═══════════════════════════════════════════════════════════════
# LOGIQUE GÉOLOCALISATION
# ═══════════════════════════════════════════════════════════════
if run_geo:
    if n_unique_pub == 0:
        st.error("❌ Aucune IP publique dans la sélection — la carte nécessite des IPs publiques (non RFC1918) pour la géolocalisation.")
        st.stop()
    # Top N IPs publiques par fréquence (déjà calculé, filtré private)
    all_ips = _pub_freq.head(geo_n).index.tolist()

    pbar = st.progress(0, text="🌐 Interrogation de ip-api.com…")
    with st.spinner(""):
        geo = geolocate_ips(all_ips)
    st.session_state.geo_cache = geo
    pbar.progress(40, text="🔗 Construction des arcs et flèches…")

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
                "angle": arrow_angle(sg["lat"], sg["lon"], dg["lat"], dg["lon"]),
                "color": sc[:3] + [240],
                "size": 18,
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

    pbar.progress(70, text="📊 Construction des statistiques IP…")

    # Table IP enrichie (pour distribution géo + scatter)
    df_str = df.assign(
        _src=df["ip_src"].astype(str),
        _dst=df["ip_dst"].astype(str),
    )
    src_set = set(df_str["_src"])
    dst_set = set(df_str["_dst"])
    ip_rows = []
    for ip, data in geo.items():
        in_src = ip in src_set
        in_dst = ip in dst_set
        role = (
            "Source & Dest."
            if (in_src and in_dst)
            else ("Source" if in_src else "Destination")
        )
        ip_sub = df_str[(df_str["_src"] == ip) | (df_str["_dst"] == ip)]
        ip_rows.append(
            {
                "IP": ip,
                "Rôle": role,
                "Pays": data["country"],
                "Ville": data["city"],
                "Lat": data["lat"],
                "Lon": data["lon"],
                "Nb Source": int((df_str["_src"] == ip).sum()) if in_src else 0,
                "Nb Dest.": int((df_str["_dst"] == ip).sum()) if in_dst else 0,
                "% Bloqué": round(blocked_ratio(ip_sub), 1),
            }
        )

    arc_df_b = pd.DataFrame(arcs)
    arrow_df_b = pd.DataFrame(arrows)
    scat_df_b = pd.DataFrame(list(scatter.values()))

    st.session_state.arc_df = arc_df_b
    st.session_state.arrow_df = arrow_df_b
    st.session_state.scatter_df = scat_df_b
    st.session_state.geo_count = len(geo)
    st.session_state.flow_log = flow_log

    if ip_rows:
        ip_geo = pd.DataFrame(ip_rows)
        ip_geo["Volume total"] = ip_geo["Nb Source"] + ip_geo["Nb Dest."]
        st.session_state.ip_geo_df = ip_geo

    st.session_state.top_src_df = (
        df.groupby(["ip_src", "action"])
        .size()
        .reset_index(name="Nb connexions")
        .sort_values("Nb connexions", ascending=False)
        .head(15)
    )
    if not arc_df_b.empty:
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
        st.session_state.country_src = arc_df_b["src_country"].value_counts().head(10)
        st.session_state.country_dst = arc_df_b["dst_country"].value_counts().head(10)

    pbar.progress(100, text="✅ Analyse terminée !")
    time.sleep(0.4)
    pbar.empty()
    st.success(
        f"✅ **{len(geo)} IPs géolocalisées** — **{len(arcs)} connexions** — **{len(arrows)} flèches**"
    )

# ═══════════════════════════════════════════════════════════════
# CARTE + FLUX EN DIRECT
# ═══════════════════════════════════════════════════════════════
map_col, feed_col = st.columns([3, 1], gap="large")

with map_col:
    st.markdown(
        "<div class='section-hd'>🗺 Carte des flux — Source ──▶ Destination</div>",
        unsafe_allow_html=True,
    )

    # Contrôles inline
    mc1, mc2, mc3, mc4 = st.columns([3, 3, 2, 2])
    with mc1:
        map_style = st.selectbox(
            "🗺 Style de carte",
            ["Voyager (colorée)", "Positron (claire)", "Dark Matter (sombre)"],
            index=0,
        )
    with mc2:
        arc_width = st.slider("⚡ Épaisseur des arcs", 1, 10, 3, key="arc_w_inline")
    with mc3:
        map_pitch = st.slider("🌍 Inclinaison (°)", 0, 55, 30, key="pitch_inline")
    with mc4:
        arrow_size = st.slider("▶ Taille flèches", 10, 35, 18, key="arrow_inline")

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
            arrow_df_display = arrow_df_s.copy()
            arrow_df_display["size"] = arrow_size
            layers.append(
                pdk.Layer(
                    "TextLayer",
                    data=arrow_df_display,
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
              <div style='font-family:Inter,sans-serif;font-size:12px;
                          background:#0f172a;border:1px solid #334155;
                          border-radius:10px;padding:12px 16px;
                          min-width:220px;box-shadow:0 8px 24px rgba(0,0,0,.5);'>
                <div style='font-weight:700;font-size:13px;margin-bottom:8px;
                            padding-bottom:6px;border-bottom:1px solid #1e293b;
                            color:#818cf8;'>{action}</div>
                <div style='color:#818cf8;font-weight:600;
                            font-family:JetBrains Mono,monospace;font-size:11px;'>⬆ {src_ip}</div>
                <div style='color:#64748b;font-size:10px;margin-bottom:6px;'>
                  📍 {src_city}, {src_country}</div>
                <div style='color:#f472b6;font-weight:600;
                            font-family:JetBrains Mono,monospace;font-size:11px;'>⬇ {dst_ip}</div>
                <div style='color:#64748b;font-size:10px;margin-bottom:6px;'>
                  🎯 {dst_city}, {dst_country}</div>
                <div style='color:#475569;font-size:10px;
                            border-top:1px solid #1e293b;padding-top:4px;'>
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

        # Légende riche
        action_counts = arc_df_s["action"].value_counts().to_dict()
        legend_defs = {
            "PERMIT": ("#2ECC71", "Trafic autorisé"),
            "DENY": ("#E74C3C", "Trafic bloqué"),
        }
        action_legend_html = ""
        for act_name, (hex_c, _label) in legend_defs.items():
            cnt = action_counts.get(act_name, 0)
            if cnt > 0:
                action_legend_html += f"""
                <div style='display:flex;align-items:center;gap:8px;'>
                  <div style='width:32px;height:4px;background:{hex_c};border-radius:2px;
                              flex-shrink:0;box-shadow:0 0 8px {hex_c}88;'></div>
                  <span style='color:#cbd5e1;font-size:.75rem;'>
                    <b style='color:{hex_c};'>{act_name}</b>
                    <span style='color:#475569;'>&nbsp;({cnt} flux)</span>
                  </span>
                </div>"""

        n_pts = len(scatter_df_s) if scatter_df_s is not None else 0
        st.markdown(
            f"""
        <div style="
            background:linear-gradient(135deg,#0f172a,#1e293b);
            border:1px solid #334155;border-radius:14px;
            padding:16px 24px;margin-top:12px;
            box-shadow:0 4px 20px rgba(0,0,0,.3);
            display:flex;flex-wrap:wrap;gap:24px;align-items:center;
        ">
          <div style='color:#475569;font-size:.65rem;font-weight:700;
                      letter-spacing:1.5px;text-transform:uppercase;'>LÉGENDE</div>
          <div style='width:1px;height:40px;background:#1e293b;'></div>
          <div>
            <div style='color:#64748b;font-size:.62rem;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:8px;'>Couleur des arcs = Action</div>
            <div style='display:flex;flex-wrap:wrap;gap:12px;'>{action_legend_html}</div>
          </div>
          <div style='width:1px;height:40px;background:#1e293b;'></div>
          <div>
            <div style='color:#64748b;font-size:.62rem;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:8px;'>Points IP</div>
            <div style='display:flex;flex-direction:column;gap:6px;'>
              <div style='display:flex;align-items:center;gap:8px;'>
                <div style='width:12px;height:12px;border-radius:50%;background:#ef4444;
                            box-shadow:0 0 8px #ef444488;flex-shrink:0;'></div>
                <span style='color:#94a3b8;font-size:.75rem;'>Source (DENY)</span>
              </div>
              <div style='display:flex;align-items:center;gap:8px;'>
                <div style='width:12px;height:12px;border-radius:50%;background:#10b981;
                            box-shadow:0 0 8px #10b98188;flex-shrink:0;'></div>
                <span style='color:#94a3b8;font-size:.75rem;'>Source (PERMIT)</span>
              </div>
            </div>
          </div>
          <div style='width:1px;height:40px;background:#1e293b;'></div>
          <div>
            <div style='color:#64748b;font-size:.62rem;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:8px;'>Épaisseur des arcs</div>
            <div style='display:flex;align-items:center;gap:6px;'>
              <div style='width:20px;height:2px;background:#94a3b8;border-radius:1px;'></div>
              <span style='color:#64748b;font-size:.72rem;'>Faible</span>
              <div style='width:20px;height:6px;background:#94a3b8;border-radius:3px;'></div>
              <span style='color:#94a3b8;font-size:.72rem;'>Fort volume</span>
            </div>
          </div>
          <div style='width:1px;height:40px;background:#1e293b;'></div>
          <div>
            <div style='color:#64748b;font-size:.62rem;letter-spacing:1px;
                        text-transform:uppercase;margin-bottom:8px;'>Direction</div>
            <div style='color:#94a3b8;font-size:.8rem;'>
              🔴 Source <span style='color:#475569;'>──▶</span> 🟢 Destination
            </div>
          </div>
          <div style='margin-left:auto;text-align:right;'>
            <div style='background:rgba(99,102,241,.1);border:1px solid rgba(99,102,241,.25);
                        border-radius:10px;padding:8px 14px;'>
              <div style='color:#818cf8;font-size:1.1rem;font-weight:800;'>{len(arc_df_s)}</div>
              <div style='color:#475569;font-size:.62rem;text-transform:uppercase;
                          letter-spacing:1px;'>arcs</div>
            </div>
          </div>
          <div>
            <div style='background:rgba(236,72,153,.08);border:1px solid rgba(236,72,153,.2);
                        border-radius:10px;padding:8px 14px;'>
              <div style='color:#f472b6;font-size:1.1rem;font-weight:800;'>{n_pts}</div>
              <div style='color:#475569;font-size:.62rem;text-transform:uppercase;
                          letter-spacing:1px;'>points IP</div>
            </div>
          </div>
        </div>
        """,
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
            Définissez le nombre d'IPs et cliquez sur
            <b>🔍 Géolocaliser & Afficher la carte</b></div>
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
          <span style='background:rgba(255,60,110,.15);border:1px solid #ff3c6e;color:#ff3c6e;
            padding:4px 12px;border-radius:20px;font-size:.78rem;font-weight:700;'>
            🚫 DENY {deny_n}</span>
          <span style='background:rgba(0,255,157,.12);border:1px solid #00ff9d;color:#00ff9d;
            padding:4px 12px;border-radius:20px;font-size:.78rem;font-weight:700;'>
            ✅ PERMIT {permit_n}</span></div>""",
            unsafe_allow_html=True,
        )
        html = "<div style='max-height:480px;overflow-y:auto;padding-right:4px;'>"
        for f in logs:
            is_deny = f["action"] == "DENY"
            accent = "#ff3c6e" if is_deny else "#00ff9d"
            border = "rgba(255,60,110,.35)" if is_deny else "rgba(0,255,157,.25)"
            bg = "rgba(255,60,110,.07)" if is_deny else "rgba(0,255,157,.05)"
            ico = "🔴" if is_deny else "🟢"
            html += f"""<div style='background:{bg};border:1px solid {border};
              border-radius:10px;padding:10px 12px;margin-bottom:8px;font-size:.78rem;'>
              <span style='color:{accent};font-weight:700;'>{ico} {f['action']}</span><br>
              <span style='color:#00d4ff;font-weight:600;'>{f['src_ip']}</span>
              <span style='color:#475569;'> ──▶ </span>
              <span style='color:#a259ff;font-weight:600;'>{f['dst_ip']}</span><br>
              <span style='color:#94a3b8;'>📍 {f['src_city'][:13]}, {f['src_country'][:11]}</span><br>
              <span style='color:#94a3b8;'>🎯 {f['dst_city'][:13]}, {f['dst_country'][:11]}</span><br>
              <span style='color:#475569;font-size:.65rem;'>{f['protocol']} :{f['port']}</span>
            </div>"""
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(
            """<div style='background:rgba(15,23,42,.6);border:2px dashed rgba(0,212,255,.25);
            border-radius:12px;text-align:center;padding:40px 16px;'>
          <div style='font-size:2rem;margin-bottom:8px;'>📡</div>
          <div style='color:#94a3b8;font-size:.8rem;font-weight:500;'>En attente…</div>
        </div>""",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════
# ANALYTICS — Top IPs + Connexions détaillées
# ═══════════════════════════════════════════════════════════════
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
            st.session_state.top_src_df.style.applymap(color_action, subset=["action"]),
            use_container_width=True,
            hide_index=True,
            height=320,
        )
    with a2:
        st.markdown(
            "<div class='section-hd'>📋 Connexions géolocalisées</div>",
            unsafe_allow_html=True,
        )
        if (
            st.session_state.detail_df is not None
            and not st.session_state.detail_df.empty
        ):
            st.dataframe(
                st.session_state.detail_df.style.applymap(
                    color_action, subset=["Action"]
                ),
                use_container_width=True,
                hide_index=True,
                height=320,
            )
        else:
            st.info("Aucune connexion géolocalisée — IPs privées ou API indisponible.")

# ═══════════════════════════════════════════════════════════════
# ANALYTICS — Distribution géographique + Scatter
# ═══════════════════════════════════════════════════════════════
if st.session_state.ip_geo_df is not None:
    geo_df = st.session_state.ip_geo_df.copy()

    st.markdown("---")
    st.markdown(
        "<div class='section-hd'>🌍 Distribution géographique</div>",
        unsafe_allow_html=True,
    )

    col_c1, col_c2, col_c3 = st.columns([2, 2, 3])
    with col_c1:
        src_geo = geo_df[geo_df["Rôle"].isin(["Source", "Source & Dest."])]
        if not src_geo.empty:
            cs = src_geo["Pays"].value_counts().reset_index()
            cs.columns = ["Pays", "count"]
            fig = pie_chart(cs, names="Pays", title="🌐 Pays — Sources")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_c2:
        dst_geo = geo_df[geo_df["Rôle"].isin(["Destination", "Source & Dest."])]
        if not dst_geo.empty:
            cd = dst_geo["Pays"].value_counts().reset_index()
            cd.columns = ["Pays", "count"]
            fig = pie_chart(cd, names="Pays", title="🎯 Pays — Destinations")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_c3:
        fig = px.scatter(
            geo_df,
            x="Volume total",
            y="% Bloqué",
            size="Volume total",
            color="Pays",
            hover_name="IP",
            hover_data=["Ville", "Rôle", "Nb Source", "Nb Dest."],
            labels={"Volume total": "Volume", "% Bloqué": "Taux bloqué (%)"},
            title="📊 Volume vs. Taux de blocage",
            size_max=50,
            template="plotly_dark",
        )
        fig.update_layout(
            plot_bgcolor="#0f172a",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tableau IP enrichi
    st.markdown("---")
    st.markdown(
        "<div class='section-hd'>📋 Tableau IP enrichi</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        geo_df.drop(columns=["Lat", "Lon"]),
        use_container_width=True,
        hide_index=True,
        height=240,
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
