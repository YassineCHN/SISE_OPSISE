import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.express as px
from modules.preprocessing import load_data
from modules.stats import top_n
from modules.charts import bar_chart, pie_chart
from modules.components.filters import render_sidebar_filters
from config import ACTION_COLORS, TOP_N_DEFAULT

st.set_page_config(
    page_title="Visualisation IP",
    page_icon="🌐",
    layout="wide",
)


@st.cache_data
def get_data():
    return load_data()


df_full = get_data()
df, params = render_sidebar_filters(df_full)

st.title("🌐 Visualisation IP")

if df.empty:
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()

n = st.slider("Nombre de top IPs à afficher", 5, 50, TOP_N_DEFAULT)

# ── Top IPs source / destination ──────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader(f"Top {n} IP sources")
    src_df = top_n(df, "ip_src", n)
    fig = bar_chart(src_df, x="ip_src", horizontal=True)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader(f"Top {n} IP destinations")
    dst_df = top_n(df, "ip_dst", n)
    fig = bar_chart(dst_df, x="ip_dst", horizontal=True)
    st.plotly_chart(fig, use_container_width=True)

# ── Actions par top IP source ─────────────────────────────────────────────
st.subheader("Distribution des actions pour les top IP sources")
top_src_ips = top_n(df, "ip_src", 10)["ip_src"].tolist()
df_top = df[df["ip_src"].isin(top_src_ips)]

action_by_ip = (
    df_top.groupby(["ip_src", "action"])
    .size()
    .reset_index(name="count")
)

fig = px.bar(
    action_by_ip,
    x="ip_src", y="count", color="action",
    barmode="stack",
    color_discrete_map=ACTION_COLORS,
    labels={"ip_src": "IP Source", "count": "Événements", "action": "Action"},
)
st.plotly_chart(fig, use_container_width=True)

# ── Interfaces réseau ─────────────────────────────────────────────────────
st.subheader("Distribution des interfaces réseau")
col_c, col_d = st.columns(2)

with col_c:
    iface_in = top_n(df, "interface_in", 10)
    fig = pie_chart(iface_in, names="interface_in", title="Interface entrée")
    st.plotly_chart(fig, use_container_width=True)

with col_d:
    iface_out = top_n(df, "interface_out", 10)
    fig = pie_chart(iface_out, names="interface_out", title="Interface sortie")
    st.plotly_chart(fig, use_container_width=True)
