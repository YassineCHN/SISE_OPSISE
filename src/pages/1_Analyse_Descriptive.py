import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from modules.preprocessing import load_data
from modules.stats import (
    action_distribution,
    top_n,
    traffic_by_period,
    unique_counts,
    blocked_ratio,
)
from modules.charts import pie_chart, bar_chart, area_chart
from modules.components.filters import render_sidebar_filters
from config import ACTION_COLORS, TOP_N_DEFAULT

st.set_page_config(
    page_title="Analyse Descriptive",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def get_data():
    return load_data()


df_full = get_data()
df, params = render_sidebar_filters(df_full)

st.title("📊 Analyse Descriptive")

if df.empty:
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────
ucounts = unique_counts(df)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Entrées totales",       f"{len(df):,}")
col2.metric("IP sources uniques",    f"{ucounts['ip_src']:,}")
col3.metric("IP destinations uniq.", f"{ucounts['ip_dst']:,}")
col4.metric("Protocoles",            f"{ucounts['protocol']:,}")
col5.metric("Trafic bloqué",         f"{blocked_ratio(df):.1f} %")

st.markdown("---")

# ── Ligne 1 : Action + Protocoles ─────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribution des actions")
    act_df = action_distribution(df)
    fig = pie_chart(act_df, names="action", color_map=ACTION_COLORS)
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader(f"Top {TOP_N_DEFAULT} protocoles")
    proto_df = top_n(df, "protocol", TOP_N_DEFAULT)
    fig = bar_chart(proto_df, x="protocol")
    st.plotly_chart(fig, use_container_width=True)

# ── Timeline ──────────────────────────────────────────────────────────────
st.subheader("Volume de trafic dans le temps")
freq_map = {"Heure": "h", "Jour": "D", "Semaine": "W"}
freq_label = st.radio("Granularité", list(freq_map.keys()), horizontal=True)
timeline_df = traffic_by_period(df, freq=freq_map[freq_label])
fig = area_chart(timeline_df, x="datetime", y="count",
                 title=f"Trafic par {freq_label.lower()}")
st.plotly_chart(fig, use_container_width=True)

# ── Top ports ─────────────────────────────────────────────────────────────
st.subheader(f"Top {TOP_N_DEFAULT} ports destination")
ports_df = top_n(df, "port_dst", TOP_N_DEFAULT)
ports_df["port_dst"] = ports_df["port_dst"].astype(str)
fig = bar_chart(ports_df, x="port_dst")
st.plotly_chart(fig, use_container_width=True)
