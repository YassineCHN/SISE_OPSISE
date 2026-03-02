import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import plotly.express as px
from modules.preprocessing import load_data
from modules.stats import (
    protocol_action_crosstab,
    top_n,
    port_category_distribution,
    traffic_by_hour,
    traffic_by_weekday,
)
from modules.charts import heatmap, bar_chart, pie_chart
from modules.components.filters import render_sidebar_filters
from config import TOP_N_DEFAULT

st.set_page_config(
    page_title="Statistiques",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def get_data():
    return load_data()


df_full = get_data()
df, params = render_sidebar_filters(df_full)

st.title("📈 Statistiques avancées")

if df.empty:
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()

# ── Crosstab Protocole × Action ───────────────────────────────────────────
st.subheader("Protocole × Action")
ct = protocol_action_crosstab(df)
fig = heatmap(ct)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Ligne 1 : Top règles + Distribution ports ─────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader(f"Top {TOP_N_DEFAULT} règles déclenchées")
    rules_df = top_n(df, "rule_id", TOP_N_DEFAULT)
    rules_df["rule_id"] = "Règle " + rules_df["rule_id"].astype(str)
    fig = bar_chart(rules_df, x="rule_id")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Distribution des plages de ports")
    port_cat_df = port_category_distribution(df)
    fig = pie_chart(port_cat_df, names="Catégorie")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Ligne 2 : Trafic par heure + par jour ────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Trafic par heure de la journée")
    hourly = traffic_by_hour(df)
    fig = px.bar(
        hourly, x="hour", y="count",
        labels={"hour": "Heure", "count": "Événements"},
        color_discrete_sequence=["#3498DB"],
    )
    fig.update_xaxes(dtick=1, title="Heure")
    st.plotly_chart(fig, use_container_width=True)

with col_d:
    st.subheader("Trafic par jour de la semaine")
    weekly = traffic_by_weekday(df)
    fig = px.bar(
        weekly, x="jour", y="count",
        labels={"jour": "Jour", "count": "Événements"},
        color_discrete_sequence=["#9B59B6"],
    )
    st.plotly_chart(fig, use_container_width=True)
