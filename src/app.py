import streamlit as st

from components.top_nav import render_top_nav
from components.sentinel_theme import apply_sentinel_theme
from app_config import APP_ICON, APP_TITLE, LAYOUT
from modules.preprocessing import load_data
from modules.stats import blocked_ratio, unique_counts

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
)

render_top_nav("home")
apply_sentinel_theme()


@st.cache_data
def get_data():
    return load_data()


df = get_data()
ucounts = unique_counts(df)

st.title(f"{APP_ICON} {APP_TITLE}")
st.markdown(
    "Tableau de bord d'analyse des logs firewall. "
    "Naviguez entre les pages via le menu en haut."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📦 Flux total",        f"{len(df):,}")
c2.metric("🌐 IP sources",        f"{ucounts['ip_src']:,}")
c3.metric("🎯 IP destinations",   f"{ucounts['ip_dst']:,}")
c4.metric("🔌 Protocoles",        f"{ucounts['protocol']:,}")
c5.metric("🚫 Trafic bloqué",     f"{blocked_ratio(df):.1f} %")

st.markdown("---")
st.subheader("📋 Pages disponibles")

pages = [
    ("📊", "Visualisation", "Analyse descriptive, DataTable, top IPs, statistiques"),
    ("🌍", "Carte",         "Cartographie géographique des flux réseau"),
    ("🛡️", "Sentinel",     "Détection d'anomalies ML + analyse temporelle"),
]

cols = st.columns(len(pages))
for col, (icon, name, desc) in zip(cols, pages):
    with col:
        st.markdown(f"### {icon} {name}")
        st.caption(desc)
