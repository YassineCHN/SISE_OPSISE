import streamlit as st

from components.top_nav import render_top_nav
from app_config import APP_ICON, APP_TITLE, LAYOUT
from modules.preprocessing import load_data
from modules.stats import blocked_ratio, unique_counts

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
)

render_top_nav("home")


@st.cache_data
def get_data():
    return load_data()


df = get_data()
ucounts = unique_counts(df)

st.title(f"{APP_ICON} {APP_TITLE}")
st.markdown(
    "Tableau de bord d'analyse des logs firewall. "
    "La navigation des pages est maintenant en haut."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Entrees totales", f"{len(df):,}")
c2.metric("IP sources uniques", f"{ucounts['ip_src']:,}")
c3.metric("IP destinations uniq.", f"{ucounts['ip_dst']:,}")
c4.metric("Protocoles", f"{ucounts['protocol']:,}")
c5.metric("Trafic bloque", f"{blocked_ratio(df):.1f} %")

st.markdown("---")
st.subheader("Pages disponibles")

pages = [
    ("[VIZ]", "Visualisation", "4 onglets: descriptive, table, IP, statistiques"),
    ("[MAP]", "Carte", "Cartographie geographique des flux"),
    ("[SEN]", "Sentinel", "Analyse avancee ML + IA Mistral"),
]

cols = st.columns(len(pages))
for col, (tag, name, desc) in zip(cols, pages):
    with col:
        st.markdown(f"### {tag} {name}")
        st.caption(desc)
