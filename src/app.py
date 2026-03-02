import streamlit as st
from modules.preprocessing import load_data
from modules.stats import unique_counts, blocked_ratio, action_distribution
from config import APP_TITLE, APP_ICON, LAYOUT

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
)


@st.cache_data
def get_data():
    return load_data()


df = get_data()

st.title(f"{APP_ICON} {APP_TITLE}")
st.markdown(
    "Tableau de bord d'analyse des logs firewall. "
    "Utilisez le menu de gauche pour naviguer entre les pages."
)

st.markdown("---")

# ── KPIs globaux ─────────────────────────────────────────────────────────
ucounts = unique_counts(df)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Entrées totales",       f"{len(df):,}")
col2.metric("IP sources uniques",    f"{ucounts['ip_src']:,}")
col3.metric("IP destinations uniq.", f"{ucounts['ip_dst']:,}")
col4.metric("Protocoles",            f"{ucounts['protocol']:,}")
col5.metric("Trafic bloqué",         f"{blocked_ratio(df):.1f} %")

st.markdown("---")

# ── Résumé des pages ──────────────────────────────────────────────────────
st.subheader("Pages disponibles")

pages = [
    ("📊", "Analyse Descriptive",
     "Vue d'ensemble : distributions des actions, protocoles, ports et timeline du trafic."),
    ("📋", "Table de données",
     "Exploration tabulaire avec recherche, sélection de colonnes et export CSV."),
    ("🌐", "Visualisation IP",
     "Top IP sources/destinations, actions par IP et distribution des interfaces réseau."),
    ("📈", "Statistiques",
     "Statistiques avancées : crosstab protocole×action, distribution horaire/hebdomadaire, règles."),
]

cols = st.columns(len(pages))
for col, (icon, name, desc) in zip(cols, pages):
    with col:
        st.markdown(f"### {icon} {name}")
        st.caption(desc)
