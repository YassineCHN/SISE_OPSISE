import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from modules.preprocessing import load_data
from modules.components.filters import render_sidebar_filters
from config import COLUMN_LABELS

st.set_page_config(
    page_title="Table de données",
    page_icon="📋",
    layout="wide",
)


@st.cache_data
def get_data():
    return load_data()


df_full = get_data()
df, params = render_sidebar_filters(df_full)

st.title("📋 Table de données")

if df.empty:
    st.warning("Aucune donnée pour les filtres sélectionnés.")
    st.stop()

# ── Barre de recherche ────────────────────────────────────────────────────
search = st.text_input("Recherche rapide (IP, protocole, action…)", "")

display_df = df.rename(columns=COLUMN_LABELS)

if search:
    mask = display_df.apply(
        lambda col: col.astype(str).str.contains(search, case=False, na=False)
    )
    display_df = display_df[mask.any(axis=1)]

# ── Sélection de colonnes ─────────────────────────────────────────────────
all_cols = list(display_df.columns)
selected_cols = st.multiselect("Colonnes à afficher", all_cols, default=all_cols)
display_df = display_df[selected_cols]

st.caption(f"{len(display_df):,} lignes affichées")
st.dataframe(display_df, use_container_width=True, height=600)

# ── Export CSV ────────────────────────────────────────────────────────────
csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Télécharger CSV",
    data=csv,
    file_name="logs_firewall_filtres.csv",
    mime="text/csv",
)
