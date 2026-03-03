import streamlit as st
import pandas as pd
from modules.preprocessing import (
    filter_by_date,
    filter_by_protocol,
    filter_by_action,
    filter_by_port_range,
)
from app_config import PORT_RANGES


def render_sidebar_filters(df: pd.DataFrame) -> tuple:
    """
    Affiche les filtres interactifs dans la sidebar Streamlit.

    Retourne
    --------
    filtered : pd.DataFrame
        Sous-ensemble de df selon les critères choisis.
    params : dict
        Dictionnaire des valeurs sélectionnées (start_date, end_date,
        protocols, actions, port_range).
    """
    st.sidebar.markdown("### ⚙️ Filtres")
    st.sidebar.markdown("---")

    # ── Plage de dates ─────────────────────────────────────────────────────
    min_date = df["datetime"].min().date()
    max_date = df["datetime"].max().date()

    date_input = st.sidebar.date_input(
        "📅 Période",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_input, (list, tuple)) and len(date_input) == 2:
        start_date, end_date = date_input
    else:
        start_date = end_date = date_input

    # ── Protocoles ─────────────────────────────────────────────────────────
    all_protocols = sorted(df["protocol"].dropna().unique().tolist())
    protocols = st.sidebar.multiselect("🔌 Protocoles", all_protocols)

    # ── Actions ────────────────────────────────────────────────────────────
    all_actions = sorted(df["action"].dropna().unique().tolist())
    actions = st.sidebar.multiselect("🎯 Actions", all_actions)

    # ── Plage de ports ─────────────────────────────────────────────────────
    port_label = st.sidebar.selectbox("🔢 Plage de ports", list(PORT_RANGES.keys()))
    port_range = PORT_RANGES[port_label]

    # ── Application des filtres ────────────────────────────────────────────
    filtered = filter_by_date(df, start_date, end_date)
    filtered = filter_by_protocol(filtered, protocols)
    filtered = filter_by_action(filtered, actions)
    filtered = filter_by_port_range(filtered, port_range)

    st.sidebar.markdown("---")
    st.sidebar.metric("📊 Lignes sélectionnées", f"{len(filtered):,}")
    st.sidebar.caption(f"📦 Total dataset : {len(df):,} lignes")

    params = {
        "start_date": start_date,
        "end_date":   end_date,
        "protocols":  protocols,
        "actions":    actions,
        "port_range": port_range,
    }

    return filtered, params
