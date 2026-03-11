import os

import streamlit as st

from modules.preprocessing import get_data_source_info

_LOCAL_TABLES = ["original_data", "generated_data"]
_LOCAL_LABELS = {
    "original_data":  "original_data (parquet)",
    "generated_data": "generated_data (csv)",
}


def render_inline_table_selector() -> str | None:
    """
    Render the dataset selector inline (main content area, not sidebar).
    Uses the same session_state keys as render_motherduck_table_selector()
    so the selection persists across all pages.
    """
    info = get_data_source_info()
    configured = info.get("configured_source", "parquet")

    if configured == "motherduck":
        options = info.get("available_tables", [])
        if not options:
            return info.get("motherduck_table") or None
        default_table = info.get("motherduck_table")
        default_idx = options.index(default_table) if default_table in options else 0
        selected = st.selectbox(
            "🧩 Table MotherDuck",
            options,
            index=default_idx,
            key="motherduck_table_selected",
        )
    else:
        labels = [_LOCAL_LABELS[t] for t in _LOCAL_TABLES]
        selected_label = st.selectbox(
            "📂 Source de données",
            labels,
            index=0,
            key="local_table_selected",
        )
        selected = _LOCAL_TABLES[labels.index(selected_label)]

    return selected


def render_motherduck_table_selector() -> str | None:
    """
    Render a sidebar dataset selector.
    - MotherDuck mode : shows available MotherDuck tables.
    - Local mode      : shows the two local files as selectable datasets.
    Returns the selected table/dataset name.
    """
    info = get_data_source_info()
    configured = info.get("configured_source", "parquet")

    if configured == "motherduck":
        options = info.get("available_tables", [])
        if not options:
            return info.get("motherduck_table") or None
        default_table = info.get("motherduck_table")
        default_idx = options.index(default_table) if default_table in options else 0
        st.sidebar.markdown("### 🧩 Table MotherDuck")
        selected = st.sidebar.selectbox(
            "Jeu de données",
            options,
            index=default_idx,
            key="motherduck_table_selected",
        )
        row_limit = int(os.getenv("MOTHERDUCK_ROW_LIMIT", "0"))
        if row_limit > 0:
            st.sidebar.caption(f"⚠️ Données limitées à {row_limit:,} lignes.")
    else:
        st.sidebar.markdown("### 🗂️ Jeu de données local")
        labels = [_LOCAL_LABELS[t] for t in _LOCAL_TABLES]
        selected_label = st.sidebar.selectbox(
            "Fichier source",
            labels,
            index=0,
            key="local_table_selected",
        )
        selected = _LOCAL_TABLES[labels.index(selected_label)]

    st.sidebar.markdown("---")
    return selected
