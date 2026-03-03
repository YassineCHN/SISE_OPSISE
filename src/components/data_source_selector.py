import streamlit as st

from modules.preprocessing import get_data_source_info


def render_motherduck_table_selector() -> str | None:
    """
    Render a sidebar selector when source is MotherDuck.
    Returns selected table name, or None for non-MotherDuck sources.
    """
    info = get_data_source_info()
    configured = info.get("configured_source", "parquet")
    if configured != "motherduck":
        return None

    options = info.get("available_tables", [])
    if not options:
        # No options declared: fallback to configured table from env.
        return info.get("motherduck_table") or None

    default_table = info.get("motherduck_table")
    if default_table in options:
        default_idx = options.index(default_table)
    else:
        default_idx = 0

    st.sidebar.markdown("### 🧩 Table MotherDuck")
    selected = st.sidebar.selectbox(
        "Jeu de donnees",
        options,
        index=default_idx,
        key="motherduck_table_selected",
    )
    st.sidebar.markdown("---")
    return selected
