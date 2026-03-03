import streamlit as st


def render_top_nav(active: str) -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    items = [
        ("home",     "app.py",                       "🏠 Accueil"),
        ("viz",      "pages/1_Visualisation.py",     "📊 Visualisation"),
        ("map",      "pages/5_Carte.py",              "🌍 Carte"),
        ("sentinel", "pages/6_Sentinel_Avance.py",   "🛡️ Sentinel"),
    ]

    cols = st.columns(len(items))
    for col, (key, page, label) in zip(cols, items):
        with col:
            st.page_link(
                page,
                label=(f"[{label}]" if key == active else label),
                disabled=(key == active),
                use_container_width=True,
            )

    st.markdown("---")
