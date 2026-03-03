"""
Composants UI réutilisables — Sentinel Theme.

Usage:
    from components.ui import section_hd, fw_card, stat_block, kpi_chip, badge
"""
import streamlit as st


# ── En-têtes de section ──────────────────────────────────────────────────────

def section_hd(text: str) -> None:
    """Affiche un titre de section avec l'accent cyan."""
    st.markdown(f"<div class='section-hd'>{text}</div>", unsafe_allow_html=True)


# ── Cards ────────────────────────────────────────────────────────────────────

def fw_card(content: str, variant: str = "") -> None:
    """
    Card générique avec bordure optionnelle colorée.

    variant : '' | 'accent' (cyan) | 'danger' (rouge) | 'info' (violet) | 'warn' (orange)
    """
    cls = f"fw-card fw-card-{variant}" if variant else "fw-card"
    st.markdown(f"<div class='{cls}'>{content}</div>", unsafe_allow_html=True)


def stat_block(value: str, label: str) -> None:
    """Grande valeur centrée avec libellé sous-jacent (style .stat-block)."""
    st.markdown(
        f"""<div class='stat-block'>
          <div class='val'>{value}</div>
          <div class='lbl'>{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def story_banner(html_content: str) -> None:
    """Bannière narrative avec bordure gauche violette."""
    st.markdown(
        f"<div class='story-banner'>{html_content}</div>",
        unsafe_allow_html=True,
    )


def report_box(text: str) -> None:
    """Zone de rapport IA avec bordure gauche violette."""
    text_html = text.replace("\n", "<br>")
    st.markdown(
        f"<div class='report-box'>{text_html}</div>",
        unsafe_allow_html=True,
    )


def ai_panel(header: str, content_fn) -> None:
    """
    Panneau IA avec en-tête violet.

    content_fn : callable qui affiche le contenu à l'intérieur du panneau.
    """
    st.markdown(
        f"""<div class='ai-panel'>
          <div class='ai-panel-hd'>⬡ {header}</div>""",
        unsafe_allow_html=True,
    )
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


# ── Badges & chips ───────────────────────────────────────────────────────────

def badge(text: str, variant: str = "info") -> None:
    """
    Badge inline coloré.

    variant : 'info' (cyan) | 'ok' (vert) | 'deny' (rouge) | 'warn' (orange)
    """
    st.markdown(
        f"<span class='kpi-chip {variant}'>{text}</span>",
        unsafe_allow_html=True,
    )


def profile_badge(text: str, variant: str = "normal") -> None:
    """
    Badge de profil comportemental.

    variant : 'normal' | 'scan' | 'ddos' | 'nocturne' | 'blocked' | 'targeted'
    """
    st.markdown(
        f"<span class='profile-badge pb-{variant}'>{text}</span>",
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple]) -> None:
    """
    Ligne de KPI chips.

    items : liste de (label, variant) — variant : 'info' | 'ok' | 'deny' | 'warn'

    Exemple : kpi_row([("1 234 DENY", "deny"), ("5 678 PERMIT", "ok")])
    """
    chips = "".join(
        f"<span class='kpi-chip {v}'>{lbl}</span>" for lbl, v in items
    )
    st.markdown(f"<div class='kpi-row'>{chips}</div>", unsafe_allow_html=True)
