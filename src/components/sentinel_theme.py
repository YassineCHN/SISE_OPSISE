import streamlit as st


def apply_sentinel_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap');
        :root {
          --bg:#07090f; --bg2:#0d1117; --bg3:#12181f; --border:#1e2a38;
          --accent:#00d4ff; --accent2:#ff3c6e; --accent3:#a259ff; --accent4:#00ff9d;
          --text:#c8d8e8; --text-dim:#4a6072; --text-hi:#e8f4ff;
          --card-glow:0 0 20px rgba(0,212,255,0.08);
        }
        html,body,[class*="css"]{font-family:'Space Mono',monospace!important;background-color:var(--bg)!important;color:var(--text)!important}
        ::-webkit-scrollbar{width:4px}
        ::-webkit-scrollbar-track{background:var(--bg2)}
        ::-webkit-scrollbar-thumb{background:var(--accent);border-radius:4px}
        .main .block-container{background:var(--bg)!important;padding:1.2rem 2rem!important;max-width:1600px}
        [data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border)!important}
        [data-testid="stSidebar"] *{color:var(--text)!important}
        [data-testid="stSidebar"] .stMarkdown h3{color:var(--accent)!important;font-family:'Syne',sans-serif!important;letter-spacing:2px;text-transform:uppercase;font-size:0.7rem}
        [data-testid="stSidebar"] hr{border-color:var(--border)!important}
        [data-testid="metric-container"]{background:var(--bg2)!important;border:1px solid var(--border)!important;border-radius:6px!important;box-shadow:var(--card-glow)!important;padding:1rem 1.2rem!important;position:relative;overflow:hidden}
        [data-testid="metric-container"]::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),var(--accent3))}
        [data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Syne',sans-serif!important;font-size:2rem!important;font-weight:800!important;color:var(--text-hi)!important}
        [data-testid="metric-container"] [data-testid="stMetricLabel"]{font-size:0.62rem!important;color:var(--text-dim)!important;letter-spacing:2px!important;text-transform:uppercase!important}
        .stButton>button{background:rgba(0,0,0,0)!important;color:var(--accent)!important;border:1px solid var(--accent)!important;border-radius:4px!important;font-family:'Space Mono',monospace!important;font-size:0.75rem!important;letter-spacing:1px!important;padding:10px 24px!important;text-transform:uppercase!important;transition:all .2s!important}
        .stButton>button:hover{background:rgba(0,212,255,0.08)!important;box-shadow:0 0 20px rgba(0,212,255,0.25)!important;transform:translateY(-1px)!important}
        [data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:6px!important}
        .stProgress>div>div{background:linear-gradient(90deg,var(--accent),var(--accent3))!important}
        [data-testid="stFileUploader"]{background:var(--bg2)!important;border:1px dashed var(--border)!important;border-radius:6px!important}
        [data-testid="stTabs"] [data-baseweb="tab-list"]{background:var(--bg2)!important;border-bottom:1px solid var(--border)!important;gap:0!important}
        [data-testid="stTabs"] [data-baseweb="tab"]{background:rgba(0,0,0,0)!important;color:var(--text-dim)!important;font-family:'Space Mono',monospace!important;font-size:0.72rem!important;letter-spacing:1px!important;text-transform:uppercase!important;padding:12px 24px!important;border:none!important;border-bottom:2px solid rgba(0,0,0,0)!important;transition:all .2s!important}
        [data-testid="stTabs"] [aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important}
        [data-testid="stAlert"]{border-radius:4px!important;border-left:3px solid var(--accent)!important}
        label{color:var(--text-dim)!important;font-size:0.7rem!important;letter-spacing:1px!important;text-transform:uppercase}
        hr{border-color:var(--border)!important}

        /* ── Cards génériques ───────────────────────────────────── */
        .fw-card{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:20px 22px;margin-bottom:14px;box-shadow:var(--card-glow)}
        .fw-card-accent{border-left:3px solid var(--accent)}
        .fw-card-danger{border-left:3px solid var(--accent2)}
        .fw-card-info  {border-left:3px solid var(--accent3)}
        .fw-card-warn  {border-left:3px solid #ffb800}

        /* ── En-têtes de section ────────────────────────────────── */
        .section-hd{font-family:'Syne',sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:var(--text-dim);border-bottom:1px solid var(--border);padding-bottom:8px;margin:24px 0 16px 0;display:flex;align-items:center;gap:8px}
        .section-hd::before{content:'';display:inline-block;width:16px;height:2px;background:var(--accent)}

        /* ── KPI chips ──────────────────────────────────────────── */
        .kpi-row{display:flex;gap:12px;margin:12px 0;flex-wrap:wrap}
        .kpi-chip{background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:6px 14px;font-size:0.68rem;letter-spacing:1px;display:inline-flex;align-items:center;gap:8px}
        .kpi-chip.deny{border-color:var(--accent2);color:var(--accent2)}
        .kpi-chip.ok  {border-color:var(--accent4);color:var(--accent4)}
        .kpi-chip.info{border-color:var(--accent);color:var(--accent)}
        .kpi-chip.warn{border-color:#ffb800;color:#ffb800}

        /* ── Feed cards ─────────────────────────────────────────── */
        .feed-card{background:var(--bg2);border:1px solid var(--border);border-radius:4px;padding:8px 12px;margin:4px 0;font-size:0.65rem;line-height:1.8;animation:fadeIn .3s ease}
        .feed-card.deny{border-left:2px solid var(--accent2)}
        .feed-card.ok  {border-left:2px solid var(--accent4)}
        @keyframes fadeIn{from{opacity:0;transform:translateX(-4px)}to{opacity:1;transform:translateX(0)}}

        /* ── Story banner ───────────────────────────────────────── */
        .story-banner{background:linear-gradient(135deg,var(--bg2),var(--bg3));border:1px solid var(--border);border-left:3px solid var(--accent3);border-radius:6px;padding:20px 24px;margin:12px 0 24px 0;font-size:0.82rem;line-height:1.8;color:var(--text)}
        .story-banner .highlight{color:var(--accent);font-weight:700}
        .story-banner .danger{color:var(--accent2);font-weight:700}

        /* ── Profile badges ─────────────────────────────────────── */
        .profile-badge{display:inline-block;padding:3px 10px;border-radius:3px;font-size:0.62rem;letter-spacing:1px;font-weight:700;text-transform:uppercase}
        .pb-normal  {background:rgba(0,255,157,0.1);color:var(--accent4);border:1px solid rgba(0,255,157,0.3)}
        .pb-scan    {background:rgba(0,212,255,0.1);color:var(--accent);border:1px solid rgba(0,212,255,0.3)}
        .pb-ddos    {background:rgba(255,60,110,0.1);color:var(--accent2);border:1px solid rgba(255,60,110,0.3)}
        .pb-nocturne{background:rgba(162,89,255,0.1);color:var(--accent3);border:1px solid rgba(162,89,255,0.3)}
        .pb-blocked {background:rgba(255,184,0,0.1);color:#ffb800;border:1px solid rgba(255,184,0,0.3)}
        .pb-targeted{background:rgba(255,60,110,0.15);color:#ff6b6b;border:1px solid rgba(255,100,100,0.4)}

        /* ── Report box ─────────────────────────────────────────── */
        .report-box{background:var(--bg2);border:1px solid var(--border);border-left:3px solid var(--accent3);border-radius:6px;padding:24px 28px;font-size:0.83rem;line-height:1.9;color:var(--text)}
        .report-box h2{color:var(--accent);font-family:'Syne',sans-serif;font-size:0.95rem;margin-top:1.2em;letter-spacing:1px}
        .report-box strong{color:var(--text-hi)}

        /* ── AI panel ───────────────────────────────────────────── */
        .ai-panel{background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--accent3);border-radius:6px;padding:20px 24px;margin-top:24px}
        .ai-panel-hd{font-family:'Syne',sans-serif;font-size:0.7rem;font-weight:700;color:var(--accent3);letter-spacing:3px;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}

        /* ── Stat block ─────────────────────────────────────────── */
        .stat-block{background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:18px 20px;text-align:center}
        .stat-block .val{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:var(--accent2);line-height:1}
        .stat-block .lbl{font-size:0.6rem;color:var(--text-dim);letter-spacing:2px;text-transform:uppercase;margin-top:4px}

        /* ── Divers ─────────────────────────────────────────────── */
        .map-wait{background:var(--bg2);border:1px dashed var(--border);border-radius:6px;text-align:center;padding:80px 20px}
        .ip-selector{background:var(--bg3);border:1px solid var(--border);border-radius:4px;padding:12px 16px;margin:8px 0;cursor:pointer;font-size:0.72rem;transition:border-color .15s}
        </style>
        """,
        unsafe_allow_html=True,
    )
