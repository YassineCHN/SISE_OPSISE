import streamlit as st
import plotly.express as px

from components.top_nav import render_top_nav
from components.sentinel_theme import apply_sentinel_theme
from components.ui import neon_metric
from components.data_source_selector import render_motherduck_table_selector
from app_config import APP_ICON, APP_TITLE, COLUMN_LABELS, LAYOUT
from modules.preprocessing import load_data
from modules.stats import blocked_ratio, unique_counts
from utils.sentinel_utils import port_label

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
)

render_top_nav("home")
apply_sentinel_theme()


@st.cache_data
def get_data(selected_table: str | None):
    return load_data(selected_table=selected_table)


selected_table = render_motherduck_table_selector()
df = get_data(selected_table)

# ── Métriques ─────────────────────────────────────────────────────────────────
_total      = len(df)
_deny       = int((df["action"] == "DENY").sum())
_permit     = int((df["action"] == "PERMIT").sum())
_src        = df["ip_src"].nunique()
_dst        = df["ip_dst"].nunique()
_pct_deny   = (_deny   / _total * 100) if _total else 0
_pct_permit = (_permit / _total * 100) if _total else 0

_pcol       = "protocol_clean" if "protocol_clean" in df.columns else "protocol"
_top_port   = int(df["port_dst"].mode()[0]) if "port_dst" in df.columns else "N/A"
_top_src_ip = df["ip_src"].value_counts().idxmax() if "ip_src" in df.columns else "N/A"
_rules      = df["rule_id"].nunique() if "rule_id" in df.columns else "N/A"

_date_start = df["datetime"].min().strftime("%d %b %Y") if "datetime" in df.columns else "—"
_date_end   = df["datetime"].max().strftime("%d %b %Y") if "datetime" in df.columns else "—"
_days       = (df["datetime"].max() - df["datetime"].min()).days if "datetime" in df.columns else 0

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-wrap{
  background:linear-gradient(135deg,#07090f 0%,#0d1117 40%,#0f1a24 100%);
  border:1px solid #1e2a38;border-radius:10px;
  padding:40px 44px 36px 44px;margin-bottom:28px;
  position:relative;overflow:hidden;
}
.hero-wrap::before{
  content:'';position:absolute;top:-60px;right:-60px;
  width:340px;height:340px;border-radius:50%;
  background:radial-gradient(circle,rgba(0,212,255,0.07) 0%,transparent 65%);
  pointer-events:none;
}
.hero-wrap::after{
  content:'';position:absolute;bottom:-80px;left:30%;
  width:280px;height:280px;border-radius:50%;
  background:radial-gradient(circle,rgba(162,89,255,0.06) 0%,transparent 65%);
  pointer-events:none;
}
.hero-tag{
  font-size:0.58rem;letter-spacing:4px;text-transform:uppercase;
  color:#4a6072;margin-bottom:14px;display:flex;align-items:center;gap:10px;
}
.hero-tag::before{content:'';display:inline-block;width:24px;height:1px;background:#00d4ff;}
.hero-title{
  font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;
  color:#e8f4ff;line-height:1.1;margin-bottom:12px;letter-spacing:-1px;
}
.hero-title span{color:#00d4ff;}
.hero-desc{
  font-size:0.82rem;color:#6a8a9a;line-height:1.8;max-width:680px;
  margin-bottom:28px;
}
.hero-badges{display:flex;gap:10px;flex-wrap:wrap;}
.hbadge{
  background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.2);
  border-radius:20px;padding:5px 14px;
  font-size:0.62rem;letter-spacing:1.5px;color:#00d4ff;text-transform:uppercase;
}
.hbadge.red{background:rgba(255,60,110,0.06);border-color:rgba(255,60,110,0.2);color:#ff3c6e;}
.hbadge.purple{background:rgba(162,89,255,0.06);border-color:rgba(162,89,255,0.2);color:#a259ff;}
.hbadge.green{background:rgba(0,255,157,0.06);border-color:rgba(0,255,157,0.2);color:#00ff9d;}

.timeline{position:relative;padding-left:32px;margin:8px 0 24px 0;}
.timeline::before{
  content:'';position:absolute;left:8px;top:0;bottom:0;
  width:1px;background:linear-gradient(to bottom,#00d4ff,#a259ff,rgba(255,60,110,0.3));
}
.tl-item{position:relative;margin-bottom:22px;animation:fadeInLeft .4s ease both;}
.tl-item:nth-child(1){animation-delay:.05s}
.tl-item:nth-child(2){animation-delay:.12s}
.tl-item:nth-child(3){animation-delay:.19s}
.tl-item:nth-child(4){animation-delay:.26s}
.tl-item:nth-child(5){animation-delay:.33s}
.tl-item:nth-child(6){animation-delay:.40s}
@keyframes fadeInLeft{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}
.tl-dot{
  position:absolute;left:-28px;top:4px;
  width:12px;height:12px;border-radius:50%;
  border:2px solid var(--dot-color);background:var(--dot-bg);
  box-shadow:0 0 8px var(--dot-color);
}
.tl-head{
  font-family:'Syne',sans-serif;font-size:0.7rem;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;color:var(--dot-color);margin-bottom:5px;
}
.tl-body{
  background:#0d1117;border:1px solid #1e2a38;
  border-left:2px solid var(--dot-color);border-radius:0 6px 6px 0;
  padding:12px 16px;font-size:0.78rem;line-height:1.7;color:#c8d8e8;
}
.tl-body b{color:#e8f4ff;}
.tl-body .hl{color:var(--dot-color);font-weight:700;}

.insight-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:20px 0;}
.ins-card{
  background:#0d1117;border:1px solid #1e2a38;border-radius:8px;
  padding:16px 18px;position:relative;overflow:hidden;
}
.ins-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--ins-color);}
.ins-num{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:800;color:var(--ins-color);line-height:1;margin-bottom:4px;}
.ins-label{font-size:0.6rem;letter-spacing:2px;text-transform:uppercase;color:#4a6072;}
.ins-desc{font-size:0.72rem;color:#6a8a9a;margin-top:6px;line-height:1.5;}

.nav-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:20px;}
.nav-card{
  background:#0d1117;border:1px solid #1e2a38;border-radius:8px;
  padding:18px 20px;cursor:default;
  transition:border-color .2s,box-shadow .2s;position:relative;overflow:hidden;
}
.nav-card:hover{
  border-color:var(--nav-color);
  box-shadow:0 0 20px color-mix(in srgb,var(--nav-color) 12%,transparent);
}
.nav-card::after{
  content:'';position:absolute;top:0;right:0;width:50px;height:50px;
  background:radial-gradient(circle at top right,color-mix(in srgb,var(--nav-color) 10%,transparent),transparent 70%);
}
.nav-icon{font-size:1.4rem;margin-bottom:10px;}
.nav-title{
  font-family:'Syne',sans-serif;font-size:0.75rem;font-weight:700;
  letter-spacing:1px;text-transform:uppercase;color:var(--nav-color);margin-bottom:6px;
}
.nav-desc{font-size:0.68rem;color:#4a6072;line-height:1.5;}

.kpi-strip{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:0 0 28px 0}
.kpi-card{background:#0d1117;border:1px solid #1e2a38;border-radius:8px;
  padding:18px 20px 14px 20px;position:relative;overflow:hidden;
  transition:border-color .2s,box-shadow .2s;}
.kpi-card:hover{border-color:var(--kpi-color);
  box-shadow:0 0 22px color-mix(in srgb,var(--kpi-color) 16%,transparent);}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--kpi-color);}
.kpi-card::after{content:'';position:absolute;top:0;right:0;width:60px;height:60px;
  background:radial-gradient(circle at top right,color-mix(in srgb,var(--kpi-color) 10%,transparent),transparent 70%);}
.kpi-icon{font-size:1.05rem;margin-bottom:10px;}
.kpi-label{font-size:0.57rem;letter-spacing:2.5px;text-transform:uppercase;color:#4a6072;margin-bottom:7px;}
.kpi-value{font-family:'Syne',sans-serif;font-size:1.85rem;font-weight:800;color:#e8f4ff;line-height:1;margin-bottom:10px;}
.kpi-sub{display:flex;align-items:center;justify-content:space-between;
  font-size:0.6rem;color:#4a6072;letter-spacing:1px;margin-bottom:7px;}
.kpi-pct{color:var(--kpi-color);font-weight:700;font-size:0.67rem;}
.kpi-bar-track{height:3px;background:#1e2a38;border-radius:2px;overflow:hidden;}
.kpi-bar-fill{height:100%;border-radius:2px;
  background:linear-gradient(90deg,var(--kpi-color),color-mix(in srgb,var(--kpi-color) 55%,#a259ff));}
</style>
""", unsafe_allow_html=True)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-tag">Threat Intelligence Platform — SISE / OPSIE 2026</div>
  <div class="hero-title">
    Un réseau sous <span>surveillance.</span><br>Des menaces sous analyse.
  </div>
  <div class="hero-desc">
    <b style="color:#e8f4ff;">{_total:,} connexions réseau</b> observées du
    <b style="color:#00d4ff;">{_date_start}</b> au <b style="color:#00d4ff;">{_date_end}</b>
    sur un firewall Iptables cloud.
    Chaque paquet raconte une histoire : tentative d'intrusion, scan furtif, flood applicatif,
    ou simple navigation légitime. NetFlow Sentinel les écoute toutes.
  </div>
  <div class="hero-badges">
    <span class="hbadge">🛡 Isolation Forest</span>
    <span class="hbadge purple">⬡ Random Forest</span>
    <span class="hbadge red">⚡ {_deny:,} menaces bloquées</span>
    <span class="hbadge green">✅ {_permit:,} accès légitimes</span>
    <span class="hbadge">{_days} jours d'observation</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-card" style="--kpi-color:#00d4ff">
    <div class="kpi-icon"></div>
    <div class="kpi-label">Flux total</div>
    <div class="kpi-value">{_total:,}</div>
    <div class="kpi-sub"><span>Toutes actions</span><span class="kpi-pct">100 %</span></div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
  </div>
  <div class="kpi-card" style="--kpi-color:#ff3c6e">
    <div class="kpi-icon"></div>
    <div class="kpi-label">DENY — Bloqués</div>
    <div class="kpi-value">{_deny:,}</div>
    <div class="kpi-sub"><span>Connexions bloquées</span><span class="kpi-pct">{_pct_deny:.1f} %</span></div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:{_pct_deny:.1f}%"></div></div>
  </div>
  <div class="kpi-card" style="--kpi-color:#00ff9d">
    <div class="kpi-icon"></div>
    <div class="kpi-label">PERMIT — Autorisés</div>
    <div class="kpi-value">{_permit:,}</div>
    <div class="kpi-sub"><span>Connexions autorisées</span><span class="kpi-pct">{_pct_permit:.1f} %</span></div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:{_pct_permit:.1f}%"></div></div>
  </div>
  <div class="kpi-card" style="--kpi-color:#a259ff">
    <div class="kpi-icon"></div>
    <div class="kpi-label">Sources uniques</div>
    <div class="kpi-value">{_src:,}</div>
    <div class="kpi-sub"><span>IPs source distinctes</span><span class="kpi-pct">—</span></div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
  </div>
  <div class="kpi-card" style="--kpi-color:#ffb800">
    <div class="kpi-icon"></div>
    <div class="kpi-label">Destinations</div>
    <div class="kpi-value">{_dst:,}</div>
    <div class="kpi-sub"><span>IPs destination distinctes</span><span class="kpi-pct">—</span></div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Storytelling + mini charts ────────────────────────────────────────────────
left_col, right_col = st.columns([5, 4], gap="large")

with left_col:
    st.markdown("<div class='section-hd'>Chronologie de l'analyse</div>", unsafe_allow_html=True)

    _proto_top = df[_pcol].value_counts().idxmax() if _pcol in df.columns else "TCP"
    _proto_cnt = df[_pcol].value_counts().max()    if _pcol in df.columns else 0
    _port_name = port_label(_top_port)
    _deny_hour = ""
    if "datetime" in df.columns and "action" in df.columns:
        _deny_df  = df[df["action"] == "DENY"].copy()
        _deny_df["hour"] = _deny_df["datetime"].dt.hour
        if not _deny_df.empty:
            _peak_h   = int(_deny_df["hour"].value_counts().idxmax())
            _peak_cnt = int(_deny_df["hour"].value_counts().max())
            _deny_hour = f"Pic d'activité hostile à <b class='hl'>{_peak_h:02d}h</b> : {_peak_cnt:,} tentatives."

    st.markdown(f"""
    <div class="timeline">
      <div class="tl-item" style="--dot-color:#00d4ff;--dot-bg:#07090f">
        <div class="tl-dot"></div>
        <div class="tl-head">Collecte des données</div>
        <div class="tl-body">
          <b>{_total:,} événements firewall</b> collectés du <b>{_date_start}</b> au <b>{_date_end}</b>.
          Logs d'un firewall <b>Iptables on-cloud</b>, couvrant <b>{_days} jours</b> d'observation.
          <b>{_src:,}</b> IPs sources distinctes recensées.
        </div>
      </div>
      <div class="tl-item" style="--dot-color:#a259ff;--dot-bg:#07090f">
        <div class="tl-dot"></div>
        <div class="tl-head">Profil du trafic</div>
        <div class="tl-body">
          <b class="hl">{_pct_deny:.1f} %</b> du trafic <b>bloqué</b> — soit <b>{_deny:,}</b> connexions refusées.
          Protocole dominant : <b class="hl">{_proto_top}</b> ({_proto_cnt:,} flux).
          Port le plus ciblé : <b class="hl">:{_top_port} ({_port_name})</b>. {_deny_hour}
        </div>
      </div>
      <div class="tl-item" style="--dot-color:#ff3c6e;--dot-bg:#07090f">
        <div class="tl-dot"></div>
        <div class="tl-head">Détection d'anomalies</div>
        <div class="tl-body">
          Chaque IP encodée en un <b>vecteur à 7 dimensions</b>.
          <b>Isolation Forest</b> isole les comportements déviants.
          <b>DBSCAN</b> regroupe les signatures similaires en clusters.
        </div>
      </div>
      <div class="tl-item" style="--dot-color:#00ff9d;--dot-bg:#07090f">
        <div class="tl-dot"></div>
        <div class="tl-head">Classification ML</div>
        <div class="tl-body">
          <b>Random Forest (200 arbres)</b> entraîné sur les vecteurs comportementaux.
          Courbes ROC, matrices de confusion et feature importance disponibles dans Sentinel.
        </div>
      </div>
      <div class="tl-item" style="--dot-color:#ffb800;--dot-bg:#07090f">
        <div class="tl-dot"></div>
        <div class="tl-head">Analyse temporelle</div>
        <div class="tl-body">
          Flux projetés sur l'axe du temps pour révéler les <b class="hl">patterns horaires et hebdomadaires</b>.
          Détection de pics par <b>Z-score</b>. Heatmaps PERMIT/DENY.
        </div>
      </div>
      <div class="tl-item" style="--dot-color:#a259ff;--dot-bg:#07090f">
        <div class="tl-dot"></div>
        <div class="tl-head">Intelligence IA — Mistral</div>
        <div class="tl-body">
          Synthèse par <b>Mistral AI</b> : résumé exécutif, menaces, géographie des attaques,
          <b class="hl">recommandations opérationnelles</b> et score de risque /100.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown("<div class='section-hd'>Distribution des actions</div>", unsafe_allow_html=True)
    _act = df["action"].value_counts().reset_index()
    _act.columns = ["action", "count"]
    _colors = {"DENY": "#ff3c6e", "PERMIT": "#00ff9d"}
    _fig_pie = px.pie(_act, names="action", values="count",
                      color="action", color_discrete_map=_colors, hole=0.6)
    _fig_pie.update_traces(textinfo="percent+label",
                           textfont=dict(family="Space Mono", size=10),
                           marker=dict(line=dict(color="#07090f", width=2)))
    _fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#c8d8e8", height=210,
                           margin=dict(t=0, b=0, l=0, r=0),
                           showlegend=True,
                           legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    st.plotly_chart(_fig_pie, use_container_width=True)

    if _pcol in df.columns:
        st.markdown("<div class='section-hd'>Protocoles</div>", unsafe_allow_html=True)
        _prot = df[_pcol].value_counts().reset_index()
        _prot.columns = ["proto", "count"]
        _fig_bar = px.bar(_prot, x="proto", y="count",
                          color_discrete_sequence=["#00d4ff", "#a259ff", "#ffb800", "#ff3c6e"])
        _fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                               font_color="#c8d8e8", height=180, showlegend=False,
                               margin=dict(t=0, b=0, l=0, r=0),
                               xaxis=dict(gridcolor="#1e2a38", title=""),
                               yaxis=dict(gridcolor="#1e2a38", title=""))
        _fig_bar.update_traces(texttemplate="%{y:,}", textposition="outside",
                               textfont=dict(size=9, color="#4a6072"))
        st.plotly_chart(_fig_bar, use_container_width=True)

    st.markdown(f"""
    <div class="insight-grid">
      <div class="ins-card" style="--ins-color:#ff3c6e">
        <div class="ins-num">{_pct_deny:.0f}%</div>
        <div class="ins-label">Taux de blocage</div>
        <div class="ins-desc">Du trafic total refusé par le firewall</div>
      </div>
      <div class="ins-card" style="--ins-color:#00d4ff">
        <div class="ins-num">{_rules}</div>
        <div class="ins-label">Règles actives</div>
        <div class="ins-desc">Politiques de filtrage distinctes</div>
      </div>
      <div class="ins-card" style="--ins-color:#a259ff">
        <div class="ins-num">:{_top_port}</div>
        <div class="ins-label">Port le + ciblé</div>
        <div class="ins-desc">{_port_name}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Navigation rapide ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-hd'>Pages disponibles</div>", unsafe_allow_html=True)
st.markdown("""
<div class="nav-grid">
  <div class="nav-card" style="--nav-color:#00d4ff">
    <div class="nav-icon">📊</div>
    <div class="nav-title">Visualisation</div>
    <div class="nav-desc">Analyse descriptive, DataTable interactive, top IPs et répartition des protocoles.</div>
  </div>
  <div class="nav-card" style="--nav-color:#a259ff">
    <div class="nav-icon">🌍</div>
    <div class="nav-title">Carte</div>
    <div class="nav-desc">Cartographie géographique des flux réseau. Géolocalisation des IPs publiques, arcs src→dst.</div>
  </div>
  <div class="nav-card" style="--nav-color:#ff3c6e">
    <div class="nav-icon">🛡️</div>
    <div class="nav-title">Sentinel</div>
    <div class="nav-desc">Détection d'anomalies ML, classification, analyse temporelle, Threat Analyst IA (Mistral).</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Explorateur de données ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='section-hd'>Explorateur de données brutes</div>", unsafe_allow_html=True)

# Filtres rapides
f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
with f1:
    _f_action = st.multiselect(
        "Action", df["action"].unique().tolist(),
        default=df["action"].unique().tolist(), key="home_action"
    )
with f2:
    _f_proto = st.multiselect(
        "Protocole", df[_pcol].dropna().unique().tolist() if _pcol in df.columns else [],
        default=df[_pcol].dropna().unique().tolist() if _pcol in df.columns else [],
        key="home_proto"
    ) if _pcol in df.columns else None
with f3:
    _f_ip = st.text_input("IP source contient", placeholder="ex: 192.168", key="home_ip")
with f4:
    _port_min = int(df["port_dst"].min()) if "port_dst" in df.columns else 0
    _port_max = int(df["port_dst"].max()) if "port_dst" in df.columns else 65535
    _f_port = st.slider("Port destination", _port_min, _port_max, (_port_min, _port_max), key="home_port")

# Application des filtres
_df_table = df.copy()
if _f_action:
    _df_table = _df_table[_df_table["action"].isin(_f_action)]
if _f_proto and _pcol in _df_table.columns:
    _df_table = _df_table[_df_table[_pcol].isin(_f_proto)]
if _f_ip.strip():
    _df_table = _df_table[_df_table["ip_src"].astype(str).str.contains(_f_ip.strip(), na=False)]
if "port_dst" in _df_table.columns:
    _df_table = _df_table[_df_table["port_dst"].between(_f_port[0], _f_port[1])]

# Recherche full-text + sélection colonnes
sa, sb = st.columns([3, 2])
with sa:
    _search = st.text_input("🔍 Recherche (IP, action, protocole...)", "", key="home_search")
with sb:
    _all_cols = list(_df_table.rename(columns=COLUMN_LABELS).columns)
    _sel_cols = st.multiselect("📑 Colonnes", _all_cols, default=_all_cols, key="home_cols")

# Appliquer la recherche full-text
_display_df = _df_table.rename(columns=COLUMN_LABELS)
if _search.strip():
    _mask = _display_df.apply(lambda col: col.astype(str).str.contains(_search.strip(), case=False, na=False))
    _display_df = _display_df[_mask.any(axis=1)]
if _sel_cols:
    _display_df = _display_df[_sel_cols]

# Compteur + pagination
_n_match  = len(_display_df)
_n_deny_f = int((_df_table["action"] == "DENY").sum())
_n_perm_f = int((_df_table["action"] == "PERMIT").sum())
_page_size = 1000
_total_pages = max(1, -(-_n_match // _page_size))

pc1, pc2 = st.columns([4, 1])
with pc1:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:8px 0;">
      <span style="font-size:0.68rem;color:#4a6072;letter-spacing:1px;">
        <b style="color:#e8f4ff;">{_n_match:,}</b> résultats
      </span>
      <span style="background:rgba(255,60,110,0.1);border:1px solid rgba(255,60,110,0.3);
        border-radius:12px;padding:3px 10px;font-size:0.62rem;color:#ff3c6e;">
        🚫 {_n_deny_f:,} DENY
      </span>
      <span style="background:rgba(0,255,157,0.1);border:1px solid rgba(0,255,157,0.3);
        border-radius:12px;padding:3px 10px;font-size:0.62rem;color:#00ff9d;">
        ✅ {_n_perm_f:,} PERMIT
      </span>
    </div>
    """, unsafe_allow_html=True)
with pc2:
    _page = st.number_input("Page", min_value=1, max_value=_total_pages, value=1, step=1, key="home_page")

_start = (_page - 1) * _page_size
_end   = _start + _page_size
st.dataframe(
    _display_df.iloc[_start:_end].reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
    height=600,
)
st.caption(f"📄 Page {_page}/{_total_pages} · {_n_match:,} lignes au total")

_csv_data = _display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"⬇ Exporter {_n_match:,} lignes en CSV",
    data=_csv_data,
    file_name="netflow_export.csv",
    mime="text/csv",
    key="home_export",
)
