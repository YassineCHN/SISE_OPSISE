"""
NetFlow Sentinel v2 — Threat Intelligence Platform
LLM intégré dans chaque onglet pour interprétation et génération de rapports.
"""

import math, time, json, os, re
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from modules.preprocessing import load_data as load_parquet_data, get_data_source_info
from utils.sentinel_utils import (
    MISTRAL_API_KEY_ENV,
    MISTRAL_MODEL_ENV,
    port_label,
    is_public,
)
from utils.sentinel_llm_analyst import generate_analysis
from components.top_nav import render_top_nav
from components.sentinel_theme import apply_sentinel_theme
from components.data_source_selector import render_motherduck_table_selector

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NetFlow Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_top_nav("sentinel")
apply_sentinel_theme()
selected_table = render_motherduck_table_selector()


# ═══════════════════════════════════════════════════════════════
# HELPERS UI
# ═══════════════════════════════════════════════════════════════
def render_ai_panel(key: str, label: str, generate_fn, requires_key=True):
    """
    Composant réutilisable : panneau IA avec bouton + zone de rapport streamé.
    key         : clé unique session_state pour stocker le rapport
    label       : texte du bouton
    generate_fn : callable(api_key, model) → generator de chunks str
    """
    sk = f"llm_{key}"
    if sk not in st.session_state:
        st.session_state[sk] = None

    st.markdown(
        f"""<div class='ai-panel'>
      <div class='ai-panel-hd'>🤖 Interprétation IA — Mistral</div>""",
        unsafe_allow_html=True,
    )

    if requires_key and not mistral_key:
        st.markdown(
            "<span style='color:#4a6072;font-size:0.72rem;'>💡 Ajoutez votre clé Mistral dans la sidebar pour activer l'analyse IA.</span>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    btn_col, info_col = st.columns([1, 3])
    with btn_col:
        run = st.button(label, key=f"btn_{key}")
    with info_col:
        if st.session_state[sk]:
            if st.button("🔄 Régénérer", key=f"regen_{key}"):
                st.session_state[sk] = None
                st.rerun()
        else:
            st.caption(f"Modèle : **{mistral_model}**")

    if run:
        st.session_state[sk] = None
        report_box = st.empty()
        full_text = ""
        try:
            for chunk in generate_fn(mistral_key, mistral_model):
                full_text += chunk
                report_box.markdown(
                    f"<div class='report-box'>{full_text}▌</div>",
                    unsafe_allow_html=True,
                )
            report_box.markdown(
                f"<div class='report-box'>{full_text}</div>", unsafe_allow_html=True
            )
            st.session_state[sk] = full_text
        except Exception as e:
            st.error(f"❌ Erreur : {e}")
    elif st.session_state[sk]:
        st.markdown(
            f"<div class='report-box'>{st.session_state[sk]}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _train_random_forest(ipf: pd.DataFrame):
    """Entraîne un Random Forest sur les features IP. Résultat mis en cache."""
    FEATURES = [
        "nb_connexions", "nb_ports_distincts", "nb_ips_dst",
        "ratio_deny", "nb_ports_sensibles", "activite_nuit", "port_dst_std",
    ]
    X = ipf[FEATURES]
    y = ipf["profil"]
    if y.nunique() < 2:
        return None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    classes = rf.classes_
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    cm = confusion_matrix(y_te, y_pred, labels=classes)
    imp_df = (
        pd.DataFrame({"feature": FEATURES, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
    )
    rep_dict = classification_report(y_te, y_pred, output_dict=True)
    try:
        yb = label_binarize(y_te, classes=classes)
        ys = rf.predict_proba(X_te)
        roc_data = [
            (cls, *roc_curve(yb[:, i], ys[:, i])[:2], auc(*roc_curve(yb[:, i], ys[:, i])[:2]))
            for i, cls in enumerate(classes)
        ]
    except Exception:
        roc_data = None
    return {"rep": rep_dict, "cm": cm, "classes": classes, "imp_df": imp_df, "cv": cv_scores, "roc": roc_data}


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
defaults = {
    "ip_features": None,
    "ts_data": None,
    "ts_pics": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
now = datetime.now().strftime("%d/%m/%Y  %H:%M:%S")
st.markdown(
    f"""
<div style="background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--accent);
            border-radius:6px;padding:24px 32px;margin-bottom:20px;
            display:flex;align-items:center;justify-content:space-between;">
  <div>
    <div style="font-family:'Syne',sans-serif;font-size:0.6rem;letter-spacing:4px;color:var(--text-dim);text-transform:uppercase;margin-bottom:6px;">
      ▌ Threat Intelligence Platform
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;color:var(--text-hi);letter-spacing:-1px;line-height:1;">
      NetFlow <span style="color:var(--accent);">Sentinel</span>
    </div>
    <div style="color:var(--text-dim);font-size:0.68rem;margin-top:6px;letter-spacing:1px;">
      Détection · Classification · Analyse temporelle · Cartographie · IA
    </div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:0.62rem;color:var(--text-dim);letter-spacing:2px;">SYSTEM TIME</div>
    <div style="font-family:'Syne',sans-serif;color:var(--accent);font-size:0.85rem;margin-top:2px;">{now}</div>
    <div style="margin-top:8px;">
      <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent4);margin-right:6px;box-shadow:0 0 6px var(--accent4);"></span>
      <span style="font-size:0.6rem;color:var(--accent4);letter-spacing:1px;">ONLINE</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    source_info = get_data_source_info()
    source_label = (
        "MotherDuck"
        if source_info.get("active_source") == "motherduck"
        else "Parquet local"
    )
    st.markdown("### 🗄️ Source des données")
    st.caption(f"Source active : **{source_label}**")
    if source_info.get("active_source") == "motherduck":
        db = source_info.get("motherduck_database", "")
        table = selected_table or source_info.get("motherduck_table", "")
        if db and table:
            st.caption(f"Table : `{db}.{table}`")
        elif table:
            st.caption(f"Table : `{table}`")
    if source_info.get("fallback_used"):
        st.warning("Fallback activé : lecture parquet local.")
    st.markdown("---")


    if st.button("🗑 Réinitialiser tout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 Mistral AI")
    if MISTRAL_API_KEY_ENV:
        st.success("✅ Clé API configurée (.env)")
        mistral_key = MISTRAL_API_KEY_ENV
    else:
        st.info("💡 Sans clé : rapports de secours activés")
        mistral_key = st.text_input(
            "🔑 Clé API Mistral",
            type="password",
            help="Laissez vide pour utiliser les rapports de secours intégrés",
        )
        mistral_key = mistral_key.strip().strip('"').strip("'")

    _models = ["mistral-small-latest", "mistral-medium-latest"]
    _def = _models.index(MISTRAL_MODEL_ENV) if MISTRAL_MODEL_ENV in _models else 0
    mistral_model = st.selectbox("🧠 Modèle", _models, index=_def)

    if st.button("🔍 Tester la clé API", use_container_width=True):
        if not mistral_key:
            st.warning("⚠️ Aucune clé fournie.")
        else:
            try:
                resp = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {mistral_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": mistral_model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 5,
                        "temperature": 0,
                        "stream": False,
                    },
                    timeout=20,
                )
                if resp.status_code == 200:
                    st.success("✅ Clé API valide.")
                elif resp.status_code == 401:
                    st.error("❌ Clé API invalide (401).")
                else:
                    st.error(f"❌ Test API échoué (HTTP {resp.status_code}).")
            except requests.RequestException as e:
                st.error(f"❌ Impossible de joindre l'API Mistral : {e}")

    st.caption("Sans clé : fallback templates activés automatiquement")


df_raw = load_parquet_data(selected_table=selected_table).copy()


# ═══════════════════════════════════════════════════════════════
# KPI STRIP — cards custom
# ═══════════════════════════════════════════════════════════════
_total   = len(df_raw)
_deny    = int((df_raw['action'] == 'DENY').sum())
_permit  = int((df_raw['action'] == 'PERMIT').sum())
_src     = df_raw['ip_src'].nunique()
_dst     = df_raw['ip_dst'].nunique()
_pct_deny   = (_deny   / _total * 100) if _total else 0
_pct_permit = (_permit / _total * 100) if _total else 0

st.markdown(f"""
<style>
.kpi-strip{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:0 0 20px 0}}
.kpi-card{{
  background:var(--bg2);
  border:1px solid var(--border);
  border-radius:8px;
  padding:18px 20px 14px 20px;
  position:relative;
  overflow:hidden;
  transition:border-color .2s, box-shadow .2s;
}}
.kpi-card:hover{{border-color:var(--kpi-color);box-shadow:0 0 24px color-mix(in srgb,var(--kpi-color) 18%,transparent)}}
.kpi-card::before{{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:var(--kpi-color);
}}
.kpi-card::after{{
  content:'';position:absolute;top:0;right:0;width:60px;height:60px;
  background:radial-gradient(circle at top right,color-mix(in srgb,var(--kpi-color) 12%,transparent),transparent 70%);
}}
.kpi-icon{{font-size:1.1rem;margin-bottom:10px;opacity:.85}}
.kpi-label{{
  font-size:0.58rem;letter-spacing:2.5px;text-transform:uppercase;
  color:var(--text-dim);margin-bottom:8px;font-family:'Space Mono',monospace;
}}
.kpi-value{{
  font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;
  color:var(--text-hi);line-height:1;margin-bottom:10px;
}}
.kpi-sub{{
  display:flex;align-items:center;justify-content:space-between;
  font-size:0.6rem;color:var(--text-dim);letter-spacing:1px;margin-bottom:7px;
}}
.kpi-sub .kpi-pct{{
  color:var(--kpi-color);font-weight:700;font-size:0.68rem;
}}
.kpi-bar-track{{
  height:3px;background:var(--border);border-radius:2px;overflow:hidden;
}}
.kpi-bar-fill{{
  height:100%;border-radius:2px;
  background:linear-gradient(90deg,var(--kpi-color),color-mix(in srgb,var(--kpi-color) 60%,var(--accent3)));
  transition:width .6s ease;
}}
</style>

<div class="kpi-strip">

  <div class="kpi-card" style="--kpi-color:#00d4ff">
    <div class="kpi-icon"></div>
    <div class="kpi-label">Flux total</div>
    <div class="kpi-value">{_total:,}</div>
    <div class="kpi-sub">
      <span>Toutes actions</span>
      <span class="kpi-pct">100 %</span>
    </div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
  </div>

  <div class="kpi-card" style="--kpi-color:#ff3c6e">
    <div class="kpi-icon"></div>
    <div class="kpi-label">DENY</div>
    <div class="kpi-value">{_deny:,}</div>
    <div class="kpi-sub">
      <span>Connexions bloquées</span>
      <span class="kpi-pct">{_pct_deny:.1f} %</span>
    </div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:{_pct_deny:.1f}%"></div></div>
  </div>

  <div class="kpi-card" style="--kpi-color:#00ff9d">
    <div class="kpi-icon"></div>
    <div class="kpi-label">PERMIT</div>
    <div class="kpi-value">{_permit:,}</div>
    <div class="kpi-sub">
      <span>Connexions autorisées</span>
      <span class="kpi-pct">{_pct_permit:.1f} %</span>
    </div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:{_pct_permit:.1f}%"></div></div>
  </div>

  <div class="kpi-card" style="--kpi-color:#a259ff">
    <div class="kpi-icon"></div>
    <div class="kpi-label">Sources uniques</div>
    <div class="kpi-value">{_src:,}</div>
    <div class="kpi-sub">
      <span>IPs source distinctes</span>
      <span class="kpi-pct">—</span>
    </div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
  </div>

  <div class="kpi-card" style="--kpi-color:#ffb800">
    <div class="kpi-icon"></div>
    <div class="kpi-label">Destinations</div>
    <div class="kpi-value">{_dst:,}</div>
    <div class="kpi-sub">
      <span>IPs destination distinctes</span>
      <span class="kpi-pct">—</span>
    </div>
    <div class="kpi-bar-track"><div class="kpi-bar-fill" style="width:100%"></div></div>
  </div>

</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_anomaly, tab_classif, tab_temporal, tab_behavior, tab_ia = st.tabs([
    "🔬  Détection d'anomalies",
    "🤖  Classification ML",
    "📈  Analyse temporelle",
    "🎯  Comportement des attaques",
    "🛡  Threat Analyst IA",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — DÉTECTION D'ANOMALIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_anomaly:
    st.markdown(
        """<div class='story-banner'>
      <b style='font-family:Syne,sans-serif;color:#00d4ff;letter-spacing:2px;font-size:0.7rem;'>SCÉNARIO 1 — DÉTECTION D'ANOMALIES</b><br><br>
      Chaque IP est transformée en un <span class='highlight'>vecteur comportemental</span> à 7 dimensions.
      <span class='highlight'>Isolation Forest</span> isole les comportements statistiquement anormaux dans cet hyperespace.
      <span class='highlight'>DBSCAN</span> regroupe ensuite les IPs par signature comportementale similaire.
      Le résultat : chaque IP reçoit un <span class='danger'>profil de menace</span> et un score d'anomalie.
    </div>""",
        unsafe_allow_html=True,
    )

    if st.button("🔬 Lancer la détection d'anomalies"):
        with st.spinner("Feature engineering & Isolation Forest…"):
            dw = df_raw.copy()
            if "datetime" in dw.columns:
                dw["datetime"] = pd.to_datetime(dw["datetime"], errors="coerce")
                dw["hour"] = dw["datetime"].dt.hour
            else:
                dw["hour"] = 0
            ip_feat = (
                dw.groupby("ip_src")
                .agg(
                    nb_connexions=("ip_src", "count"),
                    nb_ports_distincts=("port_dst", "nunique"),
                    nb_ips_dst=("ip_dst", "nunique"),
                    ratio_deny=("action", lambda x: (x == "DENY").mean()),
                    nb_ports_sensibles=(
                        "port_dst",
                        lambda x: x.isin([21, 22, 23, 3306]).sum(),
                    ),
                    activite_nuit=("hour", lambda x: ((x >= 0) & (x < 6)).mean()),
                    port_dst_std=("port_dst", "std"),
                )
                .reset_index()
            )
            ip_feat["port_dst_std"] = ip_feat["port_dst_std"].fillna(0)

            FEATURES = [
                "nb_connexions",
                "nb_ports_distincts",
                "nb_ips_dst",
                "ratio_deny",
                "nb_ports_sensibles",
                "activite_nuit",
                "port_dst_std",
            ]
            X = ip_feat[FEATURES].values
            X_s = StandardScaler().fit_transform(X)
            iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
            ip_feat["anomaly_iso"] = iso.fit_predict(X_s)
            ip_feat["anomaly_score"] = iso.decision_function(X_s)

            q99 = ip_feat["nb_connexions"].quantile(0.99)

            def profil(row):
                if row["nb_ports_distincts"] > 100:
                    return "Port Scan"
                elif row["nb_connexions"] > q99:
                    return "DDoS / Flood"
                elif row["nb_ports_sensibles"] > 10 and row["ratio_deny"] > 0.8:
                    return "Attaque ciblée"
                elif row["activite_nuit"] > 0.7:
                    return "Activité nocturne suspecte"
                elif row["ratio_deny"] > 0.9:
                    return "Comportement bloqué"
                else:
                    return "Normal"

            ip_feat["profil"] = ip_feat.apply(profil, axis=1)

            # DBSCAN (sur échantillon)
            try:
                anom = ip_feat[ip_feat["anomaly_iso"] == -1]
                norm = ip_feat[ip_feat["anomaly_iso"] == 1].sample(
                    n=min(2000, len(ip_feat[ip_feat["anomaly_iso"] == 1])),
                    random_state=42,
                )
                samp = pd.concat([anom, norm]).reset_index(drop=True)
                Xdb = StandardScaler().fit_transform(
                    samp[["nb_connexions", "nb_ports_distincts", "ratio_deny"]]
                )
                db = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1)
                samp["cluster_dbscan"] = db.fit_predict(Xdb)
                ip_feat = ip_feat.merge(
                    samp[["ip_src", "cluster_dbscan"]], on="ip_src", how="left"
                )
                ip_feat["cluster_dbscan"] = (
                    ip_feat["cluster_dbscan"].fillna(-9).astype(int)
                )
            except:
                ip_feat["cluster_dbscan"] = 0

            st.session_state.ip_features = ip_feat
        st.success("✅ Détection terminée !")

    ipf = st.session_state.ip_features
    if ipf is not None:
        n_total = len(ipf)
        n_anom = (ipf["anomaly_iso"] == -1).sum()
        n_suspects = (ipf["profil"] != "Normal").sum()
        profil_counts = ipf["profil"].value_counts()

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(
            f"<div class='stat-block'><div class='val'>{n_total:,}</div><div class='lbl'>IPs analysées</div></div>",
            unsafe_allow_html=True,
        )
        c2.markdown(
            f"<div class='stat-block'><div class='val' style='color:#ff3c6e;'>{n_anom:,}</div><div class='lbl'>Anomalies ISO Forest</div></div>",
            unsafe_allow_html=True,
        )
        c3.markdown(
            f"<div class='stat-block'><div class='val' style='color:#ffb800;'>{n_suspects:,}</div><div class='lbl'>IPs suspectes</div></div>",
            unsafe_allow_html=True,
        )
        c4.markdown(
            f"<div class='stat-block'><div class='val' style='color:#00ff9d;'>{(ipf['profil']=='Normal').sum():,}</div><div class='lbl'>Comportements normaux</div></div>",
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        v1, v2 = st.columns([3, 2], gap="large")
        with v1:
            st.markdown(
                "<div class='section-hd'>Isolation Forest — Connexions vs Ports distincts</div>",
                unsafe_allow_html=True,
            )
            fig = px.scatter(
                ipf,
                x="nb_connexions",
                y="nb_ports_distincts",
                color=ipf["anomaly_iso"].map({1: "Normal", -1: "Anomalie"}),
                color_discrete_map={"Normal": "#00d4ff", "Anomalie": "#ff3c6e"},
                hover_data=["ip_src", "profil", "ratio_deny"],
                log_x=True,
                opacity=0.65,
            )
            fig.update_traces(marker=dict(size=5))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1117",
                font_color="#c8d8e8",
                height=340,
                legend_title="",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#1e2a38", title="Nb connexions (log)"),
                yaxis=dict(gridcolor="#1e2a38", title="Ports distincts"),
            )
            st.plotly_chart(fig, use_container_width=True)
        with v2:
            st.markdown(
                "<div class='section-hd'>Profils comportementaux</div>",
                unsafe_allow_html=True,
            )
            pal = {
                "Normal": "#00ff9d",
                "Port Scan": "#00d4ff",
                "DDoS / Flood": "#ff3c6e",
                "Activité nocturne suspecte": "#a259ff",
                "Comportement bloqué": "#ffb800",
                "Attaque ciblée": "#ff6b6b",
            }
            fig2 = px.bar(
                profil_counts.reset_index(),
                x="count",
                y="profil",
                orientation="h",
                color="profil",
                color_discrete_map=pal,
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1117",
                font_color="#c8d8e8",
                height=340,
                showlegend=False,
                xaxis=dict(gridcolor="#1e2a38"),
                yaxis=dict(gridcolor="#1e2a38"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Distribution score anomalie
        st.markdown(
            "<div class='section-hd'>Distribution du score d'anomalie</div>",
            unsafe_allow_html=True,
        )
        fig3 = go.Figure()
        for grp, color, name in [(-1, "#ff3c6e", "Anomalie"), (1, "#00d4ff", "Normal")]:
            sub = ipf[ipf["anomaly_iso"] == grp]["anomaly_score"]
            fig3.add_trace(
                go.Histogram(
                    x=sub, name=name, marker_color=color, opacity=0.75, nbinsx=50
                )
            )
        fig3.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0d1117",
            font_color="#c8d8e8",
            height=230,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1e2a38", title="Score d'anomalie"),
            yaxis=dict(gridcolor="#1e2a38"),
            margin=dict(t=0, b=0, l=0, r=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Top suspects table
        st.markdown(
            "<div class='section-hd'>Top IPs suspectes</div>", unsafe_allow_html=True
        )
        suspects = (
            ipf[ipf["anomaly_iso"] == -1]
            .sort_values("anomaly_score")
            .head(20)[
                [
                    "ip_src",
                    "nb_connexions",
                    "nb_ports_distincts",
                    "ratio_deny",
                    "nb_ports_sensibles",
                    "profil",
                    "anomaly_score",
                ]
            ]
        )
        pb_map = {
            "Normal": "pb-normal",
            "Port Scan": "pb-scan",
            "DDoS / Flood": "pb-ddos",
            "Activité nocturne suspecte": "pb-nocturne",
            "Comportement bloqué": "pb-blocked",
            "Attaque ciblée": "pb-targeted",
        }
        rows_html = ""
        for _, r in suspects.iterrows():
            cls = pb_map.get(r["profil"], "pb-normal")
            rows_html += f"""<tr>
              <td style='color:#00d4ff;font-size:0.68rem;padding:6px 10px;'>{r['ip_src']}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['nb_connexions']:,}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['nb_ports_distincts']}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['ratio_deny']:.2f}</td>
              <td style='text-align:right;padding:6px 10px;'>{r['nb_ports_sensibles']}</td>
              <td style='padding:6px 10px;'><span class='profile-badge {cls}'>{r['profil']}</span></td>
              <td style='text-align:right;padding:6px 10px;color:#ff3c6e;'>{r['anomaly_score']:.4f}</td>
            </tr>"""
        st.markdown(
            f"""<table style='width:100%;border-collapse:collapse;font-size:0.7rem;'>
          <thead><tr style='color:#4a6072;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1e2a38;'>
            <th style='text-align:left;padding:6px 10px;'>IP Source</th><th style='text-align:right;padding:6px 10px;'>Connexions</th>
            <th style='text-align:right;padding:6px 10px;'>Ports</th><th style='text-align:right;padding:6px 10px;'>Ratio DENY</th>
            <th style='text-align:right;padding:6px 10px;'>Ports sensibles</th><th style='padding:6px 10px;'>Profil</th>
            <th style='text-align:right;padding:6px 10px;'>Score anomalie</th>
          </tr></thead><tbody style='border-top:1px solid #1e2a38;'>{rows_html}</tbody>
        </table>""",
            unsafe_allow_html=True,
        )

        # ── Panneau IA global anomalies
        anom_stats = {
            "n_total": int(n_total),
            "n_anomalies": int(n_anom),
            "n_suspects": int(n_suspects),
            "profil_counts": {k: int(v) for k, v in profil_counts.items()},
            "top_suspects": [
                {
                    "ip": r["ip_src"],
                    "nb_connexions": int(r["nb_connexions"]),
                    "nb_ports_distincts": int(r["nb_ports_distincts"]),
                    "ratio_deny": float(r["ratio_deny"]),
                    "profil": r["profil"],
                    "anomaly_score": float(r["anomaly_score"]),
                }
                for _, r in suspects.head(8).iterrows()
            ],
        }
        render_ai_panel(
            key="anomaly_global",
            label="🔬 Interpréter les anomalies",
            generate_fn=lambda key, model: generate_analysis(
                "anomaly", key, model, stats=anom_stats
            ),
            requires_key=False,
        )

        # ── Rapport d'incident par IP
        st.markdown("---")
        st.markdown(
            "<div class='section-hd'>Rapport d'incident par IP</div>",
            unsafe_allow_html=True,
        )
        suspect_ips = suspects["ip_src"].tolist()
        if suspect_ips:
            selected_ip = st.selectbox(
                "Choisir une IP suspecte", suspect_ips, key="ip_select_anomaly"
            )
            if selected_ip:
                ip_row = ipf[ipf["ip_src"] == selected_ip].iloc[0]
                geo_info = st.session_state.get("geo_cache", {}).get(selected_ip, {})

                # Exemples d'événements pour cette IP
                ip_events = df_raw[df_raw["ip_src"] == selected_ip].head(6)
                examples = ip_events.apply(
                    lambda r: f"{r.get('datetime','')} → {r.get('ip_dst','')}:{r.get('port_dst','')} [{r.get('action','')}]",
                    axis=1,
                ).tolist()

                inc_stats = {
                    "nb_connexions": int(ip_row["nb_connexions"]),
                    "nb_ports_distincts": int(ip_row["nb_ports_distincts"]),
                    "nb_ips_dst": int(ip_row["nb_ips_dst"]),
                    "ratio_deny": float(ip_row["ratio_deny"]),
                    "nb_ports_sensibles": int(ip_row["nb_ports_sensibles"]),
                    "activite_nuit": float(ip_row["activite_nuit"]),
                    "port_dst_std": float(ip_row["port_dst_std"]),
                    "profil": ip_row["profil"],
                    "anomaly_score": float(ip_row["anomaly_score"]),
                    "geo": geo_info,
                }
                render_ai_panel(
                    key=f"incident_{selected_ip}",
                    label=f"📋 Générer le rapport d'incident — {selected_ip}",
                    generate_fn=lambda key, model, _ip=selected_ip, _stats=inc_stats, _ex=examples: generate_analysis(
                        "incident", key, model, stats=_stats, ip=_ip, examples=_ex
                    ),
                    requires_key=False,
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — CLASSIFICATION ML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_classif:
    st.markdown(
        """<div class='story-banner'>
      <b style='font-family:Syne,sans-serif;color:#00d4ff;letter-spacing:2px;font-size:0.7rem;'>SCÉNARIO 2 — CLASSIFICATION RANDOM FOREST</b><br><br>
      Un <span class='highlight'>Random Forest à 200 arbres</span> apprend les signatures de chaque profil d'attaque.
      Objectif : créer un <span class='danger'>classificateur déployable en temps réel</span> capable de scorer
      instantanément toute nouvelle IP source. Les courbes ROC et la matrice de confusion quantifient la fiabilité opérationnelle.
    </div>""",
        unsafe_allow_html=True,
    )

    ipf = st.session_state.ip_features
    if ipf is None:
        st.info("💡 Lancez d'abord **Détection d'anomalies** pour générer les features IP.")
        st.markdown(
            """<div style='background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.2);
            border-radius:10px;padding:20px 24px;margin-top:16px;font-size:.82rem;color:#94a3b8;line-height:1.7;'>
            <b style='color:#00d4ff;'>Comment fonctionne cette étape ?</b><br><br>
            Une fois la détection d'anomalies lancée, chaque IP source est représentée par
            <b style='color:#c8d8e8;'>7 features comportementales</b> (volume de connexions, diversité des ports,
            ratio DENY, activité nocturne…). Un <b style='color:#c8d8e8;'>Random Forest à 200 arbres</b>
            apprend à distinguer automatiquement les profils d'attaque (Port Scan, DDoS, Attaque ciblée, etc.).<br><br>
            Les résultats — matrice de confusion, courbes ROC, importance des features — permettent d'évaluer
            la <b style='color:#00ff9d;'>fiabilité opérationnelle</b> du classificateur sur vos données.
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Entraînement Random Forest en cours… (mis en cache)"):
            result = _train_random_forest(ipf)

        if result is None:
            st.warning("⚠️ Pas assez de classes distinctes pour entraîner le modèle.")
        else:
            rep = result["rep"]
            cm, cls = result["cm"], result["classes"]
            imp_df = result["imp_df"]
            cv = result["cv"]
            acc = rep.get("accuracy", 0)

            st.markdown(
                f"""<div style='background:rgba(0,255,157,.06);border:1px solid rgba(0,255,157,.2);
                border-radius:10px;padding:16px 20px;margin-bottom:20px;font-size:.8rem;color:#94a3b8;line-height:1.6;'>
                <b style='color:#00ff9d;'>✅ Modèle entraîné automatiquement</b> &nbsp;·&nbsp;
                Random Forest 200 arbres &nbsp;·&nbsp; Split 75 % train / 25 % test &nbsp;·&nbsp;
                Validation croisée 5-fold &nbsp;·&nbsp;
                <b style='color:#c8d8e8;'>{len(ipf):,} IPs</b> · <b style='color:#c8d8e8;'>{len(cls)} classes</b><br>
                <span style='color:#4a6072;font-size:.72rem;'>
                Les résultats sont déterministes (random_state=42) et mis en cache — aucun recalcul à chaque visite.
                </span></div>""",
                unsafe_allow_html=True,
            )

            c1, c2, c3 = st.columns(3)
            c1.markdown(
                f"<div class='stat-block'><div class='val' style='color:#00ff9d;'>{acc:.1%}</div><div class='lbl'>Accuracy globale</div></div>",
                unsafe_allow_html=True,
            )
            tf = imp_df.iloc[0]
            c2.markdown(
                f"<div class='stat-block'><div class='val' style='color:#00d4ff;font-size:1rem;'>{tf['feature']}</div><div class='lbl'>Feature discriminante</div></div>",
                unsafe_allow_html=True,
            )
            c3.markdown(
                f"<div class='stat-block'><div class='val' style='color:#a259ff;'>{cv.mean():.3f} ±{cv.std():.3f}</div><div class='lbl'>CV 5-fold</div></div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                """<div style='background:rgba(162,89,255,.06);border:1px solid rgba(162,89,255,.2);
                border-radius:10px;padding:16px 20px;margin:16px 0;font-size:.8rem;color:#94a3b8;line-height:1.7;'>
                <b style='color:#a259ff;'>📐 Méthodologie</b><br>
                Les 7 features comportementales sont calculées par agrégation sur chaque IP source :
                <b style='color:#c8d8e8;'>nb_connexions</b> (volume total),
                <b style='color:#c8d8e8;'>nb_ports_distincts</b> (diversité des ports ciblés),
                <b style='color:#c8d8e8;'>nb_ips_dst</b> (spread réseau),
                <b style='color:#c8d8e8;'>ratio_deny</b> (taux de blocage),
                <b style='color:#c8d8e8;'>nb_ports_sensibles</b> (ports critiques : 21, 22, 23, 3306),
                <b style='color:#c8d8e8;'>activite_nuit</b> (proportion 0h–6h),
                <b style='color:#c8d8e8;'>port_dst_std</b> (variance des ports).
                Le label cible est le <b style='color:#c8d8e8;'>profil d'attaque</b> assigné par Isolation Forest.
                </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            col_cm, col_feat = st.columns(2, gap="large")
            with col_cm:
                st.markdown(
                    "<div class='section-hd'>Matrice de confusion</div>",
                    unsafe_allow_html=True,
                )
                st.caption("Chaque cellule indique combien d'IPs du profil réel (ligne) ont été classées dans le profil prédit (colonne). La diagonale = bonnes prédictions.")
                fig_cm = px.imshow(
                    cm,
                    x=list(cls),
                    y=list(cls),
                    color_continuous_scale=[
                        [0, "#0d1117"],
                        [0.5, "#1e3a5f"],
                        [1, "#00d4ff"],
                    ],
                    text_auto=True,
                    aspect="auto",
                )
                fig_cm.update_traces(textfont_size=11)
                fig_cm.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0d1117",
                    font_color="#c8d8e8",
                    height=360,
                    xaxis=dict(title="Prédit", tickangle=-30),
                    yaxis=dict(title="Réel"),
                    margin=dict(t=0, b=40, l=0, r=0),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_cm, use_container_width=True)
            with col_feat:
                st.markdown(
                    "<div class='section-hd'>Importance des features (Gini)</div>",
                    unsafe_allow_html=True,
                )
                st.caption("Score d'importance Gini : contribution moyenne de chaque feature à la réduction d'impureté dans les 200 arbres. Plus le score est élevé, plus la feature est discriminante.")
                colors_f = [
                    "#ff3c6e" if i == 0 else "#00d4ff" for i in range(len(imp_df))
                ]
                fig_fi = px.bar(
                    imp_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="feature",
                    color_discrete_sequence=colors_f,
                    text="importance",
                )
                fig_fi.update_traces(
                    texttemplate="%{text:.4f}", textposition="outside", textfont_size=9
                )
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0d1117",
                    font_color="#c8d8e8",
                    height=360,
                    showlegend=False,
                    xaxis=dict(gridcolor="#1e2a38"),
                    yaxis=dict(gridcolor="#1e2a38"),
                    margin=dict(t=0, b=0, l=0, r=60),
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            # ROC
            if result["roc"]:
                st.markdown(
                    "<div class='section-hd'>Courbes ROC (one-vs-rest)</div>",
                    unsafe_allow_html=True,
                )
                st.caption("Chaque courbe mesure la capacité du modèle à détecter un profil (TPR) sans générer de fausses alarmes (FPR). AUC = 1.0 → classification parfaite · AUC = 0.5 → aléatoire.")
                cr = ["#ff3c6e", "#00d4ff", "#00ff9d", "#ffb800", "#a259ff", "#ff6b6b"]
                fig_r = go.Figure()
                for i, (c_name, fpr, tpr, roc_auc) in enumerate(
                    result["roc"]
                ):
                    fig_r.add_trace(
                        go.Scatter(
                            x=fpr,
                            y=tpr,
                            name=f"{c_name} (AUC={roc_auc:.3f})",
                            line=dict(color=cr[i % len(cr)], width=2),
                        )
                    )
                fig_r.add_trace(
                    go.Scatter(
                        x=[0, 1],
                        y=[0, 1],
                        name="Aléatoire",
                        line=dict(color="#4a6072", dash="dash", width=1),
                    )
                )
                fig_r.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0d1117",
                    font_color="#c8d8e8",
                    height=340,
                    legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
                    xaxis=dict(gridcolor="#1e2a38", title="FPR"),
                    yaxis=dict(gridcolor="#1e2a38", title="TPR"),
                    margin=dict(t=0, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_r, use_container_width=True)

            # Métriques par classe
            st.markdown(
                "<div class='section-hd'>Rapport de classification par classe</div>",
                unsafe_allow_html=True,
            )
            st.caption("Précision = parmi les IPs classées dans ce profil, combien le sont vraiment · Rappel = parmi les vraies IPs de ce profil, combien sont détectées · F1 = moyenne harmonique (score vert > 0.8, orange > 0.5, rouge ≤ 0.5).")
            rows_m = ""
            for c_name in cls:
                d = rep.get(c_name, {})
                f1 = d.get("f1-score", 0)
                col_f = "#00ff9d" if f1 > 0.8 else "#ffb800" if f1 > 0.5 else "#ff3c6e"
                rows_m += f"""<tr>
                  <td style='padding:7px 12px;'>{c_name}</td>
                  <td style='text-align:right;padding:7px 12px;color:{col_f};'>{d.get('precision',0):.3f}</td>
                  <td style='text-align:right;padding:7px 12px;color:{col_f};'>{d.get('recall',0):.3f}</td>
                  <td style='text-align:right;padding:7px 12px;color:{col_f};font-weight:700;'>{f1:.3f}</td>
                  <td style='text-align:right;padding:7px 12px;color:#4a6072;'>{int(d.get('support',0))}</td>
                </tr>"""
            st.markdown(
                f"""<table style='width:100%;border-collapse:collapse;font-size:0.72rem;'>
              <thead><tr style='color:#4a6072;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1e2a38;'>
                <th style='text-align:left;padding:7px 12px;'>Classe</th><th style='text-align:right;padding:7px 12px;'>Précision</th>
                <th style='text-align:right;padding:7px 12px;'>Rappel</th><th style='text-align:right;padding:7px 12px;'>F1-score</th>
                <th style='text-align:right;padding:7px 12px;'>Support</th>
              </tr></thead><tbody style='border-top:1px solid #1e2a38;'>{rows_m}</tbody>
            </table>""",
                unsafe_allow_html=True,
            )

            # ── Panneau IA classification
            clf_stats = {
                "accuracy": float(acc),
                "top_feature": str(imp_df.iloc[0]["feature"]),
                "top_feat_score": float(imp_df.iloc[0]["importance"]),
                "n_classes": len(cls),
                "classes": list(cls),
                "per_class": {
                    k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in rep.items()
                    if isinstance(v, dict)
                },
                "importance": {
                    row["feature"]: float(row["importance"])
                    for _, row in imp_df.iterrows()
                },
                "cv_mean": float(cv.mean()),
            }
            render_ai_panel(
                key="classif_global",
                label="📊 Interpréter les résultats ML",
                generate_fn=lambda key, model: generate_analysis(
                    "classification", key, model, stats=clf_stats
                ),
                requires_key=False,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — ANALYSE TEMPORELLE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_temporal:
    st.markdown(
        """<div class='story-banner'>
      <b style='font-family:Syne,sans-serif;color:#00d4ff;letter-spacing:2px;font-size:0.7rem;'>SCÉNARIO 3 — ANALYSE TEMPORELLE</b><br><br>
      Le temps révèle ce que la snapshot statique cache.
      Heatmaps <span class='highlight'>heure × jour</span>, séries temporelles <span class='highlight'>PERMIT vs DENY</span>,
      et détection de pics par <span class='highlight'>Z-score</span> exposent les patterns d'attaque récurrents
      et les <span class='danger'>fenêtres temporelles critiques</span> à surveiller.
    </div>""",
        unsafe_allow_html=True,
    )

    if "datetime" not in df_raw.columns or df_raw["datetime"].isna().all():
        st.warning("⚠️ Colonne `datetime` absente ou invalide.")
    else:
        if st.button("📈 Lancer l'analyse temporelle"):
            with st.spinner("Extraction des composantes temporelles…"):
                dts = df_raw.copy()
                dts["datetime"] = pd.to_datetime(dts["datetime"], errors="coerce")
                dts = dts.dropna(subset=["datetime"])
                dts["hour"] = dts["datetime"].dt.hour
                dts["day_of_week"] = dts["datetime"].dt.day_name()
                dts["date"] = dts["datetime"].dt.date
                DAY_ORDER = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]

                hm_vol = (
                    dts.groupby(["day_of_week", "hour"])
                    .size()
                    .reset_index(name="count")
                    .pivot(index="day_of_week", columns="hour", values="count")
                    .reindex(DAY_ORDER)
                    .fillna(0)
                )
                hm_deny = (
                    dts[dts["action"] == "DENY"]
                    .groupby(["day_of_week", "hour"])
                    .size()
                    .reset_index(name="count")
                    .pivot(index="day_of_week", columns="hour", values="count")
                    .reindex(DAY_ORDER)
                    .fillna(0)
                )

                ts_h = (
                    dts.groupby([pd.Grouper(key="datetime", freq="h"), "action"])
                    .size()
                    .reset_index(name="count")
                )
                ts_total = (
                    dts.groupby(pd.Grouper(key="datetime", freq="h"))
                    .size()
                    .reset_index(name="count")
                    .rename(columns={"datetime": "hour_ts"})
                )
                ts_total["zscore"] = scipy_stats.zscore(ts_total["count"])
                ts_total["is_pic"] = ts_total["zscore"] > 2.5
                pics = ts_total[ts_total["is_pic"]].sort_values(
                    "zscore", ascending=False
                )

                # Profils horaires (si disponible)
                hourly_profil = None
                if st.session_state.ip_features is not None:
                    dp = dts.merge(
                        st.session_state.ip_features[["ip_src", "profil"]],
                        on="ip_src",
                        how="left",
                    )
                    dp["profil"] = dp["profil"].fillna("Normal")
                    hourly_profil = (
                        dp.groupby(["hour", "profil"]).size().reset_index(name="count")
                    )

                # Stats pour LLM
                deny_by_h = (
                    dts[dts["action"] == "DENY"]
                    .groupby("hour")
                    .size()
                    .nlargest(5)
                    .to_dict()
                )
                permit_by_h = (
                    dts[dts["action"] == "PERMIT"]
                    .groupby("hour")
                    .size()
                    .nlargest(5)
                    .to_dict()
                    if "PERMIT" in dts["action"].values
                    else {}
                )
                top_day = dts.groupby("day_of_week").size().reindex(DAY_ORDER).idxmax()
                pics_details = [
                    {
                        "horodatage": str(r["hour_ts"])[:16],
                        "count": int(r["count"]),
                        "zscore": float(r["zscore"]),
                    }
                    for _, r in pics.head(5).iterrows()
                ]
                profil_hours = {}
                if hourly_profil is not None:
                    for p in hourly_profil["profil"].unique():
                        sub = hourly_profil[hourly_profil["profil"] == p]
                        if not sub.empty:
                            profil_hours[p] = int(
                                sub.loc[sub["count"].idxmax(), "hour"]
                            )

                st.session_state.ts_data = {
                    "heatmap_vol": hm_vol,
                    "heatmap_deny": hm_deny,
                    "ts_hourly": ts_h,
                    "ts_total": ts_total,
                    "day_order": DAY_ORDER,
                    "hourly_profil": hourly_profil,
                    "n_days": dts["date"].nunique(),
                    "t_start": str(dts["datetime"].min())[:16],
                    "t_end": str(dts["datetime"].max())[:16],
                    "deny_by_h": deny_by_h,
                    "permit_by_h": permit_by_h,
                    "top_day": top_day,
                    "pics_details": pics_details,
                    "profil_hours": profil_hours,
                }
                st.session_state.ts_pics = pics
            st.success("✅ Analyse temporelle terminée !")

        ts = st.session_state.ts_data
        pics = st.session_state.ts_pics
        if ts is not None:
            n_pics = len(pics) if pics is not None else 0
            ts_total = ts["ts_total"]
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(
                f"<div class='stat-block'><div class='val' style='color:#00d4ff;'>{ts['n_days']}</div><div class='lbl'>Jours analysés</div></div>",
                unsafe_allow_html=True,
            )
            c2.markdown(
                f"<div class='stat-block'><div class='val' style='color:#ff3c6e;'>{n_pics}</div><div class='lbl'>Pics Z-score>2.5</div></div>",
                unsafe_allow_html=True,
            )
            hmax = ts_total.loc[ts_total["count"].idxmax(), "hour_ts"]
            c3.markdown(
                f"<div class='stat-block'><div class='val' style='color:#ffb800;font-size:1rem;'>{str(hmax)[:13]}</div><div class='lbl'>Pic max d'activité</div></div>",
                unsafe_allow_html=True,
            )
            c4.markdown(
                f"<div class='stat-block'><div class='val' style='color:#a259ff;font-size:0.9rem;'>{ts['top_day'][:3]}</div><div class='lbl'>Jour le + actif</div></div>",
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # Heatmaps
            st.markdown(
                "<div class='section-hd'>Heatmaps — Volume total & DENY par heure × jour</div>",
                unsafe_allow_html=True,
            )
            hm1, hm2 = st.columns(2, gap="large")
            with hm1:
                fig_hm = go.Figure(
                    go.Heatmap(
                        z=ts["heatmap_vol"].values,
                        x=list(range(24)),
                        y=ts["heatmap_vol"].index.tolist(),
                        colorscale=[[0, "#07090f"], [0.5, "#1e3a5f"], [1, "#00d4ff"]],
                    )
                )
                fig_hm.update_layout(
                    title="Volume total",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0d1117",
                    font_color="#c8d8e8",
                    height=280,
                    margin=dict(t=30, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_hm, use_container_width=True)
            with hm2:
                fig_hm2 = go.Figure(
                    go.Heatmap(
                        z=ts["heatmap_deny"].values,
                        x=list(range(24)),
                        y=ts["heatmap_deny"].index.tolist(),
                        colorscale=[[0, "#07090f"], [0.5, "#3a1015"], [1, "#ff3c6e"]],
                    )
                )
                fig_hm2.update_layout(
                    title="Connexions DENY",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0d1117",
                    font_color="#c8d8e8",
                    height=280,
                    margin=dict(t=30, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_hm2, use_container_width=True)

            # Séries temporelles
            st.markdown(
                "<div class='section-hd'>Séries temporelles — PERMIT vs DENY par heure</div>",
                unsafe_allow_html=True,
            )
            ts_h = ts["ts_hourly"]
            permit_ts = ts_h[ts_h["action"] == "PERMIT"].set_index("datetime")["count"]
            deny_ts = ts_h[ts_h["action"] == "DENY"].set_index("datetime")["count"]
            fig_ts = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08
            )
            fig_ts.add_trace(
                go.Scatter(
                    x=permit_ts.index,
                    y=permit_ts.values,
                    name="PERMIT",
                    line=dict(color="#00d4ff", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(0,212,255,0.08)",
                ),
                row=1,
                col=1,
            )
            fig_ts.add_trace(
                go.Scatter(
                    x=deny_ts.index,
                    y=deny_ts.values,
                    name="DENY",
                    line=dict(color="#ff3c6e", width=1.5),
                    fill="tozeroy",
                    fillcolor="rgba(255,60,110,0.08)",
                ),
                row=2,
                col=1,
            )
            fig_ts.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1117",
                font_color="#c8d8e8",
                height=360,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=0, b=0, l=0, r=0),
            )
            for i in [1, 2]:
                fig_ts.update_xaxes(gridcolor="#1e2a38", row=i, col=1)
                fig_ts.update_yaxes(gridcolor="#1e2a38", row=i, col=1)
            st.plotly_chart(fig_ts, use_container_width=True)

            # Z-score
            st.markdown(
                "<div class='section-hd'>Détection de pics — Z-score (seuil = 2.5)</div>",
                unsafe_allow_html=True,
            )
            seuil = ts_total["count"].mean() + 2.5 * ts_total["count"].std()
            fig_z = go.Figure()
            fig_z.add_trace(
                go.Scatter(
                    x=ts_total["hour_ts"],
                    y=ts_total["count"],
                    name="Volume horaire",
                    line=dict(color="#4a6072", width=1.2),
                    fill="tozeroy",
                    fillcolor="rgba(74,96,114,0.1)",
                )
            )
            if not ts_total[ts_total["is_pic"]].empty:
                fig_z.add_trace(
                    go.Scatter(
                        x=ts_total.loc[ts_total["is_pic"], "hour_ts"],
                        y=ts_total.loc[ts_total["is_pic"], "count"],
                        mode="markers",
                        name="Pic anormal",
                        marker=dict(
                            color="#ff3c6e", size=8, line=dict(color="#fff", width=1)
                        ),
                    )
                )
            fig_z.add_hline(
                y=seuil,
                line_dash="dash",
                line_color="#ff3c6e",
                annotation_text="Seuil z=2.5",
                annotation_font_color="#ff3c6e",
            )
            fig_z.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#0d1117",
                font_color="#c8d8e8",
                height=280,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#1e2a38"),
                yaxis=dict(gridcolor="#1e2a38"),
                margin=dict(t=0, b=0, l=0, r=0),
            )
            st.plotly_chart(fig_z, use_container_width=True)

            # Profils horaires
            if ts["hourly_profil"] is not None:
                st.markdown(
                    "<div class='section-hd'>Distribution horaire par profil comportemental</div>",
                    unsafe_allow_html=True,
                )
                pal = {
                    "Normal": "#00d4ff",
                    "Activité nocturne suspecte": "#a259ff",
                    "Comportement bloqué": "#ffb800",
                    "DDoS / Flood": "#ff3c6e",
                    "Port Scan": "#00ff9d",
                    "Attaque ciblée": "#ff6b6b",
                }
                fig_p = px.line(
                    ts["hourly_profil"],
                    x="hour",
                    y="count",
                    color="profil",
                    color_discrete_map=pal,
                    markers=True,
                )
                fig_p.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#0d1117",
                    font_color="#c8d8e8",
                    height=300,
                    legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
                    xaxis=dict(gridcolor="#1e2a38", title="Heure", tickmode="linear"),
                    yaxis=dict(gridcolor="#1e2a38"),
                    margin=dict(t=0, b=0, l=0, r=0),
                )
                st.plotly_chart(fig_p, use_container_width=True)

            if n_pics > 0:
                st.markdown(
                    "<div class='section-hd'>Top pics détectés</div>",
                    unsafe_allow_html=True,
                )
                st.dataframe(
                    pics.head(10)[["hour_ts", "count", "zscore"]]
                    .rename(
                        columns={
                            "hour_ts": "Horodatage",
                            "count": "Nb connexions",
                            "zscore": "Z-score",
                        }
                    )
                    .style.format({"Z-score": "{:.3f}", "Nb connexions": "{:,}"}),
                    use_container_width=True,
                    hide_index=True,
                    height=260,
                )

            # ── Panneau IA temporel
            temp_stats = {
                "n_days": ts["n_days"],
                "t_start": ts["t_start"],
                "t_end": ts["t_end"],
                "n_pics": n_pics,
                "peak_hour": str(hmax)[:16],
                "low_hour": str(ts_total.loc[ts_total["count"].idxmin(), "hour_ts"])[
                    :16
                ],
                "top_day": ts["top_day"],
                "deny_by_hour": {str(k): int(v) for k, v in ts["deny_by_h"].items()},
                "permit_by_hour": {
                    str(k): int(v) for k, v in ts["permit_by_h"].items()
                },
                "pics_details": ts["pics_details"],
                "profil_hours": {k: str(v) for k, v in ts["profil_hours"].items()},
            }
            render_ai_panel(
                key="temporal_global",
                label="📈 Interpréter les patterns temporels",
                generate_fn=lambda key, model: generate_analysis(
                    "temporal", key, model, stats=temp_stats
                ),
                requires_key=False,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — COMPORTEMENT DES ATTAQUES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_behavior:

    st.markdown("""
    <div style='background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--accent4);
                border-radius:6px;padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;gap:16px;'>
      <div style='font-size:2.5rem;'>🎯</div>
      <div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:var(--text-hi);letter-spacing:1px;'>
          Comportement des Attaques
        </div>
        <div style='color:var(--text-dim);font-size:0.68rem;margin-top:4px;letter-spacing:1px;'>
          IP Sources · IP Destinations · Top Attaquants · Corrélations & Radar
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Nécessite ip_features — guider l'utilisateur si absent
    ipf_b = st.session_state.ip_features
    if ipf_b is None:
        st.info("💡 Lancez d'abord **🔬 Détection d'anomalies** pour calculer les features comportementales.")
        st.stop()

    PROFIL_COLORS = {
        "Normal":                    "#00ff9d",
        "Port Scan":                 "#00d4ff",
        "DDoS / Flood":              "#ff3c6e",
        "Activité nocturne suspecte":"#a259ff",
        "Comportement bloqué":       "#ffb800",
        "Attaque ciblée":            "#ff6b6b",
    }
    PORT_NAMES_B = {21:"FTP",22:"SSH",23:"Telnet",25:"SMTP",53:"DNS",80:"HTTP",
                    110:"POP3",143:"IMAP",443:"HTTPS",445:"SMB",1433:"MSSQL",
                    3306:"MySQL",3389:"RDP",5432:"PostgreSQL",5900:"VNC",
                    6379:"Redis",8080:"HTTP-Alt",8443:"HTTPS-Alt"}

    def is_private_ip(ip: str) -> bool:
        try:
            p = [int(x) for x in str(ip).split(".")]
            return (p[0]==10 or (p[0]==172 and 16<=p[1]<=31)
                    or (p[0]==192 and p[1]==168) or p[0]==127)
        except Exception:
            return False

    bt1, bt2, bt3, bt4 = st.tabs([
        "🖥 IP Sources", "🎯 IP Destinations",
        "⚡ Top Attaquants", "📊 Corrélations & Radar",
    ])

    # ────────────────────────────────────────────────────────────
    # BT1 — IP SOURCES
    # ────────────────────────────────────────────────────────────
    with bt1:
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#00ff9d;letter-spacing:2px;font-size:0.7rem;'>IP SOURCES — QUI ATTAQUE ?</b><br><br>
          Chaque adresse IP source laisse une <span class='highlight'>empreinte comportementale unique</span> :
          volume de connexions, diversité de ports, ratio de blocage, activité nocturne.
          Ce sont ces 7 dimensions combinées qui distinguent un utilisateur légitime d'un
          <span class='danger'>attaquant automatisé</span>.
        </div>""", unsafe_allow_html=True)

        profils_dispo = ipf_b["profil"].unique().tolist()
        profils_sel   = st.multiselect(
            "Filtrer par profil",
            profils_dispo,
            default=[p for p in profils_dispo if p != "Normal"],
            key="beh_src_filter"
        )
        ip_sel = ipf_b[ipf_b["profil"].isin(profils_sel)] if profils_sel else ipf_b

        sc1, sc2 = st.columns(2, gap="large")
        with sc1:
            st.markdown("<div class='section-hd'>Connexions vs Ports distincts — taille = ratio DENY</div>", unsafe_allow_html=True)
            fig_s1 = px.scatter(
                ip_sel, x="nb_connexions", y="nb_ports_distincts",
                color="profil", color_discrete_map=PROFIL_COLORS,
                size="ratio_deny", size_max=22,
                hover_data=["ip_src","nb_ports_sensibles","activite_nuit","ratio_deny"],
                log_x=True, opacity=0.75
            )
            fig_s1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                  font_color="#c8d8e8", height=360, legend_title="",
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
                                  xaxis=dict(gridcolor="#1e2a38", title="Nb connexions (log)"),
                                  yaxis=dict(gridcolor="#1e2a38", title="Ports distincts"))
            st.plotly_chart(fig_s1, use_container_width=True)

        with sc2:
            st.markdown("<div class='section-hd'>Activité nocturne vs Ratio DENY — taille = volume</div>", unsafe_allow_html=True)
            fig_s2 = px.scatter(
                ip_sel, x="activite_nuit", y="ratio_deny",
                color="profil", color_discrete_map=PROFIL_COLORS,
                size="nb_connexions", size_max=25,
                hover_data=["ip_src","nb_ports_distincts"],
                opacity=0.75
            )
            fig_s2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                  font_color="#c8d8e8", height=360, legend_title="",
                                  legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
                                  xaxis=dict(gridcolor="#1e2a38", title="% activité 0h–6h", tickformat=".0%"),
                                  yaxis=dict(gridcolor="#1e2a38", title="Ratio DENY", tickformat=".0%"))
            st.plotly_chart(fig_s2, use_container_width=True)

        st.markdown("<div class='section-hd'>Distribution du volume de connexions par profil (log)</div>", unsafe_allow_html=True)
        fig_box = px.box(
            ip_sel, x="profil", y="nb_connexions",
            color="profil", color_discrete_map=PROFIL_COLORS, log_y=True
        )
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                               font_color="#c8d8e8", height=300, showlegend=False,
                               xaxis=dict(gridcolor="#1e2a38", title=""),
                               yaxis=dict(gridcolor="#1e2a38", title="Nb connexions (log)"),
                               margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_box, use_container_width=True)

        # Top 5 IP sources
        st.markdown("<div class='section-hd'>🖥 Top 5 IP sources les plus émettrices</div>", unsafe_allow_html=True)
        top5_src = df_raw["ip_src"].value_counts().head(5).reset_index()
        top5_src.columns = ["ip_src", "nb_connexions"]
        ip_map = ipf_b.set_index("ip_src")
        top5_src["ratio_deny"] = top5_src["ip_src"].map(ip_map["ratio_deny"])
        top5_src["profil"]     = top5_src["ip_src"].map(ip_map["profil"])
        top5_src["score"]      = top5_src["ip_src"].map(ip_map["anomaly_score"])

        pb_map_b = {"Normal":"pb-normal","Port Scan":"pb-scan","DDoS / Flood":"pb-ddos",
                    "Activité nocturne suspecte":"pb-nocturne","Comportement bloqué":"pb-blocked","Attaque ciblée":"pb-targeted"}
        rows_t5 = ""
        for _, r in top5_src.iterrows():
            cls = pb_map_b.get(r.get("profil","Normal"), "pb-normal")
            rows_t5 += f"""<tr>
              <td style='padding:7px 12px;color:#00d4ff;'>{r['ip_src']}</td>
              <td style='text-align:right;padding:7px 12px;'>{r['nb_connexions']:,}</td>
              <td style='text-align:right;padding:7px 12px;'>{r['ratio_deny']:.1%}</td>
              <td style='padding:7px 12px;'><span class='profile-badge {cls}'>{r.get('profil','—')}</span></td>
              <td style='text-align:right;padding:7px 12px;color:#ff3c6e;'>{r['score']:.4f}</td>
            </tr>"""
        st.markdown(f"""<table style='width:100%;border-collapse:collapse;font-size:0.72rem;'>
          <thead><tr style='color:#4a6072;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1e2a38;'>
            <th style='text-align:left;padding:7px 12px;'>IP Source</th>
            <th style='text-align:right;padding:7px 12px;'>Connexions</th>
            <th style='text-align:right;padding:7px 12px;'>Ratio DENY</th>
            <th style='padding:7px 12px;'>Profil</th>
            <th style='text-align:right;padding:7px 12px;'>Score anomalie</th>
          </tr></thead>
          <tbody style='border-top:1px solid #1e2a38;'>{rows_t5}</tbody>
        </table>""", unsafe_allow_html=True)

        # LLM
        src_stats = {
            "tab": "src",
            "top5_src": [{"ip":r["ip_src"],"connexions":int(r["nb_connexions"]),
                           "ratio_deny":float(r["ratio_deny"]) if pd.notna(r["ratio_deny"]) else 0,
                           "profil":str(r.get("profil","Normal"))} for _,r in top5_src.iterrows()],
            "profil_dist": {k:int(v) for k,v in ipf_b["profil"].value_counts().items()},
            "n_suspects": int((ipf_b["profil"]!="Normal").sum()),
        }
        render_ai_panel(
            key="behavior_src",
            label="🌍 Analyser les IP sources",
            generate_fn=lambda key, model: generate_analysis("behavior", key, model, stats=src_stats),
            requires_key=False,
        )

    # ────────────────────────────────────────────────────────────
    # BT2 — IP DESTINATIONS
    # ────────────────────────────────────────────────────────────
    with bt2:
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#00ff9d;letter-spacing:2px;font-size:0.7rem;'>IP DESTINATIONS — QUI EST CIBLÉ ?</b><br><br>
          Les IP destinations révèlent les <span class='highlight'>actifs les plus exposés</span> du réseau.
          Une destination recevant beaucoup de DENY est activement attaquée mais protégée.
          Les IPs sources <span class='danger'>hors plan d'adressage RFC 1918</span> indiquent des connexions
          depuis Internet non filtrées — risque d'intrusion directe.
        </div>""", unsafe_allow_html=True)

        dst_agg = df_raw.groupby("ip_dst").agg(
            nb_connexions =("ip_dst",   "count"),
            nb_src_uniques=("ip_src",   "nunique"),
            ratio_deny    =("action",   lambda x: (x=="DENY").mean()),
            ports_uniques =("port_dst", "nunique"),
            nb_permit     =("action",   lambda x: (x=="PERMIT").sum()),
            nb_deny       =("action",   lambda x: (x=="DENY").sum()),
        ).reset_index().sort_values("nb_connexions", ascending=False)

        st.markdown("<div class='section-hd'>Top 10 IP destinations — PERMIT vs DENY</div>", unsafe_allow_html=True)
        top10_dst = dst_agg.head(10)
        fig_dst = go.Figure()
        fig_dst.add_trace(go.Bar(x=top10_dst["ip_dst"], y=top10_dst["nb_permit"],
                                  name="PERMIT", marker_color="#00d4ff", opacity=0.85))
        fig_dst.add_trace(go.Bar(x=top10_dst["ip_dst"], y=top10_dst["nb_deny"],
                                  name="DENY", marker_color="#ff3c6e", opacity=0.9))
        fig_dst.update_layout(barmode="stack", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                               font_color="#c8d8e8", height=320,
                               legend=dict(bgcolor="rgba(0,0,0,0)"),
                               xaxis=dict(gridcolor="#1e2a38", tickangle=-30, tickfont=dict(size=9)),
                               yaxis=dict(gridcolor="#1e2a38", title="Nb connexions"),
                               margin=dict(t=0,b=40,l=0,r=0))
        st.plotly_chart(fig_dst, use_container_width=True)

        dc1, dc2 = st.columns([3,2], gap="large")
        with dc1:
            st.markdown("<div class='section-hd'>Table détaillée des destinations (Top 20)</div>", unsafe_allow_html=True)
            st.dataframe(
                dst_agg.head(20)[["ip_dst","nb_connexions","nb_src_uniques","ratio_deny","ports_uniques","nb_permit","nb_deny"]]
                .rename(columns={"ip_dst":"IP Dest","nb_connexions":"Connexions","nb_src_uniques":"Sources uniques",
                                  "ratio_deny":"Ratio DENY","ports_uniques":"Ports","nb_permit":"PERMIT","nb_deny":"DENY"})
                .style.format({"Connexions":"{:,}","Sources uniques":"{:,}",
                               "Ratio DENY":"{:.2%}","PERMIT":"{:,}","DENY":"{:,}"}),
                use_container_width=True, hide_index=True, height=340
            )

        with dc2:
            st.markdown("<div class='section-hd'>🌐 Accès depuis IPs hors RFC 1918</div>", unsafe_allow_html=True)
            ext_src = df_raw[~df_raw["ip_src"].apply(is_private_ip)]["ip_src"].value_counts().head(15).reset_index()
            ext_src.columns = ["ip_ext", "nb_connexions"]
            if not ext_src.empty:
                fig_ext = go.Figure(go.Bar(
                    x=ext_src["nb_connexions"], y=ext_src["ip_ext"],
                    orientation="h", marker_color="#ff3c6e", opacity=0.85
                ))
                fig_ext.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                       font_color="#c8d8e8", height=340,
                                       xaxis=dict(gridcolor="#1e2a38", title="Nb connexions"),
                                       yaxis=dict(gridcolor="#1e2a38", tickfont=dict(size=9)),
                                       margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_ext, use_container_width=True)
                st.markdown(f"""<div class='kpi-row'>
                  <span class='kpi-chip deny'>⚠️ {len(ext_src)} IPs externes détectées</span>
                </div>""", unsafe_allow_html=True)
            else:
                st.success("✅ Aucune IP source externe détectée.")

        # LLM
        dst_stats = {
            "tab": "dst",
            "top10_dst": [{"ip":r["ip_dst"],"connexions":int(r["nb_connexions"]),
                           "ratio_deny":float(r["ratio_deny"]),"nb_src":int(r["nb_src_uniques"])}
                          for _,r in top10_dst.iterrows()],
            "n_ext_ips": len(ext_src) if not ext_src.empty else 0,
            "top_ext": [{"ip":r["ip_ext"],"connexions":int(r["nb_connexions"])}
                        for _,r in ext_src.head(8).iterrows()] if not ext_src.empty else [],
        }
        render_ai_panel(
            key="behavior_dst",
            label="🎯 Analyser les IP destinations",
            generate_fn=lambda key, model: generate_analysis("behavior", key, model, stats=dst_stats),
            requires_key=False,
        )

    # ────────────────────────────────────────────────────────────
    # BT3 — TOP ATTAQUANTS
    # ────────────────────────────────────────────────────────────
    with bt3:
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#00ff9d;letter-spacing:2px;font-size:0.7rem;'>TOP ATTAQUANTS — LES PLUS DANGEREUX</b><br><br>
          Le classement par <span class='highlight'>score d'anomalie Isolation Forest</span> concentre les
          menaces les plus sérieuses en tête de liste. La taille des bulles représente les ports sensibles ciblés —
          un grand bulle rouge signifie une IP combinant <span class='danger'>volume + ciblage précis + ratio DENY élevé</span>.
        </div>""", unsafe_allow_html=True)

        n_att = st.slider("Nombre d'attaquants à afficher", 5, 30, 15, key="beh_top_slider")
        suspects_b = (ipf_b[ipf_b["profil"] != "Normal"]
                      .sort_values("anomaly_score").head(n_att))

        st.markdown("<div class='section-hd'>Carte comportementale des attaquants</div>", unsafe_allow_html=True)
        fig_att = px.scatter(
            suspects_b, x="nb_connexions", y="nb_ports_distincts",
            color="profil", color_discrete_map=PROFIL_COLORS,
            size="nb_ports_sensibles", size_max=45,
            hover_data=["ip_src","ratio_deny","activite_nuit","anomaly_score"],
            text="ip_src", log_x=True, opacity=0.85
        )
        fig_att.update_traces(textposition="top center", textfont=dict(size=8, color="#c8d8e8"))
        fig_att.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                               font_color="#c8d8e8", height=420, legend_title="",
                               legend=dict(bgcolor="rgba(0,0,0,0)", font_size=10),
                               xaxis=dict(gridcolor="#1e2a38", title="Nb connexions (log)"),
                               yaxis=dict(gridcolor="#1e2a38", title="Nb ports distincts"))
        st.plotly_chart(fig_att, use_container_width=True)

        st.markdown("<div class='section-hd'>Tableau comportemental détaillé</div>", unsafe_allow_html=True)
        st.dataframe(
            suspects_b[["ip_src","nb_connexions","nb_ports_distincts","ratio_deny",
                         "nb_ports_sensibles","activite_nuit","profil","anomaly_score"]]
            .rename(columns={"ip_src":"IP","nb_connexions":"Connexions","nb_ports_distincts":"Ports",
                              "ratio_deny":"Ratio DENY","nb_ports_sensibles":"Ports sensibles",
                              "activite_nuit":"Activité nuit","profil":"Profil","anomaly_score":"Score"})
            .style.format({"Connexions":"{:,.0f}","Ports":"{:,.0f}","Ratio DENY":"{:.2%}",
                           "Activité nuit":"{:.2%}","Score":"{:.4f}"}),
            use_container_width=True, hide_index=True, height=360
        )

        # Top 10 ports <1024 avec accès autorisé
        st.markdown("<div class='section-hd'>🔑 Top 10 ports sensibles (&lt;1024) avec accès PERMIT</div>", unsafe_allow_html=True)
        df_permit_b = df_raw[df_raw["action"]=="PERMIT"] if "action" in df_raw.columns else pd.DataFrame()
        if not df_permit_b.empty and "port_dst" in df_permit_b.columns:
            top_perm_ports = (df_permit_b[df_permit_b["port_dst"] < 1024]["port_dst"]
                              .value_counts().head(10).reset_index())
            top_perm_ports.columns = ["port","count"]
            top_perm_ports["service"] = top_perm_ports["port"].map(lambda p: PORT_NAMES_B.get(int(p),"Inconnu"))
            top_perm_ports["risk"]    = top_perm_ports["port"].map(
                lambda p: "🔴 Critique" if int(p) in [22,23,3389,445,1433,3306,5900]
                else ("🟡 Modéré" if int(p) in [80,443,21,25,53] else "⚪ Faible"))

            tc1, tc2 = st.columns([2,3], gap="large")
            with tc1:
                rows_pp = ""
                for _, r in top_perm_ports.iterrows():
                    rows_pp += f"""<tr>
                      <td style='padding:6px 10px;color:#00d4ff;'>:{int(r['port'])}</td>
                      <td style='padding:6px 10px;'>{r['service']}</td>
                      <td style='padding:6px 10px;text-align:right;'>{r['count']:,}</td>
                      <td style='padding:6px 10px;text-align:center;'>{r['risk']}</td>
                    </tr>"""
                st.markdown(f"""<table style='width:100%;border-collapse:collapse;font-size:0.7rem;'>
                  <thead><tr style='color:#4a6072;font-size:0.6rem;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid #1e2a38;'>
                    <th style='text-align:left;padding:6px 10px;'>Port</th>
                    <th style='text-align:left;padding:6px 10px;'>Service</th>
                    <th style='text-align:right;padding:6px 10px;'>Accès PERMIT</th>
                    <th style='text-align:center;padding:6px 10px;'>Risque</th>
                  </tr></thead>
                  <tbody style='border-top:1px solid #1e2a38;'>{rows_pp}</tbody>
                </table>""", unsafe_allow_html=True)
            with tc2:
                colors_pp = ["#ff3c6e" if int(r["port"]) in [22,23,3389,445,1433,3306,5900]
                             else "#ffb800" if int(r["port"]) in [80,443,21,25,53] else "#00d4ff"
                             for _,r in top_perm_ports.iterrows()]
                fig_pp = go.Figure(go.Bar(
                    x=top_perm_ports["count"],
                    y=[f":{int(r['port'])} {r['service']}" for _,r in top_perm_ports.iterrows()],
                    orientation="h", marker_color=colors_pp, opacity=0.9
                ))
                fig_pp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                      font_color="#c8d8e8", height=300, showlegend=False,
                                      xaxis=dict(gridcolor="#1e2a38", title="Nb accès PERMIT"),
                                      yaxis=dict(gridcolor="#1e2a38"),
                                      margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_pp, use_container_width=True)

        # LLM
        att_stats = {
            "tab": "top_attackers", "n": n_att,
            "top_attackers": [{"ip":r["ip_src"],"score":float(r["anomaly_score"]),
                               "connexions":int(r["nb_connexions"]),"ports":int(r["nb_ports_distincts"]),
                               "ratio_deny":float(r["ratio_deny"]),"profil":r["profil"]}
                              for _,r in suspects_b.head(8).iterrows()],
            "top_perm_ports": [{"port":int(r["port"]),"service":r["service"],"count":int(r["count"])}
                               for _,r in top_perm_ports.iterrows()] if not df_permit_b.empty else [],
        }
        render_ai_panel(
            key="behavior_top_att",
            label="⚡ Analyser les top attaquants",
            generate_fn=lambda key, model: generate_analysis("behavior", key, model, stats=att_stats),
            requires_key=False,
        )

    # ────────────────────────────────────────────────────────────
    # BT4 — CORRÉLATIONS & RADAR
    # ────────────────────────────────────────────────────────────
    with bt4:
        st.markdown("""<div class='story-banner'>
          <b style='font-family:Syne,sans-serif;color:#00ff9d;letter-spacing:2px;font-size:0.7rem;'>CORRÉLATIONS & RADAR — LA STRUCTURE DE LA MENACE</b><br><br>
          La matrice de corrélation révèle quelles features varient <span class='highlight'>ensemble</span> —
          clé pour comprendre pourquoi Isolation Forest est plus efficace qu'un seuillage simple.
          Le radar visualise l'<span class='danger'>empreinte comportementale normalisée</span> de chaque profil d'attaque.
        </div>""", unsafe_allow_html=True)

        FEAT_COLS = ["nb_connexions","nb_ports_distincts","nb_ips_dst","ratio_deny",
                     "nb_ports_sensibles","activite_nuit","port_dst_std"]
        feat_available = [f for f in FEAT_COLS + ["anomaly_score"] if f in ipf_b.columns]
        corr = ipf_b[feat_available].corr()

        st.markdown("<div class='section-hd'>Matrice de corrélation des features comportementales</div>", unsafe_allow_html=True)
        fig_corr = px.imshow(
            corr, color_continuous_scale="RdBu_r",
            text_auto=".2f", zmin=-1, zmax=1, aspect="auto"
        )
        fig_corr.update_traces(textfont_size=10)
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                font_color="#c8d8e8", height=420,
                                xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
                                yaxis=dict(tickfont=dict(size=9)),
                                coloraxis_colorbar=dict(tickfont=dict(color="#c8d8e8")),
                                margin=dict(t=0,b=40,l=0,r=0))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Radar chart
        st.markdown("<div class='section-hd'>Radar — Empreinte comportementale normalisée par profil</div>", unsafe_allow_html=True)
        radar_cats = [f for f in ["nb_connexions","nb_ports_distincts","ratio_deny","nb_ports_sensibles","activite_nuit"] if f in ipf_b.columns]
        pm = ipf_b.groupby("profil")[radar_cats].mean()
        pn = (pm - pm.min()) / (pm.max() - pm.min() + 1e-9)

        def hex_to_rgba(hex_color: str, alpha: float = 0.12) -> str:
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            return f"rgba({r},{g},{b},{alpha})"

        fig_radar = go.Figure()
        pal_r = ["#00c896","#4a90d9","#ff4b4b","#ff8c42","#a78bfa","#34d399"]
        for (pf, row), col in zip(pn.iterrows(), pal_r):
            v = row.tolist(); v.append(v[0])
            c_final = PROFIL_COLORS.get(pf, col)
            fig_radar.add_trace(go.Scatterpolar(
                r=v, theta=radar_cats + [radar_cats[0]],
                fill="toself", name=pf,
                line_color=c_final, opacity=0.75,
                fillcolor=hex_to_rgba(c_final, 0.12),
            ))
        fig_radar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8d8e8", family="Space Mono"),
            polar=dict(
                bgcolor="#0d1117",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e2a38", tickfont=dict(size=8)),
                angularaxis=dict(gridcolor="#1e2a38", tickfont=dict(size=9))
            ),
            height=480,
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2a38", font_size=11),
            margin=dict(t=20,b=20,l=20,r=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Évolution temporelle des DENY par protocole
        if "datetime" in df_raw.columns and not df_raw["datetime"].isna().all():
            st.markdown("<div class='section-hd'>Évolution temporelle des DENY par protocole (granularité 6h)</div>", unsafe_allow_html=True)
            pcol_b = "protocol_clean" if "protocol_clean" in df_raw.columns else "protocol"
            if pcol_b in df_raw.columns:
                deny_ts_b = (df_raw[df_raw["action"]=="DENY"]
                             .groupby([pd.Grouper(key="datetime", freq="6h"), pcol_b])
                             .size().reset_index(name="count"))
                fig_ev = px.area(
                    deny_ts_b, x="datetime", y="count", color=pcol_b,
                    color_discrete_map={"TCP":"#00d4ff","UDP":"#ffb800","OTHER":"#4a6072"}
                )
                fig_ev.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                                      font_color="#c8d8e8", height=300,
                                      legend=dict(bgcolor="rgba(0,0,0,0)"),
                                      xaxis=dict(gridcolor="#1e2a38"),
                                      yaxis=dict(gridcolor="#1e2a38", title="Nb connexions DENY"),
                                      margin=dict(t=0,b=0,l=0,r=0))
                st.plotly_chart(fig_ev, use_container_width=True)

        # LLM
        corr_flat = []
        for i, fa in enumerate(corr.columns):
            for j, fb in enumerate(corr.columns):
                if j > i and abs(corr.iloc[i,j]) > 0.2:
                    corr_flat.append({"feat_a":fa,"feat_b":fb,"corr":float(corr.iloc[i,j])})
        corr_flat.sort(key=lambda x: abs(x["corr"]), reverse=True)

        corr_stats = {
            "tab": "correlations",
            "top_correlations": corr_flat[:12],
        }
        render_ai_panel(
            key="behavior_corr",
            label="📊 Interpréter les corrélations",
            generate_fn=lambda key, model: generate_analysis("behavior", key, model, stats=corr_stats),
            requires_key=False,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — THREAT ANALYST IA (rapport global)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ia:
    st.markdown(
        """
    <div style='background:var(--bg2);border:1px solid var(--border);border-top:2px solid var(--accent3);
                border-radius:6px;padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;gap:16px;'>
      <div style='font-size:2.5rem;'>🛡</div>
      <div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:800;color:var(--text-hi);letter-spacing:1px;'>
          Threat Analyst IA — Rapport global de menaces
        </div>
        <div style='color:var(--text-dim);font-size:0.68rem;margin-top:4px;letter-spacing:1px;'>
          Synthèse exécutive · Analyse des menaces · Recommandations · Scoring de risque
        </div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    df_deny = (
        df_raw[df_raw["action"] == "DENY"]
        if "action" in df_raw.columns
        else pd.DataFrame()
    )
    n_deny = len(df_deny)
    n_ips = df_deny["ip_src"].nunique() if not df_deny.empty else 0
    n_ports = df_deny["port_dst"].nunique() if not df_deny.empty else 0
    top_port_val = (
        df_deny["port_dst"].value_counts().idxmax() if not df_deny.empty else "—"
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(
        f"<div class='stat-block'><div class='val'>{n_deny:,}</div><div class='lbl'>Tentatives DENY</div></div>",
        unsafe_allow_html=True,
    )
    d2.markdown(
        f"<div class='stat-block'><div class='val'>{n_ips:,}</div><div class='lbl'>IPs suspectes</div></div>",
        unsafe_allow_html=True,
    )
    d3.markdown(
        f"<div class='stat-block'><div class='val'>{n_ports}</div><div class='lbl'>Ports ciblés</div></div>",
        unsafe_allow_html=True,
    )
    d4.markdown(
        f"<div class='stat-block'><div class='val' style='font-size:1.4rem;'>:{top_port_val}</div><div class='lbl'>Port n°1 ciblé</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if df_deny.empty:
        st.warning("⚠️ Aucune connexion DENY dans les données.")
    else:
        geo_cache = st.session_state.get("geo_cache", {})
        render_ai_panel(
            key="global_threat",
            label="🛡 Générer le rapport global de menaces",
            generate_fn=lambda key, model: generate_analysis(
                "global_threat",
                key,
                model,
                stats={},
                df_deny=df_deny,
                geo_cache=geo_cache,
            ),
            requires_key=True,
        )

        st.markdown("---")
        st.markdown(
            "<div class='section-hd'>Aperçu des données DENY</div>",
            unsafe_allow_html=True,
        )
        preview_cols = [
            c
            for c in [
                "datetime",
                "ip_src",
                "ip_dst",
                "port_dst",
                "protocol_clean",
                "rule_id",
            ]
            if c in df_deny.columns
        ]
        if preview_cols:
            st.dataframe(
                df_deny[preview_cols].head(50),
                use_container_width=True,
                hide_index=True,
                height=280,
            )

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    """
<div style='text-align:center;color:#2a3a4a;font-size:0.62rem;letter-spacing:2px;padding:8px 0;'>
  NETFLOW SENTINEL v2 &nbsp;·&nbsp; Isolation Forest &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp;
  ip-api.com &nbsp;·&nbsp; Mistral AI &nbsp;·&nbsp; pydeck &nbsp;·&nbsp; Streamlit
</div>
""",
    unsafe_allow_html=True,
)
