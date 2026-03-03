import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from components.top_nav import render_top_nav
from components.sentinel_theme import apply_sentinel_theme
from components.ui import neon_metric
from app_config import ACTION_COLORS, COLUMN_LABELS, TOP_N_DEFAULT
from modules.charts import area_chart, bar_chart, heatmap, pie_chart
from modules.components.filters import render_sidebar_filters
from modules.preprocessing import load_data
from modules.stats import (
    action_distribution,
    blocked_ratio,
    port_category_distribution,
    protocol_action_crosstab,
    top_n,
    traffic_by_hour,
    traffic_by_period,
    traffic_by_weekday,
    unique_counts,
)

st.set_page_config(
    page_title="Visualisation",
    page_icon="📊",
    layout="wide",
)

render_top_nav("viz")
apply_sentinel_theme()


@st.cache_data
def get_data():
    return load_data()


df_full = get_data()
df, _params = render_sidebar_filters(df_full)

st.title("📊 Visualisation unifiée")
st.caption("Analyse descriptive · DataTable · Visualisation IP · Statistiques")

if df.empty:
    st.warning("⚠️ Aucune donnée pour les filtres sélectionnés.")
    st.stop()

ucounts = unique_counts(df)
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    neon_metric("📦 Flux total",       f"{len(df):,}")
with c2:
    neon_metric("🌐 IP sources",       f"{ucounts['ip_src']:,}", color="var(--accent3)")
with c3:
    neon_metric("🎯 IP destinations",  f"{ucounts['ip_dst']:,}", color="var(--accent3)")
with c4:
    neon_metric("🔌 Protocoles",       f"{ucounts['protocol']:,}", color="var(--accent4)")
with c5:
    neon_metric("🚫 Trafic bloqué",    f"{blocked_ratio(df):.1f} %", color="var(--accent2)")

(tab_desc, tab_table, tab_ip, tab_stats) = st.tabs(
    ["📊 Analyse descriptive", "📋 DataTable", "🌐 Visualisation IP", "📈 Statistiques"]
)

with tab_desc:
    ca, cb = st.columns(2)
    with ca:
        st.subheader("🎯 Distribution des actions")
        act_df = action_distribution(df)
        st.plotly_chart(
            pie_chart(act_df, names="action", color_map=ACTION_COLORS),
            use_container_width=True,
        )
    with cb:
        st.subheader(f"🔌 Top {TOP_N_DEFAULT} protocoles")
        proto_df = top_n(df, "protocol", TOP_N_DEFAULT)
        st.plotly_chart(bar_chart(proto_df, x="protocol"), use_container_width=True)

    st.subheader("📈 Volume de trafic dans le temps")
    freq_map = {"Heure": "h", "Jour": "D", "Semaine": "W"}
    freq_label = st.radio("Granularité", list(freq_map.keys()), horizontal=True, key="desc_freq")
    timeline_df = traffic_by_period(df, freq=freq_map[freq_label])
    st.plotly_chart(
        area_chart(timeline_df, x="datetime", y="count", title=f"Trafic par {freq_label.lower()}"),
        use_container_width=True,
    )

    st.subheader(f"🔢 Top {TOP_N_DEFAULT} ports destination")
    ports_df = top_n(df, "port_dst", TOP_N_DEFAULT)
    ports_df["port_dst"] = ports_df["port_dst"].astype(str)
    st.plotly_chart(bar_chart(ports_df, x="port_dst"), use_container_width=True)

with tab_table:
    st.subheader("📋 Table de données")
    search = st.text_input("🔍 Recherche rapide (IP, protocole, action...)", "", key="table_search")

    display_df = df.rename(columns=COLUMN_LABELS)
    if search:
        mask = display_df.apply(lambda col: col.astype(str).str.contains(search, case=False, na=False))
        display_df = display_df[mask.any(axis=1)]

    all_cols = list(display_df.columns)
    selected_cols = st.multiselect("📑 Colonnes à afficher", all_cols, default=all_cols, key="table_cols")
    display_df = display_df[selected_cols]

    page_size = 1000
    total_rows = len(display_df)
    total_pages = max(1, -(-total_rows // page_size))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="table_page")

    start = (page - 1) * page_size
    end = start + page_size
    st.dataframe(display_df.iloc[start:end], use_container_width=True, height=600)
    st.caption(f"📄 Page {page}/{total_pages} · {total_rows:,} lignes au total")

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Télécharger CSV",
        data=csv,
        file_name="logs_firewall_filtres.csv",
        mime="text/csv",
    )

with tab_ip:
    st.subheader("🌐 Visualisation IP")
    n = st.slider("Nombre de top IPs à afficher", 5, 50, TOP_N_DEFAULT, key="ip_topn")

    ia, ib = st.columns(2)
    with ia:
        st.subheader(f"⬆️ Top {n} IP sources")
        src_df = top_n(df, "ip_src", n)
        st.plotly_chart(bar_chart(src_df, x="ip_src", horizontal=True), use_container_width=True)
    with ib:
        st.subheader(f"⬇️ Top {n} IP destinations")
        dst_df = top_n(df, "ip_dst", n)
        st.plotly_chart(bar_chart(dst_df, x="ip_dst", horizontal=True), use_container_width=True)

    st.subheader("🎯 Distribution des actions — Top IP sources")
    top_src_ips = top_n(df, "ip_src", 10)["ip_src"].tolist()
    df_top = df[df["ip_src"].isin(top_src_ips)]
    action_by_ip = df_top.groupby(["ip_src", "action"]).size().reset_index(name="count")
    fig = px.bar(
        action_by_ip,
        x="ip_src",
        y="count",
        color="action",
        barmode="stack",
        color_discrete_map=ACTION_COLORS,
        labels={"ip_src": "IP Source", "count": "Événements", "action": "Action"},
    )
    st.plotly_chart(fig, use_container_width=True)

    ic, id_ = st.columns(2)
    with ic:
        st.subheader("🔀 Interfaces d'entrée")
        iface_in = top_n(df, "interface_in", 10)
        st.plotly_chart(pie_chart(iface_in, names="interface_in", title="Interface entrée"), use_container_width=True)
    with id_:
        st.subheader("🔀 Interfaces de sortie")
        iface_out = top_n(df, "interface_out", 10)
        st.plotly_chart(pie_chart(iface_out, names="interface_out", title="Interface sortie"), use_container_width=True)

with tab_stats:
    st.subheader("🔥 Protocole × Action")
    ct = protocol_action_crosstab(df)
    st.plotly_chart(heatmap(ct), use_container_width=True)

    sa, sb = st.columns(2)
    with sa:
        st.subheader(f"📋 Top {TOP_N_DEFAULT} règles déclenchées")
        rules_df = top_n(df, "rule_id", TOP_N_DEFAULT)
        rules_df["rule_id"] = "Règle " + rules_df["rule_id"].astype(str)
        st.plotly_chart(bar_chart(rules_df, x="rule_id"), use_container_width=True)
    with sb:
        st.subheader("🔢 Distribution des plages de ports")
        port_cat_df = port_category_distribution(df)
        port_label_col = port_cat_df.columns[0]
        st.plotly_chart(pie_chart(port_cat_df, names=port_label_col), use_container_width=True)

    sc, sd = st.columns(2)
    with sc:
        st.subheader("🕐 Trafic par heure de la journée")
        hourly = traffic_by_hour(df)
        fig_h = px.bar(
            hourly,
            x="hour",
            y="count",
            labels={"hour": "Heure", "count": "Événements"},
            color_discrete_sequence=["#00d4ff"],
        )
        fig_h.update_xaxes(dtick=1, title="Heure")
        st.plotly_chart(fig_h, use_container_width=True)
    with sd:
        st.subheader("📅 Trafic par jour de la semaine")
        weekly = traffic_by_weekday(df)
        fig_w = px.bar(
            weekly,
            x="jour",
            y="count",
            labels={"jour": "Jour", "count": "Événements"},
            color_discrete_sequence=["#a259ff"],
        )
        st.plotly_chart(fig_w, use_container_width=True)
