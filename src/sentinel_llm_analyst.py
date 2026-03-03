"""
llm_analyst.py — Module d'analyse IA pour NetFlow Sentinel
Gère 4 modes : incident_report, anomaly_analysis, classification_insight, temporal_analysis
Fallback templates si pas de clé API. Streaming SSE via Mistral.
"""

import json
import os
import requests
from typing import Generator

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _port_name(p) -> str:
    PORTS = {21:"FTP",22:"SSH",23:"Telnet",25:"SMTP",53:"DNS",80:"HTTP",
             110:"POP3",143:"IMAP",443:"HTTPS",445:"SMB",1433:"MSSQL",
             3306:"MySQL",3389:"RDP",5432:"PostgreSQL",5900:"VNC",
             6379:"Redis",8080:"HTTP-Alt",8443:"HTTPS-Alt"}
    try:
        return PORTS.get(int(p), str(p))
    except Exception:
        return str(p)


# ═══════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════

def build_anomaly_prompt(stats: dict) -> str:
    n_total       = stats.get("n_total", 0)
    n_anomalies   = stats.get("n_anomalies", 0)
    n_suspects    = stats.get("n_suspects", 0)
    profil_counts = stats.get("profil_counts", {})
    top_suspects  = stats.get("top_suspects", [])   # liste de dicts {ip, nb_connexions, nb_ports_distincts, ratio_deny, profil, anomaly_score}
    pct_anomalies = (n_anomalies / n_total * 100) if n_total else 0

    profil_lines = "\n".join([f"  • {p} : {c} IPs" for p, c in profil_counts.items()])
    suspect_lines = "\n".join([
        f"  • {s['ip']} — {s['nb_connexions']} cnx — {s['nb_ports_distincts']} ports — "
        f"ratio_deny={s['ratio_deny']:.0%} — profil={s['profil']} — score={s['anomaly_score']:.4f}"
        for s in top_suspects[:8]
    ])

    return f"""Tu es analyste SOC senior. Interprète les résultats d'une détection d'anomalies Isolation Forest sur du trafic réseau.

=== RÉSULTATS ISOLATION FOREST ===
IPs analysées       : {n_total:,}
Anomalies détectées : {n_anomalies:,} ({pct_anomalies:.1f}% du total)
IPs profil suspect  : {n_suspects:,}

=== RÉPARTITION DES PROFILS ===
{profil_lines}

=== TOP IPs SUSPECTES (les plus anormales) ===
{suspect_lines}

=== INSTRUCTIONS ===
Rédige une interprétation structurée en français avec ces sections (titres ##) :

## 🔬 Synthèse de la détection
(2-3 phrases : volume d'anomalies, proportion, gravité globale)

## 🧠 Analyse comportementale par profil
(Pour chaque profil non-Normal détecté, explique le comportement typique et le risque associé)

## 🎯 IPs prioritaires à investiguer
(Liste des 3-5 IPs les plus critiques avec justification basée sur les données)

## 🛡️ Actions recommandées
(Actions concrètes par profil : blocage, surveillance renforcée, corrélation SIEM)

Sois précis, cite les données réelles. Pas de contenu offensif."""


def build_classification_prompt(stats: dict) -> str:
    accuracy      = stats.get("accuracy", 0)
    top_feature   = stats.get("top_feature", "")
    top_feat_score= stats.get("top_feat_score", 0)
    n_classes     = stats.get("n_classes", 0)
    classes       = stats.get("classes", [])
    per_class     = stats.get("per_class", {})   # {classe: {precision, recall, f1, support}}
    importance    = stats.get("importance", {})  # {feature: score}
    cv_mean       = stats.get("cv_mean", None)

    class_lines = "\n".join([
        f"  • {cls} → F1={m.get('f1-score',0):.3f} | Précision={m.get('precision',0):.3f} | Rappel={m.get('recall',0):.3f} | Support={int(m.get('support',0))}"
        for cls, m in per_class.items() if cls not in ["accuracy","macro avg","weighted avg"]
    ])
    feat_lines = "\n".join([f"  • {f} : {s:.4f}" for f, s in list(importance.items())[:7]])
    cv_str = f"{cv_mean:.4f}" if cv_mean else "N/A"

    return f"""Tu es expert en machine learning appliqué à la cybersécurité. Interprète les résultats d'un classificateur Random Forest entraîné pour détecter des profils d'attaque réseau.

=== PERFORMANCE DU MODÈLE ===
Accuracy globale          : {accuracy:.2%}
Validation croisée (CV)   : {cv_str}
Nombre de classes         : {n_classes}
Feature la + discriminante : {top_feature} (importance={top_feat_score:.4f})

=== MÉTRIQUES PAR CLASSE ===
{class_lines}

=== IMPORTANCE DES FEATURES ===
{feat_lines}

=== INSTRUCTIONS ===
Rédige une interprétation structurée en français avec ces sections (titres ##) :

## 📊 Performance globale du modèle
(Évalue l'accuracy et la fiabilité du modèle pour un usage opérationnel)

## 🔍 Analyse des classes difficiles à classifier
(Identifie les classes avec F1 faible, explique pourquoi et le risque de faux négatifs)

## ⚙️ Interprétation des features discriminantes
(Explique pourquoi la feature dominante est si puissante pour distinguer les attaquants)

## 🚀 Intégration opérationnelle recommandée
(Comment déployer ce modèle en production : seuils, alertes, workflow SOC)

Sois précis, cite les scores réels. Pas de contenu offensif."""


def build_temporal_prompt(stats: dict) -> str:
    n_days        = stats.get("n_days", 0)
    t_start       = stats.get("t_start", "N/A")
    t_end         = stats.get("t_end", "N/A")
    n_pics        = stats.get("n_pics", 0)
    peak_hour     = stats.get("peak_hour", "N/A")
    low_hour      = stats.get("low_hour", "N/A")
    top_day       = stats.get("top_day", "N/A")
    deny_by_hour  = stats.get("deny_by_hour", {})   # {heure: count} top 5
    permit_by_hour= stats.get("permit_by_hour", {}) # {heure: count} top 5
    pics_details  = stats.get("pics_details", [])   # [{horodatage, count, zscore}]
    profil_hours  = stats.get("profil_hours", {})   # {profil: heure_pic}

    deny_lines   = "\n".join([f"  • {h}h : {c:,} DENY" for h, c in deny_by_hour.items()])
    permit_lines = "\n".join([f"  • {h}h : {c:,} PERMIT" for h, c in permit_by_hour.items()])
    pics_lines   = "\n".join([f"  • {p['horodatage']} — {p['count']:,} cnx — z={p['zscore']:.2f}" for p in pics_details[:5]])
    profil_lines = "\n".join([f"  • {prof} → pic à {h}h" for prof, h in profil_hours.items()])

    return f"""Tu es analyste cybersécurité spécialisé en threat hunting temporel. Interprète les patterns temporels d'un trafic réseau.

=== CONTEXTE ===
Période analysée : {t_start} → {t_end} ({n_days} jours)
Heure de pointe  : {peak_hour}
Heure creuse     : {low_hour}
Jour le + actif  : {top_day}
Pics Z-score>2.5 : {n_pics} détectés

=== HEURES DE POINTE — DENY ===
{deny_lines if deny_lines else "Données insuffisantes"}

=== HEURES DE POINTE — PERMIT ===
{permit_lines if permit_lines else "Données insuffisantes"}

=== PICS D'ACTIVITÉ ANORMAUX ===
{pics_lines if pics_lines else "Aucun pic détecté"}

=== PROFILS COMPORTEMENTAUX — HEURES DE PICS ===
{profil_lines if profil_lines else "Analyse de profils non disponible"}

=== INSTRUCTIONS ===
Rédige une interprétation structurée en français avec ces sections (titres ##) :

## ⏱️ Synthèse des patterns temporels
(Vue d'ensemble : quand le réseau est-il le plus exposé ?)

## 🌙 Analyse des heures critiques
(Interprète les pics DENY nocturnes ou hors-heures — indicateurs d'activité automatisée)

## 🚨 Interprétation des pics Z-score
(Pour chaque pic majeur : hypothèse sur la cause, corrélation avec profils connus)

## 📅 Patterns hebdomadaires détectés
(Y a-t-il des jours ou périodes récurrents à surveiller ?)

## ⏰ Recommandations de surveillance
(Créneaux horaires à monitorer en priorité, fenêtres d'alerte à configurer)

Sois précis, cite les heures et dates réelles. Pas de contenu offensif."""


def build_incident_prompt(ip: str, stats: dict, examples: list) -> str:
    """Rapport d'incident ciblé sur une IP spécifique (inspiré du template fourni)."""
    events       = stats.get("nb_connexions", 0)
    uniq_dst     = stats.get("nb_ips_dst", 0)
    uniq_ports   = stats.get("nb_ports_distincts", 0)
    ratio_deny   = stats.get("ratio_deny", 0)
    nb_sensibles = stats.get("nb_ports_sensibles", 0)
    activite_nuit= stats.get("activite_nuit", 0)
    port_std     = stats.get("port_dst_std", 0)
    profil       = stats.get("profil", "Normal")
    score        = stats.get("anomaly_score", 0)
    geo_info     = stats.get("geo", {})

    country = geo_info.get("country", "Inconnu")
    city    = geo_info.get("city", "")
    isp     = geo_info.get("isp", "")

    ex_lines = "\n".join([f"  {e}" for e in examples[:6]])

    return f"""Tu es analyste SOC senior. Génère un rapport d'incident détaillé et professionnel en français pour l'IP suivante.

=== IP ANALYSÉE ===
Adresse IP   : {ip}
Localisation : {city}, {country} (ISP: {isp})
Profil détecté : {profil}
Score d'anomalie Isolation Forest : {score:.4f}

=== COMPORTEMENT OBSERVÉ ===
Total connexions      : {events:,}
Destinations uniques  : {uniq_dst}
Ports distincts       : {uniq_ports}
Ratio DENY            : {ratio_deny:.1%}
Ports sensibles ciblés: {nb_sensibles}
Activité nocturne     : {activite_nuit:.1%}
Variance ports (std)  : {port_std:.1f}

=== EXEMPLES D'ÉVÉNEMENTS ===
{ex_lines if ex_lines else "Aucun exemple disponible."}

=== INSTRUCTIONS ===
Rédige un rapport d'incident structuré en français avec ces sections (titres ##) :

## 🔍 Résumé exécutif
(2-3 phrases résumant l'activité et la menace potentielle)

## 📊 Analyse comportementale
(Interprète chaque métrique : que signifie ce ratio DENY, ce nombre de ports, cette activité nocturne ?)

## 🎯 Classification de la menace
(Confirme ou nuance le profil détecté : {profil}. Niveau de confiance et justification)

## 🛡️ Recommandations immédiates
(Actions prioritaires : blocage, surveillance, corrélation avec d'autres sources)

## ⚠️ Score de risque : XX/100
(Note avec justification en 1 phrase)

Cite les données réelles. Pas de contenu offensif."""


def build_global_threat_prompt(df_deny, geo_cache: dict) -> str:
    """Rapport global de menaces (onglet Threat Analyst — version améliorée)."""
    import pandas as pd
    total     = len(df_deny)
    top_ips   = df_deny["ip_src"].value_counts().head(8)
    top_ports = df_deny["port_dst"].value_counts().head(8)
    top_protos= df_deny["protocol_clean"].value_counts().head(3) if "protocol_clean" in df_deny.columns else pd.Series()

    try:
        df_deny = df_deny.copy()
        df_deny["datetime"] = pd.to_datetime(df_deny["datetime"])
        t_start  = df_deny["datetime"].min().strftime("%H:%M:%S")
        t_end    = df_deny["datetime"].max().strftime("%H:%M:%S")
        date_str = df_deny["datetime"].min().strftime("%d/%m/%Y")
    except Exception:
        t_start = t_end = date_str = "N/A"

    ip_lines = []
    for ip, cnt in top_ips.items():
        info    = geo_cache.get(str(ip)) or {}
        country = info.get("country","Inconnu")
        city    = info.get("city","")
        isp     = info.get("isp","")
        ip_lines.append(f"  • {ip} — {cnt} tentatives — {city}, {country} (ISP: {isp})")

    port_lines = [f"  • Port {p} ({_port_name(p)}) — {c} tentatives" for p, c in top_ports.items()]
    proto_str  = ", ".join(f"{p}: {c}" for p, c in top_protos.items()) if not top_protos.empty else "TCP majoritaire"

    return f"""Tu es analyste expert en cybersécurité réseau. Analyse les données de trafic BLOQUÉ (DENY) et rédige un rapport de menaces exécutif en français.

=== CONTEXTE ===
Date          : {date_str}
Plage horaire : {t_start} → {t_end}
Total DENY    : {total:,} connexions bloquées

=== TOP IPs SOURCES (attaquants potentiels) ===
{chr(10).join(ip_lines)}

=== TOP PORTS CIBLÉS ===
{chr(10).join(port_lines)}

=== PROTOCOLES ===
{proto_str}

=== INSTRUCTIONS ===
Rédige un rapport structuré avec ces sections (titres ##) :

## 📋 Résumé exécutif
(3-4 phrases narratives : volume, heures, menace globale)

## 🔍 Analyse des menaces détectées
(Patterns identifiés : scans de ports, brute-force SSH/RDP, reconnaissance, botnets…)

## 🎯 Top menaces
(Liste numérotée des 5 menaces principales : IP · pays · port · type d'attaque probable)

## 🌍 Géographie des attaques
(Analyse des origines géographiques et leur signification opérationnelle)

## 🛡️ Recommandations
(Actions concrètes : CIDRs à bloquer, règles firewall, alertes SIEM)

## 🔴 Score de risque global : XX/100
(Note avec justification en 1 phrase)

Cite les IPs et ports réels. Sois précis et professionnel. Pas de contenu offensif."""


# ═══════════════════════════════════════════════════════════════
# FALLBACK TEMPLATES (sans clé API)
# ═══════════════════════════════════════════════════════════════

def _fallback_anomaly(stats: dict) -> str:
    n_anomalies = stats.get("n_anomalies", 0)
    n_total     = stats.get("n_total", 0)
    n_suspects  = stats.get("n_suspects", 0)
    profil_counts = stats.get("profil_counts", {})
    pct = (n_anomalies / n_total * 100) if n_total else 0

    profil_lines = "\n".join([f"- **{p}** : {c} IPs" for p, c in profil_counts.items()])
    severity = "élevée" if pct > 10 else "modérée" if pct > 3 else "faible"

    return f"""## 🔬 Synthèse de la détection

L'Isolation Forest a analysé **{n_total:,} IPs** et détecté **{n_anomalies:,} anomalies** ({pct:.1f}% du trafic) — gravité **{severity}**. {n_suspects:,} IPs présentent un profil comportemental suspect.

## 🧠 Analyse comportementale par profil

{profil_lines}

- **Port Scan** : IP contactant un grand nombre de ports distincts → reconnaissance automatisée
- **DDoS/Flood** : volume de connexions anormalement élevé → saturation possible
- **Attaque ciblée** : ports sensibles + ratio DENY élevé → intrusion intentionnelle
- **Activité nocturne** : activité concentrée entre 0h et 6h → automatisation suspecte
- **Comportement bloqué** : quasi-totalité des connexions bloquées → IP blacklistée ou mal configurée

## 🎯 IPs prioritaires à investiguer

Consultez la table ci-dessus : priorisez les IPs avec `anomaly_score` le plus bas (plus négatif = plus suspect) ET un profil non-Normal.

## 🛡️ Actions recommandées

- **Port Scan** → bloquer l'IP + configurer rate-limiting sur le firewall
- **DDoS/Flood** → activer la protection anti-flood, scrubbing si possible
- **Attaque ciblée** → investigation immédiate + corrélation WAF/IDS
- **Activité nocturne** → alerte SIEM sur les plages 0h-6h pour ces IPs

*(Rapport généré sans clé API — ajoutez votre clé Mistral pour une analyse IA approfondie)*"""


def _fallback_classification(stats: dict) -> str:
    accuracy  = stats.get("accuracy", 0)
    top_feat  = stats.get("top_feature", "N/A")
    top_score = stats.get("top_feat_score", 0)
    per_class = stats.get("per_class", {})
    cv_mean   = stats.get("cv_mean", None)

    classe_lines = "\n".join([
        f"- **{cls}** : F1={m.get('f1-score',0):.3f} — {'✅ Bonne détection' if m.get('f1-score',0) > 0.75 else '⚠️ À améliorer'}"
        for cls, m in per_class.items() if cls not in ["accuracy","macro avg","weighted avg"]
    ])
    cv_str = f"{cv_mean:.3f}" if cv_mean else "N/A"
    quality = "excellent" if accuracy > 0.9 else "bon" if accuracy > 0.75 else "acceptable"

    return f"""## 📊 Performance globale du modèle

Accuracy globale : **{accuracy:.2%}** (validation croisée : {cv_str}) — qualité **{quality}**.
Le modèle peut être utilisé {'en production avec confiance' if accuracy > 0.85 else 'avec supervision humaine'}.

## 🔍 Analyse des classes difficiles à classifier

{classe_lines}

Les classes avec F1 < 0.5 indiquent un manque d'exemples d'entraînement ou une similarité comportementale forte avec d'autres classes.

## ⚙️ Interprétation des features discriminantes

La feature **`{top_feat}`** (importance={top_score:.4f}) est la plus déterminante pour distinguer les comportements.
Cela signifie que {"le nb de ports distincts est le signal le plus fort d'un scan" if 'port' in top_feat else "le comportement réseau de base est le meilleur indicateur"}.

## 🚀 Intégration opérationnelle recommandée

- Scorer chaque nouvelle IP en temps réel dès son premier événement réseau
- Déclencher une alerte SIEM si `profil != Normal` ET score < -0.05
- Retrainer le modèle mensuellement avec les nouvelles données validées

*(Rapport généré sans clé API — ajoutez votre clé Mistral pour une analyse IA approfondie)*"""


def _fallback_temporal(stats: dict) -> str:
    n_pics   = stats.get("n_pics", 0)
    peak_hour= stats.get("peak_hour", "N/A")
    top_day  = stats.get("top_day", "N/A")
    n_days   = stats.get("n_days", 0)

    return f"""## ⏱️ Synthèse des patterns temporels

Sur **{n_days} jours** analysés, le trafic présente des patterns temporels nets. Le pic d'activité se situe à **{peak_hour}** et le jour le plus chargé est **{top_day}**.

## 🌙 Analyse des heures critiques

Les DENY concentrés en dehors des heures ouvrées (avant 8h / après 20h) indiquent une activité automatisée — bots, scanners, ou acteurs dans d'autres fuseaux horaires.

## 🚨 Interprétation des pics Z-score

**{n_pics} pic(s)** ont été détectés avec un Z-score > 2.5, soit une activité statistiquement anormale. Ces pics correspondent généralement à des campagnes de scan, des tentatives de brute-force coordonnées ou des incidents de déni de service.

## 📅 Patterns hebdomadaires détectés

Les jours **lundi et mardi** concentrent généralement plus d'activité suspecte (reprise des campagnes automatisées après le weekend). Les weekends montrent souvent plus d'activité nocturne.

## ⏰ Recommandations de surveillance

- Configurer des alertes renforcées sur les plages **22h-6h**
- Déclencher un rapport automatique lors de tout Z-score > 3.0
- Corréler les pics avec les logs applicatifs et les événements métier

*(Rapport généré sans clé API — ajoutez votre clé Mistral pour une analyse IA approfondie)*"""


def _fallback_incident(ip: str, stats: dict) -> str:
    profil   = stats.get("profil","Normal")
    events   = stats.get("nb_connexions", 0)
    ratio    = stats.get("ratio_deny", 0)
    ports    = stats.get("nb_ports_distincts", 0)
    score    = stats.get("anomaly_score", 0)
    country  = stats.get("geo",{}).get("country","Inconnu")

    risk_score = min(100, int(abs(score) * 200 + ratio * 30 + min(ports, 200) * 0.2))

    return f"""## 🔍 Résumé exécutif

L'IP **{ip}** ({country}) a généré **{events:,} connexions** avec un ratio DENY de **{ratio:.1%}**. Elle est classée comme **{profil}** par le moteur de détection.

## 📊 Analyse comportementale

- **Volume** : {events:,} connexions — {'anormalement élevé' if events > 500 else 'volume modéré'}
- **Ratio DENY** : {ratio:.1%} — {'très suspect' if ratio > 0.8 else 'préoccupant' if ratio > 0.5 else 'normal'}
- **Ports distincts** : {ports} — {'indicateur de scan actif' if ports > 50 else 'comportement ciblé'}

## 🎯 Classification de la menace

Profil confirmé : **{profil}**. Score d'anomalie Isolation Forest : {score:.4f} (plus négatif = plus suspect).

## 🛡️ Recommandations immédiates

- Vérifier la réputation de cette IP sur AbuseIPDB et Shodan
- {'Blocage immédiat recommandé' if ratio > 0.9 else 'Surveillance renforcée pendant 24h'}
- Corréler avec les logs WAF, IDS et SIEM
- Documenter dans le registre d'incidents

## ⚠️ Score de risque : {risk_score}/100

{'Risque critique — action immédiate requise' if risk_score > 75 else 'Risque élevé — surveillance prioritaire' if risk_score > 50 else 'Risque modéré — à monitorer'}

*(Rapport généré sans clé API — ajoutez votre clé Mistral pour une analyse IA approfondie)*"""


# ═══════════════════════════════════════════════════════════════
# STREAMING ENGINE
# ═══════════════════════════════════════════════════════════════

def stream_analysis(api_key: str, model: str, prompt: str) -> Generator[str, None, None]:
    """Générateur streaming SSE vers Mistral."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens":  2000,
        "stream":      True,
    }
    with requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=90,
        stream=True,
    ) as resp:
        # Lever une HTTPError pour tout code >= 400 (429, 401, 403…)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and line != b"data: [DONE]":
                raw = line.decode("utf-8")
                if raw.startswith("data: "):
                    try:
                        chunk   = json.loads(raw[6:])
                        delta   = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        pass


# ═══════════════════════════════════════════════════════════════
# MAIN GENERATE FUNCTION
# ═══════════════════════════════════════════════════════════════

def generate_analysis(mode: str, api_key: str, model: str,
                      stats: dict, ip: str = "", examples: list = None,
                      df_deny=None, geo_cache: dict = None):
    """
    Génère une analyse IA en streaming ou retourne un fallback.

    mode : "anomaly" | "classification" | "temporal" | "incident" | "global_threat"

    Yields str chunks si streaming, sinon retourne str directement.
    """
    examples = examples or []
    geo_cache = geo_cache or {}

    # Sélection du prompt
    if mode == "anomaly":
        prompt   = build_anomaly_prompt(stats)
        fallback = _fallback_anomaly(stats)
    elif mode == "classification":
        prompt   = build_classification_prompt(stats)
        fallback = _fallback_classification(stats)
    elif mode == "temporal":
        prompt   = build_temporal_prompt(stats)
        fallback = _fallback_temporal(stats)
    elif mode == "incident":
        prompt   = build_incident_prompt(ip, stats, examples)
        fallback = _fallback_incident(ip, stats)
    elif mode == "global_threat":
        prompt   = build_global_threat_prompt(df_deny, geo_cache)
        fallback = None
    elif mode == "behavior":
        tab = stats.get("tab", "src")
        prompt   = build_behavior_prompt(tab, stats)
        fallback = _fallback_behavior(tab, stats)
    else:
        yield "Mode inconnu."
        return

    if not api_key:
        yield fallback or "❌ Pas de clé API ni de fallback disponible."
        return

    try:
        yield from stream_analysis(api_key, model, prompt)

    except requests.exceptions.HTTPError as e:
        code = e.response.status_code if hasattr(e, "response") else "?"

        if code == 429:
            # Rate limit : on affiche un message clair et on retourne le fallback proprement
            retry_after = ""
            try:
                ra = e.response.headers.get("Retry-After") or e.response.headers.get("x-ratelimit-reset-requests")
                if ra:
                    retry_after = f" Réessayez dans ~{ra}s."
            except Exception:
                pass
            yield (
                f"⚠️ **Quota Mistral dépassé (429 — Rate Limit).{retry_after}**\n\n"
                f"Votre clé est valide mais le plan a atteint sa limite de requêtes par minute/jour.\n"
                f"👉 Attendez quelques secondes puis cliquez sur **Régénérer**.\n\n"
                f"---\n\n"
                f"*En attendant, voici l'analyse de secours :*\n\n"
            )
            if fallback:
                yield fallback
        elif code == 401:
            yield "❌ **Clé API invalide (401).** Vérifiez la clé dans la sidebar."
        elif code == 403:
            yield "❌ **Accès refusé (403).** Votre clé n'a pas accès à ce modèle."
        else:
            yield f"❌ **Erreur API Mistral ({code}).**\n\n"
            if fallback:
                yield fallback

    except requests.exceptions.ConnectionError:
        yield "❌ **Impossible de joindre l'API Mistral.** Vérifiez votre connexion internet.\n\n"
        if fallback:
            yield fallback

    except requests.exceptions.Timeout:
        yield "⏱️ **Timeout — l'API Mistral met trop de temps à répondre.** Réessayez dans quelques instants.\n\n"
        if fallback:
            yield fallback

    except Exception as e:
        yield f"❌ **Erreur inattendue :** `{e}`\n\n"
        if fallback:
            yield fallback


def build_behavior_prompt(tab: str, stats: dict) -> str:
    """Prompts pour l'onglet Comportement des attaques — 4 sous-analyses."""

    if tab == "src":
        top5        = stats.get("top5_src", [])
        profil_dist = stats.get("profil_dist", {})
        n_suspects  = stats.get("n_suspects", 0)
        top5_lines  = "\n".join([
            f"  • {r['ip']} — {r['connexions']:,} cnx — ratio_deny={r['ratio_deny']:.0%} — profil={r['profil']}"
            for r in top5
        ])
        profil_lines = "\n".join([f"  • {p} : {c} IPs" for p,c in profil_dist.items()])
        return f"""Tu es analyste SOC. Analyse le comportement des IP sources d'un réseau d'entreprise.

=== TOP 5 IP SOURCES LES PLUS ÉMETTRICES ===
{top5_lines}

=== RÉPARTITION DES PROFILS ({n_suspects} suspects) ===
{profil_lines}

Rédige en français avec ces sections (##) :
## 🌍 Analyse des IP sources suspectes
## 🔁 Patterns comportementaux identifiés
## 🛡️ Recommandations ciblées par profil

Cite les IPs réelles. 2-3 phrases par section. Pas de contenu offensif."""

    elif tab == "dst":
        top10_dst = stats.get("top10_dst", [])
        n_ext     = stats.get("n_ext_ips", 0)
        top_ext   = stats.get("top_ext", [])
        dst_lines = "\n".join([
            f"  • {r['ip']} — {r['connexions']:,} cnx — ratio_deny={r['ratio_deny']:.0%} — {r['nb_src']} sources"
            for r in top10_dst
        ])
        ext_lines = "\n".join([f"  • {r['ip']} — {r['connexions']:,} cnx" for r in top_ext[:8]])
        return f"""Tu es analyste SOC. Analyse les IP destinations d'un réseau et les accès hors plan d'adressage.

=== TOP 10 IP DESTINATIONS ===
{dst_lines}

=== IPs SOURCES HORS RFC 1918 ({n_ext} IPs externes détectées) ===
{ext_lines}

Rédige en français avec ces sections (##) :
## 🎯 Analyse des destinations les plus ciblées
## 🚨 Évaluation des accès hors plan d'adressage
## 🛡️ Recommandations de segmentation réseau

2-3 phrases par section. Pas de contenu offensif."""

    elif tab == "top_attackers":
        top_att   = stats.get("top_attackers", [])
        top_ports = stats.get("top_perm_ports", [])
        n         = stats.get("n", 15)
        att_lines = "\n".join([
            f"  • {r['ip']} — score={r['score']:.4f} — {r['connexions']:,} cnx — {r['ports']} ports — ratio_deny={r['ratio_deny']:.0%} — profil={r['profil']}"
            for r in top_att[:8]
        ])
        port_lines = "\n".join([
            f"  • Port {r['port']} ({r['service']}) — {r['count']:,} accès autorisés"
            for r in top_ports[:10]
        ])
        return f"""Tu es analyste SOC senior. Analyse les top {n} attaquants et les ports sensibles exposés.

=== TOP ATTAQUANTS (trié par score d'anomalie) ===
{att_lines}

=== PORTS SENSIBLES (<1024) AVEC ACCÈS AUTORISÉ ===
{port_lines}

Rédige en français avec ces sections (##) :
## 🔥 Profil des attaquants les plus dangereux
## ⚠️ Risques liés aux ports sensibles autorisés
## 🛡️ Contre-mesures prioritaires

Sois précis, cite les données réelles. 2-3 phrases par section. Pas de contenu offensif."""

    elif tab == "correlations":
        top_corr  = stats.get("top_correlations", [])
        corr_lines= "\n".join([
            f"  • {r['feat_a']} ↔ {r['feat_b']} : r={r['corr']:.3f}"
            for r in top_corr
        ])
        return f"""Tu es expert ML cybersécurité. Interprète une matrice de corrélation de features comportementales réseau.

=== CORRÉLATIONS LES PLUS SIGNIFICATIVES ===
{corr_lines}

Features : nb_connexions, nb_ports_distincts, nb_ips_dst, ratio_deny,
nb_ports_sensibles, activite_nuit, port_dst_std, anomaly_score

Rédige en français avec ces sections (##) :
## 🔗 Corrélations comportementales clés
## 🧠 Ce que révèle la structure des données
## ⚙️ Implications pour la détection automatisée

2-3 phrases par section. Pas de contenu offensif."""

    return "Mode inconnu."


def _fallback_behavior(tab: str, stats: dict) -> str:
    """Fallback templates pour l'onglet comportement."""
    if tab == "src":
        top5 = stats.get("top5_src", [])
        lines = "\n".join([f"- **{r['ip']}** : {r['connexions']:,} connexions — {r['profil']}" for r in top5])
        return f"""## 🌍 Analyse des IP sources suspectes

Les IP sources les plus actives montrent des signatures comportementales variées :

{lines}

## 🔁 Patterns comportementaux identifiés

Un ratio DENY élevé indique que le firewall bloque systématiquement cette IP — comportement typique d'une IP connue des blacklists ou d'un scanner automatisé. Les IP avec de nombreux ports distincts correspondent à de la reconnaissance active.

## 🛡️ Recommandations ciblées par profil

- **Port Scan** → bloquer le CIDR source, activer le rate-limiting
- **DDoS/Flood** → scrubbing + null-route temporaire
- **Attaque ciblée** → investigation immédiate WAF/IDS + corrélation SIEM

*(Rapport de secours — ajoutez une clé Mistral pour l'analyse IA)*"""

    elif tab == "dst":
        n_ext = stats.get("n_ext_ips", 0)
        return f"""## 🎯 Analyse des destinations les plus ciblées

Les IP destinations concentrant le plus de tentatives sont probablement des serveurs exposés (web, SSH, base de données). Un ratio DENY élevé sur une destination indique qu'elle est activement ciblée mais correctement protégée.

## 🚨 Évaluation des accès hors plan d'adressage

**{n_ext} IPs sources externes** (hors RFC 1918) ont été détectées. Ces adresses publiques contournent la logique d'adressage interne et représentent un risque d'intrusion ou de latéralisation.

## 🛡️ Recommandations de segmentation réseau

- Activer des règles de blocage sur les plages non attendues
- Implémenter du micro-segmentation sur les serveurs les plus ciblés
- Vérifier les règles NAT/PAT pour les accès entrants

*(Rapport de secours — ajoutez une clé Mistral pour l'analyse IA)*"""

    elif tab == "top_attackers":
        top_ports = stats.get("top_perm_ports", [])
        plines = "\n".join([f"- Port **{r['port']}** ({r['service']}) : {r['count']:,} accès" for r in top_ports[:5]])
        return f"""## 🔥 Profil des attaquants les plus dangereux

Les attaquants avec le score d'anomalie le plus bas (le plus négatif) combinent plusieurs signaux : volume élevé, diversité de ports, activité nocturne. Ce cumul de signaux indique une activité automatisée et persistante.

## ⚠️ Risques liés aux ports sensibles autorisés

{plines}

Ces ports privilégiés (<1024) autorisés représentent une surface d'attaque réelle. Tout service exposé sans MFA ou sans rate-limiting est une cible prioritaire.

## 🛡️ Contre-mesures prioritaires

- Bloquer les IPs avec score < -0.10 ET ratio_deny > 80%
- Activer MFA sur tous les ports sensibles exposés (SSH, RDP, MSSQL)
- Configurer des alertes SIEM sur toute connexion réussie depuis ces IPs

*(Rapport de secours — ajoutez une clé Mistral pour l'analyse IA)*"""

    elif tab == "correlations":
        return """## 🔗 Corrélations comportementales clés

Les features les plus corrélées entre elles tendent à co-varier dans les mêmes comportements suspects. Une forte corrélation entre `nb_ports_distincts` et `anomaly_score` confirme que la diversité des ports est le signal le plus discriminant.

## 🧠 Ce que révèle la structure des données

Le `ratio_deny` corrèle fortement avec `nb_ports_sensibles`, indiquant que les IPs ciblant des ports critiques sont aussi les plus bloquées — signe que la politique firewall est bien configurée sur ces vecteurs.

## ⚙️ Implications pour la détection automatisée

Ces corrélations justifient l'utilisation d'Isolation Forest plutôt qu'un simple seuillage : les anomalies sont multidimensionnelles et ne peuvent pas être détectées feature par feature.

*(Rapport de secours — ajoutez une clé Mistral pour l'analyse IA)*"""

    return "Mode non reconnu."


# """
# llm_analyst.py — Module d'analyse IA pour NetFlow Sentinel
# Gère 4 modes : incident_report, anomaly_analysis, classification_insight, temporal_analysis
# Fallback templates si pas de clé API. Streaming SSE via Mistral.
# """

# import json
# import os
# import requests
# from typing import Generator

# # ─────────────────────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────────────────────
# def _port_name(p) -> str:
#     PORTS = {21:"FTP",22:"SSH",23:"Telnet",25:"SMTP",53:"DNS",80:"HTTP",
#              110:"POP3",143:"IMAP",443:"HTTPS",445:"SMB",1433:"MSSQL",
#              3306:"MySQL",3389:"RDP",5432:"PostgreSQL",5900:"VNC",
#              6379:"Redis",8080:"HTTP-Alt",8443:"HTTPS-Alt"}
#     try:
#         return PORTS.get(int(p), str(p))
#     except Exception:
#         return str(p)


# # ═══════════════════════════════════════════════════════════════
# # PROMPTS
# # ═══════════════════════════════════════════════════════════════

# def build_anomaly_prompt(stats: dict) -> str:
#     n_total       = stats.get("n_total", 0)
#     n_anomalies   = stats.get("n_anomalies", 0)
#     n_suspects    = stats.get("n_suspects", 0)
#     profil_counts = stats.get("profil_counts", {})
#     top_suspects  = stats.get("top_suspects", [])   # liste de dicts {ip, nb_connexions, nb_ports_distincts, ratio_deny, profil, anomaly_score}
#     pct_anomalies = (n_anomalies / n_total * 100) if n_total else 0

#     profil_lines = "\n".join([f"  • {p} : {c} IPs" for p, c in profil_counts.items()])
#     suspect_lines = "\n".join([
#         f"  • {s['ip']} — {s['nb_connexions']} cnx — {s['nb_ports_distincts']} ports — "
#         f"ratio_deny={s['ratio_deny']:.0%} — profil={s['profil']} — score={s['anomaly_score']:.4f}"
#         for s in top_suspects[:8]
#     ])

#     return f"""Tu es analyste SOC senior. Interprète les résultats d'une détection d'anomalies Isolation Forest sur du trafic réseau.

# === RÉSULTATS ISOLATION FOREST ===
# IPs analysées       : {n_total:,}
# Anomalies détectées : {n_anomalies:,} ({pct_anomalies:.1f}% du total)
# IPs profil suspect  : {n_suspects:,}

# === RÉPARTITION DES PROFILS ===
# {profil_lines}

# === TOP IPs SUSPECTES (les plus anormales) ===
# {suspect_lines}

# === INSTRUCTIONS ===
# Rédige une interprétation structurée en français avec ces sections (titres ##) :

# ## 🔬 Synthèse de la détection
# (2-3 phrases : volume d'anomalies, proportion, gravité globale)

# ## 🧠 Analyse comportementale par profil
# (Pour chaque profil non-Normal détecté, explique le comportement typique et le risque associé)

# ## 🎯 IPs prioritaires à investiguer
# (Liste des 3-5 IPs les plus critiques avec justification basée sur les données)

# ## 🛡️ Actions recommandées
# (Actions concrètes par profil : blocage, surveillance renforcée, corrélation SIEM)

# Sois précis, cite les données réelles. Pas de contenu offensif."""


# def build_classification_prompt(stats: dict) -> str:
#     accuracy      = stats.get("accuracy", 0)
#     top_feature   = stats.get("top_feature", "")
#     top_feat_score= stats.get("top_feat_score", 0)
#     n_classes     = stats.get("n_classes", 0)
#     classes       = stats.get("classes", [])
#     per_class     = stats.get("per_class", {})   # {classe: {precision, recall, f1, support}}
#     importance    = stats.get("importance", {})  # {feature: score}
#     cv_mean       = stats.get("cv_mean", None)

#     class_lines = "\n".join([
#         f"  • {cls} → F1={m.get('f1-score',0):.3f} | Précision={m.get('precision',0):.3f} | Rappel={m.get('recall',0):.3f} | Support={int(m.get('support',0))}"
#         for cls, m in per_class.items() if cls not in ["accuracy","macro avg","weighted avg"]
#     ])
#     feat_lines = "\n".join([f"  • {f} : {s:.4f}" for f, s in list(importance.items())[:7]])
#     cv_str = f"{cv_mean:.4f}" if cv_mean else "N/A"

#     return f"""Tu es expert en machine learning appliqué à la cybersécurité. Interprète les résultats d'un classificateur Random Forest entraîné pour détecter des profils d'attaque réseau.

# === PERFORMANCE DU MODÈLE ===
# Accuracy globale          : {accuracy:.2%}
# Validation croisée (CV)   : {cv_str}
# Nombre de classes         : {n_classes}
# Feature la + discriminante : {top_feature} (importance={top_feat_score:.4f})

# === MÉTRIQUES PAR CLASSE ===
# {class_lines}

# === IMPORTANCE DES FEATURES ===
# {feat_lines}

# === INSTRUCTIONS ===
# Rédige une interprétation structurée en français avec ces sections (titres ##) :

# ## 📊 Performance globale du modèle
# (Évalue l'accuracy et la fiabilité du modèle pour un usage opérationnel)

# ## 🔍 Analyse des classes difficiles à classifier
# (Identifie les classes avec F1 faible, explique pourquoi et le risque de faux négatifs)

# ## ⚙️ Interprétation des features discriminantes
# (Explique pourquoi la feature dominante est si puissante pour distinguer les attaquants)

# ## 🚀 Intégration opérationnelle recommandée
# (Comment déployer ce modèle en production : seuils, alertes, workflow SOC)

# Sois précis, cite les scores réels. Pas de contenu offensif."""


# def build_temporal_prompt(stats: dict) -> str:
#     n_days        = stats.get("n_days", 0)
#     t_start       = stats.get("t_start", "N/A")
#     t_end         = stats.get("t_end", "N/A")
#     n_pics        = stats.get("n_pics", 0)
#     peak_hour     = stats.get("peak_hour", "N/A")
#     low_hour      = stats.get("low_hour", "N/A")
#     top_day       = stats.get("top_day", "N/A")
#     deny_by_hour  = stats.get("deny_by_hour", {})   # {heure: count} top 5
#     permit_by_hour= stats.get("permit_by_hour", {}) # {heure: count} top 5
#     pics_details  = stats.get("pics_details", [])   # [{horodatage, count, zscore}]
#     profil_hours  = stats.get("profil_hours", {})   # {profil: heure_pic}

#     deny_lines   = "\n".join([f"  • {h}h : {c:,} DENY" for h, c in deny_by_hour.items()])
#     permit_lines = "\n".join([f"  • {h}h : {c:,} PERMIT" for h, c in permit_by_hour.items()])
#     pics_lines   = "\n".join([f"  • {p['horodatage']} — {p['count']:,} cnx — z={p['zscore']:.2f}" for p in pics_details[:5]])
#     profil_lines = "\n".join([f"  • {prof} → pic à {h}h" for prof, h in profil_hours.items()])

#     return f"""Tu es analyste cybersécurité spécialisé en threat hunting temporel. Interprète les patterns temporels d'un trafic réseau.

# === CONTEXTE ===
# Période analysée : {t_start} → {t_end} ({n_days} jours)
# Heure de pointe  : {peak_hour}
# Heure creuse     : {low_hour}
# Jour le + actif  : {top_day}
# Pics Z-score>2.5 : {n_pics} détectés

# === HEURES DE POINTE — DENY ===
# {deny_lines if deny_lines else "Données insuffisantes"}

# === HEURES DE POINTE — PERMIT ===
# {permit_lines if permit_lines else "Données insuffisantes"}

# === PICS D'ACTIVITÉ ANORMAUX ===
# {pics_lines if pics_lines else "Aucun pic détecté"}

# === PROFILS COMPORTEMENTAUX — HEURES DE PICS ===
# {profil_lines if profil_lines else "Analyse de profils non disponible"}

# === INSTRUCTIONS ===
# Rédige une interprétation structurée en français avec ces sections (titres ##) :

# ## ⏱️ Synthèse des patterns temporels
# (Vue d'ensemble : quand le réseau est-il le plus exposé ?)

# ## 🌙 Analyse des heures critiques
# (Interprète les pics DENY nocturnes ou hors-heures — indicateurs d'activité automatisée)

# ## 🚨 Interprétation des pics Z-score
# (Pour chaque pic majeur : hypothèse sur la cause, corrélation avec profils connus)

# ## 📅 Patterns hebdomadaires détectés
# (Y a-t-il des jours ou périodes récurrents à surveiller ?)

# ## ⏰ Recommandations de surveillance
# (Créneaux horaires à monitorer en priorité, fenêtres d'alerte à configurer)

# Sois précis, cite les heures et dates réelles. Pas de contenu offensif."""


# def build_incident_prompt(ip: str, stats: dict, examples: list) -> str:
#     """Rapport d'incident ciblé sur une IP spécifique (inspiré du template fourni)."""
#     events       = stats.get("nb_connexions", 0)
#     uniq_dst     = stats.get("nb_ips_dst", 0)
#     uniq_ports   = stats.get("nb_ports_distincts", 0)
#     ratio_deny   = stats.get("ratio_deny", 0)
#     nb_sensibles = stats.get("nb_ports_sensibles", 0)
#     activite_nuit= stats.get("activite_nuit", 0)
#     port_std     = stats.get("port_dst_std", 0)
#     profil       = stats.get("profil", "Normal")
#     score        = stats.get("anomaly_score", 0)
#     geo_info     = stats.get("geo", {})

#     country = geo_info.get("country", "Inconnu")
#     city    = geo_info.get("city", "")
#     isp     = geo_info.get("isp", "")

#     ex_lines = "\n".join([f"  {e}" for e in examples[:6]])

#     return f"""Tu es analyste SOC senior. Génère un rapport d'incident détaillé et professionnel en français pour l'IP suivante.

# === IP ANALYSÉE ===
# Adresse IP   : {ip}
# Localisation : {city}, {country} (ISP: {isp})
# Profil détecté : {profil}
# Score d'anomalie Isolation Forest : {score:.4f}

# === COMPORTEMENT OBSERVÉ ===
# Total connexions      : {events:,}
# Destinations uniques  : {uniq_dst}
# Ports distincts       : {uniq_ports}
# Ratio DENY            : {ratio_deny:.1%}
# Ports sensibles ciblés: {nb_sensibles}
# Activité nocturne     : {activite_nuit:.1%}
# Variance ports (std)  : {port_std:.1f}

# === EXEMPLES D'ÉVÉNEMENTS ===
# {ex_lines if ex_lines else "Aucun exemple disponible."}

# === INSTRUCTIONS ===
# Rédige un rapport d'incident structuré en français avec ces sections (titres ##) :

# ## 🔍 Résumé exécutif
# (2-3 phrases résumant l'activité et la menace potentielle)

# ## 📊 Analyse comportementale
# (Interprète chaque métrique : que signifie ce ratio DENY, ce nombre de ports, cette activité nocturne ?)

# ## 🎯 Classification de la menace
# (Confirme ou nuance le profil détecté : {profil}. Niveau de confiance et justification)

# ## 🛡️ Recommandations immédiates
# (Actions prioritaires : blocage, surveillance, corrélation avec d'autres sources)

# ## ⚠️ Score de risque : XX/100
# (Note avec justification en 1 phrase)

# Cite les données réelles. Pas de contenu offensif."""


# def build_global_threat_prompt(df_deny, geo_cache: dict) -> str:
#     """Rapport global de menaces (onglet Threat Analyst — version améliorée)."""
#     import pandas as pd
#     total     = len(df_deny)
#     top_ips   = df_deny["ip_src"].value_counts().head(8)
#     top_ports = df_deny["port_dst"].value_counts().head(8)
#     top_protos= df_deny["protocol_clean"].value_counts().head(3) if "protocol_clean" in df_deny.columns else pd.Series()

#     try:
#         df_deny = df_deny.copy()
#         df_deny["datetime"] = pd.to_datetime(df_deny["datetime"])
#         t_start  = df_deny["datetime"].min().strftime("%H:%M:%S")
#         t_end    = df_deny["datetime"].max().strftime("%H:%M:%S")
#         date_str = df_deny["datetime"].min().strftime("%d/%m/%Y")
#     except Exception:
#         t_start = t_end = date_str = "N/A"

#     ip_lines = []
#     for ip, cnt in top_ips.items():
#         info    = geo_cache.get(str(ip)) or {}
#         country = info.get("country","Inconnu")
#         city    = info.get("city","")
#         isp     = info.get("isp","")
#         ip_lines.append(f"  • {ip} — {cnt} tentatives — {city}, {country} (ISP: {isp})")

#     port_lines = [f"  • Port {p} ({_port_name(p)}) — {c} tentatives" for p, c in top_ports.items()]
#     proto_str  = ", ".join(f"{p}: {c}" for p, c in top_protos.items()) if not top_protos.empty else "TCP majoritaire"

#     return f"""Tu es analyste expert en cybersécurité réseau. Analyse les données de trafic BLOQUÉ (DENY) et rédige un rapport de menaces exécutif en français.

# === CONTEXTE ===
# Date          : {date_str}
# Plage horaire : {t_start} → {t_end}
# Total DENY    : {total:,} connexions bloquées

# === TOP IPs SOURCES (attaquants potentiels) ===
# {chr(10).join(ip_lines)}

# === TOP PORTS CIBLÉS ===
# {chr(10).join(port_lines)}

# === PROTOCOLES ===
# {proto_str}

# === INSTRUCTIONS ===
# Rédige un rapport structuré avec ces sections (titres ##) :

# ## 📋 Résumé exécutif
# (3-4 phrases narratives : volume, heures, menace globale)

# ## 🔍 Analyse des menaces détectées
# (Patterns identifiés : scans de ports, brute-force SSH/RDP, reconnaissance, botnets…)

# ## 🎯 Top menaces
# (Liste numérotée des 5 menaces principales : IP · pays · port · type d'attaque probable)

# ## 🌍 Géographie des attaques
# (Analyse des origines géographiques et leur signification opérationnelle)

# ## 🛡️ Recommandations
# (Actions concrètes : CIDRs à bloquer, règles firewall, alertes SIEM)

# ## 🔴 Score de risque global : XX/100
# (Note avec justification en 1 phrase)

# Cite les IPs et ports réels. Sois précis et professionnel. Pas de contenu offensif."""


# # ═══════════════════════════════════════════════════════════════
# # FALLBACK TEMPLATES (sans clé API)
# # ═══════════════════════════════════════════════════════════════

# def _fallback_anomaly(stats: dict) -> str:
#     n_anomalies = stats.get("n_anomalies", 0)
#     n_total     = stats.get("n_total", 0)
#     n_suspects  = stats.get("n_suspects", 0)
#     profil_counts = stats.get("profil_counts", {})
#     pct = (n_anomalies / n_total * 100) if n_total else 0

#     profil_lines = "\n".join([f"- **{p}** : {c} IPs" for p, c in profil_counts.items()])
#     severity = "élevée" if pct > 10 else "modérée" if pct > 3 else "faible"

#     return f"""## 🔬 Synthèse de la détection

# L'Isolation Forest a analysé **{n_total:,} IPs** et détecté **{n_anomalies:,} anomalies** ({pct:.1f}% du trafic) — gravité **{severity}**. {n_suspects:,} IPs présentent un profil comportemental suspect.

# ## 🧠 Analyse comportementale par profil

# {profil_lines}

# - **Port Scan** : IP contactant un grand nombre de ports distincts → reconnaissance automatisée
# - **DDoS/Flood** : volume de connexions anormalement élevé → saturation possible
# - **Attaque ciblée** : ports sensibles + ratio DENY élevé → intrusion intentionnelle
# - **Activité nocturne** : activité concentrée entre 0h et 6h → automatisation suspecte
# - **Comportement bloqué** : quasi-totalité des connexions bloquées → IP blacklistée ou mal configurée

# ## 🎯 IPs prioritaires à investiguer

# Consultez la table ci-dessus : priorisez les IPs avec `anomaly_score` le plus bas (plus négatif = plus suspect) ET un profil non-Normal.

# ## 🛡️ Actions recommandées

# - **Port Scan** → bloquer l'IP + configurer rate-limiting sur le firewall
# - **DDoS/Flood** → activer la protection anti-flood, scrubbing si possible
# - **Attaque ciblée** → investigation immédiate + corrélation WAF/IDS
# - **Activité nocturne** → alerte SIEM sur les plages 0h-6h pour ces IPs

# *(Rapport généré sans clé API — ajoutez votre clé Mistral pour une analyse IA approfondie)*"""


# def _fallback_classification(stats: dict) -> str:
#     accuracy  = stats.get("accuracy", 0)
#     top_feat  = stats.get("top_feature", "N/A")
#     top_score = stats.get("top_feat_score", 0)
#     per_class = stats.get("per_class", {})
#     cv_mean   = stats.get("cv_mean", None)

#     classe_lines = "\n".join([
#         f"- **{cls}** : F1={m.get('f1-score',0):.3f} — {'✅ Bonne détection' if m.get('f1-score',0) > 0.75 else '⚠️ À améliorer'}"
#         for cls, m in per_class.items() if cls not in ["accuracy","macro avg","weighted avg"]
#     ])
#     cv_str = f"{cv_mean:.3f}" if cv_mean else "N/A"
#     quality = "excellent" if accuracy > 0.9 else "bon" if accuracy > 0.75 else "acceptable"

#     return f"""## 📊 Performance globale du modèle

# Accuracy globale : **{accuracy:.2%}** (validation croisée : {cv_str}) — qualité **{quality}**.
# Le modèle peut être utilisé {'en production avec confiance' if accuracy > 0.85 else 'avec supervision humaine'}.

# ## 🔍 Analyse des classes difficiles à classifier

# {classe_lines}

# Les classes avec F1 < 0.5 indiquent un manque d'exemples d'entraînement ou une similarité comportementale forte avec d'autres classes.

# ## ⚙️ Interprétation des features discriminantes

# La feature **`{top_feat}`** (importance={top_score:.4f}) est la plus déterminante pour distinguer les comportements.
# Cela signifie que {'le nombre de ports distincts contactés est le signal le plus fort dune activité de scan' if 'port' in top_feat else 'le comportement réseau de base est le meilleur indicateur'}.

# ## 🚀 Intégration opérationnelle recommandée

# - Scorer chaque nouvelle IP en temps réel dès son premier événement réseau
# - Déclencher une alerte SIEM si `profil != Normal` ET score < -0.05
# - Retrainer le modèle mensuellement avec les nouvelles données validées

# *(Rapport généré sans clé API — ajoutez votre clé Mistral pour une analyse IA approfondie)*"""


# def _fallback_temporal(stats: dict) -> str:
#     n_pics   = stats.get("n_pics", 0)
#     peak_hour= stats.get("peak_hour", "N/A")
#     top_day  = stats.get("top_day", "N/A")
#     n_days   = stats.get("n_days", 0)

#     return f"""## ⏱️ Synthèse des patterns temporels

# Sur **{n_days} jours** analysés, le trafic présente des patterns temporels nets. Le pic d'activité se situe à **{peak_hour}** et le jour le plus chargé est **{top_day}**.

# ## 🌙 Analyse des heures critiques

# Les DENY concentrés en dehors des heures ouvrées (avant 8h / après 20h) indiquent une activité automatisée — bots, scanners, ou acteurs dans d'autres fuseaux horaires.

# ## 🚨 Interprétation des pics Z-score

# **{n_pics} pic(s)** ont été détectés avec un Z-score > 2.5, soit une activité statistiquement anormale. Ces pics correspondent généralement à des campagnes de scan, des tentatives de brute-force coordonnées ou des incidents de déni de service.

# ## 📅 Patterns hebdomadaires détectés

# Les jours **lundi et mardi** concentrent généralement plus d'activité suspecte (reprise des campagnes automatisées après le weekend). Les weekends montrent souvent plus d'activité nocturne.

# ## ⏰ Recommandations de surveillance

# - Configurer des alertes renforcées sur les plages **22h-6h**
# - Déclencher un rapport automatique lors de tout Z-score > 3.0
# - Corréler les pics avec les logs applicatifs et les événements métier

# *(Rapport généré sans clé API — ajoutez votre clé Mistral pour une analyse IA approfondie)*"""


# def _fallback_incident(ip: str, stats: dict) -> str:
#     profil   = stats.get("profil","Normal")
#     events   = stats.get("nb_connexions", 0)
#     ratio    = stats.get("ratio_deny", 0)
#     ports    = stats.get("nb_ports_distincts", 0)
#     score    = stats.get("anomaly_score", 0)
#     country  = stats.get("geo",{}).get("country","Inconnu")

#     risk_score = min(100, int(abs(score) * 200 + ratio * 30 + min(ports, 200) * 0.2))

#     return f"""## 🔍 Résumé exécutif

# L'IP **{ip}** ({country}) a généré **{events:,} connexions** avec un ratio DENY de **{ratio:.1%}**. Elle est classée comme **{profil}** par le moteur de détection.

# ## 📊 Analyse comportementale

# - **Volume** : {events:,} connexions — {'anormalement élevé' if events > 500 else 'volume modéré'}
# - **Ratio DENY** : {ratio:.1%} — {'très suspect' if ratio > 0.8 else 'préoccupant' if ratio > 0.5 else 'normal'}
# - **Ports distincts** : {ports} — {'indicateur de scan actif' if ports > 50 else 'comportement ciblé'}

# ## 🎯 Classification de la menace

# Profil confirmé : **{profil}**. Score d'anomalie Isolation Forest : {score:.4f} (plus négatif = plus suspect).

# ## 🛡️ Recommandations immédiates

# - Vérifier la réputation de cette IP sur AbuseIPDB et Shodan
# - {'Blocage immédiat recommandé' if ratio > 0.9 else 'Surveillance renforcée pendant 24h'}
# - Corréler avec les logs WAF, IDS et SIEM
# - Documenter dans le registre d'incidents

# ## ⚠️ Score de risque : {risk_score}/100

# {'Risque critique — action immédiate requise' if risk_score > 75 else 'Risque élevé — surveillance prioritaire' if risk_score > 50 else 'Risque modéré — à monitorer'}

# *(Rapport généré sans clé API — ajoutez votre clé Mistral pour une analyse IA approfondie)*"""


# # ═══════════════════════════════════════════════════════════════
# # STREAMING ENGINE
# # ═══════════════════════════════════════════════════════════════

# def stream_analysis(api_key: str, model: str, prompt: str) -> Generator[str, None, None]:
#     """Générateur streaming SSE vers Mistral."""
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type":  "application/json",
#     }
#     payload = {
#         "model":       model,
#         "messages":    [{"role": "user", "content": prompt}],
#         "temperature": 0.3,
#         "max_tokens":  2000,
#         "stream":      True,
#     }
#     with requests.post(
#         "https://api.mistral.ai/v1/chat/completions",
#         headers=headers,
#         json=payload,
#         timeout=90,
#         stream=True,
#     ) as resp:
#         resp.raise_for_status()
#         for line in resp.iter_lines():
#             if line and line != b"data: [DONE]":
#                 raw = line.decode("utf-8")
#                 if raw.startswith("data: "):
#                     try:
#                         chunk   = json.loads(raw[6:])
#                         delta   = chunk["choices"][0].get("delta", {})
#                         content = delta.get("content", "")
#                         if content:
#                             yield content
#                     except Exception:
#                         pass


# # ═══════════════════════════════════════════════════════════════
# # MAIN GENERATE FUNCTION
# # ═══════════════════════════════════════════════════════════════

# def generate_analysis(mode: str, api_key: str, model: str,
#                       stats: dict, ip: str = "", examples: list = None,
#                       df_deny=None, geo_cache: dict = None):
#     """
#     Génère une analyse IA en streaming ou retourne un fallback.

#     mode : "anomaly" | "classification" | "temporal" | "incident" | "global_threat"

#     Yields str chunks si streaming, sinon retourne str directement.
#     """
#     examples = examples or []
#     geo_cache = geo_cache or {}

#     # Sélection du prompt
#     if mode == "anomaly":
#         prompt   = build_anomaly_prompt(stats)
#         fallback = _fallback_anomaly(stats)
#     elif mode == "classification":
#         prompt   = build_classification_prompt(stats)
#         fallback = _fallback_classification(stats)
#     elif mode == "temporal":
#         prompt   = build_temporal_prompt(stats)
#         fallback = _fallback_temporal(stats)
#     elif mode == "incident":
#         prompt   = build_incident_prompt(ip, stats, examples)
#         fallback = _fallback_incident(ip, stats)
#     elif mode == "global_threat":
#         prompt   = build_global_threat_prompt(df_deny, geo_cache)
#         fallback = None
#     else:
#         yield "Mode inconnu."
#         return

#     if not api_key:
#         yield fallback or "❌ Pas de clé API ni de fallback disponible."
#         return

#     try:
#         yield from stream_analysis(api_key, model, prompt)
#     except requests.exceptions.HTTPError as e:
#         code = e.response.status_code if hasattr(e, "response") else "?"
#         yield (fallback or "") + f"\n\n*(Erreur API Mistral {code} — vérifiez votre clé)*"
#     except Exception as e:
#         yield (fallback or "") + f"\n\n*(Erreur : {e})*"