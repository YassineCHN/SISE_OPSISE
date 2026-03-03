"""
utils.py — Fonctions utilitaires NetFlow Visualizer
Géolocalisation, flèches, prompt Mistral, streaming SSE.
"""
import math
import time
import json
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Chargement des variables d'environnement
# ─────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Normalize values to avoid false "key detected" due to quotes/spaces.
MISTRAL_API_KEY_ENV = os.getenv("MISTRAL_API_KEY", "").strip().strip('"').strip("'")
MISTRAL_MODEL_ENV = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip()

# ─────────────────────────────────────────────
# Mapping port → nom de service
# ─────────────────────────────────────────────
PORT_NAMES: dict[int, str] = {
    21:    "FTP",
    22:    "SSH",
    23:    "Telnet",
    25:    "SMTP",
    53:    "DNS",
    80:    "HTTP",
    110:   "POP3",
    143:   "IMAP",
    443:   "HTTPS",
    445:   "SMB/CIFS",
    1433:  "MSSQL",
    3306:  "MySQL",
    3389:  "RDP",
    5432:  "PostgreSQL",
    5900:  "VNC",
    6379:  "Redis",
    8080:  "HTTP-Alt",
    8443:  "HTTPS-Alt",
    27017: "MongoDB",
    9200:  "Elasticsearch",
}


def port_label(p) -> str:
    """Retourne le nom du service associé au port, ou le numéro brut."""
    try:
        return PORT_NAMES.get(int(p), str(p))
    except Exception:
        return str(p)


# ─────────────────────────────────────────────
# Géolocalisation
# ─────────────────────────────────────────────
def is_public(ip: str) -> bool:
    """Retourne True si l'IP est publique (non RFC-1918/loopback)."""
    try:
        parts = [int(x) for x in str(ip).strip().split(".")]
        if len(parts) != 4:
            return False
        a, b = parts[0], parts[1]
        return not (
            a == 10
            or a == 127
            or (a == 172 and 16 <= b <= 31)
            or (a == 192 and b == 168)
            or a == 0
            or a >= 240
        )
    except Exception:
        return False


def geolocate_ips(ip_list: list) -> dict:
    """
    Géolocalise une liste d'IPs via ip-api.com (batch, 100/req).
    Utilise le cache session Streamlit pour éviter les appels redondants.
    """
    results   = dict(st.session_state.geo_cache)
    to_fetch  = list({ip for ip in ip_list if is_public(ip) and ip not in results})

    for i in range(0, len(to_fetch), 100):
        batch   = to_fetch[i:i + 100]
        payload = [
            {"query": ip, "fields": "query,lat,lon,country,city,isp,status"}
            for ip in batch
        ]
        try:
            resp = requests.post(
                "http://ip-api.com/batch?fields=query,lat,lon,country,city,isp,status",
                json=payload,
                timeout=12,
            )
            if resp.status_code == 200:
                for item in resp.json():
                    if item.get("status") == "success":
                        results[item["query"]] = {
                            "lat":     item["lat"],
                            "lon":     item["lon"],
                            "country": item.get("country", "?"),
                            "city":    item.get("city",    "?"),
                            "isp":     item.get("isp",     "?"),
                        }
        except Exception as e:
            st.warning(f"⚠️ Erreur ip-api.com : {e}")

        if i + 100 < len(to_fetch):
            time.sleep(1.2)

    st.session_state.geo_cache = results
    return results


# ─────────────────────────────────────────────
# Calcul de l'angle pour les flèches pydeck
# ─────────────────────────────────────────────
def arrow_angle(slat: float, slon: float, dlat: float, dlon: float) -> float:
    """Angle CCW depuis l'est (convention TextLayer pydeck) pour pointer src→dst."""
    return math.degrees(math.atan2(dlat - slat, dlon - slon))


# ─────────────────────────────────────────────
# Prompt Mistral
# ─────────────────────────────────────────────
def build_threat_prompt(df_deny, geo_cache: dict) -> str:
    """
    Construit le prompt d'analyse de menaces envoyé au LLM Mistral.
    Intègre les Top IPs, ports ciblés et informations géographiques.
    """
    import pandas as pd

    total      = len(df_deny)
    top_ips    = df_deny["ip_src"].value_counts().head(5)   # réduit pour limiter les tokens
    top_ports  = df_deny["port_dst"].value_counts().head(5)  # réduit pour limiter les tokens
    top_protos = (
        df_deny["protocol_clean"].value_counts().head(3)
        if "protocol_clean" in df_deny.columns
        else pd.Series()
    )

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
        country = info.get("country", "Inconnu")
        city    = info.get("city",    "")
        isp     = info.get("isp",     "")
        ip_lines.append(f"  • {ip} — {cnt} tentatives — {city}, {country} (ISP: {isp})")

    port_lines = [
        f"  • Port {port} ({port_label(port)}) — {cnt} tentatives"
        for port, cnt in top_ports.items()
    ]

    proto_str = (
        ", ".join(f"{p}: {c}" for p, c in top_protos.items())
        if not top_protos.empty
        else "TCP majoritaire"
    )

    return f"""Tu es un analyste expert en cybersécurité réseau.
Analyse les données suivantes de trafic réseau BLOQUÉ (DENY) et rédige un rapport de menaces professionnel en français.

=== CONTEXTE ===
Date          : {date_str}
Plage horaire : {t_start} → {t_end}
Total DENY    : {total} connexions bloquées

=== TOP 10 IPs SOURCES (attaquants) ===
{chr(10).join(ip_lines)}

=== TOP PORTS CIBLÉS ===
{chr(10).join(port_lines)}

=== PROTOCOLES ===
{proto_str}

=== INSTRUCTIONS ===
Rédige un rapport structuré avec exactement ces sections (titres Markdown ##) :

## 📋 Résumé exécutif
(2-3 phrases narratives décrivant l'activité suspecte globale, avec heures et volume)

## 🔍 Analyse des menaces détectées
(Patterns identifiés : scans de ports, brute-force SSH/RDP, reconnaissance, etc.)

## 🎯 Top menaces
(Liste numérotée des 3-5 menaces principales : IP source · pays · port ciblé · type d'attaque probable)

## 🛡️ Recommandations
(Actions concrètes : CIDRs à bloquer, règles firewall iptables/ufw, alertes à configurer)

## 🔴 Score de risque global : XX/100
(Note de 0 à 100 avec justification en 1 phrase)

Sois précis et professionnel. Cite les IPs et ports réels des données."""


# ─────────────────────────────────────────────
# Appel Mistral en streaming SSE
# ─────────────────────────────────────────────
def stream_mistral(api_key: str, model: str, prompt: str):
    """
    Générateur : émet les chunks de texte du rapport Mistral en streaming SSE.
    Utilisation : for chunk in stream_mistral(key, model, prompt): ...
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens":  2500,
        "stream":      True,
    }
    with requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=90,
        stream=True,
    ) as resp:
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
