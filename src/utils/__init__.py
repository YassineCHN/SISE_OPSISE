import math
import time
import requests
import streamlit as st


# ── Utilitaires réseau ────────────────────────────────────────────────────────

def port_label(port) -> str:
    """Retourne un libellé lisible pour un port destination."""
    known = {
        21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
        80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS",
        3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
        6379: "Redis", 8080: "HTTP-Alt", 8443: "HTTPS-Alt",
        27017: "MongoDB",
    }
    try:
        p = int(port)
        return f"{p} ({known[p]})" if p in known else str(p)
    except (ValueError, TypeError):
        return str(port)


def is_public(ip: str) -> bool:
    """Retourne True si l'IP est une adresse publique (non RFC1918/loopback)."""
    parts = ip.strip().split(".")
    if len(parts) != 4:
        return False
    try:
        a, b = int(parts[0]), int(parts[1])
        return not (
            a == 10
            or a == 127
            or (a == 172 and 16 <= b <= 31)
            or (a == 192 and b == 168)
        )
    except ValueError:
        return False


def geolocate_ips(ip_list: list) -> dict:
    """
    Géolocalise une liste d'IPs via ip-api.com (batch, 100 IP/requête).
    Utilise st.session_state.geo_cache pour éviter les requêtes redondantes.
    """
    cache  = st.session_state.get("geo_cache", {})
    unique = list({str(ip) for ip in ip_list if is_public(str(ip))} - set(cache))
    results = dict(cache)

    for i in range(0, len(unique), 100):
        batch = unique[i : i + 100]
        payload = [
            {"query": ip, "fields": "query,lat,lon,country,city,status"}
            for ip in batch
        ]
        try:
            resp = requests.post(
                "http://ip-api.com/batch?fields=query,lat,lon,country,city,status",
                json=payload,
                timeout=10,
            )
            if resp.status_code == 200:
                for item in resp.json():
                    if item.get("status") == "success":
                        results[item["query"]] = {
                            "lat":     item["lat"],
                            "lon":     item["lon"],
                            "country": item.get("country", "?"),
                            "city":    item.get("city", "?"),
                        }
        except Exception as e:
            st.warning(f"Erreur API ip-api.com : {e}")
        if i + 100 < len(unique):
            time.sleep(1.5)  # respecter la limite 45 req/min

    st.session_state.geo_cache = results
    return results


def arrow_angle(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcule l'angle de rotation (degrés) d'une flèche
    pointant de (lat1, lon1) vers (lat2, lon2). 0° = Est, sens anti-horaire.
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return math.degrees(math.atan2(dlat, dlon))
