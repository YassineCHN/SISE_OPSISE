from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/log_clean.parquet")

# ── App ──────────────────────────────────────────────────────────────────
APP_TITLE = "SISE – Analyse des logs Firewall"
APP_ICON = "🔥"
LAYOUT = "wide"

# ── Libellés des colonnes (fr) ────────────────────────────────────────────
COLUMN_LABELS = {
    "datetime":      "Horodatage",
    "ip_src":        "IP Source",
    "ip_dst":        "IP Destination",
    "protocol":      "Protocole",
    "port_dst":      "Port Destination",
    "action":        "Action",
    "rule_id":       "Règle",
    "interface_in":  "Interface Entrée",
    "interface_out": "Interface Sortie",
}

# ── Couleurs par action ───────────────────────────────────────────────────
ACTION_COLORS = {
    "ACCEPT": "#2ECC71",
    "DROP":   "#E74C3C",
    "REJECT": "#E67E22",
    "LOG":    "#3498DB",
}

# ── Palette générale (Plotly) ─────────────────────────────────────────────
COLOR_SEQUENCE = [
    "#3498DB", "#E74C3C", "#2ECC71", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#34495E",
    "#E91E63", "#00BCD4",
]

# ── Plages de ports (RFC 6056) ────────────────────────────────────────────
PORT_RANGES = {
    "Tous":                    None,
    "Well-known (0–1023)":     "well_known",
    "Registered (1024–49151)": "registered",
    "Dynamic (49152–65535)":   "dynamic",
}

# ── Valeurs par défaut ────────────────────────────────────────────────────
TOP_N_DEFAULT = 10
