from pathlib import Path

DATA_PATH = Path("data/processed/log_clean.parquet")

APP_TITLE = "SISE - Analyse des logs Firewall"
APP_ICON = "F"
LAYOUT = "wide"

COLUMN_LABELS = {
    "datetime": "Horodatage",
    "ip_src": "IP Source",
    "ip_dst": "IP Destination",
    "protocol": "Protocole",
    "port_dst": "Port Destination",
    "action": "Action",
    "rule_id": "Regle",
    "interface_in": "Interface Entree",
    "interface_out": "Interface Sortie",
}

ACTION_COLORS = {
    "ACCEPT": "#2ECC71",
    "DROP": "#E74C3C",
    "REJECT": "#E67E22",
    "LOG": "#3498DB",
}

COLOR_SEQUENCE = [
    "#3498DB",
    "#E74C3C",
    "#2ECC71",
    "#F39C12",
    "#9B59B6",
    "#1ABC9C",
    "#E67E22",
    "#34495E",
    "#E91E63",
    "#00BCD4",
]

PORT_RANGES = {
    "Tous": None,
    "Well-known (0-1023)": "well_known",
    "Registered (1024-49151)": "registered",
    "Dynamic (49152-65535)": "dynamic",
}

TOP_N_DEFAULT = 10
