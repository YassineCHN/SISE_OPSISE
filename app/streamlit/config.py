from pathlib import Path

# ── Chemins ───────────────────────────────────────────────────────────────
DATA_PATH = Path("data/processed/log_clean.parquet")

# ── App ───────────────────────────────────────────────────────────────────
APP_TITLE   = "SISE – Analyse des logs Firewall"
APP_ICON    = "🔥"
LAYOUT      = "wide"
APP_VERSION = "2.0"

# ── Colonnes (labels FR) ─────────────────────────────────────────────────
COLUMN_LABELS = {
    "datetime":     "Horodatage",
    "ip_src":       "IP Source",
    "ip_dst":       "IP Destination",
    "protocol":     "Protocole",
    "port_src":     "Port Source",
    "port_dst":     "Port Destination",
    "action":       "Action",
    "rule_id":      "Règle",
    "interface_in": "Interface Entrée",
}

# ── Couleurs actions ──────────────────────────────────────────────────────
ACTION_COLORS = {
    "PERMIT": "#00C896",
    "DENY":   "#FF4B4B",
    "ACCEPT": "#00C896",
    "DROP":   "#FF4B4B",
    "REJECT": "#FF8C42",
    "LOG":    "#4A90D9",
}

# ── Profils ML ────────────────────────────────────────────────────────────
PROFIL_COLORS = {
    "Normal":                     "#56d364",
    "Port Scan":                  "#79c0ff",
    "DDoS / Flood":               "#f78166",
    "Attaque ciblée":             "#ffa657",
    "Activité nocturne suspecte": "#d2a8ff",
    "Comportement bloqué":        "#e3b341",
}

# ── Palette Plotly ────────────────────────────────────────────────────────
COLOR_SEQ = [
    "#00C896","#4A90D9","#FF4B4B","#FF8C42",
    "#A78BFA","#34D399","#F59E0B","#64748B",
    "#EC4899","#06B6D4",
]

# ── Ports RFC 6056 ────────────────────────────────────────────────────────
PORT_RANGES = {
    "Tous":                    None,
    "Well-known (0–1023)":     "well_known",
    "Registered (1024–49151)": "registered",
    "Dynamic (49152–65535)":   "dynamic",
}

# ── Services connus ───────────────────────────────────────────────────────
SERVICE_MAP = {
    20:"FTP-data", 21:"FTP", 22:"SSH", 23:"Telnet", 25:"SMTP",
    53:"DNS", 67:"DHCP", 68:"DHCP", 80:"HTTP", 110:"POP3",
    123:"NTP", 143:"IMAP", 161:"SNMP", 443:"HTTPS", 445:"SMB",
    514:"Syslog", 3306:"MySQL", 3389:"RDP", 5353:"mDNS",
    8080:"HTTP-Alt", 8443:"HTTPS-Alt", 5900:"VNC",
}

# ── Ports UDP connus ─────────────────────────────────────────────────────
UDP_PORTS = {53,67,68,69,123,161,162,514,520,1900,5353}

# ── Features ML ──────────────────────────────────────────────────────────
FEATURES_COLS = [
    "nb_connexions","nb_ports_distincts","nb_ips_dst",
    "ratio_deny","nb_ports_sensibles","activite_nuit","port_dst_std"
]

TOP_N_DEFAULT = 10
