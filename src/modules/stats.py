import pandas as pd


def action_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution des actions (ACCEPT, DROP, REJECT…)."""
    return df["action"].value_counts().rename_axis("action").reset_index(name="count")


def top_n(df: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    """Top N valeurs d'une colonne."""
    return df[col].value_counts().head(n).rename_axis(col).reset_index(name="count")


def traffic_by_period(df: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Volume de trafic agrégé par période (h, D, W…)."""
    return (
        df.set_index("datetime")
        .resample(freq)
        .size()
        .rename("count")
        .reset_index()
    )


def protocol_action_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau croisé Protocole × Action."""
    return pd.crosstab(df["protocol"], df["action"])


def unique_counts(df: pd.DataFrame) -> dict:
    """Nombre de valeurs uniques par dimension clé."""
    return {
        "ip_src":   df["ip_src"].nunique(),
        "ip_dst":   df["ip_dst"].nunique(),
        "protocol": df["protocol"].nunique(),
        "rule_id":  df["rule_id"].nunique(),
        "port_dst": df["port_dst"].nunique(),
    }


def blocked_ratio(df: pd.DataFrame) -> float:
    """Pourcentage de trafic bloqué (DROP ou REJECT)."""
    if len(df) == 0:
        return 0.0
    blocked = df["action"].isin(["DROP", "REJECT"]).sum()
    return blocked / len(df) * 100


def _port_category(port) -> str:
    try:
        p = int(port)
    except (ValueError, TypeError):
        return "Inconnu"
    if p <= 1023:
        return "Well-known"
    if p <= 49151:
        return "Registered"
    return "Dynamic"


def port_category_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution des plages de ports destination."""
    categories = df["port_dst"].dropna().map(_port_category)
    return categories.value_counts().rename_axis("Catégorie").reset_index(name="count")


def traffic_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Volume de trafic par heure de la journée (0–23)."""
    tmp = df.copy()
    tmp["hour"] = tmp["datetime"].dt.hour
    return tmp.groupby("hour").size().rename("count").reset_index()


def traffic_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """Volume de trafic par jour de la semaine."""
    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    tmp = df.copy()
    tmp["weekday"] = tmp["datetime"].dt.dayofweek
    counts = tmp.groupby("weekday").size().rename("count").reset_index()
    counts["jour"] = counts["weekday"].map(lambda x: days[x])
    return counts[["jour", "count"]]
