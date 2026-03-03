from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv

try:
    import duckdb
except ImportError:  # handled when DATA_SOURCE=motherduck
    duckdb = None


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)

# Local parquet source
DATA_PATH = Path("data/processed/log_clean.parquet")

# MotherDuck config env vars
DATA_SOURCE_ENV = "DATA_SOURCE"
MD_TOKEN_ENV = "MOTHERDUCK_TOKEN"
MD_DATABASE_ENV = "MOTHERDUCK_DATABASE"
MD_TABLE_ENV = "MOTHERDUCK_TABLE"
MD_TABLES_LEGACY_ENV = "MOTHERDUCK_TABLES"
MD_TABLE_OPTIONS_ENV = "MOTHERDUCK_TABLE_OPTIONS"
MD_FALLBACK_ENV = "MOTHERDUCK_FALLBACK_TO_PARQUET"
_LAST_LOAD_INFO = {
    "loaded": False,
    "configured_source": "parquet",
    "active_source": "parquet",
    "motherduck_database": "",
    "motherduck_table": "",
    "fallback_used": False,
    "error": "",
}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_md_table() -> str:
    table = os.getenv(MD_TABLE_ENV, "").strip()
    if table:
        return table

    # Backward compatibility: accept MOTHERDUCK_TABLES only if one table is provided.
    legacy = os.getenv(MD_TABLES_LEGACY_ENV, "").strip()
    if not legacy:
        return ""
    items = [part.strip() for part in legacy.split(",") if part.strip()]
    if len(items) != 1:
        raise ValueError(
            f"{MD_TABLE_ENV} must contain a single table. "
            f"Legacy {MD_TABLES_LEGACY_ENV} currently has {len(items)} tables."
        )
    return items[0]


def get_available_motherduck_tables() -> list[str]:
    raw = os.getenv(MD_TABLE_OPTIONS_ENV, "").strip()
    items = [part.strip() for part in raw.split(",") if part.strip()] if raw else []

    # Ensure configured default table is part of the selectable options.
    try:
        default_table = _get_md_table()
    except Exception:
        default_table = ""
    if default_table and default_table not in items:
        items.append(default_table)
    return items


def _load_from_parquet() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


def _qualify_table(table: str, database: str) -> str:
    # If already qualified (db.schema.table or schema.table), keep as-is.
    if "." in table:
        return table
    return f"{database}.{table}"


def _load_from_motherduck(selected_table: str | None = None) -> pd.DataFrame:
    if duckdb is None:
        raise ImportError("duckdb is required for MotherDuck source. Install package 'duckdb'.")

    token = os.getenv(MD_TOKEN_ENV, "").strip().strip('"').strip("'")
    database = os.getenv(MD_DATABASE_ENV, "").strip()
    table = selected_table.strip() if selected_table else _get_md_table()

    if not token:
        raise ValueError(f"Missing env var {MD_TOKEN_ENV}.")
    if not database:
        raise ValueError(f"Missing env var {MD_DATABASE_ENV}.")
    if not table:
        raise ValueError(f"Missing env var {MD_TABLE_ENV}. Provide exactly one table name.")

    conn = duckdb.connect(f"md:{database}?motherduck_token={token}")
    try:
        qualified_table = _qualify_table(table, database)
        return conn.execute(f"SELECT * FROM {qualified_table}").df()
    finally:
        conn.close()


def load_data(selected_table: str | None = None) -> pd.DataFrame:
    """
    Load data from configured source, then apply cleaning and optimizations.

    DATA_SOURCE values:
    - parquet (default)
    - motherduck
    """
    source = os.getenv(DATA_SOURCE_ENV, "parquet").strip().lower()
    _LAST_LOAD_INFO["configured_source"] = source
    _LAST_LOAD_INFO["motherduck_database"] = os.getenv(MD_DATABASE_ENV, "").strip()
    _LAST_LOAD_INFO["fallback_used"] = False
    _LAST_LOAD_INFO["error"] = ""
    try:
        _LAST_LOAD_INFO["motherduck_table"] = _get_md_table()
    except Exception:
        _LAST_LOAD_INFO["motherduck_table"] = ""

    if source == "motherduck":
        try:
            df = _load_from_motherduck(selected_table=selected_table)
            _LAST_LOAD_INFO["active_source"] = "motherduck"
            _LAST_LOAD_INFO["motherduck_table"] = selected_table.strip() if selected_table else _LAST_LOAD_INFO["motherduck_table"]
        except Exception as exc:
            if _env_bool(MD_FALLBACK_ENV, default=True):
                df = _load_from_parquet()
                _LAST_LOAD_INFO["active_source"] = "parquet"
                _LAST_LOAD_INFO["fallback_used"] = True
                _LAST_LOAD_INFO["error"] = str(exc)
            else:
                raise RuntimeError(f"MotherDuck loading failed: {exc}") from exc
    else:
        df = _load_from_parquet()
        _LAST_LOAD_INFO["active_source"] = "parquet"

    df = clean_columns(df)
    df = optimize_types(df)
    _LAST_LOAD_INFO["loaded"] = True
    return df


def get_data_source_info() -> dict:
    info = dict(_LAST_LOAD_INFO)
    configured = os.getenv(DATA_SOURCE_ENV, "parquet").strip().lower()
    info["configured_source"] = configured
    info["motherduck_database"] = os.getenv(MD_DATABASE_ENV, "").strip()
    try:
        info["motherduck_table"] = _get_md_table()
    except Exception:
        pass
    info["available_tables"] = get_available_motherduck_tables()
    if not info.get("loaded"):
        info["active_source"] = configured
    return info


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage et typage des colonnes.
    """
    # Normalize heterogeneous schemas across MotherDuck tables.
    rename_map = {
        "proto": "protocol",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    # Table 1 uses a single "interface" column.
    if "interface" in df.columns:
        if "interface_in" not in df.columns:
            df["interface_in"] = df["interface"]
        if "interface_out" not in df.columns:
            df["interface_out"] = df["interface"]

    # Ensure expected columns exist for downstream modules/charts.
    for col in ["protocol", "action", "rule_id", "port_dst", "ip_src", "ip_dst"]:
        if col not in df.columns:
            df[col] = pd.NA

    if "interface_in" not in df.columns:
        df["interface_in"] = pd.NA
    if "interface_out" not in df.columns:
        df["interface_out"] = pd.NA

    # Sentinel pages also rely on protocol_clean.
    if "protocol_clean" not in df.columns:
        df["protocol_clean"] = df["protocol"]

    if "protocol" in df.columns:
        df["protocol"] = df["protocol"].astype(str).str.upper().replace({"<NA>": pd.NA, "NAN": pd.NA})
    if "protocol_clean" in df.columns:
        df["protocol_clean"] = (
            df["protocol_clean"]
            .astype(str)
            .str.upper()
            .replace({"<NA>": pd.NA, "NAN": pd.NA})
        )
    if "action" in df.columns:
        df["action"] = df["action"].astype(str).str.upper().replace({"<NA>": pd.NA, "NAN": pd.NA})

    # Datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Suppression Firewall ID si present
    if "FW" in df.columns:
        df = df.drop(columns=["FW"])

    return df


def optimize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimisation memoire pour gros dataset.
    """

    categorical_cols = ["protocol", "action", "interface_in", "interface_out"]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    int_cols = ["port_dst", "rule_id"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


# ---------------------------------------------------------
# FONCTIONS UTILES POUR L'APP
# ---------------------------------------------------------


def filter_by_date(df, start_date=None, end_date=None):
    if start_date:
        df = df[df["datetime"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["datetime"] <= pd.to_datetime(end_date)]
    return df


def filter_by_protocol(df, protocols):
    if protocols:
        df = df[df["protocol"].isin(protocols)]
    return df


def filter_by_action(df, actions):
    if actions:
        df = df[df["action"].isin(actions)]
    return df


def filter_by_port_range(df, port_range):
    """
    RFC 6056 ranges :
    - well_known: 0-1023
    - registered: 1024-49151
    - dynamic: 49152-65535
    """
    if port_range == "well_known":
        return df[df["port_dst"] <= 1023]

    if port_range == "registered":
        return df[(df["port_dst"] >= 1024) & (df["port_dst"] <= 49151)]

    if port_range == "dynamic":
        return df[df["port_dst"] >= 49152]

    return df


def detect_external_ips(df, university_prefix="159.84."):
    """
    Detecte les IP hors plan d'adressage universite.
    """
    return df[~df["ip_src"].str.startswith(university_prefix)]
