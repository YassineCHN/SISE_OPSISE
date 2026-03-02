from pathlib import Path
import pandas as pd
import ipaddress


# Chemin du fichier parquet
DATA_PATH = Path("data/processed/log_clean.parquet")


def load_data():
    """
    Charge les données parquet et applique les optimisations.
    """
    df = pd.read_parquet(DATA_PATH)

    df = clean_columns(df)
    df = optimize_types(df)

    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage et typage des colonnes.
    """

    # Datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Suppression Firewall ID si présent
    if "FW" in df.columns:
        df = df.drop(columns=["FW"])

    return df


def optimize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimisation mémoire pour gros dataset.
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
    Détecte les IP hors plan d'adressage université.
    """
    return df[~df["ip_src"].str.startswith(university_prefix)]
