import importlib.util
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SENTINEL_DIR = PROJECT_ROOT / "app" / "streamlit"
SENTINEL_APP = SENTINEL_DIR / "app_sentinel.py"
SENTINEL_UTILS = SENTINEL_DIR / "utils.py"

if not SENTINEL_APP.exists():
    st.error(f"Fichier introuvable: {SENTINEL_APP}")
    st.stop()

if not SENTINEL_UTILS.exists():
    st.error(f"Fichier utils introuvable: {SENTINEL_UTILS}")
    st.stop()

# Make sibling imports from app/streamlit resolve first.
if str(SENTINEL_DIR) not in sys.path:
    sys.path.insert(0, str(SENTINEL_DIR))


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    # Force the sentinel-local utils module name to avoid collision with src/utils.
    sys.modules["utils"] = _load_module("sentinel_utils", SENTINEL_UTILS)

    spec = importlib.util.spec_from_file_location("sentinel_legacy_app", SENTINEL_APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
except ModuleNotFoundError as exc:
    st.error(
        "Dependance manquante: "
        f"`{exc.name}`. Installez les deps avec `pip install -r src/requirements.txt`."
    )
    st.stop()
except Exception as exc:
    st.exception(exc)
