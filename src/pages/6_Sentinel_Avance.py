import importlib.util
import sys
from pathlib import Path

import streamlit as st

from components.top_nav import render_top_nav

st.set_page_config(
    page_title="Sentinel Avance",
    page_icon="S",
    layout="wide",
)
render_top_nav("sentinel")

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

if str(SENTINEL_DIR) not in sys.path:
    sys.path.insert(0, str(SENTINEL_DIR))


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    sys.modules["utils"] = _load_module("sentinel_utils", SENTINEL_UTILS)

    # app_sentinel.py calls st.set_page_config() internally.
    # We no-op it here because page config is already set by this wrapper.
    original_set_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None
    try:
        spec = importlib.util.spec_from_file_location("sentinel_legacy_app", SENTINEL_APP)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        st.set_page_config = original_set_page_config
except ModuleNotFoundError as exc:
    st.error(
        "Dependance manquante: "
        f"`{exc.name}`. Installez les deps avec `pip install -r src/requirements.txt`."
    )
    st.stop()
except Exception as exc:
    st.exception(exc)
