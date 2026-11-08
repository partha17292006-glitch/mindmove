"""Shared helpers for the Streamlit app: paths, cached loaders, small widgets."""
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config as C  # noqa: E402
from mindmove import data  # noqa: E402

LOSO_KEY = "LOSO (unseen patient)"
CLASS_COLORS = {"left_hand": "#2a78d6", "right_hand": "#e5484d", "feet": "#30a46c", "tongue": "#e09f3e"}


def page(title, icon="🧠"):
    st.set_page_config(page_title=f"MindMove · {title}", page_icon=icon, layout="wide")


def subject_label(s):
    return f"A{s:02d}"


def subject_picker(key="subject", default=3, label="Subject"):
    return st.sidebar.selectbox(label, C.SUBJECTS, index=C.SUBJECTS.index(default),
                                format_func=subject_label, key=key)


@st.cache_resource(show_spinner="Loading EEG ...")
def load_subject(s):
    return data.load_subject(s)


@st.cache_resource(show_spinner="Loading models ...")
def load_models(s):
    p = C.MODELS / f"A{s:02d}.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_resource
def load_covs():
    p = C.DATA_PROCESSED / "covs_8_30.npz"
    if not p.exists():
        return None, None
    with np.load(p) as z:
        covs = {(s, se): z[f"cov_{s}_{se}"] for s in C.SUBJECTS for se in C.SESSIONS}
        ys = {(s, se): z[f"y_{s}_{se}"] for s in C.SUBJECTS for se in C.SESSIONS}
    return covs, ys


@st.cache_data
def _read_csv(path, mtime):  # mtime in the cache key -> re-running a script refreshes the app
    return pd.read_csv(path)


def load_csv(name):
    p = C.RESULTS / name
    return _read_csv(str(p), p.stat().st_mtime) if p.exists() else None


def best_pipelines():
    p = C.MODELS / "best.json"
    return {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def require(obj, what, script):
    """Stop the page with a helpful message if a precomputed artefact is missing."""
    if obj is None:
        st.warning(f"{what} not found. Run `python {script}` first.")
        st.stop()
    return obj


def figure(name, caption=None):
    p = C.FIGURES / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"figure {name} not generated yet")
