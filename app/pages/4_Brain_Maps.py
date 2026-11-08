import common as U  # noqa: I001
import numpy as np
import plotly.express as px
import streamlit as st

import config as C
from mindmove import data, viz

U.page("Brain Maps", "🗺️")
st.title("🗺️ Brain Maps — what is the decoder looking at?")

s = U.subject_picker()
info = data.make_info()

st.subheader("1 · Spatial patterns per class (one-vs-rest CSP, 8–30 Hz)")
src = st.radio("Show", ["Selected subject", "Mean of 9 subjects"], horizontal=True)
pats_file = C.RESULTS / "csp_patterns.npz"
if pats_file.exists():
    with np.load(pats_file) as z:
        pats = {c: (z[c][s - 1] if src == "Selected subject" else z[c].mean(0)) for c in C.CLASSES}
    st.pyplot(viz.plot_topomaps(pats, info, cmap="Reds", symmetric=False, unit="|pattern|",
                                title="Where each imagined movement suppresses rhythm most, relative to the others"))
    peaks = {C.CLASS_LABELS[c]: C.CHANNELS[int(np.argmax(v))] for c, v in pats.items()}
    st.caption("Peak channel: " + " · ".join(f"{k}: **{v}**" for k, v in peaks.items())
               + ". Expected from motor-cortex anatomy: left hand → right hemisphere (C4), right hand → "
                 "left hemisphere (C3), feet → midline (Cz). Tongue sits laterally and lower.")
else:
    st.info("Run `python scripts/06_train_final.py` to compute patterns.")


@st.cache_data(show_spinner="Computing mu-band power maps ...")
def mu_topo(subject, band):
    X, y = U.load_subject(subject)["T"]
    return viz.band_erd_topo(X, y, band)


st.subheader("2 · Rhythm change maps (imagery vs. pre-cue baseline)")
BANDS = {"8–13 Hz (mu)": (8, 13), "13–30 Hz (beta)": (13, 30), "8–30 Hz (mu+beta)": (8, 30)}
band = BANDS[st.radio("Band", list(BANDS), horizontal=True)]
st.pyplot(viz.plot_topomaps(mu_topo(s, band), info, unit="dB (blue = ERD)",
                            title=f"{U.subject_label(s)} session T: power change in {band[0]}–{band[1]} Hz"))

st.subheader("3 · The manifold: each dot is one EEG trial (t-SNE of tangent vectors)")
tsne = U.load_csv("tsne.csv")
if tsne is None:
    st.info("Run `python scripts/04_transfer.py` to compute the embedding.")
else:
    colour = st.radio("Colour by", ["subject", "class"], horizontal=True)
    t = tsne.copy()
    t["subject"] = t.subject.map(U.subject_label)
    t["class"] = t["class"].map(C.CLASS_LABELS)
    c1, c2 = st.columns(2)
    cmap = {C.CLASS_LABELS[k]: v for k, v in U.CLASS_COLORS.items()} if colour == "class" else None
    for col, when in ((c1, "before"), (c2, "after")):
        fig = px.scatter(t, x=f"x_{when}", y=f"y_{when}", color=colour, opacity=0.6, height=420,
                         color_discrete_map=cmap, title=f"{when} re-centering")
        fig.update_traces(marker_size=4)
        fig.update_layout(xaxis_visible=False, yaxis_visible=False)
        col.plotly_chart(fig, width="stretch")
    st.caption("Before re-centering, trials cluster by **person**: the between-subject shift dominates. "
               "After moving every subject's Riemannian mean to the identity, the subject clusters merge, "
               "which is why a model trained on other people starts to work for a new one.")

st.subheader("4 · Which frequency bands matter? (FBTS+LR coefficients)")
imp = U.load_csv("band_importance.csv")
if imp is not None:
    order = [f"{lo}-{hi} Hz" for lo, hi in C.FBTS_BANDS]
    imp["Subject"] = imp.subject.map(U.subject_label)
    fig = px.bar(imp, x="band", y="importance", color="Subject", barmode="group",
                 category_orders={"band": order}, height=360)
    st.plotly_chart(fig, width="stretch")
    st.caption("Share of the logistic-regression weight mass per band. Different people rely on different "
               "bands, which is exactly why a filter bank beats a single fixed 8–30 Hz band.")
