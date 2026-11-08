import common as U  # noqa: I001
import numpy as np
import plotly.express as px
import streamlit as st

import config as C
from mindmove import robustness
from mindmove.viz import plot_topomaps

U.page("Robustness", "🛠️")
st.title("🛠️ Robustness: broken electrodes and noise")
st.caption("Electrodes detach and gel dries. A real BCI must survive that. Break channels or add noise "
           "to the day-2 data and watch the day-1 decoder react, live.")

s = U.subject_picker()
models = U.require(U.load_models(s), "Trained models", "scripts/06_train_final.py")
pipe = st.sidebar.selectbox("Decoder", ["CSP+LDA", "TS+LR", "FBTS+LR"], index=1)
k = st.sidebar.slider("Broken electrodes", 0, 8, 2)
repair = st.sidebar.toggle("Repair with spherical-spline interpolation", value=True)
snr = st.sidebar.select_slider("Extra noise SNR (dB)", options=["clean", 30, 20, 10, 5, 0], value="clean")
seed = st.sidebar.number_input("Which electrodes (random seed)", 0, 999, 0)

XE, yE = U.load_subject(s)["E"]


@st.cache_data(show_spinner="Corrupting and decoding ...")
def run(subject, pipe, k, repair, snr, seed):
    rng = np.random.default_rng(seed)
    X = U.load_subject(subject)["E"][0]
    Xc, bad = robustness.corrupt_electrodes(X, k, rng, repair=repair)
    if snr != "clean":
        Xc = robustness.add_noise(Xc, float(snr), rng)
    clean = float((U.load_models(subject)[pipe].predict(X) == yE).mean())
    acc = float((U.load_models(subject)[pipe].predict(Xc) == yE).mean())
    return clean, acc, bad.tolist()


clean, acc, bad = run(s, pipe, k, repair, snr, seed)
c1, c2, c3 = st.columns(3)
c1.metric("Clean accuracy", f"{clean:.1%}")
c2.metric("Accuracy now", f"{acc:.1%}", f"{acc - clean:+.1%}")
c3.metric("Broken channels", ", ".join(C.CHANNELS[i] for i in bad) or "none")

mask = np.zeros(len(C.CHANNELS))
mask[bad] = 1
if bad:
    from mindmove.data import make_info
    st.pyplot(plot_topomaps({"broken electrodes": mask}, make_info(), cmap="Reds", symmetric=False,
                            title="Broken electrodes (red)"), width="content")

st.divider()
rob = U.load_csv("robustness.csv")
if rob is not None:
    st.subheader("Precomputed curves (all subjects, random electrode sets)")
    el = rob[rob.test == "electrodes"]
    who = st.radio("Show", ["Mean of 9 subjects", "Selected subject"], horizontal=True)
    if who == "Selected subject":
        el = el[el.subject == s]
    g = el.groupby(["pipeline", "condition", "level"]).accuracy.mean().reset_index()
    g["series"] = g.pipeline + " · " + g.condition
    fig = px.line(g, x="level", y="accuracy", color="series", markers=True, height=380,
                  labels={"level": "broken electrodes (of 22)"})
    fig.add_hline(y=0.25, line_dash="dot", annotation_text="chance")
    st.plotly_chart(fig, width="stretch")
    no = rob[rob.test == "noise"]
    if who == "Selected subject":
        no = no[no.subject == s]
    g = no.groupby(["pipeline", "level"]).accuracy.mean().reset_index()
    g["SNR (dB)"] = g.level.map(lambda v: "clean" if v >= 99 else f"{v:g}")
    fig = px.line(g.sort_values("level", ascending=False), x="SNR (dB)", y="accuracy", color="pipeline",
                  markers=True, height=320)
    st.plotly_chart(fig, width="stretch")
    st.caption("A single broken electrode recording large noise dominates the covariance matrix and pushes "
               "every decoder towards chance. Marking it bad and interpolating it from its neighbours "
               "recovers most of the accuracy, so detecting bad channels matters more than the classifier.")
