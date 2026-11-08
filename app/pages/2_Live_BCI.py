import time

import common as U  # noqa: I001
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config as C
from mindmove import data, stream

U.page("Live BCI", "🦽")
st.title("🦽 Live BCI simulator")
st.caption("Held-out day-2 trials are replayed one at a time. The decoder only sees each trial's EEG and "
           "outputs class probabilities; a confident decision moves the wheelchair.")

s = U.subject_picker()
models = U.require(U.load_models(s), "Trained models", "scripts/06_train_final.py")
best = U.best_pipelines().get(s, "FBTS+LR")
options = [best] + [k for k in models if k != best]
model_name = st.sidebar.selectbox("Decoder", options,
                                  format_func=lambda k: f"{k} (best for this subject)" if k == best else k)
threshold = st.sidebar.slider("Confidence threshold", 0.25, 0.95, 0.45, 0.05,
                              help="Below this max-probability the chair does NOT move (safety).")
n_trials = st.sidebar.slider("Trials to play", 12, 288, 48, 12)
delay = st.sidebar.slider("Seconds per trial", 0.0, 1.5, 0.25, 0.05)
shuffle = st.sidebar.checkbox("Shuffle trial order", value=False)

st.sidebar.markdown("**Mapping:** left hand → turn left · right hand → turn right · feet → forward · "
                    "tongue → stop")
if model_name == U.LOSO_KEY:
    st.info("**Unseen-patient mode:** this decoder never saw a single labelled trial of this subject. It was "
            "trained on the other 8 people; this subject's data is only *re-centered* on the manifold using "
            "unlabelled trials.")

XE, yE = U.load_subject(s)["E"]
NOBAR = {"displayModeBar": False}


@st.cache_data(show_spinner="Decoding ...")
def all_probs(subject, name):
    m = U.load_models(subject)[name]
    if name == U.LOSO_KEY:                       # unsupervised: centre of the unlabelled day-2 trials
        m.fit_reference(U.load_subject(subject)["E"][0])
    return m.predict_proba(U.load_subject(subject)["E"][0])


probs = all_probs(s, model_name)
order = np.random.default_rng(C.SEED).permutation(len(yE)) if shuffle else np.arange(len(yE))
order = order[:n_trials]

col_eeg, col_chair = st.columns([3, 2])
eeg_ph, prob_ph = col_eeg.empty(), col_eeg.empty()
chair_ph, stat_ph = col_chair.empty(), col_chair.empty()
log_ph = st.empty()


def eeg_fig(trial, title):
    fig = go.Figure()
    for i, ch in enumerate(["C3", "Cz", "C4"]):
        fig.add_trace(go.Scatter(x=data.TIMES, y=trial[C.CHANNELS.index(ch)] * 1e6 - 40 * i, name=ch,
                                 mode="lines", line=dict(width=1)))
    fig.add_vline(x=0, line_dash="dash")
    fig.add_vrect(x0=C.CLF_TMIN, x1=C.CLF_TMAX, fillcolor="grey", opacity=0.12, line_width=0)
    fig.update_layout(height=230, title=title, margin=dict(t=40, b=20, l=40, r=10), showlegend=True,
                      xaxis_title="s", yaxis=dict(showticklabels=False))
    return fig


def heat_fig(trial):
    z = trial[:, ::5] * 1e6
    lim = 3 * float(z.std())                     # robust colour range: outliers don't wash it out
    fig = go.Figure(go.Heatmap(z=z, x=data.TIMES[::5], y=C.CHANNELS, colorscale="RdBu_r",
                               zmin=-lim, zmax=lim, showscale=False))
    fig.update_layout(height=230, margin=dict(t=10, b=20, l=40, r=10), title=None)
    return fig


def prob_fig(p, threshold, true_c):
    colors = [U.CLASS_COLORS[c] for c in C.CLASSES]
    fig = go.Figure(go.Bar(x=[C.CLASS_LABELS[c] for c in C.CLASSES], y=p, marker_color=colors,
                           text=[f"{v:.2f}" for v in p], textposition="outside"))
    fig.add_hline(y=threshold, line_dash="dot", annotation_text="threshold")
    fig.update_layout(height=230, yaxis_range=[0, 1.1], margin=dict(t=30, b=20),
                      title=f"Decoder probabilities (true: {C.CLASS_LABELS[true_c]})")
    return fig


def chair_fig(chair):
    xs, ys = zip(*chair.path)
    dx, dy = stream.HEADINGS[chair.heading]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="#8fa8c8"),
                             marker=dict(size=4), name="path"))
    fig.add_trace(go.Scatter(x=[chair.x], y=[chair.y], mode="markers+text", text=["🦽"],
                             textfont=dict(size=22), marker=dict(size=1), name="chair"))
    fig.add_annotation(x=chair.x + 0.8 * dx, y=chair.y + 0.8 * dy, ax=chair.x, ay=chair.y,
                       xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3)
    fig.update_layout(height=330, showlegend=False, margin=dict(t=30, b=10),
                      xaxis=dict(range=[-0.5, stream.GRID - 0.5], dtick=1, showticklabels=False),
                      yaxis=dict(range=[-0.5, stream.GRID - 0.5], dtick=1, showticklabels=False,
                                 scaleanchor="x"), title="Wheelchair (10×10 room)")
    return fig


if st.button("▶ Play", type="primary"):
    chair = stream.Wheelchair()
    results = []
    for n, i in enumerate(order, 1):
        p = probs[i]
        pred, conf, cmd = stream.decide(p, threshold)
        r = stream.TrialResult(int(i), p, pred, C.CLASSES[int(yE[i])], conf, cmd)
        results.append(r)
        chair.step(cmd)
        eeg_ph.plotly_chart(eeg_fig(XE[i], f"Trial {n}/{len(order)} — EEG at C3 · Cz · C4"),
                            width="stretch", key=f"eeg{n}", config=NOBAR)
        prob_ph.plotly_chart(prob_fig(p, threshold, r.true), width="stretch", key=f"prob{n}", config=NOBAR)
        chair_ph.plotly_chart(chair_fig(chair), width="stretch", key=f"chair{n}", config=NOBAR)
        sm = stream.stream_summary(results)
        verdict = ("✅ correct" if r.correct else "❌ wrong") if conf else "⏸ uncertain, no movement"
        stat_ph.markdown(
            f"**Decision:** {C.CLASS_LABELS[pred]} → *{cmd or 'no movement'}* · {verdict}  \n"
            f"**Running accuracy:** {sm['accuracy_all']:.1%} · **acted on** {sm['coverage']:.0%} of trials · "
            f"**accuracy when acting:** {sm['accuracy_when_acting']:.1%} · wrong moves: {sm['wrong_moves']}")
        if delay:
            time.sleep(delay)
    st.success(f"Session finished: {len(results)} trials, accuracy {sm['accuracy_all']:.1%}, "
               f"acted on {sm['coverage']:.0%}, accuracy when acting {sm['accuracy_when_acting']:.1%}.")
else:
    i = order[0]
    eeg_ph.plotly_chart(eeg_fig(XE[i], "First trial (press Play)"), width="stretch", config=NOBAR)
    prob_ph.plotly_chart(heat_fig(XE[i]), width="stretch")
    chair_ph.plotly_chart(chair_fig(stream.Wheelchair()), width="stretch")

st.divider()
st.subheader("Safety trade-off: threshold vs. wrong moves (whole session)")
ths = np.round(np.arange(0.25, 0.96, 0.05), 2)
true = yE
rows = []
for t in ths:
    pm = probs.max(1)
    act = pm >= t
    pred = probs.argmax(1)
    rows.append({"threshold": t, "coverage": act.mean(),
                 "accuracy_when_acting": (pred[act] == true[act]).mean() if act.any() else np.nan})
tr = pd.DataFrame(rows)
fig = go.Figure()
fig.add_trace(go.Scatter(x=tr.threshold, y=tr.coverage, name="share of trials acted on"))
fig.add_trace(go.Scatter(x=tr.threshold, y=tr.accuracy_when_acting, name="accuracy when acting"))
fig.add_vline(x=threshold, line_dash="dot")
fig.update_layout(height=300, xaxis_title="confidence threshold", yaxis_range=[0, 1.02], margin=dict(t=20))
st.plotly_chart(fig, width="stretch")
st.caption("Raising the threshold makes the chair move less often but more reliably: the classic BCI "
           "speed-versus-safety trade-off.")
