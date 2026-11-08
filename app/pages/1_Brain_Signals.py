import common as U  # noqa: I001
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config as C
from mindmove import data, viz

U.page("Brain Signals", "📈")
st.title("📈 Brain Signals")
st.caption("What does motor imagery look like in EEG? Pick a subject, class and trial.")

s = U.subject_picker()
session = st.sidebar.radio("Session", C.SESSIONS, format_func=lambda x: {"T": "T (day 1)", "E": "E (day 2)"}[x])
cls = st.sidebar.selectbox("Imagined movement", C.CLASSES, format_func=C.CLASS_LABELS.get)
X, y = U.load_subject(s)[session]
idx = np.flatnonzero(y == C.CLASSES.index(cls))
trial_no = st.sidebar.slider("Trial of this class", 1, len(idx), 1)
channels = st.sidebar.multiselect("Channels", C.CHANNELS, default=["C3", "Cz", "C4"])

trial = X[idx[trial_no - 1]]
st.subheader(f"Raw EEG — {U.subject_label(s)}{session}, {C.CLASS_LABELS[cls]}, trial {trial_no}")
fig = go.Figure()
for i, ch in enumerate(channels):
    fig.add_trace(go.Scatter(x=data.TIMES, y=trial[C.CHANNELS.index(ch)] * 1e6 - i * 40, name=ch,
                             mode="lines", line=dict(width=1)))
fig.add_vline(x=0, line_dash="dash", annotation_text="cue")
fig.add_vrect(x0=C.CLF_TMIN, x1=C.CLF_TMAX, fillcolor="grey", opacity=0.12, line_width=0,
              annotation_text="classification window", annotation_position="top left")
fig.update_layout(height=360, xaxis_title="time relative to cue (s)", yaxis_title="µV (offset per channel)",
                  margin=dict(t=30, b=40))
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Power spectrum per class")
    ch = st.selectbox("Channel", C.CHANNELS, index=C.CHANNELS.index("C3"), key="psd_ch")
    f, psd = viz.class_psd(X, y, ch)
    fig = go.Figure()
    for c, p in psd.items():
        fig.add_trace(go.Scatter(x=f, y=p, name=C.CLASS_LABELS[c], line=dict(color=U.CLASS_COLORS[c])))
    fig.add_vrect(x0=8, x1=13, fillcolor="grey", opacity=0.1, line_width=0, annotation_text="mu")
    fig.add_vrect(x0=13, x1=30, fillcolor="blue", opacity=0.04, line_width=0, annotation_text="beta")
    fig.update_layout(height=340, xaxis_title="Hz", yaxis_title="dB", margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
    st.caption("Motor imagery *desynchronises* the mu (8–13 Hz) and beta (13–30 Hz) rhythms over the "
               "motor cortex controlling the imagined body part: that power drop is what the decoder exploits.")


@st.cache_data(show_spinner="Computing time-frequency maps ...")
def erd_for(subject, session):
    X, y = U.load_subject(subject)[session]
    return viz.compute_erd(X, y)


with c2:
    st.subheader("ERD/ERS map (C3 · Cz · C4)")
    freqs, times, erd = erd_for(s, session)
    st.pyplot(viz.plot_erd(freqs, times, erd, title=f"{U.subject_label(s)}{session}: % power vs baseline"))
    st.caption("Blue = power decrease (ERD) relative to the pre-cue baseline. The red blob at 0–0.5 s "
               "is the visual evoked response to the cue, which is why classification starts at 0.5 s.")
