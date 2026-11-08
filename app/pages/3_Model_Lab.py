import common as U  # noqa: I001
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
from mindmove import evaluate
from mindmove.pipelines import OURS, PIPELINE_NAMES

U.page("Model Lab", "🧪")
st.title("🧪 Model Lab")
st.caption("Six classical pipelines, three evaluation protocols, nine subjects.")

bench = U.require(U.load_csv("benchmark.csv"), "results/benchmark.csv", "scripts/03_benchmark.py")
protocols = {"cross_session": "Cross-session (train day 1 → test day 2, official)",
             "within_session_T": "Within-session 5-fold CV, day 1",
             "within_session_E": "Within-session 5-fold CV, day 2"}
proto = st.sidebar.radio("Protocol", list(protocols), format_func=protocols.get)
chosen = st.sidebar.multiselect("Pipelines", PIPELINE_NAMES, default=PIPELINE_NAMES)
metric = st.sidebar.radio("Metric", ["kappa", "accuracy"])
show_winner = st.sidebar.checkbox("Show 2008 competition winner", value=True)
if not chosen:
    st.stop()

sub = bench[(bench.protocol == proto) & bench.pipeline.isin(chosen)].copy()
sub["Subject"] = sub.subject.map(U.subject_label)
fig = px.bar(sub, x="Subject", y=metric, color="pipeline", barmode="group",
             category_orders={"pipeline": chosen}, height=400)
if show_winner and metric == "kappa" and proto == "cross_session":
    fig.add_trace(go.Scatter(x=[U.subject_label(s) for s in C.SUBJECTS], y=C.WINNER_KAPPA,
                             mode="markers", marker=dict(symbol="line-ew-open", size=26, color="black",
                                                         line=dict(width=3)),
                             name="2008 winner (FBCSP)"))
st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Mean ± sd over 9 subjects")
    tab = (bench[bench.pipeline.isin(chosen)].groupby(["pipeline", "protocol"])[metric]
           .agg(lambda v: f"{v.mean():.3f} ± {v.std():.3f}").unstack()[list(protocols)])
    tab.columns = ["cross-session", "within day 1", "within day 2"]
    st.dataframe(tab, width="stretch")
    drop = (bench[bench.protocol == "within_session_E"].groupby("pipeline")[metric].mean()
            - bench[bench.protocol == "cross_session"].groupby("pipeline")[metric].mean())
    n_lower = int((drop > 0).sum())
    st.caption(f"Cross-session vs. within-session (day 2) {metric}: lower for {n_lower}/{len(drop)} pipelines ("
               + ", ".join(f"{p} {-d:+.3f}" for p, d in drop.items() if p in chosen)
               + "). That gap is EEG **non-stationarity**: the brain signal changes between days.")
with c2:
    st.subheader("Significance (paired Wilcoxon, n = 9)")
    stats = U.load_csv("stats.csv")
    if stats is not None:
        st.dataframe(stats[["a", "b", "mean_diff", "a_wins", "p_value"]].round(4), hide_index=True,
                     width="stretch")
        st.caption("Only 9 subjects means low statistical power: a p-value above 0.05 is *not* "
                   "evidence that two methods are equal.")
    a = st.selectbox("Compare A", PIPELINE_NAMES, index=PIPELINE_NAMES.index(OURS))
    b = st.selectbox("with B", PIPELINE_NAMES, index=PIPELINE_NAMES.index("FBCSP+SVM"))
    if a != b:
        r = evaluate.wilcoxon_compare(bench, a, b, protocol=proto, metric=metric)
        st.write(f"{a} − {b}: mean diff **{r['mean_diff']:+.3f}**, A wins {r['a_wins']}/9, "
                 f"p = **{r['p_value']:.4f}**")

st.divider()
st.subheader("Confusion matrix (cross-session)")
preds = U.load_csv("predictions_cross_session.csv")
if preds is not None:
    c1, c2 = st.columns([1, 2])
    pipe = c1.selectbox("Pipeline", PIPELINE_NAMES, index=PIPELINE_NAMES.index(OURS), key="cm_pipe")
    who = c1.selectbox("Subject", ["all"] + C.SUBJECTS,
                       format_func=lambda s: "all subjects" if s == "all" else U.subject_label(s))
    p = preds[preds.pipeline == pipe]
    if who != "all":
        p = p[p.subject == who]
    cm = evaluate.confusion(p.y_true, p.y_pred)
    labels = [C.CLASS_LABELS[c] for c in C.CLASSES]
    fig = px.imshow(np.round(cm, 2), x=labels, y=labels, text_auto=True, color_continuous_scale="Blues",
                    zmin=0, zmax=1, labels=dict(x="predicted", y="true", color="share"))
    fig.update_layout(height=380)
    c2.plotly_chart(fig, width="stretch")
    c1.caption("Rows sum to 1. Feet and tongue are the classes most often confused, because both "
               "are represented near the midline (Cz) rather than in one hemisphere.")
