import common as U  # noqa: I001
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import config as C
from mindmove import transfer

U.page("New Patient", "🆕")
st.title("🆕 New patient: cutting calibration time with transfer learning")
st.markdown("""
A new patient normally has to perform **hundreds of labelled imagery trials** before the BCI works,
which is exhausting for someone with ALS. Here the decoder (TS+LR on 8–30 Hz covariances) is retrained
**live** using only *k* labelled trials per class from the new patient's day 1, then tested on their
day 2. *Transfer* adds the data of the 8 other subjects after **Riemannian re-centering**, which needs no labels.
""")

covs, ys = U.load_covs()
U.require(covs, "Covariance cache data/processed/covs_8_30.npz", "scripts/04_transfer.py")
s = U.subject_picker()
k = st.sidebar.select_slider("Labelled calibration trials per class (k)",
                             options=[0, 1, 2, 3, 5, 10, 20, 36, 72], value=5)
seed = st.sidebar.number_input("Random draw", 0, 99, 0)


@st.cache_resource(show_spinner="Re-centering all subjects on the manifold (once) ...")
def recentered():
    return transfer.recenter_all(covs)


@st.cache_data(show_spinner="Retraining decoders ...")
def live(subject, k, seed):
    rng_seed = C.SEED + seed
    rc = recentered()
    out = {}
    for m in ("subject_only", "subject_only_rc", "transfer_rc"):
        out[m] = transfer.calibration_point(covs, ys, subject, k, m, np.random.default_rng(rng_seed), rc=rc)
    return out


res = live(s, k, seed)
names = {"subject_only": "Subject only (standard)", "subject_only_rc": "Subject only + re-centering",
         "transfer_rc": "Transfer from 8 others + re-centering"}
cols = st.columns(3)
for col, (m, label) in zip(cols, names.items()):
    v = res[m]
    col.metric(label, "n/a (no data)" if np.isnan(v) else f"{v:.1%}",
               help="accuracy on the new patient's day-2 session (chance = 25%)")
if k == 0:
    st.success("k = 0: the transfer decoder has never seen a labelled trial from this patient, yet it "
               f"reaches {res['transfer_rc']:.1%} accuracy (chance 25%).")

st.divider()
cal = U.load_csv("calibration.csv")
if cal is not None:
    st.subheader("Precomputed calibration curves (5 random draws per k)")
    who = st.radio("Show", ["Selected subject", "Mean of 9 subjects"], horizontal=True)
    d = cal[cal.subject == s] if who == "Selected subject" else cal
    fig = go.Figure()
    for m, g in d.groupby("method", sort=False):
        a = g.groupby("k_per_class").accuracy.agg(["mean", "sem"]).reset_index()
        fig.add_trace(go.Scatter(x=a.k_per_class, y=a["mean"], error_y=dict(array=a["sem"].fillna(0)),
                                 mode="lines+markers", name=m))
    fig.add_hline(y=0.25, line_dash="dot", annotation_text="chance")
    fig.update_layout(height=420, xaxis_title="labelled trials per class from the new patient",
                      yaxis_title="accuracy on day 2", yaxis_range=[0.15, 1])
    st.plotly_chart(fig, width="stretch")

    mean = cal.groupby(["method", "k_per_class"]).accuracy.mean().unstack()
    so = mean.loc[mean.index.str.startswith("Subject only (")].iloc[0]
    tr = mean.loc[mean.index.str.startswith("Transfer")].iloc[0]
    target = tr.get(10, np.nan)
    need = so[so >= target]
    msg = (f"With transfer, **10 trials/class** give {target:.1%} on average. Standard calibration needs "
           + (f"**{int(need.index.min())} trials/class** to match it." if len(need)
              else "more than all 72 trials/class to match it."))
    st.info(msg)
