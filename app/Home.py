"""MindMove — Streamlit entry point.   Run:  streamlit run app/Home.py"""
import common as U  # noqa: I001  (sets sys.path)
import streamlit as st

import config as C

U.page("Home")

st.title("🧠 MindMove")
st.subheader("Decoding imagined movement from brain signals with classical (Riemannian) ML")

c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("""
**The problem.** People with ALS, spinal-cord injury or locked-in syndrome cannot move, but their
motor cortex still produces *movement intentions*. A Brain-Computer Interface (BCI) reads those
intentions from EEG so the patient can steer a wheelchair by **imagining** a movement.

**Why it is hard.** EEG is microvolts of signal buried in noise, it changes from day to day, and it
differs hugely between people, so a model trained on Monday fails on Tuesday, and every new
patient needs a long calibration session.

**What MindMove does.**
- treats each EEG trial as a **covariance matrix**, a point on the curved manifold of SPD matrices,
  and classifies with **Riemannian geometry** (MDM, tangent space), benchmarked against the
  **CSP / FBCSP** methods that won BCI Competition IV;
- **re-centers** each person's data on the manifold so a new patient can use models trained on
  others (**transfer learning**, no labels needed);
- tests **robustness** to broken electrodes and noise, and shows **what the brain is doing** (ERD maps,
  spatial patterns).

Data: **BCI Competition IV 2a**. 9 subjects, 22 EEG channels, 4 imagined movements
(left hand, right hand, feet, tongue), 2 sessions on different days, 288 trials each.
""")
with c2:
    st.graphviz_chart("""
digraph G {
  rankdir=TB; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=11];
  eeg [label="22-channel EEG trial\\n(0.5–2.5 s after cue)"];
  fb  [label="Band-pass filter bank\\n4–40 Hz"];
  cov [label="OAS covariance per band\\n(22×22 SPD matrix)"];
  rc  [label="Riemannian re-centering\\nM^-1/2 C M^-1/2", fillcolor="#fdf0e0"];
  ts  [label="Tangent space projection\\nat the Riemannian mean"];
  lr  [label="Logistic regression"];
  out [label="left · right · feet · tongue\\n→ wheelchair command", fillcolor="#e6f4ea"];
  eeg -> fb -> cov -> ts -> lr -> out; cov -> rc [style=dashed, label=" new patient"]; rc -> ts [style=dashed];
}""", width="stretch")

st.divider()
st.header("Headline results — official cross-session protocol (train day 1 → test day 2)")
summ = U.load_csv("benchmark_summary.csv")
if summ is None:
    st.info("Run `python scripts/03_benchmark.py` to produce the results.")
else:
    cs = summ[summ.protocol == "cross_session"].sort_values("kappa_mean", ascending=False)
    cols = st.columns(len(cs))
    for col, (_, r) in zip(cols, cs.iterrows()):
        delta = r.kappa_mean - 0.57
        col.metric(r.pipeline, f"κ = {r.kappa_mean:.2f}", f"{delta:+.2f} vs winner",
                   help=f"accuracy {r.accuracy_mean:.1%} ± {r.accuracy_std:.1%} (mean ± sd over 9 subjects)")
    st.caption("κ = Cohen's kappa, the official competition metric (0 = chance, 1 = perfect). "
               "The BCI Competition IV 2a winner (FBCSP, Ang et al. 2008) reached κ = 0.57.")
    U.figure("benchmark_cross_session.png")

st.divider()
st.markdown("""
### Explore
| Page | What you can do |
|---|---|
| **1 Brain Signals** | Look at raw EEG, spectra and ERD maps for any subject / class / trial |
| **2 Live BCI** | Replay a held-out session: the decoder drives a virtual wheelchair trial by trial |
| **3 Model Lab** | Compare all six pipelines, confusion matrices, significance tests |
| **4 Brain Maps** | Spatial patterns on the scalp, the manifold before/after re-centering |
| **5 New Patient** | How many calibration trials does a new patient need, with and without transfer? |
| **6 Robustness** | Break electrodes / add noise and watch accuracy (and repair it) live |
""")
st.caption(f"Classification window {C.CLF_TMIN}–{C.CLF_TMAX} s after cue · seed {C.SEED} · "
           "all methods are classical ML (no deep learning).")
