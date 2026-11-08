"""Step 2 — exploration figures + a numeric check of contralateral ERD.

  figures/raw_trials_A03.png       one trial per class at C3/Cz/C4
  figures/psd_grand_average.png    class-wise spectra during imagery
  figures/erd_grand_average.png    Morlet ERD/ERS maps at C3/Cz/C4 per class (all subjects, session T)
  figures/erd_topomap_mu.png       mu-band (8-13 Hz) power change topomaps per class
  results/erd_check.csv            mu ERD at C3 vs C4 for left/right-hand imagery per subject
"""
import warnings

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import config as C
from mindmove import data, viz

warnings.filterwarnings("ignore")


def main():
    info = data.make_info()
    psd_acc, erd_acc, topo_acc, rows = {}, [], [], []
    for s in C.SUBJECTS:
        X, y = data.load_subject(s, sessions=("T",))["T"]
        if s == 3:
            for i, c in enumerate(C.CLASSES):
                trial = X[np.flatnonzero(y == i)[0]]
                viz.plot_raw_trial(trial, data.TIMES, title=f"A03 — {C.CLASS_LABELS[c]}").savefig(
                    C.FIGURES / f"raw_trial_A03_{c}.png", dpi=110)
        for ch in viz.MOTOR_CHANNELS:
            f, psd = viz.class_psd(X, y, ch)
            for c, p in psd.items():
                psd_acc.setdefault((ch, c), []).append(p)
        freqs, times, erd = viz.compute_erd(X, y)
        erd_acc.append(erd)
        topo = viz.band_erd_topo(X, y, (8, 13))
        topo_acc.append(topo)
        c3, c4 = C.CHANNELS.index("C3"), C.CHANNELS.index("C4")
        for band in ((8, 13), (8, 30)):
            t = topo if band == (8, 13) else viz.band_erd_topo(X, y, band)
            rows.append({"subject": s, "band": f"{band[0]}-{band[1]} Hz",
                         "left_hand_C3_dB": t["left_hand"][c3], "left_hand_C4_dB": t["left_hand"][c4],
                         "right_hand_C3_dB": t["right_hand"][c3], "right_hand_C4_dB": t["right_hand"][c4]})
        print(f"  A{s:02d} done", flush=True)

    # Grand-average PSD
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, ch in zip(axes, viz.MOTOR_CHANNELS):
        for c in C.CLASSES:
            ax.plot(f, np.mean(psd_acc[(ch, c)], 0), color=viz.CLASS_COLORS[c], label=C.CLASS_LABELS[c])
        ax.axvspan(8, 13, color="0.93", zorder=0)
        ax.set_title(ch)
        ax.set_xlabel("Hz")
    axes[0].set_ylabel("dB")
    axes[0].legend(fontsize=8)
    fig.suptitle("Grand-average power spectrum during imagery (9 subjects, session T)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "psd_grand_average.png", dpi=130)

    ga = {c: np.mean([e[c] for e in erd_acc], 0) for c in C.CLASSES}
    viz.plot_erd(freqs, times, ga, title="Grand-average ERD/ERS (9 subjects, session T)").savefig(
        C.FIGURES / "erd_grand_average.png", dpi=130)

    topo_ga = {c: np.mean([t[c] for t in topo_acc], 0) for c in C.CLASSES}
    viz.plot_topomaps(topo_ga, info, title="Mu-band (8-13 Hz) power change, imagery vs baseline",
                      unit="dB (blue = ERD)").savefig(C.FIGURES / "erd_topomap_mu.png", dpi=130)

    # Lateralisation check: compare the two hand classes *at the same electrode*
    # (removes channel-specific amplitude bias). Contralateral organisation means
    # left-hand imagery desynchronises C4 more than right-hand imagery does, and vice versa.
    df = pd.DataFrame(rows)
    df["C4_left_stronger"] = df.left_hand_C4_dB < df.right_hand_C4_dB
    df["C3_right_stronger"] = df.right_hand_C3_dB < df.left_hand_C3_dB
    df.to_csv(C.RESULTS / "erd_check.csv", index=False)
    print(df.round(2).to_string(index=False))
    for band, g in df.groupby("band"):
        print(f"\n[{band}] per subject: C4 left>right ERD {g.C4_left_stronger.sum()}/9, "
              f"C3 right>left ERD {g.C3_right_stronger.sum()}/9")
    c3, c4 = C.CHANNELS.index("C3"), C.CHANNELS.index("C4")
    print(f"[8-13 Hz grand average] C4: left {topo_ga['left_hand'][c4]:.2f} dB vs right "
          f"{topo_ga['right_hand'][c4]:.2f} dB | C3: right {topo_ga['right_hand'][c3]:.2f} dB vs left "
          f"{topo_ga['left_hand'][c3]:.2f} dB")


if __name__ == "__main__":
    main()
