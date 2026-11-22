# -*- coding: utf-8 -*-
"""Fig. 7 — Real-world video evaluation (V1-V6) on public Xiph DERF footage.

Panels
  (a) MAC elimination per regime on real video for all six backbones, with the
      synthetic R1-R6 reference stream (MBV3-Small) drawn dashed for comparison.
  (b) P2 wall-clock speedup vs the dense M0 rerun of the same frames; values below
      1.0 are genuine slowdowns and are drawn as such.
  (c) pre-GAP spatial fidelity of P2 against its equal-rate random control and the
      T=4 periodic keyframe baseline (reference backbone, log scale).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "cm-cnn-production"
OUT = Path(__file__).resolve().parent / "figures" / "fig7_real_video.png"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 10.6,
    "axes.labelsize": 8.5, "legend.fontsize": 9.0, "figure.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})

REG = ["V1", "V2", "V3", "V4", "V5", "V6"]
LABEL = {
    "V1": "V1\nstatic", "V2": "V2\nlocal motion", "V3": "V3\nfast object",
    "V4": "V4\ncamera pan", "V5": "V5\nillum. step", "V6": "V6\nshot cuts",
}
MODELS = [
    ("resnet18", "ResNet-18", "#1f5fa8", "o"),
    ("resnet50", "ResNet-50", "#0e7a3d", "s"),
    ("mobilenet_v2", "MobileNetV2", "#8c3fa8", "^"),
    ("mobilenetv3_small", "MBV3-Small", "#d97706", "D"),
    ("shufflenetv2_x0_5", "ShuffleNetV2-x0.5", "#b91c1c", "v"),
    ("mbv3_large_320", "MBV3-Large@320", "#0f766e", "P"),
]


def main():
    rv = json.load(open(EXP / "real_video_results.json"))
    rd = json.load(open(EXP / "reviewer_defense.json"))
    x = np.arange(len(REG))

    fig = plt.figure(figsize=(10.0, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.68, wspace=0.26)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    axes = [ax_a, ax_b, ax_c]

    # ---- (a) MAC elimination on real video -------------------------------------
    ax = axes[0]
    for key, name, color, mk in MODELS:
        m = rv["models"][key]
        y = [r["p2"]["mac_elimination"] * 100 for r in m["regimes"]]
        ax.plot(x, y, marker=mk, ms=5.0, lw=1.4, color=color, label=name, zorder=4)
    syn = rd["models"]["mobilenetv3_small"]["regimes"]
    ys = [r["p2"]["mac_elimination"] * 100 for r in syn]
    ax.plot(x, ys, ls="--", lw=1.1, color="#9ca3af", marker="x", ms=5,
            label="MBV3-Small, synthetic R1-R6", zorder=3)
    ax.axhline(0, color="#6b7280", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[r] for r in REG], fontsize=9.6)
    ax.set_ylabel("MAC elimination (%)")
    ax.set_ylim(-4, 104)
    ax.set_title("(a)  Real video: five regimes keep elimination, the global pan flushes (V4 = 0%)",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False,
              columnspacing=1.2, handlelength=1.6)

    # ---- (b) speedup vs dense M0 ------------------------------------------------
    ax = axes[1]
    for key, name, color, mk in MODELS:
        m = rv["models"][key]
        y = [r["p2"]["speedup_vs_m0"] for r in m["regimes"]]
        ax.plot(x, y, marker=mk, ms=4.8, lw=1.3, color=color, label=name)
    ax.axhline(1.0, color="#374151", lw=0.9, ls=":")
    ax.text(5.30, 1.03, "M0", fontsize=8.6, color="#374151")
    ax.set_xticks(x); ax.set_xticklabels([r for r in REG], fontsize=9.8)
    ax.set_ylabel("P2 speedup vs dense M0 (x)")
    ax.set_ylim(0.35, 2.45)
    ax.set_title("(b)  Wall clock: 0.50-2.47x", loc="left", fontweight="bold")

    # ---- (c) spatial fidelity, reference backbone ------------------------------
    ax = axes[2]
    ref = rv["models"]["mobilenetv3_small"]
    p2 = [r["p2"]["spatial_rel_l2"] for r in ref["regimes"]]
    rnd = [r["p2_random"]["spatial_rel_l2"] for r in ref["regimes"]]
    key = [r["p2_keyframe"]["spatial_rel_l2"] for r in ref["regimes"]]
    w = 0.26
    ax.bar(x - w, p2, w, color="#d97706", label="P2 (CM dirty-tile cache)")
    ax.bar(x, rnd, w, color="#9ca3af", label="equal-rate random")
    ax.bar(x + w, key, w, color="#60a5fa", label="keyframe T=4")
    floor = 5e-4
    ax.set_yscale("log")
    ax.set_ylim(floor, 1.0)
    for i, v in enumerate(p2):
        if v == 0.0:
            ax.text(i - w, floor * 1.15, "exact\n0.0", ha="center", fontsize=7.6, color="#d97706")
    ax.set_xticks(x); ax.set_xticklabels([r for r in REG], fontsize=9.8)
    ax.set_ylabel("pre-GAP spatialRelL2 vs M0 (log)")
    ax.set_title("(c)  Fidelity: exact on flush, ghosted on reuse", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)

    fig.savefig(OUT, dpi=300)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
