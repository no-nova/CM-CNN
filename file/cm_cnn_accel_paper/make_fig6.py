# -*- coding: utf-8 -*-
"""Fig. 6 — Latency–accuracy / fidelity Pareto frontier."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 9.2,
    "axes.labelsize": 8.5, "legend.fontsize": 6.4, "figure.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.7), gridspec_kw={"wspace": 0.24})

# ---------------- Panel (a): 32x32 phases — ms vs test accuracy ----------------
ax = axes[0]
pts = [  # (ms, acc%, label, color, marker)
    (3.461, 94.72, "VGG-4 dense-avx512", "#4b5563", "o"),
    (0.834, 94.44, "VGG-4 v3.1 M5 (4.15x)", "#1f5fa8", "o"),
    (0.799, 95.00, "VGG-4 v4 8-lane cosine\n(4.33x, 84.14% MAC cut)", "#0e7a3d", "*"),
    (8.475, 97.78, "ResNet-18 M0 (32x32)", "#4b5563", "s"),
    (3.141, 98.33, "ResNet-18 S1 pruned\n(2.70x, +0.56 pp)", "#d97706", "s"),
    (0.621, 96.94, "MobileNetV2 M0 (32x32)", "#4b5563", "^"),
    (0.374, 98.33, "MobileNetV2 S1 pruned\n(1.66x v3 / 1.70x v2, +1.39 pp)", "#d97706", "^"),
]
for ms, acc, lab, c, m in pts:
    ax.scatter(ms, acc, s=120 if m == "*" else 55, color=c, marker=m,
               edgecolors="white", linewidths=0.8, zorder=5)
ax.set_xscale("log")
ax.set_xlabel("Single-core latency (ms / image, log scale)")
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(93.5, 99.2); ax.set_xlim(0.3, 12)
ax.set_title("(a)  32x32 regime: latency–accuracy Pareto\n(Phase A VGG-4 · Phase C pruned backbones)", loc="left", fontweight="bold")
ax.annotate("VGG-4 dense-avx512", (3.461, 94.72), xytext=(4.6, 95.15), fontsize=6.2,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#4b5563"))
ax.annotate("v4 cosine 4.33x", (0.799, 95.00), xytext=(0.46, 95.85), fontsize=6.2,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#0e7a3d"))
ax.annotate("v3.1 M5 0.828 ms", (0.834, 94.44), xytext=(1.15, 94.28), fontsize=6.0,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#1f5fa8"))
ax.annotate("ResNet-18 S1 2.70x", (3.141, 98.33), xytext=(1.6, 97.6), fontsize=6.2,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#d97706"))
ax.annotate("MobileNetV2 S1", (0.374, 98.33), xytext=(0.56, 98.72), fontsize=6.2,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#d97706"))
ax.annotate("MobileNetV2 M0 (0.62 ms, 96.94%)", (0.621, 96.94), xytext=(0.36, 96.30), fontsize=6.2,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#4b5563"))
ax.annotate("ResNet-18 M0 (8.48 ms, 97.78%)", (8.475, 97.78), xytext=(5.4, 96.15), fontsize=6.2,
            arrowprops=dict(arrowstyle="-", lw=0.7, color="#4b5563"))
ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

# ---------------- Panel (b): production — ms vs logit cosine vs M0 ----------------
ax = axes[1]
# torch reference (y=1.0, black x), M0 (y=1.0), P1, P1-random, P2, P2-random, keyframe
prod = {  # model: (torch_ms, m0_ms, p1_ms, p1cos, p1r_ms, p1rcos, p2_ms, p2cos, p2r_ms, p2rcos, kf_ms, kfcos)
 "ResNet-18":        (20.37, 65.85, 40.81, 0.653, 51.19, 0.533, 37.11, 0.988, 36.91, 0.974, 33.82, 0.987),
 "ResNet-50":        (58.36, 148.55, 94.37, 0.654, 121.12, 0.606, 82.11, 0.992, 81.03, 0.982, 76.51, 0.989),
 "MobileNetV2":      (9.53, 17.60, 14.28, 0.497, 16.65, 0.419, 9.50, 0.993, 10.43, 0.976, 9.33, 0.985),
 "MBV3-Small":       (2.13, 2.89, 2.15, 0.641, 2.43, 0.490, 2.49, 0.990, 2.58, 0.965, 1.58, 0.976),
 "ShuffleNetV2-x0.5":(2.97, 1.96, 1.56, 0.570, 1.72, 0.341, 2.24, 0.992, 2.21, 0.963, 1.15, 0.981),
 "MBV3-Large@320":   (14.10, 25.74, 18.35, 0.579, 25.45, 0.599, 15.69, 0.997, 16.80, 0.990, 14.27, 0.995),
}
markers = {"ResNet-18": "o", "ResNet-50": "s", "MobileNetV2": "^",
           "MBV3-Small": "D", "ShuffleNetV2-x0.5": "v", "MBV3-Large@320": "P"}
# Display-only vertical separation for overlapping oneDNN reference glyphs; source values remain y=1.0.
y_jitter = {"ResNet-18": 0.012, "ResNet-50": -0.008, "MobileNetV2": 0.012,
            "MBV3-Small": -0.008, "ShuffleNetV2-x0.5": 0.012, "MBV3-Large@320": -0.008}
for m, (tms, m0, p1ms, p1c, p1rms, p1rc, p2ms, p2c, p2rms, p2rc, kfms, kfc) in prod.items():
    mk = markers[m]
    ax.scatter(tms, 1.0 + y_jitter[m], s=42, marker=mk, facecolors="none", edgecolors="#111827", linewidths=1.1, zorder=6)
    ax.scatter(p2ms, p2c, s=52, marker=mk, color="#1f5fa8", alpha=0.9, edgecolors="white", linewidths=0.7)
    ax.scatter(p2rms, p2rc, s=42, marker=mk, color="#b91c1c", alpha=0.75, edgecolors="white", linewidths=0.7)
    ax.scatter(p1ms, p1c, s=52, marker=mk, color="#d97706", alpha=0.9, edgecolors="white", linewidths=0.7)
    ax.scatter(p1rms, p1rc, s=42, marker=mk, color="#f97316", alpha=0.75, edgecolors="white", linewidths=0.7)
ax.set_xscale("log")
ax.set_xlabel("Single-core latency / frame (ms, log scale)")
ax.set_ylabel("Logit cosine vs M0 (fidelity)")
ax.set_ylim(0.30, 1.02)
ax.set_title("(b)  Production matrix: fidelity–latency frontier\n(open = PyTorch oneDNN @ ≈1.0; blue = P2; orange = P1; red = controls)", loc="left", fontweight="bold")
ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
leg = [
    mlines.Line2D([], [], marker="o", ls="", mfc="none", mec="#111827", label="PyTorch oneDNN (torch MSE ref)"),
    mlines.Line2D([], [], marker="o", ls="", mfc="#1f5fa8", mec="white", label="P2 temporal cache"),
    mlines.Line2D([], [], marker="o", ls="", mfc="#b91c1c", mec="white", label="P2-random control"),
    mlines.Line2D([], [], marker="o", ls="", mfc="#d97706", mec="white", label="P1 static spatial gate"),
    mlines.Line2D([], [], marker="o", ls="", mfc="#f97316", mec="white", label="P1-random control"),
]
ax.legend(handles=leg, frameon=False, loc="lower right", fontsize=6.0)
leg_shapes = [
    mlines.Line2D([], [], marker=markers[m], ls="", mfc="#4b5563", mec="white", label=lab)
    for m, lab in [("ResNet-18", "R18"), ("ResNet-50", "R50"), ("MobileNetV2", "MBV2"),
                   ("MBV3-Small", "MBV3-S"), ("ShuffleNetV2-x0.5", "SNV2"), ("MBV3-Large@320", "MBV3-L")]
]
leg_shape = ax.legend(handles=leg_shapes, frameon=False, loc="lower left",
                      bbox_to_anchor=(0.005, 0.28), fontsize=6.0, ncol=2,
                      handletextpad=0.15, columnspacing=0.7)
leg_shape.set_title("model (marker shape)", prop={"size": 6.2, "weight": "bold"})
ax.legend(handles=leg, frameon=False, loc="lower right", fontsize=6.0)
ax.add_artist(leg_shape)
plt.tight_layout()
plt.savefig("figures/fig6_pareto.png")
print("fig6 done")
