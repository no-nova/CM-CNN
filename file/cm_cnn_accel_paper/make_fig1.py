# -*- coding: utf-8 -*-
"""Fig. 1 — System overview & microarchitectural roofline (Phase A/B/D/E)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 9.2,
    "axes.labelsize": 8.5, "legend.fontsize": 7.2, "figure.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_GRAY = "#1f5fa8", "#d97706", "#0e7a3d", "#b91c1c", "#6d28d9", "#4b5563"

fig = plt.figure(figsize=(13.2, 5.0))
gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.22)

# ---------------- Panel (a): pipeline diagram ----------------
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.set_title("(a)  CM-CNN-Accel static/dynamic acceleration pipeline", loc="left", fontweight="bold")

def box(x, y, w, h, text, fc, fs=6.6, ec="#111827", tc="#111827", lw=1.1, rounded=True):
    p = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.6,rounding_size=1.6" if rounded else "square,pad=0.0",
        fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc,
            wrap=True, linespacing=1.25)

def arrow(x0, y0, x1, y1, label=None, color="#374151", lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=13, lw=lw, color=color))
    if label:
        ax.text((x0+x1)/2, (y0+y1)/2 + 2.2, label, ha="center", fontsize=6.0,
                color=color, fontweight="bold")

box(2, 74, 26, 15, "Frozen official\ncheckpoints\n(BN folded, chmod 444,\nSHA-256 pinned)", "#e5e7eb", fs=6.3)
box(34, 74, 26, 15, "Union-find DAG\nisomorphic channel\npermutation  π_L\n(zero accuracy loss)", "#dbeafe", fs=6.3)
box(66, 74, 32, 15, "conv_tile kernel\n2x2 px × 4 groups = 16 ZMM\n0.5 load uops/FMA\n(register-resident)", "#dcfce7", fs=6.3)
arrow(28, 81.5, 34, 81.5); arrow(60, 81.5, 66, 81.5)

box(2, 40, 26, 17, "8-lane co-activation\ncosine clustering\n(VGG-4: 84.14% MAC cut,\n4.33x, 95.00% acc)", "#fef3c7", fs=6.3)
box(34, 40, 26, 17, "16-lane structured\npruning S1 (Taylor/L2)\nResNet-18 2.70x, MBV2 1.66x\n(+0.56 / +1.39 pp acc)", "#ffedd5", fs=6.3)
box(66, 40, 32, 17, "Tile Arithmetic Density\nlaw  α_l = MACs/(Tiles·C_in)\n∝ C_out·K²  → dominates\nwall-clock scaling", "#ede9fe", fs=6.3)

box(2, 6, 56, 18, "Temporal dirty-mask engine (Phase E.2):\nframe diff (τ = max 2.5% range, median+4·MAD)\n→ 8-neighbour dilation → multi-scale geometric\nprojection → MAC-weighted share > 0.50 ⇒\npre-budget full flush to M0 (bit-exact frames)", "#f0fdf4", fs=6.3)
box(64, 6, 34, 18, "Honest negative paths:\nscalar compaction 2.0 uops/FMA\ndev-blind rank-16 gate (−2.22 pp)\nP1 under real dynamic range", "#fef2f2", fs=6.3)
arrow(15, 74, 15, 57); arrow(47, 74, 47, 57); arrow(82, 74, 82, 57)
arrow(15, 40, 15, 24); arrow(47, 40, 30, 24); arrow(82, 40, 81, 24)

# ---------------- Panel (b): roofline ----------------
ax = fig.add_subplot(gs[1])
# measured / derived operating points (see paper Table)
pts = [
    ("v2 im2col kernel\n(measured, ResNet-18 v2)", 1.125, 555.422720/12.030, C_GRAY, "o", 9),
    ("patch-packed tap prefilter\n(measured, −33% MACs)", 0.62, 38.0, C_RED, "s", 9),
    ("conv_tile (dense tile kernel)\n(measured, 81 GMAC/s)", 0.5, 81.2, C_BLUE, "o", 13),
    ("scalar compaction\n(2.0 uops/FMA — never\nbeats dense on R18/MBV2)", 2.0, 81.2*0.42, C_ORANGE, "^", 9),
]
for name, x, y, c, m, s in pts:
    ax.scatter(x, y, s=s**2, color=c, marker=m, zorder=5, edgecolors="white", linewidths=0.7)
ax.axhline(81.2, ls="--", lw=1.0, color=C_BLUE, alpha=0.45)
ax.text(1.32, 85, "measured dense roofline\n≈ 81 GMAC/s (conv_tile)", fontsize=6.2, color=C_BLUE, va="bottom")
ax.set_xlabel("Load µops per FMA  (lower is better)")
ax.set_ylabel("Sustained convolution throughput  (GMAC/s)")
ax.set_xlim(0.38, 2.22); ax.set_ylim(20, 125)
ax.set_title("(b)  Microarchitectural roofline:\ninstruction-throughput wall, not FLOPs", loc="left", fontweight="bold")
ax.annotate("46.2 GMAC/s\n(derived: 555.4M MAC / 12.03 ms)", xy=(1.125, 46.2), xytext=(1.28, 28.5),
            fontsize=6.2, arrowprops=dict(arrowstyle="-", lw=0.8, color=C_GRAY), color=C_GRAY)
ax.annotate("38 GMAC/s (measured)", xy=(0.62, 38.0), xytext=(0.70, 24.5),
            fontsize=6.2, arrowprops=dict(arrowstyle="-", lw=0.8, color=C_RED), color=C_RED)
ax.annotate("scalar-gather ceiling\n< dense (0.78–0.97x)", xy=(2.0, 34.1), xytext=(1.42, 56),
            fontsize=6.2, arrowprops=dict(arrowstyle="-", lw=0.8, color=C_ORANGE), color=C_ORANGE)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig1_system_roofline.png")
print("fig1 done")
