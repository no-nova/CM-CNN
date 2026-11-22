# -*- coding: utf-8 -*-
"""Fig. 2 — Co-activation clustering & multi-scale dirty projection.
Panel A: real measured heatmaps parsed from 11_PhaseA_Spatial_Heatmaps.txt (VGG-4, sample 0, label 8).
Panel B: schematic of the denoise→dilate→multi-scale geometric projection pipeline (illustration, not data).
Panel C: measured 8-lane block dead rates before/after clustering (09 JSON / main doc §9.2, §11.1).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import re

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2, "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_BLUE, C_GREEN, C_ORANGE = "#1f5fa8", "#0e7a3d", "#d97706"
RAMP = " .:-=+*#%@"   # ordinal luminance ramp used by the heatmaps

def parse_heatmaps(path):
    text = open(path).read()
    blocks = {}
    cur_key, cur_lines = None, []
    for line in text.splitlines():
        m = re.match(r"^([A-Z0-9_]+)\s*$", line.strip())
        if m and (line.strip().isupper() or m.group(1) in ("INPUT",)):
            if cur_key: blocks[cur_key] = cur_lines
            cur_key, cur_lines = line.strip(), []
            continue
        if line.strip().startswith("### sample"):
            if cur_key: blocks[cur_key] = cur_lines
            cur_key, cur_lines = None, []
            blocks["__sample__"] = line.strip()
            continue
        if cur_key is not None:
            cur_lines.append(line.rstrip("\n"))
    if cur_key: blocks[cur_key] = cur_lines
    return blocks

blocks = parse_heatmaps("/tmp/arena-coding-uploads/11_PhaseA_Spatial_Heatmaps.txt")
def grid_from(name):
    lines = [l for l in blocks.get(name, []) if l.strip() != ""]
    rows = [list(l.ljust(32)[:32]) for l in lines]
    return np.array([[RAMP.index(ch) if ch in RAMP else len(RAMP) - 1 for ch in row] for row in rows], dtype=float)

inp  = grid_from("INPUT")
dense2 = grid_from("DENSE_CONV2_ACTIVE")
cm2    = grid_from("CM_CONV2_EXECUTED")
print("shapes:", inp.shape, dense2.shape, cm2.shape)

fig = plt.figure(figsize=(14.2, 4.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 1.0, 0.98], wspace=0.42)

# ---------------- Panel (a): real heatmaps ----------------
gs_a = gs[0].subgridspec(1, 3, wspace=0.34)
titles = ["Input 32×32\n(sample 0, label 8)", "Dense Conv2 active\n(pre-ReLU truth)", "CM Conv2 executed\n(97.8% run;\ntap-level savings)"]
for i, (g, t) in enumerate(zip([inp, dense2, cm2], titles)):
    ax = fig.add_subplot(gs_a[i])
    ax.imshow(g, cmap="magma", interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(t, fontsize=6.5, pad=3)
fig.text(0.185, 1.00, "(a)  Real compute heatmaps (Phase A, frozen VGG-4) — object contour emerges",
         ha="center", fontsize=9.0, fontweight="bold")

# ---------------- Panel (b): multi-scale dirty projection schematic ----------------
ax = fig.add_subplot(gs[1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
ax.set_title("(b)  Multi-scale geometric projection (schematic)", loc="left", fontweight="bold", fontsize=9.0)
def box(x, y, w, h, text, fc, fs=6.4):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
                                         fc=fc, ec="#111827", lw=1.0))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)
def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.5, color="#374151"))
box(4, 78, 41, 14, "Frame diff in 2×2 base tiles\nτ = max(2.5% range,\nmedian + 4·MAD)", "#dbeafe", fs=6.3)
box(55, 78, 41, 14, "8-neighbour dilation\n(covers ≥ 3×3 halo of\nneighbouring 3×3 convs)", "#e0e7ff", fs=6.3)
arrow(45.5, 85, 54.5, 85)
box(4, 44, 41, 15, "Geometric projection onto\nstride-2 grids 112/56/28/\n14/7 — tile dirty iff any\nbase tile in footprint is", "#ede9fe", fs=5.9)
box(55, 44, 41, 15, "MAC-weighted share\nŝ = ΣMACs_dirty / ΣMACs\nŝ > 0.50 ⇒ flush to M0\n(2nd opinion: >50% tiles\ndiffer > 2.5% of range)", "#fef3c7", fs=5.9)
arrow(24, 77.5, 24, 59.5); arrow(75.5, 77.5, 75.5, 59.5); arrow(45.5, 51.5, 54.5, 51.5)
box(4, 8, 92, 16, "Flush ⇒ dense M0 kernels run IN-PLACE on the\npersistent cache ⇒ cut frames are bit-exact\n(all 6 models: cut spatial_rel_l2 == 0.000000)", "#f0fdf4", fs=6.2)
arrow(75.5, 43.5, 58, 24.5)

# ---------------- Panel (c): measured block dead rates before/after ----------------
ax = fig.add_subplot(gs[2])
layers = ["Conv2", "Conv3", "Conv4"]
before16 = [9.41, 37.22, 7.96]   # 16-lane all-dead, before (09 JSON dense.oracle_block16_dead[1:])
after16  = [16.74, 48.59, 34.05] # after isomorphic permutation (clustered)
before8  = [28.40, 64.59, 42.16] # 8-lane, bias-sorted (v3.1)
after8   = [30.22, 71.57, 45.09] # 8-lane, cosine co-activation (v4)
x = np.arange(3); w = 0.19
ax.bar(x - 1.5*w, before16, w, color="#9ca3af", label="16-lane before")
ax.bar(x - 0.5*w, after16,  w, color="#6b7280", label="16-lane after π (bias-sort)")
ax.bar(x + 0.5*w, before8,  w, color="#93c5fd", label="8-lane before (bias-sort)")
ax.bar(x + 1.5*w, after8,   w, color=C_BLUE,  label="8-lane after cosine clustering")
ax.set_xticks(x); ax.set_xticklabels(layers)
ax.set_ylabel("All-dead block / fiber rate (%)")
ax.set_ylim(0, 88)
ax.set_title("(c)  Reordering unlocks dead blocks\n(measured, frozen VGG-4, 360-image test)", loc="left", fontweight="bold", fontsize=8.4)
ax.legend(frameon=False, fontsize=6.0, loc="upper left")
ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
for xi, a, b in zip(x - 1.5*w, before16, [b - a for a, b in zip(before16, after16)]):
    ax.text(xi, a + 1.0, f"{a:.1f}", ha="center", fontsize=5.4, color="#374151")
for xi, v in zip(x - 0.5*w, after16):
    ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", fontsize=5.4, color="#374151")
for xi, v in zip(x + 0.5*w, before8):
    ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", fontsize=5.4, color="#374151")
for xi, v in zip(x + 1.5*w, after8):
    ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", fontsize=5.4, color="#1f5fa8", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig2_clustering_projection.png")
print("fig2 done")
