# -*- coding: utf-8 -*-
"""Fig. 3 — Tile Arithmetic Density Law across six production backbones."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 9.2,
    "axes.labelsize": 8.5, "legend.fontsize": 7.0, "figure.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})
C_P1, C_P2, C_BAR = "#1f5fa8", "#d97706", "#0e7a3d"

models  = ["ResNet-50", "ResNet-18", "MBV3-Small", "MobileNetV2", "MBV3-Large@320", "ShuffleNetV2-x0.5"]
graphM  = np.array([4089.2, 1814.1, 49.0, 280.1, 400.3, 38.7])   # MMACs
lever   = np.array([6.64, 8.98, 14.10, 20.07, 23.92, 38.97])     # tile leverage L (MAC-weighted)
deep    = np.array([78, 68, 80, 73, 66, 74])                     # deep MAC share (<=28 px), %
p1_spd  = np.array([1.57, 1.61, 1.34, 1.23, 1.40, 1.26])
p2_spd  = np.array([1.80, 1.78, 1.14, 1.85, 1.63, 0.88])
p1_cut  = np.array([49.4, 46.2, 40.0, 43.9, 59.4, 38.4])
p2_cut  = np.array([55.3, 55.2, 51.4, 54.1, 57.8, 52.2])
order   = np.argsort(lever)
bubble  = np.maximum(60.0, 4.2 * np.sqrt(graphM))   # area ~ sqrt(MACs), min 60 pt²

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.24})

# ---------------- Panel (a): leverage vs measured speedup ----------------
ax = axes[0]
ax.set_xscale("log")
sc1 = ax.scatter(lever, p1_spd, s=bubble, marker="o", color=C_P1, alpha=0.85,
                 edgecolors="white", linewidths=0.8, label="P1 static spatial gate (bubble area ∝ √MACs)")
sc2 = ax.scatter(lever, p2_spd, s=bubble, marker="s", color=C_P2, alpha=0.85,
                 edgecolors="white", linewidths=0.8, label="P2 temporal dirty-mask cache")
off_p1 = {"ResNet-50": (-4, -11), "ResNet-18": (7, 3), "MBV3-Small": (6, -10),
          "MobileNetV2": (7, -11), "MBV3-Large@320": (4, 6), "ShuffleNetV2-x0.5": (-8, -13)}
off_p2 = {"ResNet-50": (5, 5), "ResNet-18": (-2, 7), "MBV3-Small": (4, -12),
          "MobileNetV2": (6, 6), "MBV3-Large@320": (6, -12), "ShuffleNetV2-x0.5": (4, 7)}
for i in order:
    ax.annotate(models[i], (lever[i], p1_spd[i]), textcoords="offset points",
                xytext=off_p1[models[i]], fontsize=6.2, color=C_P1)
    ax.annotate(models[i], (lever[i], p2_spd[i]), textcoords="offset points",
                xytext=off_p2[models[i]], fontsize=6.2, color=C_P2)
ax.axhline(1.0, ls="--", lw=1.1, color="#b91c1c", alpha=0.75)
ax.set_xlabel("Tile arithmetic leverage  L  (log scale)\nL = (MACs_l/MACs) / (Tiles_l/Tiles)  — ᾱ = ΣMACs/ΣTiles normalised")
ax.set_ylabel("Measured single-core speedup vs M0 (x)")
ax.set_xlim(5.6, 56); ax.set_ylim(0.75, 2.15)
ax.set_title("(a)  Arithmetic density per tile vs wall-clock payoff\n(6 official torchvision checkpoints, frozen)", loc="left", fontweight="bold")
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(loc="lower left", frameon=False)

# ---------------- Panel (b): leverage vs R1 cut + deep share ----------------
ax = axes[1]
ax.set_axisbelow(True)
R1cut = {"ResNet-50": 97.174, "ResNet-18": 97.195, "MBV3-Small": 94.446,
         "MobileNetV2": 96.808, "MBV3-Large@320": 96.665, "ShuffleNetV2-x0.5": 94.763}
lever_sorted_models = [models[i] for i in order]
bars = ax.bar(np.arange(6), [R1cut[m] for m in lever_sorted_models], width=0.55,
              color=C_BAR, alpha=0.88, label="R1 static MAC cut (%)")
ax.set_xticks(np.arange(6))
ax.set_xticklabels(["R50\n6.64", "R18\n8.98", "MBV3-S\n14.10", "MBV2\n20.07", "MBV3-L\n23.92", "SNV2-x0.5\n38.97"],
                   fontsize=6.9)
ax.set_ylim(90, 100)
ax.set_ylabel("R1 MAC elimination (%)")
ax.set_xlabel("Model (tile leverage L below each bar)")
ax.set_title("(b)  R1 > 90% everywhere (deep-MACs 66–80%)", loc="left", fontweight="bold", fontsize=8.6)
ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
for i, m in enumerate(lever_sorted_models):
    ax.text(i, R1cut[m] + 0.06, f"{R1cut[m]:.1f}", ha="center", fontsize=6.6, color="#065f46")
ax2 = ax.twinx(); ax2.set_ylim(0, 130); ax2.spines["right"].set_visible(True)
ax2.plot(np.arange(6), [deep[models.index(m)] for m in lever_sorted_models],
         color="#6d28d9", marker="D", ms=5, lw=1.4, label="deep (≤28 px) MAC share (%, right axis)")
ax2.set_ylabel("Deep (≤28 px) MAC share (%)")
ax2.tick_params(axis="y", labelsize=7)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(-0.03, 1.32),
          frameon=True, framealpha=0.95, fontsize=6.3, edgecolor="#9ca3af")
plt.tight_layout()
plt.savefig("figures/fig3_density_law.png")
print("fig3 done")
