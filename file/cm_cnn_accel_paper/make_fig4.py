# -*- coding: utf-8 -*-
"""Fig. 4 — Stage-wise error propagation & near-boundary margin drift."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 9.2,
    "axes.labelsize": 8.5, "legend.fontsize": 6.8, "figure.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4), gridspec_kw={"wspace": 0.22})

# ---------------- Panel (a): 4-stage error propagation (P1) ----------------
ax = axes[0]
stages = [56, 28, 14, 7]
stage_p1 = {
 "ResNet-18":         [0.9539, 0.8149, 1.0536, 1.2027],
 "ResNet-50":         [0.9947, 1.1230, 1.0570, 0.5216],
 "MobileNetV2":       [0.6431, 0.9970, 0.9657, 1.2987],
 "MBV3-Small":        [0.2515, 0.6528, 0.8074, 1.0553],
 "ShuffleNetV2-x0.5": [0.2195, 0.4083, 1.1273, 1.0143],
 "MBV3-Large@320":    [0.0, 0.0, 0.0, 0.0],
}
styles = {"ResNet-18": ("#1f5fa8", "o"), "ResNet-50": ("#3b82f6", "s"),
          "MobileNetV2": ("#d97706", "^"), "MBV3-Small": ("#0e7a3d", "D"),
          "ShuffleNetV2-x0.5": ("#6d28d9", "v"), "MBV3-Large@320": ("#b91c1c", "P")}
for m, v in stage_p1.items():
    c, mk = styles[m]
    ax.plot([0, 1, 2, 3], v, marker=mk, ms=5.5, lw=1.5, color=c, label=m)
ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["56×56", "28×28", "14×14", "7×7"])
ax.set_xlabel("Feature-map stage (pre-downsample grid)")
ax.set_ylabel("P1 relative L2 error vs M0")
ax.set_title("(a)  Four-stage error propagation (P1), no blow-up", loc="left", fontweight="bold", fontsize=8.6)
ax.legend(frameon=False, fontsize=6.3, loc="lower center",
          bbox_to_anchor=(0.5, 1.04), ncol=6, handletextpad=0.4, columnspacing=0.8)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)

# ---------------- Panel (b): near-boundary margin drift ----------------
ax = axes[1]
models = ["R18", "R50", "MBV2", "MBV3-S", "SNV2", "MBV3-L"]
p1_dm  = [0.0581, 0.0223, 0.0222, 0.0687, 0.2035, 0.0415]
rnd_dm = [0.0698, 0.0258, 0.0281, 0.1241, 0.1431, 0.0203]
n_nb   = [77, 89, 90, 84, 44, 60]
xx = np.arange(6); w = 0.34
ax.bar(xx - w/2, p1_dm,  width=w, color="#1f5fa8", label="P1 (this work)")
ax.bar(xx + w/2, rnd_dm, width=w, color="#b91c1c", label="P1-random (equal-rate)")
ax.set_xticks(xx); ax.set_xticklabels([f"{m}\n(n={n})" for m, n in zip(models, n_nb)], fontsize=7.0)
ax.set_ylabel("near_boundary Δmargin  (M0 margin p1−p2 < 0.15)")
ax.set_title("(b)  Hardest-20% near-boundary slice — margin drift\nP1 beats equal-rate random on 4/6; loses on MBV3-L & SNV2", loc="left", fontweight="bold")
ax.legend(frameon=False)
ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
for i in range(6):
    ax.text(xx[i]-w/2, p1_dm[i]+0.003, f"{p1_dm[i]:.3f}", ha="center", fontsize=5.8, color="#1f5fa8")
    ax.text(xx[i]+w/2, rnd_dm[i]+0.003, f"{rnd_dm[i]:.3f}", ha="center", fontsize=5.8, color="#b91c1c")
plt.tight_layout()
plt.savefig("figures/fig4_error_boundary.png")
print("fig4 done")
