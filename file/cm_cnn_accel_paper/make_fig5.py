# -*- coding: utf-8 -*-
"""Fig. 5 — Six-regime stress ladder & scene-cut analysis (Phase E.2).

All numbers are read from experiments/cm-cnn-production/reviewer_defense.json so the figure cannot
drift away from the measured data.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.0, "axes.titlesize": 9.0,
    "axes.labelsize": 8.3, "legend.fontsize": 6.6, "figure.dpi": 300,
    "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
})

EXP = Path(__file__).resolve().parents[2] / "experiments" / "cm-cnn-production"
DEF = json.loads((EXP / "reviewer_defense.json").read_text())

keys = ["resnet18", "resnet50", "mobilenet_v2", "mobilenetv3_small",
        "shufflenetv2_x0_5", "mbv3_large_320"]
models = ["ResNet-18", "ResNet-50", "MobileNetV2", "MBV3-Small", "ShuffleNetV2-x0.5", "MBV3-Large@320"]
lad = DEF["meta"]["e2_status"]["ladder_cut_pct"]
cut = {lab: [lad[k][r] for r in ("R1", "R2", "R3", "R4", "R5", "R6")] for k, lab in zip(keys, models)}

regimes = ["R1\nstatic", "R2\nmotion", "R3\nfast", "R4\npan", "R5\nillum", "R6\ncut"]
colors = ["#1f77b4", "#2ca02c", "#d97706", "#b91c1c", "#2563eb", "#6d28d9"]

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), gridspec_kw={"wspace": 0.26})

# ---------------- Panel (a): grouped ladder bars ----------------
ax = axes[0]
x = np.arange(6); w = 0.13
for i, m in enumerate(models):
    ax.bar(x + (i - 2.5) * w, cut[m], width=w, label=m, color=colors[i], alpha=0.92)
ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=7.6)
ax.set_ylabel("MAC elimination (%)")
ax.set_ylim(0, 108)
ax.axhline(90, ls=":", lw=0.9, color="#374151", alpha=0.6)
ax.set_title("(a)  Six-model x six-regime ladder\nstrict  R1 > R2 > R3 > R4 = 0  everywhere", loc="left", fontweight="bold")
ax.legend(fontsize=5.7, ncol=2, frameon=False, loc="upper right")
ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)

# ---------------- Panel (b): reference-model latency per regime ----------------
ax = axes[1]
ref = DEF["models"]["mobilenetv3_small"]["regimes"]
mean_ms = [r["p2"]["mean_ms"] for r in ref]
p95_ms = [r["p2"]["p95_ms"] for r in ref]
max_ms = [r["p2"]["max_ms"] for r in ref]
m0_ms = [r["p2"]["m0_ms"] for r in ref]
xx = np.arange(6)
ax.plot(xx, mean_ms, "o-", color="#1f5fa8", lw=1.6, ms=5, label="P2 mean")
ax.plot(xx, p95_ms, "s--", color="#d97706", lw=1.3, ms=4.5, label="P2 P95")
ax.plot(xx, max_ms, "^:", color="#b91c1c", lw=1.3, ms=4.5, label="P2 max")
ax.plot(xx, m0_ms, "x--", color="#4b5563", lw=1.3, ms=5, label="M0 dense (per-regime)")
ax.set_xticks(xx); ax.set_xticklabels(regimes, fontsize=7.6)
ax.set_ylabel("ms / frame (MBV3-Small reference)")
ax.set_title("(b)  Latency ladder incl. tail - MBV3-Small\nflush frame (R4) pays dense M0 + bookkeeping", loc="left", fontweight="bold")
ax.legend(frameon=True, framealpha=0.9, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=6.4)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)

# ---------------- Panel (c): R6 scene-cut spatial error ----------------
ax = axes[2]
p2_r6 = [DEF["models"][k]["regimes"][5]["p2"]["spatial_rel_l2"] for k in keys]
rnd_r6 = [DEF["models"][k]["regimes"][5]["p2_random"]["spatial_rel_l2"] for k in keys]
kf_r6 = [DEF["models"][k]["regimes"][5]["p2_keyframe"]["spatial_rel_l2"] for k in keys]
xx = np.arange(6); w = 0.26
ax.bar(xx - w, p2_r6, width=w, color="#1f5fa8", label="P2 (this work)")
ax.bar(xx, rnd_r6, width=w, color="#b91c1c", label="P2-random (equal-rate control)")
ax.bar(xx + w, kf_r6, width=w, color="#6d28d9", label="P2-keyframe (periodic, T=4)")
ax.set_xticks(xx); ax.set_xticklabels(["R18", "R50", "MBV2", "MBV3-S", "SNV2", "MBV3-L"], fontsize=7.6)
ax.set_ylabel("R6 spatialRelL2 (108-frame mean)")
ax.set_title("(c)  R6 hard scene-cut: P2 vs controls\nflush frames bit-exact (spatialRelL2 = 0) for all 6", loc="left", fontweight="bold")
ax.legend(frameon=False, fontsize=6.2)
ax.set_ylim(0, max(max(p2_r6), max(rnd_r6), max(kf_r6)) * 1.25)
ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "figures" / "fig5_stress_ladder.png")
print("fig5 done", dict(zip(models, [[round(v, 1) for v in cut[m]] for m in models])))
