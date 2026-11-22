"""Publication figures for paper 2.  Every curve is read from results/*.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
OUT = HERE.parents[1] / "paper2_bundle" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "svg.fonttype": "path", "pdf.fonttype": 42, "figure.dpi": 200, "savefig.bbox": "tight",
})
C = {"cm": "#1b5e9b", "nat": "#c1543a", "teal": "#2e8b57", "cats": "#8a5aa8",
     "wanda": "#8c6d31", "grey": "#666666"}


def load(n):
    return json.loads((R / n).read_text())


# ----------------------------------------------------------------- fig 1 ---
def fig_ppl():
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.55), sharey=False, layout="constrained")
    for ax, (fn, title) in zip(axes, [
            ("p6_sota_qwen3_5-0_8b.json", "Qwen3.5-0.8B (dense PPL 27.17)"),
            ("p6_sota_qwen3-0_6b.json", "Qwen3-0.6B (dense PPL 32.37)")]):
        d = load(fn)
        if fn.endswith("0_6b.json"):
            d["methods"]["cm_full"] = load("p6_sota_qwen3-0_6b_coupled.json")["methods"]["cm_full"]
        taus = sorted(float(t) for t in d["methods"]["cats_block"]["delta_ppl"])
        styles = [("cats_block", "CATS-Block", C["cats"], "o", "-"),
                  ("teal_block", "TEAL-Block", C["teal"], "s", "-"),
                  ("static_wanda", "Wanda 2:4 blocks", C["wanda"], "^", "-"),
                  ("cm_uncoupled", "CM-T uncoupled", C["cm"], "D", "-"),
                  ("cm_full", "CM-T coupled", C["cm"], "v", "--")]
        for key, lab, col, mk, ls in styles:
            if key not in d["methods"]:
                continue
            y = [d["methods"][key]["delta_ppl"][f"{t:.1f}"] if f"{t:.1f}" in
                 d["methods"][key]["delta_ppl"] else
                 d["methods"][key]["delta_ppl"][f"{t}"] for t in taus]
            ax.plot(taus, y, ls, marker=mk, ms=3.2, color=col, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel(r"skip budget $\tau$")
        ax.set_title(title)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel(r"$\Delta$PPL (WikiText-2)")
    axes[0].legend(frameon=False, loc="upper left", ncol=1)
    fig.savefig(OUT / "fig1_ppl_tau.svg")
    plt.close(fig)


# ----------------------------------------------------------------- fig 2 ---
def fig_ladder():
    d = load("p6_sota_qwen3_5-0_8b.json")
    lad = d["ladder"]
    names = [s["name"].replace("_", " ") for s in lad]
    taus = [0.10, 0.20, 0.30, 0.40]
    fig, ax = plt.subplots(figsize=(6.9, 2.5), layout="constrained")
    w = 0.19
    cols = ["#bcd3e8", "#8fb4d6", "#5f92c0", "#1b5e9b"]
    for i, t in enumerate(taus):
        y = [s["delta"][f"{t:.1f}"] if f"{t:.1f}" in s["delta"] else s["delta"][f"{t}"]
             for s in lad]
        ax.bar(np.arange(len(lad)) + (i - 1.5) * w, y, w, color=cols[i],
               label=rf"$\tau={t:.2f}$")
    ax.set_xticks(range(len(lad)))
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel(r"$\Delta$PPL")
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_ylim(0, max(max(s["delta"].values()) for s in lad) * 1.18)
    fig.savefig(OUT / "fig2_ladder.svg")
    plt.close(fig)


# ----------------------------------------------------------------- fig 3 ---
def fig_iq4():
    bm = load("p4_iq4_blockmajor.json")
    sl = bm["single_layer"]
    rows = sl["rows_table"] if isinstance(sl, dict) and "rows_table" in sl else sl
    skips, r4, r8, r8mt = [], [], [], []
    for r in rows:
        skips.append(float(r["skip"]))
        r4.append(float(r.get("ms_st4", 0)))
        r8.append(float(r.get("ms_st8", 0)))
        r8mt.append(float(r.get("ms_mt", 0)))
    st = bm["coupled_ffn_stream"]
    sd = st["rows_table"] if isinstance(st, dict) and "rows_table" in st else st

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45), layout="constrained")
    ax = axes[0]
    ax.plot(skips, r4, "o-", color="#8fb4d6", label="block-major, 4-row")
    ax.plot(skips, r8, "s-", color="#5f92c0", label="block-major, 8-row")
    ax.plot(skips, r8mt, "D-", color=C["cm"], label="block-major, 8-row, 32 threads")
    ax.axhline(bm["dense_baselines"]["rowmajor_single_layer_ms"], color=C["nat"],
               ls="--", lw=1.0, label="row-major dense")
    ax.set_xlabel(r"skip rate $\tau$")
    ax.set_ylabel("ms per layer")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    xs = [float(r["skip"]) for r in sd]
    ys = [float(r.get("speedup_cold", r.get("speedup", 1))) for r in sd]
    ax.plot(xs, ys, "D-", color=C["cm"])
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel(r"skip rate $\tau$")
    ax.set_ylabel("stream speedup")
    ax.grid(alpha=0.25, lw=0.4)
    fig.savefig(OUT / "fig3_iq4_blockmajor.svg")
    plt.close(fig)


# ----------------------------------------------------------------- fig 4 ---
def fig_ceiling():
    """Amdahl byte ceiling: max speedup versus the share of weights in the FFN."""
    frac = np.linspace(0.02, 0.45, 200)
    fig, ax = plt.subplots(figsize=(3.35, 2.45), layout="constrained")
    for tau, col in ((0.2, "#bcd3e8"), (0.4, "#5f92c0"), (0.6, "#1b5e9b")):
        ax.plot(frac * 100, 1.0 / (1.0 - frac * (1 - (1 - tau))), color=col,
                label=rf"$\tau$={tau:.1f} ({int(tau*100)}% of FFN skipped)")
    ax.set_xlabel("FFN share of per-token weight bytes (%)")
    ax.set_ylabel("ceiling on speedup")
    ax.set_ylim(1.0, 2.2)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "fig4_byte_ceiling.svg")
    plt.close(fig)


# ----------------------------------------------------------------- fig 5 ---
def fig_crossmodal():
    w = load("p6_whisper_real.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4), layout="constrained")
    ax = axes[0]
    lab, wer, cer = [], [], []
    for r in w["rows"]:
        lab.append(r["label"])
        wer.append(r["wer"] if isinstance(r["wer"], float) else r["wer"]["mean"])
        cer.append(r["cer"] if isinstance(r["cer"], float) else r["cer"]["mean"])
    x = np.arange(len(lab))
    ax.bar(x - 0.19, wer, 0.38, color=C["cm"], label="WER")
    ax.bar(x + 0.19, cer, 0.38, color="#8fb4d6", label="CER")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$\\tau$={0.2*i:.1f}" for i in range(len(lab))])
    ax.set_ylabel("WER / CER")
    ax.set_ylim(0, max(wer) * 1.35)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.4)

    ax = axes[1]
    s = load("p6_smolvlm_50.json")
    rows = s["rows"]
    nice = {"dense": "dense", "text": "LM", "vision": "ViT", "both": "both"}
    labels = []
    for r in rows:
        labels.append(nice.get(r["tower"], r["tower"])
                      + ("" if r["tower"] == "dense" else f"\n${r['tau']:.1f}$"))
    acc = [float(r["acc"]) for r in rows]
    lo = [float(r["acc"]) - float(r["acc_ci_lo"]) for r in rows]
    hi = [float(r["acc_ci_hi"]) - float(r["acc"]) for r in rows]
    cols = [C["grey"] if r["tower"] == "dense" else C["cm"] for r in rows]
    ax.bar(range(len(acc)), acc, 0.62, color=cols,
           yerr=[lo, hi], ecolor="#444444", capsize=1.6,
           error_kw={"lw": 0.7})
    ax.set_xticks(range(len(acc)))
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylabel("VQA accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.savefig(OUT / "fig5_crossmodal.svg")
    plt.close(fig)


fig_ppl()
fig_ladder()
fig_iq4()
fig_ceiling()
fig_crossmodal()
print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
