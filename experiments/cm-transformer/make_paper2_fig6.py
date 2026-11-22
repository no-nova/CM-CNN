"""Figure 6: the quality-throughput frontier at three levels of the stack."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = HERE.parents[1] / "paper2_bundle" / "figures"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "svg.fonttype": "path",
                     "font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    k = load("p1_kernels_v2.json") or {}
    perm = load("p8_perm.json") or {}
    sysd = load("p8_system.json") or {}
    iq4 = load("p4_iq4_blockmajor.json") or {}
    cpu = load("p1_kernels_v2cpu.json") or {}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)

    # ---- left: kernel / operator level
    ax = axes[0]
    rates = [r for r in (k.get("rates") or []) if r > 0]
    v7 = [k["micro"]["T1"][f"{r:.2f}"]["cuda_v7_speedup"] for r in rates]
    ax.plot(v7, [r * 100 for r in rates], "o-", lw=1.4, ms=3.5, color="#1f4e79",
            label="CUDA v7, T=1")
    v4 = [k["micro"]["T1"][f"{r:.2f}"]["cuda_v4_speedup"] for r in rates]
    ax.plot(v4, [r * 100 for r in rates], "s--", lw=1.1, ms=3.0, color="#c0392b",
            label="CUDA v4 (mask)")
    cpu_p = cpu.get("kernel_sweep") or []
    if cpu_p:
        ax.plot([c["speedup"] for c in cpu_p], [c["skip"] * 100 for c in cpu_p],
                "^-.", lw=1.1, ms=3.2, color="#2e7d32", label="AVX-512, row-parallel")
    st = (perm.get("strategies") or {}).get("ours", {}).get("rates", {})
    ax.axhline(0, color="#555", lw=0.6, ls=":")
    ax.set_xlabel("measured speedup vs dense operator")
    ax.set_ylabel("skipped blocks (%)")
    ax.set_title("Operator level: one projection")
    ax.set_ylim(-6, 78)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")

    # ---- right: end to end
    ax = axes[1]
    lad = (sysd.get("ladder") or {}).get("qwen3.5-0.8b", {})
    pts = []
    for tag, label, colour, marker in (
        ("tau0.20", "$\\tau$=0.20, end-to-end", "#1f4e79", "o"),
        ("tau0.40", "$\\tau$=0.40, end-to-end", "#5b9bd5", "s"),
    ):
        if tag in lad:
            pts.append((lad[tag]["speedup"], label, colour, marker))
    iq = iq4.get("coupled_ffn_stream") or []
    if iq:
        best = max(iq, key=lambda r: r["speedup"])
        pts.append((best["speedup"], "coupled FFN, 4B IQ4-NL", "#2e7d32", "^"))
    for x, label, colour, marker in pts:
        ax.vlines(x, 0.0, 1.0, color=colour, ls="--", lw=1.0, label=label)
        ax.plot([x], [1.0], marker=marker, color=colour, ms=4)
    ax.axvline(1.0, color="#555", lw=0.8, ls=":", ymin=0.0, ymax=1.0)
    ax.set_xlim(0.5, 2.2)
    ax.set_ylim(-0.12, 1.85)
    ax.set_yticks([])
    ax.set_xlabel("measured speedup vs dense")
    ax.set_title("Whole model: where the bound bites")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=1,
              handlelength=1.6, borderaxespad=0.2)

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "fig6_frontier.svg"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
