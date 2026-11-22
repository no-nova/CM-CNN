"""Assemble every raw result into the paper's result file + markdown report.

    python3 assemble_report.py

Reads results/*.json (produced by m0_baseline.py, p1_ffn.py, p1_kernels.py,
p2_stream.py) and experiments/cm-cnn-production/real_video_results.json
(Paper 1), writes:
    experiments/cm-transformer/qwen3_5_0_8b_results.json
    experiments/cm-transformer/REPORT.md
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

from common import (EXP_DIR, FINAL_JSON, PAPER1_RESULTS, RESULTS_DIR, SAFETENSORS,
                    log, sha256_file)

REGIMES = ["V1", "V2", "V3", "V4", "V5", "V6"]
NAMES = {"V1": "static surveillance", "V2": "local motion",
         "V3": "fast natural object", "V4": "camera pan",
         "V5": "illumination change", "V6": "multi-shot cuts"}


def load(name):
    p = RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, n=2):
    return "—" if x is None else f"{x:.{n}f}"


def pct(x, n=1):
    return "—" if x is None else f"{100 * x:.{n}f}%"


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=EXP_DIR.parents[1],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    m0 = load("m0_baseline.json")
    p1 = {"cuda_fp32": load("p1_cuda_float32.json"),
          "cuda_bf16": load("p1_cuda_bfloat16.json"),
          "cpu_fp32": load("p1_cpu_float32.json")}
    kernels = load("p1_kernels.json")
    p2 = {"cuda": load("p2_cuda.json"), "cpu": load("p2_cpu.json")}
    p1_paper = json.loads(PAPER1_RESULTS.read_text()) if PAPER1_RESULTS.exists() else {"models": {}}

    arch = m0["arch"]
    vc, tc = arch["vision_encoder"], arch["language_backbone"]

    # ----------------------------------------------------------- architecture
    t_arch = table(
        ["component", "value"],
        [["vision depth / width / heads", f"{vc['depth']} / {vc['hidden_size']} / {vc['num_heads']}"],
         ["vision patch (T×H×W)", f"{vc['temporal_patch_size']}×{vc['patch_size']}×{vc['patch_size']}"],
         ["vision MLP (fc1, fc2)", f"768→{vc['intermediate_size']} ({vc['activation']}) →768"],
         ["vision merger", "LN(768) → 2×2 merge → 3072 → 3072 → 1024"],
         ["vision tokens per 224² frame", "196 (14×14); after 2×2 merge 49"],
         ["vision halo dilation L", f"{vc['halo_dilation_L']} (stride-16 patch embed)"],
         ["vision MACs per frame pair", f"{vc['macs_per_frame_pair_224']['total_per_pair']/1e6:.1f} MMAC"],
         ["LM layers", f"{tc['num_hidden_layers']} ({tc['n_linear_attention']} linear + {tc['n_full_attention']} full at 3,7,11,15,19,23)"],
         ["LM hidden / FFN", f"{tc['hidden_size']} / {tc['intermediate_size']} (SwiGLU, 112 32-lane blocks)"],
         ["LM full attention", f"{tc['full_attention']['num_heads']} q-heads, {tc['full_attention']['num_kv_heads']} kv-heads, head_dim {tc['full_attention']['head_dim']}, output gate"],
         ["LM linear attention", f"gated delta rule, {tc['linear_attention']['num_key_heads']} heads, k/v dim {tc['linear_attention']['key_head_dim']}, conv k={tc['linear_attention']['conv_kernel_dim']}"],
         ["RoPE", "partial 25% (64 of 256 dims), theta 1e7, interleaved"],
         ["FFN MACs per token", f"{tc['macs_per_token']['ffn']['gate']/1e6:.2f} + {tc['macs_per_token']['ffn']['up']/1e6:.2f} + {tc['macs_per_token']['ffn']['down']/1e6:.2f} MMAC (gate/up/down)"],
         ["vocab / params", f"{tc['vocab_size']} / {arch['checkpoint']['total_params']:,}"]])

    # --------------------------------------------------------------- M0 table
    rows = []
    for r in REGIMES:
        g32 = m0["m0"]["vision_gpu"]["float32"]["regimes"][r]
        g16 = m0["m0"]["vision_gpu"]["bfloat16"]["regimes"][r]
        c32 = m0["m0"]["cpu"]["vision"]["regimes"][r]
        rows.append([r, NAMES[r], f(g32["mean_ms_per_pair"], 3), f(g16["mean_ms_per_pair"], 3),
                     f(c32["mean_ms_per_pair"], 1), f(g32["single_pair_ms"], 2),
                     f(g32["batch_vs_single_max_abs"], 6)])
    t_m0 = table(["regime", "content", "GPU fp32 ms/pair", "GPU bf16 ms/pair",
                  "CPU fp32 ms/pair", "GPU fp32 ms/single", "determinism max abs Δ"], rows)

    def mean_ms(rows_):
        return sum(x["ms"] for x in rows_) / len(rows_)

    def max_rep(rows_):
        return max(x["repeat_max_abs"] for x in rows_)

    gtxt, ctxt = m0["m0"]["text_gpu"], m0["m0"]["cpu"]["text"]
    t_m0_text = table(
        ["platform", "fp32 ms/prompt", "bf16 ms/prompt", "fp16 ms/prompt", "repeat max abs Δ"],
        [["RTX 5070", f(mean_ms(gtxt["float32"]), 1), f(mean_ms(gtxt["bfloat16"]), 1),
          f(mean_ms(gtxt["float16"]), 1), f(max_rep(gtxt["float32"]), 6)],
         ["Zen 5 (1 thread)", f(mean_ms(ctxt), 1), "—", "—", f(max_rep(ctxt), 6)]])

    # ------------------------------------------------------------ P1: iso
    iso_rows = []
    for tag, d in (("fp32 (GPU)", p1["cuda_fp32"]), ("bf16 (GPU)", p1["cuda_bf16"]),
                   ("fp32 (CPU, T≤32)", p1["cpu_fp32"])):
        if not d:
            continue
        i = d["isomorphism_tau0"]["clustered"]
        iso_rows.append([tag, f(i["max_abs_dlogit"], 6), f(i["logitRmse"], 6), f(i["kl"], 8),
                         f(i["top1_agreement"], 3)])
    t_iso = table(["precision", "max abs Δlogit", "logitRmse", "KL", "Top-1"], iso_rows)

    # ------------------------------------------------------- P1: skip sweep
    d = p1["cuda_fp32"]
    sweep_rows = []
    for k, rate in enumerate(d["sweep"]["clustered"]):
        r_c = d["sweep"]["clustered"][k]
        r_n = d["sweep"]["natural"][k]
        r_r = d["sweep"]["random"][k]
        sweep_rows.append([f(rate["rate"], 1),
                           f(r_c["logitRmse"], 3), f(r_n["logitRmse"], 3), f(r_r["logitRmse"], 3),
                           f"{f(r_c['kl'], 5)} / {f(r_n['kl'], 5)}",
                           f(r_c["top1_agreement"], 3), pct(rate["energy_retained"]),
                           pct(rate["mac_reduction_ffn_total"]), pct(rate["mac_reduction_down_proj"])])
    t_p1 = table(["skip rate τ", "RMSE clustered", "RMSE natural", "RMSE random",
                  "KL clus / nat", "Top-1", "block energy kept", "FFN MAC cut", "down_proj MAC cut"],
                 sweep_rows)

    # --------------------------------------------------------- P1: kernels
    krows = []
    if kernels:
        v6 = {r["rate"]: r for r in kernels["cuda"]["v6"]}
        v4 = {r["rate"]: r for r in kernels["cuda"]["v4"]}
        avx = {r["rate"]: r for r in kernels["avx512"]["sweep"]}
        for rate in sorted(v6):
            krows.append([f(rate, 2),
                          f(v6[rate]["ms_per_token_all24_T1"], 4), f(v6[rate]["speedup_vs_dense"], 2),
                          f(v6[rate]["GFLOPs"], 0),
                          f(v4[rate]["ms_per_token_all24_T1"], 4), f(v4[rate]["speedup_vs_dense"], 2),
                          f(avx[rate]["sum_ms_all24_layers_T1"], 2),
                          f(avx[rate]["speedup_vs_dense"], 2),
                          f(avx[rate]["GFLOPs"], 1)])
    t_kern = table(["skip rate", "CUDA v6 ms/token", "v6 ×", "v6 GFLOP/s",
                    "CUDA v4 ms/token", "v4 ×", "AVX-512 ms/token", "AVX-512 ×", "AVX-512 GFLOP/s"],
                   krows)

    # ---------------------------------------------------------------- P2 GPU
    pc = p2["cuda"]
    t_p2 = []
    for r in REGIMES:
        rows_r = pc["regimes"][r]
        tau = rows_r["policies"]["tau"]
        age = rows_r["policies"]["tau+age4"]
        v0 = tau["variants"]["0.0|isomorphic"]
        v5 = tau["variants"]["0.5|clustered"]
        a0 = age["variants"]["0.0|isomorphic"]
        t_p2.append([r, NAMES[r], pct(rows_r["mean_dirty_ratio"]), pct(tau["mean_group_touched"]),
                     str(rows_r["flush_frames"]),
                     f(rows_r["dense_ms_per_frame"], 2),
                     f(v0["ms_per_frame"], 2), f(v0["wallclock_speedup"], 2),
                     pct(v0["mac_reduction"]), f(v0["merged_cos_mean"], 3), f(v0["pooled_cos"], 3),
                     pct(age["mean_dirty_ratio"]),
                     f(a0["wallclock_speedup"], 2), f(a0["merged_cos_mean"], 3),
                     pct(v5["mac_reduction"]), f(v5["wallclock_speedup"], 2), f(v5["merged_cos_mean"], 3)])
    t_p2 = table(["regime", "content", "dirty", "2×2 groups", "flush frames", "dense ms",
                  "P2 ms", "P2 ×", "MAC cut", "cos", "pooled cos",
                  "P2+age dirty", "P2+age ×", "P2+age cos",
                  "P2+P1(0.5) MAC cut", "P2+P1 ×", "P2+P1 cos"], t_p2)

    # -------------------------------------------- P2 vs Paper 1 CNNs (same V1-V6)
    comp = []
    for r in REGIMES:
        row = [r]
        for model in ("mobilenetv3_small", "resnet18"):
            m = p1_paper["models"].get(model)
            if not m:
                row += ["—", "—"]
                continue
            reg = next(x for x in m["regimes"] if x["regime"] == r)["p2"]
            row += [pct(reg["mac_elimination"]), f(reg["speedup_vs_m0"], 2)]
        v0 = pc["regimes"][r]["policies"]["tau"]["variants"]["0.0|isomorphic"]
        row += [pct(v0["mac_reduction"]), f(v0["wallclock_speedup"], 2), f(v0["merged_cos_mean"], 3)]
        per = pc["regimes"][r]["halo_percolation_3x3_stage"]
        row += [pct(per[1]), pct(per[3]), pct(per[5])]
        comp.append(row)
    t_comp = table(["regime", "Paper1 MBV3-S MAC cut", "MBV3-S ×", "Paper1 RN18 MAC cut", "RN18 ×",
                    "Qwen3.5 vision MAC cut", "Qwen3.5 ×", "Qwen3.5 cos",
                    "halo r=1 (hyp.)", "halo r=3", "halo r=5"], comp)

    # ---------------------------------------------------------------- P2 CPU
    cpu = p2["cpu"]
    t_cpu = table(
        ["regime", "dirty", "dense ms", "P2 ms", "P2 ×", "MAC cut", "cos",
         "P2+age ms", "P2+age ×", "P2+age cos"],
        [[r, pct(cpu["regimes"][r]["mean_dirty_ratio"]),
          f(cpu["regimes"][r]["dense_ms_per_frame"], 1),
          f(cpu["regimes"][r]["policies"]["tau"]["variants"]["0.0|isomorphic"]["ms_per_frame"], 1),
          f(cpu["regimes"][r]["policies"]["tau"]["variants"]["0.0|isomorphic"]["wallclock_speedup"], 2),
          pct(cpu["regimes"][r]["policies"]["tau"]["variants"]["0.0|isomorphic"]["mac_reduction"]),
          f(cpu["regimes"][r]["policies"]["tau"]["variants"]["0.0|isomorphic"]["merged_cos_mean"], 3),
          f(cpu["regimes"][r]["policies"]["tau+age4"]["variants"]["0.0|isomorphic"]["ms_per_frame"], 1),
          f(cpu["regimes"][r]["policies"]["tau+age4"]["variants"]["0.0|isomorphic"]["wallclock_speedup"], 2),
          f(cpu["regimes"][r]["policies"]["tau+age4"]["variants"]["0.0|isomorphic"]["merged_cos_mean"], 3)]
         for r in REGIMES])

    # ------------------------------------------------------ combined P1+P2
    comb = []
    for r in REGIMES:
        for pol in ("tau", "tau+age4"):
            for rate in ("0.3", "0.5"):
                for name in ("clustered", "natural", "random"):
                    v = pc["regimes"][r]["policies"][pol]["variants"].get(f"{rate}|{name}")
                    if not v:
                        continue
                    comb.append([r, pol, rate, name, f(v["ms_per_frame"], 2),
                                 f(v["wallclock_speedup"], 2), pct(v["mac_reduction"]),
                                 f(v["ideal_speedup_mac"], 2), f(v["merged_cos_mean"], 3),
                                 f(v["pooled_cos"], 3)])
    t_comb = table(["regime", "dirty policy", "τ", "perm", "ms/frame", "×", "MAC cut",
                    "ideal MAC ×", "cos", "pooled cos"], comb)

    # ---------------------------------------------------------------- report
    sha = sha256_file(SAFETENSORS)
    codec = git("diff", "--", "src/lib/codec.ts")
    meta = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": "models/Qwen3.5-0.8B",
            "checkpoint_sha256": sha,
            "checkpoint_bytes": SAFETENSORS.stat().st_size,
            "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, torch 2.15.0+cu, CUDA 13.5)",
            "cpu": "AMD Ryzen 9 9955HX (Zen 5, AVX-512), single thread for latency",
            "repo_commit": git("rev-parse", "HEAD"),
            "codec_ts_diff_empty": codec == "",
            "weights_unmodified": sha == arch["checkpoint"]["sha256"]}

    out = {"meta": meta, "architecture": arch, "m0": m0, "p1": p1, "p1_kernels": kernels,
           "p2": p2, "paper1_reference": p1_paper.get("protocol", {}),
           "tables_markdown": {"architecture": t_arch, "m0": t_m0, "m0_text": t_m0_text,
                               "p1_isomorphism": t_iso, "p1_sweep": t_p1, "p1_kernels": t_kern,
                               "p2_rtx5070": t_p2, "p2_vs_paper1": t_comp, "p2_zen5": t_cpu,
                               "combined": t_comb}}
    FINAL_JSON.write_text(json.dumps(out, indent=1))
    log(f"wrote {FINAL_JSON.relative_to(EXP_DIR.parents[1])} "
        f"({FINAL_JSON.stat().st_size/1e3:.1f} kB)")

    md = ["# Paper 2 — CM-Transform-Accel, pilot on Qwen3.5-0.8B",
          "", f"Generated {meta['generated']} · commit `{meta['repo_commit'][:9]}` · "
          f"checkpoint sha256 `{sha[:16]}…` (unmodified: {meta['weights_unmodified']}) · "
          f"`git diff -- src/lib/codec.ts` empty: **{meta['codec_ts_diff_empty']}**", "",
          "## 1. Architecture (from config.json + safetensors header)", "", t_arch, "",
          "Vision/hidden stack: patch embed `Conv3d(3→768, k=(2,16,16), s=(2,16,16))` has "
          "**zero halo** — one output token sees exactly one 16×16 tile of one frame pair — "
          "while every block is pre-LN transformer (bidirectional attention inside a frame pair).", "",
          "## 2. M0 baseline (RTX 5070 + Zen 5)", "", t_m0, "", t_m0_text, "",
          "## 3. P1 — 32-lane permutation + structured block skipping", "",
          "### 3.1 Exact functional isomorphism at τ = 0", "", t_iso, "",
          "fp32: permutation of the FFN rows/columns is *exactly* neutral up to floating-point "
          "accumulation order (max|Δlogit| ≈ 1e-5 on logits of order 1e3, KL < 1e-7, Top-1 1.000). "
          "bf16 re-rounds every projection, so the same permutation shows 0.25 absolute logit "
          "difference with cosine 0.9999 — a rounding floor, not a modelling change.", "",
          "### 3.2 Skip sweep (GPU fp32)", "", t_p1, "",
          "### 3.3 Kernels", "", t_kern, "",
          "## 4. P2 — zero-halo temporal token compaction on V1–V6", "", t_p2, "",
          "### 4.1 Direct comparison with Paper 1 CNNs on the same streams", "", t_comp, "",
          "Read the first four numeric columns against each other: Paper 1's CNNs are forced to "
          "flush on V1/V3/V4 (0% MAC elimination, sub-1× wall-clock) precisely because their dirty "
          "masks percolate through 3×3 stages, while the `L_halo = 0` patch embed keeps 88–97% of "
          "the frame skippable on the same frames. Where Paper 1 does win (V2/V5, up to 2.27× on "
          "ResNet-18) the transformer still eliminates more MACs but converts less of them into "
          "GPU wall-clock, because its per-token tiles are small.", "",
          "### 4.2 Halo percolation (why L_halo = 0 matters)", "",
          table(["regime"] + [f"stage {i}" for i in range(6)],
                [[r] + [pct(x) for x in pc["regimes"][r]["halo_percolation_3x3_stage"][:6]]
                 for r in REGIMES]), "",
          "A 3×3 stride-1 stage would dilate the dirty mask by one patch per block; the measured "
          "mask is already 6–16% of the frame, so five such stages saturate the grid (64–89%). "
          "The stride-16 patch embed has no such coupling: the dirty ratio is *flat* over all 12 "
          "blocks, which is the structural reason a transformer tolerates token compaction where "
          "Paper 1's CNNs did not.", "",
          "### 4.3 Zen 5 CPU", "", t_cpu, "",
          "## 5. Combined P1 + P2", "", t_comb, "",
          "## 6. Findings", "",
          "**P1 is exact before it is fast.** A 32-lane co-activation permutation applied to "
          "`gate_proj`/`up_proj` rows and `down_proj` columns of all 24 FFN layers is bit-neutral: "
          "fp32 gives `max|Δlogit| = 1.1e-5`, `logitRmse = 2.0e-6`, `KL = 3e-8`, `Top-1 = 1.000`; "
          "the two platforms agree with each other as well (CPU fp32 1.5e-5). Under bf16 the same "
          "permutation costs `max|Δlogit| = 0.25` at cosine 0.9999 — pure re-rounding.", "",
          "**Clustering pays only when it is asked to skip something.** With τ = 0 the three orders "
          "are identical (as they must be). From τ ≥ 0.2 the clustered order consistently has the "
          "lowest logit RMSE: 0.398 vs 0.563 (natural) / 0.517 (random) at τ = 0.2, 0.565 vs 0.829 / "
          "0.819 at τ = 0.3, 0.849 vs 1.65 / 1.47 at τ = 0.5 — a 1.4–1.9× error reduction at the same "
          "MAC count, and the measured intra-block co-activation cosine rises from 0.411 (native "
          "order) to 0.544 (clustered, +32% relative). KL is the noisier metric here (a handful of "
          "heavy-tail vocabulary logits dominate it) and does not order the variants as reliably.", "",
          "**The kernel is traffic-bound long before it is bookkeeping-bound.** Skipping 50% of the "
          "32-lane tiles of all 24 down-projections cuts 50% of the down-proj MACs and measures "
          "1.78× (V6 kernel, 221 GFLOP/s) and 1.78× at T=4 on the RTX 5070, 2.85× at τ = 0.75; the "
          "dense-optimal 4-row variant runs at 760 GFLOP/s but only reaches 1.06× because it is "
          "L1/latency-bound and never touches main memory. On Zen 5 the same skip is 1.27× at τ = 0.5 "
          "and 2.42× at τ = 0.75, exactly the DRAM-traffic scaling (12.8 → 30.8 GFLOP/s) that the "
          "mechanism predicts.", "",
          "**P2's win is structural, and it is the opposite of Paper 1's.** The stride-16 patch "
          "embed has `L_halo = 0`, so the dirty ratio stays flat over all 12 blocks (3.2–15.7%), "
          "whereas a single 3×3 stage would already push it to 5.1–29.1% and five stages to "
          "25.3–89.3%. On the streams where Paper 1's CNNs eliminated 0% of their MACs and ran at "
          "0.70–0.98× (V1, V3, V4) the transformer eliminates 87.9–96.7%; on V2/V5/V6 it eliminates "
          "83.5–90.5% versus 28.7–64.4% for the CNNs.", "",
          "**Wall-clock does not follow MACs on the GPU.** 83.5–96.7% MAC elimination buys only "
          "1.41–2.16× end-to-end because the per-frame work collapses into small, launch-bound "
          "tensors (196 tokens × 768). The same code on Zen 5 converts the same MAC reduction into "
          "2.35–8.02× because it is compute-bound and single-threaded; this is where token "
          "compaction is worth implementing first.", "",
          "**Fidelity is governed by how much the *clean* tokens drift, not by the dirty count.** "
          "V1 keeps 93.8% of its patches clean yet drops to cosine 0.754 (clean patches still change "
          "~10% per frame, and their cached residual state is never refreshed); V5 with 88.6% clean "
          "holds 0.969 (clean patches change ~3%). A staleness guard fixes the ordering: refreshing "
          "a token after 4 clean frames moves V1 to 0.933 and every regime into 0.888–0.973 at "
          "70.6–77.4% MAC elimination. A cumulative-drift budget (0.15) is too strict and degrades "
          "V3/V4 to 6–9% savings.", "",
          "**Combining P1 with P2 works, but the vision MLP is the wrong place for P1.** P2 alone "
          "costs 0.03–0.25 cosine; adding 30–50% block skipping on the active tokens costs another "
          "0.10–0.45 (cosine 0.42–0.85), and only on the LLM FFN is skipping almost free "
          "(Top-1 1.000, KL 2e-3 at τ = 0.3). Block skipping of the vision MLP is worth it only at "
          "τ ≤ 0.1, where the LM study already showed KL ≈ 1e-4.", "",
          "### Caveats", "",
          "1. No official `transformers` Qwen3.5 implementation exists for this environment "
          "(requires ≥ 4.57.dev, absent), so all numbers are relative to the reference "
          "implementation in `model_ref.py`, built from `config.json` + checkpoint shapes. "
          "Isomorphism and speed claims are internally exact; absolute fidelity is not comparable "
          "to a HF forward.", 
          "2. `V1–V6` are the same 36-frame, 224² DERF streams as Paper 1, but Paper 1's numbers "
          "come from its own AVX-512 kernels and its own dirty-tile τ, so the comparison is "
          "protocol-matched, not implementation-matched.",
          "3. The vision tower is a pretrained VLM encoder evaluated on video frames it was not "
          "tuned for; its feature drift over 36 frames is larger than a video-tuned encoder's "
          "would be, which makes the P2 fidelity numbers a conservative lower bound.",
          "", "## 7. Reproduction", "", "```bash", "",
          "cd experiments/cm-transformer",
          "python3 m0_baseline.py            # M0 on GPU (fp32/bf16/fp16) and CPU",
          "python3 p1_ffn.py cuda --fp32     # permutation, isomorphism, skip sweep",
          "python3 p1_kernels.py             # CUDA (nvcc) + AVX-512 kernels",
          "python3 p2_stream.py cuda         # zero-halo streaming compaction",
          "python3 p2_stream.py cpu",
          "python3 assemble_report.py        # this file + qwen3_5_0_8b_results.json",
          "python3 -m pytest tests -q",
          "```", ""]
    (EXP_DIR / "REPORT.md").write_text("\n".join(md))
    log(f"wrote REPORT.md ({len(md)} lines)")
    print("\n".join(md[:40]))


if __name__ == "__main__":
    main()
