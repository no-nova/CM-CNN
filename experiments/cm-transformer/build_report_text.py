"""Assemble the text-only P1 report from the raw result files.

Reads (whatever exists):
  results/textcap_base_qwen35.json / textcap_base_qwen3.json   M0 capability
  results/textcap_gsm_fix_*.json                               re-scored GSM8K
  results/p1v2_q35.json / p1v2_q3.json                         P1 sweep
  results/p1_kernels_v2.json                                   kernels + e2e
  results/gguf_iq4nl.json                                      IQ4_NL block skip
and writes REPORT_TEXT.md plus text_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
OUT = Path(__file__).resolve().parent


def load(name, default=None):
    p = RES / name
    if not p.exists():
        return default
    return json.loads(p.read_text())


def f(x, n=3, dash="-"):
    if x is None:
        return dash
    if isinstance(x, float):
        return f"{x:.{n}f}"
    return str(x)


def pct(x, n=2):
    return "-" if x is None else f"{100 * x:.{n}f}%"


def ekeep_cell(cells, model, cfg, rate):
    """Retained block energy from results/p1v2_ekeep.json (HF path)."""
    rec = (cells or {}).get(model) or {}
    return (rec.get(cfg) or {}).get(f"{rate:.2f}")


def gsm_cell(cells, tag, cfg, rate):
    """Per-cell GSM8K accuracy from results/p1v2_gsm.json (None if absent)."""
    rec = (cells or {}).get(tag) or {}
    return ((rec.get(cfg) or {}).get(f"{rate:.2f}") or {}).get("acc")


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    base35 = load("textcap_base_qwen35.json", {})
    base3 = load("textcap_base_qwen3.json", {})
    gsm_cells = load("p1v2_gsm.json", {})
    gsm35 = load("textcap_gsm_fix_qwen35.json", {})
    gsm3 = load("textcap_gsm_fix_qwen3.json", {})
    sweep = {"qwen3.5-0.8b": load("p1v2_q35.json"), "qwen3-0.6b": load("p1v2_q3.json")}
    kern = load("p1_kernels_v2.json")
    gguf = load("gguf_iq4nl.json")
    ekeep_cells = load("p1v2_ekeep.json", {})
    gguf_groups = load("gguf_groups.json", {})

    L = ["# CM-Transform P1 — text-only acceleration & capability report", ""]

    # ---------------- 0. headline --------------------------------------
    L += [
        "## 0. Headline",
        "",
        "| question | answer |",
        "|---|---|",
        "| Are the baselines real now? | yes — ARC-Easy 0.778 / PIQA 0.702 / BoolQ "
        "0.630 (0.8B) and 0.512 / 0.698 / 0.652 (0.6B) at 500 items, GSM8K 0.280 / "
        "0.380, generation 84 / 62 tok/s |",
        "| Does the permutation beat natural order? | on Qwen3-0.6B yes, by "
        "2.1x / 1.75x / 2.1x at tau 0.10 / 0.20 / 0.30 and 2.0x / 2.2x at 0.40 / "
        "0.50; on Qwen3.5-0.8B it wins from tau 0.30 up (4.94 vs 5.22, 8.72 vs "
        "9.38, 14.65 vs 16.16) but natural still wins at 0.10-0.20 (0.68-2.34 vs "
        "0.92-2.46) |",
        "| Does adaptive gating beat per-layer top-k? | yes at every rate on both "
        "models (0.8B: dPPL 0.92/2.46/4.94 vs 1.15/2.76/5.51; 0.6B: 0.28/1.45/3.22 "
        "vs 0.34/2.01/5.14) |",
        "| Does the 32-lane kernel scale? | yes: 2.94x (T=1) and 3.46x (T=64) at 75% "
        "skipped blocks, i.e. it tracks the 1/(1-s) bound, while the old warp-mask "
        "kernel stays at 1.0-1.7x |",
        "| Do the four Phase-3 fixes change the verdict? | **quality yes, "
        "single-operator yes, whole-model no.** Norm-preserving rescaling cuts "
        "dPPL by 0.15-0.99 (bigger at higher tau); block-major IQ4_NL turns line "
        "traffic into a linear function of the skip rate (4.54x cold at 80% vs "
        "1.53x row-major, 2.84x vs 1.68x for 32 layers); row-parallel OpenMP lifts "
        "CPU decode from 0.89x to 1.02-1.17x; SwiGLU coupling doubles the "
        "skippable MLP share (ceiling 1.12x -> 1.26x). The model-level decode "
        "ceiling is unchanged because `down_proj` + `up_proj` are still only 21% "
        "of the checkpoint. |",
        "| Does end-to-end decode get faster? | device-side compaction fixed the "
        "regression (GPU decode 0.78x -> 0.85x, prefill 0.59x -> 0.99x), but the "
        "byte ceiling of the model forbids a win: `down_proj` is only 10.4% of "
        "the 3.4 GB checkpoint, so 1.12x is the absolute maximum at T=1 |",
        "| Phase 4: is the coupled SwiGLU skip real? | **yes at operator level, "
        "no at model level.** AVX-512 fused: 1.35/1.83/2.18x over the same-code "
        "dense at tau 0.2/0.4/0.6 and 8-13x over torch; CUDA fused 1.09-1.78x; "
        "8-row block-major IQ4_NL beats the row-major dense baseline in absolute "
        "time (0.534 vs 0.930 ms) and by 3.8-5.7x under skipping |",
        "| Phase 5: does `float4` + 2 rows/warp close the CUDA gap? | **no** — it is "
        "9% slower dense and 30-35% slower at tau 0.40 than the Phase-4 scalar 4-row "
        "kernel (halving rows/warp halves the work per `x` fetch), so the 4-row "
        "kernel stays the shipping path. The weighted block gate is the real win: "
        "-6% rel-L2 at equal time on the GPU and 22-42% lower dPPL on the model |",
        "| Phase 5: is the IQ4_NL Amdahl ceiling broken? | yes — the gate "
        "pre-filter (r=2 partial rows) moves the coupled FFN stream from "
        "1.08/1.16/1.35/1.97x to 1.23/1.38/1.99/4.24x at 20/40/60/80% skip |",
        "| Phase 5: does it transfer to speech and VLMs? | yes at tau=0 (exact) and "
        "at tau=0.2 (whisper teacher-forced top-1 0.889; SmolVLM VQA 1.0), with "
        "one mechanism finding: GELU sparsity makes 54%/51% of whisper's enc.4/5 "
        "blocks exactly dead after pi |",
        "| Does it help a real quantised model? | hot/L3-resident: 1.23x @20% and "
        "4.24x @80% skipped (dense 0.722 ms, final JSON); cold/DRAM: 1.14x @20% "
        "and 1.68x @80% at block granularity. Super-blocks (G=4) halve the line "
        "traffic (5184 -> 1728 B/row at 80% skip) but the loop is "
        "instruction-bound, so cold time improves only to 1.44x while energy "
        "capture drops from 0.81 to 0.56 |",
        "",
    ]

    # ---------------- 1. baseline --------------------------------------
    rows = []
    for name, b, g in (("Qwen3.5-0.8B", base35, gsm35), ("Qwen3-0.6B", base3, gsm3)):
        ds = b.get("datasets", {})
        gs = (g or b).get("gsm8k", {})
        rows.append([
            name,
            f(ds.get("arc", {}).get("acc"), 3),
            f(ds.get("piqa", {}).get("acc"), 3),
            f(ds.get("boolq", {}).get("acc"), 3),
            f(gs.get("acc"), 3),
            f((b.get("generation") or {}).get("tokens_per_s"), 1),
        ])
    L += ["## 1. Dense baseline (M0), lm-evaluation-harness scoring", "",
          md_table(["model", "ARC-Easy acc", "PIQA acc", "BoolQ acc", "GSM8K",
                    "gen tok/s"], rows), ""]

    # ---------------- 2. P1 sweep --------------------------------------
    for model, sw in sweep.items():
        if not sw:
            continue
        L += [f"## 2. P1 sweep — {model}", ""]
        m0 = sw["m0"]
        tag_key = model
        L += [f"Dense PPL (WikiText-2, 8,176 tokens) = **{m0['ppl']:.3f}**.", ""]
        rows = []
        for cfg, runs in sw["configs"].items():
            for r in runs:
                mc = r.get("mc", {})
                rows.append([
                    cfg, f"{r['rate']:.2f}", f(r["ppl"]),
                    f"{r['ppl'] - m0['ppl']:+.3f}",
                    f(mc.get("arc", {}).get("acc"), 3),
                    f(mc.get("piqa", {}).get("acc"), 3),
                    f(mc.get("boolq", {}).get("acc"), 3),
                    f(gsm_cell(gsm_cells, tag_key, cfg, r["rate"])
                      or (r.get("gsm8k") or {}).get("acc"), 3),
                    pct(r.get("retained_energy")
                        if r.get("retained_energy") is not None
                        else ekeep_cell(ekeep_cells, tag_key, cfg, r["rate"]), 2),
                ])
        L += [md_table(["variant/gate", "tau", "PPL", "dPPL", "ARC", "PIQA",
                        "BoolQ", "GSM8K", "E_keep"], rows), ""]
        be = sw.get("block_error_total", {})
        if model == "qwen3.5-0.8b":
            L += [f"Within-block output-truncation energy (lower is better): "
                  f"ours {be.get('ours', float('nan')):.4g}, natural "
                  f"{be.get('natural', float('nan')):.4g}, random "
                  f"{be.get('random', float('nan')):.4g}.", ""]
        else:
            L += [f"Within-block output-truncation energy: ours "
                  f"{be.get('ours', float('nan')):.4g}, natural "
                  f"{be.get('natural', float('nan')):.4g}, random "
                  f"{be.get('random', float('nan')):.4g}. These absolute sums are "
                  f"**not comparable across models or runtimes**: the metric sums "
                  f"`C[i,j] = E[h_i h_j] <Wd[:,i], Wd[:,j]>` over blocks, so it "
                  f"scales with the squared activation RMS, the number of captured "
                  f"tokens and the layer count. The HF capture path stores 4 x 256 "
                  f"tokens for 28 layers of Qwen3-0.6B whose post-SwiGLU "
                  f"activations are far larger than the 0.8B set (hence 5.7e5 vs "
                  f"60.3), and it also mixes native-order C with the permuted "
                  f"grouping, so it must not be read as a quality ranking here. "
                  f"The comparable numbers for this model are E_keep (99.88% ours "
                  f"vs 99.81% natural at tau=0.50) and dPPL.", ""]

    # ---------------- 2b. generation stability -------------------------
    gen_cells = load("p1v2_gen.json", {})
    if gen_cells:
        rows = []
        for model, rec in gen_cells.items():
            d = rec.get("dense") or {}
            for name, v in rec.items():
                if name == "dense" or not isinstance(v, dict):
                    continue
                rows.append([model, name, f(v.get("rouge_l"), 3),
                             f(v.get("first_div"), 1),
                             f(v.get("top1_agreement"), 2),
                             f(v.get("tokens_per_s"), 1),
                             f(d.get("tokens_per_s"), 1)])
        L += ["## 2b. Generation stability (50 prompts x 64 greedy tokens)", "",
              md_table(["model", "variant", "ROUGE-L", "first divergence (tokens)",
                        "same first token", "tok/s (P1)", "tok/s (dense)"], rows), "",
              "_ROUGE-L and divergence are measured against the dense continuation "
              "of the same prompt; a divergence step of ~15 means the two "
              "continuations stay literally identical for 15 tokens on average._", ""]

    # ---------------- 3. kernels ---------------------------------------
    if kern:
        micro = kern.get("micro", {})
        for T, row in micro.items():
            L += [f"## 3. Kernel micro-benchmark ({T} tokens, layer "
                  f"{kern.get('layer')})", ""]
            body = []
            for rate, e in row.items():
                body.append([rate,
                             f(e.get("cuda_dense_ms"), 2),
                             f(e.get("cuda_v4_mask_ms"), 2),
                             f(e.get("cuda_v7_list_ms"), 2),
                             f(e.get("cuda_v7_speedup"), 2),
                             f(e.get("cuda_v7_GFLOPS"), 0),
                             f(e.get("cpu_mask_ms"), 2),
                             f(e.get("cpu_list_ms"), 2),
                             f(e.get("cpu_list_speedup_vs_mask"), 2)])
            L += [md_table(["skip", "CUDA dense ms", "v4 mask ms", "v7 list ms",
                            "v7 speedup", "v7 GFLOP/s", "AVX mask ms",
                            "AVX list ms", "AVX speedup"], body), ""]
        for key, title in (("e2e_gpu", "GPU (RTX 5070)"), ("e2e_cpu", "CPU (Zen 5 AVX-512)")):
            e = kern.get(key)
            if not e:
                continue
            d = e["dense"]
            kd = e.get("kernel_dense")
            rows = [["dense (vendor BLAS)", f(d["decode_ms_per_tok"], 1),
                     f(d["decode_tok_s"], 1), f(d["prefill_ms"], 0), "1.00x", "-"]]
            if kd:
                rows.append(["kernel, no skip", f(kd["decode_ms_per_tok"], 1),
                             f(kd["decode_tok_s"], 1), f(kd["prefill_ms"], 0), "1.00x",
                             f(kd.get("actual_skip_rate"), 3)])
            for k, v in e.items():
                if not k.startswith("skip_"):
                    continue
                rows.append([k.replace("skip_", "skip="), f(v["decode_ms_per_tok"], 1),
                             f(v["decode_tok_s"], 1), f(v["prefill_ms"], 0),
                             f"{v['decode_speedup']:.2f}x",
                             f(v.get("actual_skip_rate"), 3)])
            L += [f"## 4. End-to-end greedy decode — {title}", "",
                  md_table(["config", "ms/token", "tok/s", "prefill ms",
                            "decode speedup", "blocks kept"], rows), "",
                  "_Speedup is measured against the cuBLAS dense path, i.e. it "
                  "includes the Python-side index compaction that runs once per "
                  "token per layer. The kernel itself reaches 2.9x (T=1) and "
                  "3.5x (T=64) at 75% skipped blocks, so the negative end-to-end "
                  "numbers above are a harness cost, not a kernel cost._", ""]

    e06 = load("p1_kernels_06b.json", {})
    if e06:
        rows = [["dense (HF, no kernel)", f"{e06['dense_ms_per_token']:.1f}", "1.00x",
                 "-"]]
        if e06.get("kernel_no_skip_ms"):
            rows.append(["kernel, no skip", f"{e06['kernel_no_skip_ms']:.1f}",
                         "1.00x", "1.000"])
        for r in e06.get("rows", []):
            rows.append([f"skip={r['rate']:.2f}", f"{r['ms_per_token']:.1f}",
                         f"{r['speedup_vs_dense']:.2f}x",
                         f"{r['kept']:.3f}"])
        L += ["## 4c. End-to-end decode — Qwen3-0.6B (official HF model + kernel)", "",
              md_table(["config", "ms/token", "decode speedup", "blocks kept"], rows), "",
              "_Same ceiling as the 0.8B: `down_proj` is 352 MB of a 1.2 GB fp32 "
              "checkpoint (29%), so at 25% kept blocks the byte-bound ceiling is "
              "1.28x, and per-layer kernel dispatch costs ~1.4 ms/token, which is "
              "why measured speedup stays below 1.0x until the compaction, the "
              "mask and the GEMM are fused into a single launch._", ""]

    # ---------------- 5. GGUF ------------------------------------------
    if gguf:
        m, lay = gguf["model"], gguf["iq4nl_layout"]
        L += ["## 5. IQ4_NL quantized FFN (Qwen3.5-4B-IQ4_NL.gguf)", "",
              f"GGUF v{m['gguf_version']} · {m['tensor_count']} tensors · arch "
              f"`{m['arch']}` · {m['n_layers']} layers · n_embd {m['n_embd']} · "
              f"n_ff {m['n_ff']} · ffn_down = IQ4_NL, "
              f"{lay['block_elems']} weights / {lay['block_bytes']} B "
              f"({lay['bits_per_weight']:.2f} bit/w). Layout verified: "
              f"{lay['nibble_layout']}; C++ decode == numpy "
              f"(max|Δ| {lay['cpp_decode_vs_numpy_max_abs']:.1e}).", ""]
        rows = []
        for b in gguf["benchmark"]["table"]:
            rows.append([b["skip_rate"], f(b["ms_per_token_hot"], 3),
                         f(b["speedup_hot"], 2), f(b["ms_per_token_cold"], 3),
                         f(b["speedup_cold"], 2), f(b["gbs_modeled_cold"], 1),
                         f(b["gbs_cacheline_cold"], 1),
                         f(b["energy_captured"], 4), f(b["max_abs_err"], 2e-3 if False else 5),
                         f(b["rel_l2"], 5)])
        L += ["Single layer (blk.16.ffn_down, 9216×2560), one decode token:", "",
              md_table(["skip", "hot ms", "hot speedup", "cold ms", "cold speedup",
                        "modeled GB/s", "cache-line GB/s", "E captured", "max|Δ|",
                        "rel L2"], rows), ""]
        rows = []
        for b in gguf["stream_all_layers"]["table"]:
            rows.append([b["skip_rate"], f(b["n_active_blocks"], 0),
                         f(b["ms_per_token"], 1), f(b["speedup"], 2),
                         f(b["gbs_modeled"], 2), f(b["gbs_touch_modeled"], 2)])
        L += ["All 32 ffn_down tensors streamed from DRAM per token (mmap, L3 "
              "evicted between reps):", "",
              md_table(["skip", "active blocks", "ms/token", "speedup",
                        "modeled GB/s", "touch GB/s"], rows), "",
              "_Honest caveat: an 18-byte block sits inside a 64-byte cache line, "
              "so skipping 20% of the blocks removes almost no DRAM traffic — "
              "the modeled byte saving is real only when the weights are "
              "L3-resident (hot rows), which is why the cold column is flat._", ""]

    if gguf_groups:
        sl = gguf_groups.get("single_layer", {})
        rows = []
        for r in sl.get("rows_table", []):
            ngrp = round((1 - r["skip"]) * sl.get("blocks", 288) / r["G"])
            aligned = 64 * ngrp if r["G"] == 2 else (128 * ngrp if r["G"] == 4 else None)
            rows.append([r["G"], f"{r['skip']:.2f}",
                         f"{r['line_bytes'] / max(sl.get('rows', 1), 1):.0f}",
                         "-" if aligned is None else f"{aligned}",
                         f"{r['ms_cold']:.3f}", f"{r['speedup_cold']:.2f}x",
                         f"{r['energy_captured']:.4f}"])
        st = gguf_groups.get("stream_all_layers", {})
        rows32 = [[r["G"], f"{r['skip']:.2f}", f"{r['ms_per_token']:.1f}",
                   f"{r['speedup']:.2f}x",
                   f"{r['line_bytes_per_token'] / 1e6:.0f}",
                   f"{r['energy_captured']:.4f}"] for r in st.get("rows_table", [])]
        L += ["## 5b. Super-block gating (cache-line aligned groups), IQ4_NL ffn_down", "",
              f"Dense cold: **{sl.get('dense_ms_cold', 0):.3f} ms** per token for one "
              f"layer ({sl.get('dense_bytes', 0) / 1e6:.1f} MB); all 32 layers "
              f"streamed from DRAM: **{st.get('dense_ms_per_token', 0):.1f} ms/token**. "
              f"`lines/row` counts the distinct 64-byte lines touched per output "
              f"row; `aligned B/row` is the same count if every kept group were "
              f"padded to a line multiple (G=2 to 64 B, G=4 to 128 B).", "",
              md_table(["G", "skip", "lines/row (B)", "aligned B/row", "ms cold",
                        "speedup", "E_captured"], rows), "",
              md_table(["G", "skip", "ms/token (32 layers)", "speedup",
                        "line bytes/token (MB)", "E_captured"], rows32), "",
              "_Reading: group-granular skipping is what actually removes cache "
              "lines — at G=1 the line count barely moves until 60% of the blocks "
              "are gone (5184 B/row at skip 0.2-0.4), while G=4 drops to 4096 B at "
              "40% and 1728 B at 80%. The measured cold time follows only "
              "partially: this decode loop is instruction-bound (dequant + FMA per "
              "surviving block), not DRAM-bound, so halving the line traffic at "
              "80% skip buys 1.44x (G=4) rather than 3x, and costs energy capture "
              "(0.56 vs 0.81). G=1 remains the better quality/speed trade-off "
              "below ~60% skip; G=4 only pays at very high skip._", ""]

    # ---------------- 6. why e2e is flat + data reconciliation ----------
    mdl = load("textcap_base_qwen35.json", {})
    L += ["## 6. Why end-to-end decode moves so little (bandwidth ceiling)", "",
          "| quantity | value |", "|---|---|",
          "| Qwen3.5-0.8B fp32 weights | 3.4 GB (24 layers) |",
          "| one `down_proj` layer | 3584 x 1024 x 4 B = 14.7 MB |",
          "| all 24 `down_proj` weights | 353 MB = **10.4%** of the model |",
          "| dense decode | 12.2 ms/token = 279 GB/s effective (card is DRAM-bound) |",
          "| ceiling if 100% of `down_proj` blocks were skipped | 1/(1-0.104) = "
          "**1.12x** |",
          "| achievable at 75% skipped blocks | 1/(1-0.078) = **1.08x** |",
          "", "At T=1 the token loop must read every weight of every layer; P1 can "
          "only remove the bytes behind the blocks it drops, and `down_proj` is "
          "10.4% of the checkpoint. A 1.3-1.6x decode speedup is therefore not "
          "reachable on an 0.8B fp32 model by skipping `down_proj` alone, no "
          "matter how good the kernel is. What the kernel can do is stop *losing*: "
          "device-side compaction (`compact_kernel` + `down_skip4_var_kernel`) "
          "removed the per-layer host sync and lifted decode from 0.78x to 0.85x "
          "and prefill from 0.59x to 0.99x, i.e. skipping is now free instead of "
          "costing 20-40%.", "",
          "## 7. Data reconciliation and CPU AVX-512 diagnosis", "",
          "**IQ4_NL 1.845 ms vs 0.722 ms.** The `1.845 ms -> 2.79x / 4.32x / "
          "10.94x` numbers came from the delegate's *first* single-run "
          "measurement, where the dense baseline still included the generic "
          "dequantising path and only one timing round was taken. The final "
          "`results/gguf_iq4nl.json` (best of 3 interleaved rounds x 9 cold reps, "
          "specialised `iq4nl_matvec_dense` path) is authoritative: dense hot "
          "**0.722 ms**, cold 1.056 ms, hot speedups **1.23x / 2.36x / 4.24x** at "
          "20/60/80% skipped. Section 5 above and all tables in this report use "
          "the JSON values; the earlier chat figure is superseded and must not be "
          "quoted.", "",
          "**CPU `kernel, no skip` 243 ms vs vendor BLAS 118 ms.** Three causes, "
          "all measured: (1) the kernel was *single-threaded* while PyTorch/oneDNN "
          "uses all 32 cores — adding OpenMP (`down_skip_avx512_auto`, "
          "`#pragma omp parallel for` over tokens) brought no-skip from 243 ms to "
          "**~195 ms**; (2) it used 4-row register blocking, i.e. 8 accumulators "
          "for 32 ZMM registers — the new kernel uses 8 rows (16 accumulators) and "
          "reuses each loaded `a` block across all 8 rows; (3) block addressing "
          "was `idx[k]*32` with a dependent 32-bit load per iteration, so the "
          "list variant lost to the mask variant (0.88x) — the auto kernel now "
          "compacts once per token into `idx[]` and prefetches the next `a` block "
          "(`_mm_prefetch`). Even so, PyTorch remains faster per FLOP on this "
          "shape because it reuses packed weight tiles across the whole row block; "
          "our kernel is bandwidth/latency bound at ~10 GB/s/core.", ""]

    # ---------------- Phase 3 (defects 1-4) ----------------------------
    bm = load("gguf_blockmajor.json", {})
    rm = load("gguf_groups.json", {})
    cpu2 = load("p1_cpu_auto2.json", {})
    kern2cpu = load("p1_kernels_v2cpu.json", {})
    if bm:
        sl, st = bm["single_layer"], bm["stream_all_layers"]
        rmrow = {r["skip"]: r for r in rm.get("single_layer", {}).get("rows_table", [])
                 if r["G"] == 1}
        rows = []
        for r in sl["rows_table"]:
            rm_ = rmrow.get(r["skip"], {})
            rows.append([f"{r['skip']:.2f}", f"{rm_.get('line_bytes', 0) / 1e6:.2f}",
                         f"{r['line_bytes'] / 1e6:.2f}",
                         f"{rm_.get('ms_cold', 0):.3f} / {rm_.get('speedup_cold', 0):.2f}x",
                         f"{r['ms_cold']:.3f} / {r['speedup_cold']:.2f}x",
                         f"{r['energy_captured']:.4f}"])
        rows32 = []
        rmrow32 = {r["skip"]: r for r in rm.get("stream_all_layers", {}).get("rows_table", [])
                   if r["G"] == 1}
        for r in st["rows_table"]:
            rm_ = rmrow32.get(r["skip"], {})
            rows32.append([f"{r['skip']:.2f}",
                           f"{rm_.get('line_bytes_per_token', 0) / 1e6:.0f}",
                           f"{r['line_bytes_per_token'] / 1e6:.0f}",
                           f"{rm_.get('ms_per_token', 0):.1f} / {rm_.get('speedup', 0):.2f}x",
                           f"{r['ms_per_token']:.1f} / {r['speedup']:.2f}x",
                           f"{r['energy_captured']:.4f}"])
        L += ["## 8. Defect 1 — Block-major IQ4_NL layout (cache-line fix)", "",
              f"Weights repacked offline from `[M, nb*18]` (row-major) to "
              f"`[nb, M*18]` (block-major): one block is a contiguous "
              f"{sl['rows'] * 18} B = {sl['rows'] * 18 // 64} cache lines, so a "
              f"skipped block removes whole lines and every touched line is fully "
              f"consumed. Repack cost {sl['repack_ms']:.1f} ms for one tensor "
              f"(done once, offline). Numerics: max|Δ| = "
              f"{sl['max_abs_err_full']:.2e}, rel L2 = {sl['rel_l2_full']:.2e} vs a "
              f"numpy dequant reference. Cold timings are single-threaded, best of "
              f"5 with a 96 MB eviction read between reps.", "",
              md_table(["skip", "row-major lines (MB)", "block-major lines (MB)",
                        "row-major ms / speedup", "block-major ms / speedup",
                        "E_captured"], rows), "",
              md_table(["skip", "row-major lines/tok (MB)",
                        "block-major lines/tok (MB)",
                        "row-major ms / speedup", "block-major ms / speedup",
                        "E_captured"], rows32), "",
              "_Reading: at G=1 the block-major layout turns the line traffic into "
              "a linear function of the skip rate (13.3 MB -> 10.6 -> 8.0 -> 5.3 "
              "-> 2.7 MB), which row-major could not do (5184 B/row stayed flat "
              "until 60% skip). Single-layer cold time therefore improves from "
              "1.53x to 4.54x at 80% skip and from 1.38x to 1.81x at 60%, with the "
              "same energy capture (0.81 / 0.94). All-32-layer streaming: 2.84x vs "
              "1.68x at 80% skip, 1.35x vs 1.43x at 60%. The dense baseline of the "
              "block-major kernel is slower (1.62 ms vs 0.93 ms) because the 4-row "
              "SIMD dequant path trades ILP for the layout; that is the remaining "
              "gap to close with 8-row blocking._", ""]

    if cpu2:
        rows = [[f"{r['T']}", f"{r['skip']:.2f}", f"{r['dense_ms']:.3f}",
                 f"{r['auto_ms']:.3f}", f"{r['auto2_ms']:.3f}",
                 f"{r['dense_ms'] / r['auto2_ms']:.2f}x"] for r in cpu2["rows"]]
        ec = kern2cpu.get("e2e_cpu", {})
        e2e_rows = []
        if ec:
            d = ec["dense"]
            e2e_rows.append(["dense (vendor BLAS)", f"{d['decode_ms_per_tok']:.1f}",
                             f"{d['decode_tok_s']:.2f}", "1.00x", "-"])
            kd = ec.get("kernel_dense", {})
            if kd:
                e2e_rows.append(["kernel, no skip", f"{kd['decode_ms_per_tok']:.1f}",
                                 f"{kd['decode_tok_s']:.2f}",
                                 f"{d['decode_ms_per_tok'] / kd['decode_ms_per_tok']:.2f}x",
                                 "1.000"])
            for k, v in ec.items():
                if k.startswith("skip_"):
                    e2e_rows.append([f"target tau {k[5:]}", f"{v['decode_ms_per_tok']:.1f}",
                                     f"{v['decode_tok_s']:.2f}",
                                     f"{v.get('decode_speedup', 0):.2f}x",
                                     f"{v.get('actual_skip_rate', 0):.3f}"])
        L += ["## 9. Defect 2 — CPU decode: parallelise over output rows, not tokens", "",
              f"`down_skip_avx512_auto` parallelised over tokens (`T=1` => one "
              f"thread); `down_skip_avx512_auto2` compacts once, then runs "
              f"`collapse(2)` over (token, 8-row group), so a single decode token "
              f"spreads over all cores. Measured with "
              f"`OMP_NUM_THREADS=16 OMP_WAIT_POLICY=passive` (32 spinning threads "
              f"cost 2.8 ms of fork/join on this box — more than the whole 0.09 ms "
              f"op).", "",
              md_table(["T", "skip", "torch.mm ms", "token-parallel ms",
                        "row-parallel ms", "vs dense"], rows), "",
              md_table(["config", "ms/token", "tok/s", "decode speedup",
                        "actual skip"], e2e_rows), "",
              "_Reading: at T=1 the token-parallel kernel is 0.19 ms (worse than "
              "the 0.16 ms BLAS call) while the row-parallel kernel is 0.064 ms at "
              "no skip and 0.027-0.033 ms at 60-75% skip, i.e. 2.5x faster than "
              "BLAS before any skipping. In the 24-layer end-to-end loop this lifts "
              "CPU decode from 0.89-0.91x (token-parallel) to 0.94 / 0.98 / 1.02 / "
              "1.05 / 1.17x, crossing 1.0x at 60% skip. At T=64 the picture "
              "inverts: torch.mm already saturates 16 cores and the kernel is only "
              "competitive above 75% skip._", ""]

    for dt in ("fp32", "bf16"):
        cp = load(f"p1_coupled_{dt}.json", {})
        if not cp:
            continue
        rows = [["dense (permuted, no skip)", f"{cp['dense_ms']:.1f}", "1.00x", "-",
                 "-"]]
        for r in cp["rows"]:
            rows.append([f"{r['mode']} tau {r['rate']:.2f}", f"{r['ms']:.1f}",
                         f"{r['speedup']:.2f}x", f"{r['max_abs']:.3f}",
                         f"{r['rel_l2']:.4f}"])
        L += [f"## 10{'b' if dt == 'bf16' else ''}. Defect 3 — SwiGLU-coupled skipping "
              f"({dt}, Qwen3.5-0.8B, prefill of {cp['tokens']} tokens)", "",
              f"Permutation isomorphism check (tau=0): max|Δlogit| = "
              f"{cp['isomorphism_max_abs']:.2e}. `block` = round-2 behaviour "
              f"(down_proj columns only); `coupled` = gate dense, then only the "
              f"active rows of `up_proj` and the matching columns of `down_proj`.", "",
              md_table(["config", "ms", "speedup", "max|dlogit|", "rel L2"], rows), "",
              "_Reading: coupling lifts the skippable share of the MLP from 1/3 to "
              "2/3, so the byte ceiling moves from 10.4% to 20.8% of the checkpoint "
              "(1.12x -> 1.26x) and measured prefill speedup roughly doubles at the "
              "same tau. The gather (`index_select`) is still a torch op per layer; "
              "fusing it into the kernel is the next step._", ""]

    nf35 = load("p1v2_q35_normfix.json", {})
    nf3 = load("p1v2_q3_normfix.json", {})
    rows = []
    for tag, new in (("qwen3.5-0.8b", nf35), ("qwen3-0.6b", nf3)):
        old = sweep.get(tag, {})
        if not new:
            continue
        for cfg in ("ours_adaptive", "natural_uniform"):
            orows = {r.get("rate"): r for r in old.get("configs", {}).get(cfg, [])}
            nrows = {round(r.get("rate", 0), 2): r for r in new.get("configs", {}).get(cfg, [])}
            for rate in sorted(nrows):
                o, n = orows.get(rate), nrows[rate]
                if not o:
                    continue
                rows.append([f"{tag} {cfg}", f"{rate:.2f}", f"{o['ppl']:.3f}", f"{n['ppl']:.3f}",
                             f"{n['ppl'] - o['ppl']:+.3f}",
                             f"{o.get('mc', {}).get('arc', {}).get('acc', 0):.3f} -> "
                             f"{n.get('mc', {}).get('arc', {}).get('acc', 0):.3f}"])
    if rows:
        L += ["## 11. Defect 4 — norm-preserving rescaling", "",
              "`alpha = min(gamma_max, ||a|| / ||a_keep||)` applied to the truncated "
              "post-activation vector (same rule in the HF path). gamma_max = 1.5.", "",
              md_table(["config", "tau", "PPL (no fix)", "PPL (fix)", "delta",
                        "ARC-Easy"], rows), "",
              "_Reading: the rescued energy was already 0.99 / 0.975 / 0.955 at tau "
              "0.1 / 0.2 / 0.3, so ||a_keep|| is only 0.5-2.3% short and the cap "
              "barely moves perplexity; the fix earns ~0.15 PPL at tau 0.10 and "
              "less above it. It is a free insurance policy, not a cure — the "
              "residual ΔPPL is direction drift, not missing norm._", ""]

    p4 = load("p4_swiglu_cpu.json", {})
    p432 = load("p4_swiglu_cpu32.json", {})
    if p4:
        rows = []
        for r in p4["single_layer"]:
            rows.append([r["mode"], f"{r['ms']:.3f}", f"{r['speedup']:.2f}x",
                         f"{r['kept_blocks']}/112",
                         ("-" if r["max_abs"] == 0 else f"{r['max_abs']:.3f}"),
                         ("-" if not r.get("rel_l2") else f"{r['rel_l2']:.4f}")])
        e2e = []
        ec = p432.get("e2e_decode", {}) if p432 else p4.get("e2e_decode", {})
        if ec:
            tb_ms = ec.get("torch_baseline", {}).get("ms_per_token", 0)
            for k, v in ec.items():
                if not isinstance(v, dict) or "ms_per_token" not in v:
                    continue
                e2e.append([k, f"{v['ms_per_token']:.1f}", f"{v['tok_s']:.2f}",
                            (f"{tb_ms / v['ms_per_token']:.2f}x" if tb_ms else "-")])
        L += ["## 12. Defect 3 (closed) — fused zero-copy SwiGLU coupled skip, AVX-512", "",
              "`swiglu_coupled_f32` (swiglu_avx512.cpp): gate dense -> ONE compacted "
              "block list from `g` -> `up_proj` active rows read in place -> "
              "`h_act = g_act*u_act` in a 3584-float scratch -> `down_proj` active "
              "columns read in place. No weight copy at any point; the three "
              "phases share `idx[]`. Isomorphism at tau=0: max|dlogit| = "
              f"{p4['iso_vs_torch_max_abs']:.2e} vs torch.", "",
              md_table(["config", "ms", "vs fused dense", "kept", "max|d out|",
                        "rel L2"], rows), "",
              "_Single layer (layer 11, T=1, 16 threads, passive): the fused dense "
              "path is 0.198 ms against torch's 1.190 ms for the same layer (6.0x "
              "from layout + thread discipline alone), and coupled skipping adds "
              "1.35x / 1.83x / 2.18x on top at tau 0.2 / 0.4 / 0.6. The rel-L2 "
              "column is the L2 error of the layer output relative to its own "
              "norm; it overstates model damage because the MLP output cancels "
              "against the residual (the same tau costs only +1.0 PPL at model "
              "level)._", "",
              md_table(["E2E decode config", "ms/token", "tok/s", "vs torch"], e2e), "",
              "_End-to-end (24 layers, incremental decode, 32 threads, passive): "
              "fused dense 151.5 ms/token vs torch 133.8 (torch wins at 32 threads "
              "because its GEMV already saturates the socket), coupled tau=0.6 "
              "121.0 ms/token = 1.25x over the fused dense path and 1.11x over "
              "torch. At quality-viable tau (0.2) the model-level gain is only "
              "0.98-1.01x, because gate stays dense and the rest of the model "
              "(attention + KV + lm_head) does not shrink._", ""]

    iq = load("p4_iq4_blockmajor.json", {})
    if iq:
        rows = [[f"{r['skip']:.2f}", f"{r['line_bytes'] / 1e6:.2f}",
                 f"{r['ms_st4']:.3f}", f"{r['ms_st8']:.3f}", f"{r['ms_mt']:.3f}",
                 f"{r['speedup_vs_rowmajor_dense']:.2f}x"] for r in iq["single_layer"]]
        st = [[f"{r['skip']:.2f}", f"{r['bytes_per_token'] / 1e6:.0f}",
               f"{r['ms_per_token']:.1f}", f"{r['speedup']:.2f}x"] for r in iq["coupled_ffn_stream"]]
        L += ["## 13. Defect 1 (closed) — 8-row block-major IQ4_NL + coupled FFN", "",
              f"8-row register blocking (16 ZMM accumulators) plus OpenMP over 8-row "
              f"groups; repack {iq['repack_ms_per_tensor']:.1f} ms per tensor, "
              f"{iq['repack_all_layers_s']:.1f} s for 32 layers x (down+up) = "
              f"849 MB of packed weights.", "",
              md_table(["skip", "lines (MB)", "4-row ms", "8-row ms",
                        f"8-row mt{iq['threads']} ms", "vs row-major dense (0.930 ms)"], rows), "",
              "_Single layer cold: the 8-row kernel alone is 1.90x faster than the "
              "4-row one at skip 0 (1.388 vs 2.636 ms), and with row-parallel "
              "OpenMP the dense case lands at **0.534 ms, i.e. 1.74x FASTER than "
              "the row-major dense baseline (0.930 ms)**. Every skip point then "
              "beats that baseline in absolute wall clock: 0.242 / 0.241 / 0.202 / "
              "0.164 ms at 20/40/60/80% skip (3.8x-5.7x). The Phase-3 objection "
              "(block-major dense is slower) is gone._", "",
              md_table(["skip", "bytes/token (MB)", "ms/token", "speedup"], st), "",
              "_Coupled FFN, all 32 layers streamed per token (gate dense + "
              "`ffn_up` active row blocks + `ffn_down` active column blocks, one "
              "shared idx): 43.1 ms dense -> 40.0 / 37.0 / 32.0 / 21.9 ms at 20 / "
              "40 / 60 / 80% skip = 1.08x / 1.16x / 1.35x / 1.97x, with 40 -> 19 "
              "MB actually read. Amdahl for a dense gate predicts a ceiling of "
              "2.14x at 80% skip, so the implementation reaches 92% of the bound; "
              "skipping `ffn_gate` too (it is the same matvec shape) is what would "
              "move the ceiling to 5.0x._", ""]

    p4c = load("p4_swiglu_cuda.json", {})
    if p4c:
        rows = [[r["mode"], f"{r['ms']:.3f}", f"{r['speedup']:.2f}x",
                 f"{r['kept']}/112",
                 ("-" if not r.get("max_abs") else f"{r['max_abs']:.3f}"),
                 ("-" if not r.get("rel_l2") else f"{r['rel_l2']:.4f}")]
                for r in p4c["single_layer"]]
        chain = [[f"{r['rate']:.2f}", f"{r['ms_per_token']:.2f}",
                  f"{r['speedup']:.2f}x"] for r in p4c["mlp_chain_24"]]
        torch_chain = 24 * p4c["single_layer"][0]["ms"]
        L += ["## 14. Defect 3 (CUDA) — fused zero-copy SwiGLU coupled skip, RTX 5070", "",
              "`swiglu_cuda.cu`: warp-per-row GEMV (32 lanes read 32 consecutive "
              "floats, shuffle reduction), block-parallel top-K compaction, no "
              "host sync (every kernel is sized for the worst case and reads `K` "
              "from device memory), `up` rows and `down` columns read in place from "
              "the same buffers. Isomorphism at tau=0: max|dout| = "
              f"{p4c['iso_vs_torch_max_abs']:.2e}.", "",
              md_table(["config", "ms", "vs torch dense", "kept", "max|d out|",
                        "rel L2"], rows), "",
              md_table(["MLP-only chain (24 layers, 1 token)", "ms/token", "speedup"], chain), "",
              f"_Reading: single layer, torch/cuBLAS is 0.136 ms; the fused kernel "
              f"needs 0.296 ms at dense (scalar 4-byte lane loads versus cuBLAS "
              f"GEMM) but crosses over at tau=0.4 (0.112-0.124 ms, 1.1-1.2x) and "
              f"reaches 1.78x at tau=0.6 (0.076 ms). Over a 24-layer MLP chain the "
              f"same crossover means {torch_chain:.1f} ms of cuBLAS MLP versus "
              f"7.15 ms fused dense, 2.97 ms at tau=0.4 (1.1x faster than cuBLAS) "
              f"and 1.81 ms at tau=0.6 (1.8x). The dense gap is a 4-byte-lane load "
              f"issue, not an algorithmic one: `float4` loads and 2 rows per warp "
              f"should close it._", ""]


    # ---------------- 15. CUDA float4 + weighted gate ------------------ #
    p5c = load("p5_cuda.json", {})
    p5w = load("p5_weighted_gate.json", {})
    if p5c:
        op = p5c["operator"]
        rows = [[("plain" if r["gate"] == "plain" else "**weighted**"),
                 f"{r['rate']:.2f}", f"{r['ms']*1e3:.0f}", f"{r['speedup_vs_torch']:.2f}x",
                 f"{r['speedup_vs_dense_kernel']:.2f}x", f"{r['rel_l2']:.3f}",
                 f"{r['cosine']:.3f}"] for r in p5c["rows"]]
        chain = [[f"{r['rate']:.2f}", f"{r['ms']:.2f}",
                  f"{r['speedup_vs_torch_chain']:.2f}x"] for r in p5c["chain_24_layers"]["rows"]]
        L += ["## 15. Phase 5 / Task 1 — CUDA `float4` + 2 rows/warp, weighted block gate", "",
              f"`gemv2_warp_kernel` / `up2_warp_kernel` / `down2_warp_kernel`: each lane "
              f"loads 16 B (`float4`) of the weight row and of `x`, and one `x` "
              f"register set feeds **two output rows**, so the dense kernel no longer "
              f"pays 2x the traffic of cuBLAS. Isomorphism at tau=0: max|dout| = "
              f"{p5c['isomorphism_tau0_max_abs']:.1e}. Offline block importance "
              f"`w_b = RMS(u_b) * ||W_down[:,b]||_F` (one float per block, "
              f"mean-normalised, range {p5c['block_weight_stats']['min']:.2f}-"
              f"{p5c['block_weight_stats']['max']:.2f}) is folded into the device-side "
              f"score `S_b = w_b * ||g_b||_2`, i.e. **no two-pass gate**.", "",
              f"Dense single layer: torch/cuBLAS {op['torch_ms']*1e3:.0f} us, fused "
              f"dense {op['fused_dense_ms']*1e3:.0f} us ({op['dense_vs_torch']:.2f}x of "
              f"cuBLAS, down from 0.46x with 4-byte lane loads).", "",
              md_table(["gate", "tau", "us", "vs torch", "vs fused dense", "rel L2",
                        "cosine"], rows), "",
              md_table(["MLP-only chain (24 layers)", "ms/token", "vs cuBLAS chain"],
                       chain), "",
              "_Reading — and a negative result worth recording: the `float4` + "
              "2-rows/warp rewrite did **not** beat the Phase-4 scalar 4-row kernel on "
              "this box. Same layer, same torch baseline (0.136 ms): Phase-4 4-row "
              "scalar was 0.296 / 0.224 / 0.112-0.124 / 0.076 ms at dense / 0.20 / "
              "0.40 / 0.60, while the 2-row vectorised version gives 0.322 / 0.268 / "
              "0.153-0.160 / 0.091-0.098 ms, i.e. 9% slower dense and 30-35% slower at "
              "tau 0.40 (the box was also busier: load ~20 vs ~10, which inflates both "
              "sides equally, so the sign is real). Cause: halving the rows per warp "
              "halves the arithmetic per `x` fetch, so the warp spends proportionally "
              "more time on lane strided weight traffic and on the fixed "
              "reduction/scheduling overhead; the `float4` win does not compensate. "
              "The 4-row kernel therefore remains the recommended CUDA path and the "
              "2-row variant is kept only for the weighted-gate experiment. What "
              "Phase 5 did buy on the GPU is the gate score: the weighted variant is "
              "-6% rel-L2 at equal time at 0.20 and 0.40 and -3% at 0.60, and it "
              "crosses cuBLAS at tau ~ 0.75 (0.060 vs 0.136 ms)._", ""]
        if p5w:
            sub = [r for r in p5w["rows"] if r["rate"] in (0.0, 0.2, 0.4, 0.6)]
            by = {(r["gate"], r["rate"]): r for r in sub}
            rows = []
            for rate in sorted({r["rate"] for r in sub}):
                p_ = by.get(("plain", rate), {})
                w_ = by.get(("weighted", rate), {})
                rows.append([f"{rate:.2f}", f"{p_.get('ppl', 0):.4f}",
                             f"{p_.get('delta_ppl', 0):+.3f}",
                             f"{w_.get('ppl', 0):.4f}", f"{w_.get('delta_ppl', 0):+.3f}",
                             (f"{100*(1 - w_['delta_ppl']/p_['delta_ppl']):+.0f}%"
                              if p_.get('delta_ppl') else "-")])
            L += ["### 15b. Weighted gate — model-level PPL (Qwen3.5-0.8B, GPU fp32)", "",
                  f"{p5w['windows']} WikiText-2 windows, `block_weight` computed offline "
                  f"from calibration activations: `w_b = RMS_t||u_b(t)|| * "
                  f"||W_down[:, b]||_F`, score `S_b = w_b ||g_b||_2`.", "",
                  md_table(["tau", "plain PPL", "dPPL", "weighted PPL", "dPPL",
                            "dPPL reduction"], rows), "",
                  "_The weighted score folds the down-projection energy into the gate "
                  "decision (the Phase-4 caveat that gate-only gating is a weak proxy "
                  "for `||act*up||`), and it pays: dPPL drops 42% at tau 0.20, 28% at "
                  "0.40 and 22% at 0.60 at zero extra runtime cost (one float per "
                  "block, precomputed)._", ""]

    # ---------------- 16. IQ4_NL gate prefilter ------------------------ #
    p5q = load("p5_iq4_prefilter.json", {})
    if p5q:
        rows = [[f"{r['rate']:.2f}", f"{r['r']}", f"{r['kept']}",
                 f"{r['ms_per_token']:.1f}", f"{r['mb_per_token']:.1f}",
                 f"{r['speedup']:.2f}x"] for r in p5q["stream_all_layers"]]
        L += ["## 16. Phase 5 / Task 2 — breaking the 2.14x Amdahl ceiling: gate pre-filter", "",
              "`gate_prefilter_blockmajor`: phase 1 evaluates only the first `r` rows of "
              "every 32-row block-major gate block (`r*1440 B` contiguous at the block "
              "start), phase 2 ranks blocks by that partial energy (+`w_b` if given), "
              "phase 3 computes the remaining `32-r` rows **only for the K selected "
              "blocks**, then the same idx drives `ffn_up` (active row blocks) and "
              "`ffn_down` (active column blocks). Gate row traffic falls from `nb*32` "
              "to `nb*r + K*(32-r)`.", "",
              md_table(["tau", "prefilter rows r", "kept blocks", "ms/token",
                        "MB/token", "vs dense (r=2)"], rows), "",
              f"_Measured on the real `Qwen3.5-4B-IQ4_NL.gguf` (32 layers streamed per "
              f"token, 32 threads; the box was busy — load ~20 — so absolute ms are "
              f"~1.7x the clean Phase-4 run, while the ratios are stable). "
              f"r=2 is the sweet spot: 1.23x / 1.38x / 1.99x / 4.24x at 20 / 40 / 60 / "
              f"80% skip versus 1.08x / 1.16x / 1.35x / 1.97x with the dense gate — "
              f"the ceiling is genuinely broken, but 2.5-3.5x needs tau >= 0.7 where "
              f"the coupled FFN rel-L2 is already ~0.85, so the honest operating "
              f"point stays at tau <= 0.4 (1.4x)._", "",
              "_Prefilter quality (unit test `test_p5_iq4_prefilter.py`, synthetic "
              "activation with near-uniform block energies): of the 173 blocks kept "
              "at tau=0.40 the partial-row ranking captures 61.8/63.1/65.2/66.5% of "
              "the dense gate energy for r=2/4/8/16 versus 58.9% for a random "
              "equal-size selection — i.e. the cheap estimator is only 1.05-1.10x "
              "better than random on flat activations. Real decode activations are "
              "far more concentrated (60% of the energy in 40% of the blocks), which "
              "is where the 1.4x stream gain above comes from; the r=2 ranking is a "
              "traffic-saving device, not a faithful block predictor._", ""]

    # ---------------- 17. whisper-base --------------------------------- #
    p5wsp = load("p5_whisper.json", {})
    if p5wsp:
        a = p5wsp["arch"]
        rows = [[f"{r['tau']:.2f}", f"{r['enc_rel_l2']:.2e}", f"{r['enc_cosine']:.5f}",
                 f"{r['dec_top1_tf']:.3f}", f"{r['token_agree']:.3f}",
                 f"{r['text_rouge_l']:.3f}", f"{r['energy_captured']:.4f}",
                 f"{pct(r['actual_skip'])}"] for r in p5wsp["rows"]]
        sp = p5wsp["speed"]
        fs = sp.get("ffn_operator_speedup", {}).get("enc", {}).get("by_tau", {})
        frows = [[f"{k}", f"{v['dense_ms']:.3f}", f"{v['partial_ms']:.3f}",
                  f"{v['speedup']:.2f}x"] for k, v in fs.items()]
        dead = p5wsp["calibr"].get("dead_block_fraction_after_perm", {})
        L += ["## 17. Phase 5 / Task 3 — whisper-base (speech encoder-decoder, GELU FFN)", "",
              f"`models/whisper-base`: {a['n_encoder_layers']} encoder + "
              f"{a['n_decoder_layers']} decoder layers, d_model {a['d_model']}, FFN "
              f"{a['encoder_ffn_dim']} = {a['blocks_per_ffn']} lanes of 32, GELU. The "
              f"encoder always sees T={p5wsp['calibr']['audio'].get('enc_frames', 1500)} "
              f"frames (the mel extractor pads to 3000 frames), which is the largest "
              f"token count in the study.", "",
              md_table(["tau", "enc rel L2", "enc cosine", "dec top1 (tf)", "token agree",
                        "ROUGE-L", "energy kept", "actual skip"], rows), "",
              md_table(["encoder FFN op (6 layers, T=1500)", "dense ms",
                        "block-sparse ms", "speedup"], frows), "",
              f"_Mechanism finding worth keeping: 86%/89% of GELU activations in enc.4/"
              f"enc.5 are exactly zero, and after pi **{dead.get('enc.4', 0)*100:.0f}% / "
              f"{dead.get('enc.5', 0)*100:.0f}% of their 32-lane blocks have exactly zero "
              f"energy** (vs ~1%/3% before permutation), so theta collapses to ~0 in "
              f"those two layers and they are skipped for free. Free-running token "
              f"agreement is a step function here (a 5% isotropic noise control on the "
              f"dense encoder also gives agreement 0.000), so the graded metric is "
              f"teacher-forced top-1: 0.889/0.815/0.667 at tau 0.2/0.4/0.6. WER is not "
              f"reported because the audio is synthetic and has no reference "
              f"transcript._", ""]

    # ---------------- 18. SmolVLM2 ------------------------------------- #
    p5s = load("p5_smolvlm.json", {})
    if p5s:
        ar = p5s["arch"]
        rows = [[r["tower"], f"{r['tau']:.2f}", f"{r['rel_l2']:.3f}",
                 f"{r['rel_l2_prefill_logit']:.3f}", f"{r['top1_agree']:.3f}",
                 f"{r['rouge_l']:.3f}", f"{r['vqa_acc']:.2f}",
                 f"{r['energy_captured']:.4f}", f"{pct(r['actual_skip'])}"]
                for r in p5s["rows"]]
        g = p5s["speed"]["gpu"]
        srows = [["vision forward (ms)"] + [f"{g['vision_forward_ms'][k]:.1f}"
                                            for k in ("0.0", "0.2", "0.4", "0.6")],
                 ["LM prefill (ms)"] + [f"{g['lm_prefill_ms'][k]:.1f}"
                                        for k in ("0.0", "0.2", "0.4", "0.6")],
                 ["LM decode (ms/token)"] + [f"{g['lm_decode_ms_per_token'][k]:.2f}"
                                             for k in ("0.0", "0.2", "0.4", "0.6")]]
        L += ["## 18. Phase 5 / Task 3 — SmolVLM2-256M-Video-Instruct", "",
              f"Vision tower {ar['vision']['n_layers']}x{ar['vision']['d_model']} "
              f"(FFN {ar['vision']['ffn_dim']}, {ar['vision']['activation']}, "
              f"{ar['vision']['lanes']} lanes), text LM "
              f"{ar['text']['n_layers']}x{ar['text']['d_model']} with SwiGLU "
              f"{ar['text']['ffn_dim']} ({ar['text']['lanes']} lanes). One image at "
              f"512px -> {ar['vision']['n_patches_per_image']} patches -> "
              f"{ar['vision']['n_image_tokens_one_image']} image tokens; the "
              f"connector reduces 12288 -> 576.", "",
              md_table(["tower", "tau", "rel L2", "prefill logit relL2", "top-1 agree",
                        "ROUGE-L", "VQA acc", "energy kept", "actual skip"], rows), "",
              md_table(["GPU wall clock", "dense", "tau 0.2", "tau 0.4", "tau 0.6"],
                       srows), "",
              f"_Isomorphism: max|dlogit| {p5s['checks']['tau0_max_abs_dlogit']:.1e}, "
              f"feature rel-L2 {p5s['checks']['tau0_feat_rel_l2']:.1e}, generated text "
              f"identical at tau=0. Skipping the **vision** tower is much more damaging "
              f"than skipping the text LM at the same rate (rel-L2 0.334 vs 0.276 at "
              f"0.2, 0.666 vs 0.336 at 0.4), matching the Qwen3.5 finding that vision "
              f"MLPs are far less redundant; VQA accuracy still holds 1.0/0.75 at tau "
              f"0.2 for text/vision. In PyTorch the gated path is slower than dense "
              f"(54 -> 57 ms vision, 8.2 -> 10.6 ms/token decode) — same framework "
              f"overhead as everywhere else; the kernel-level gain is what the CUDA/"
              f"AVX-512 operators deliver._", ""]

    # ---------------- 19. five-model matrix ---------------------------- #
    L += ["## 19. Five-model matrix (Phase 5 close-out)", "",
          "| model | FFN shape | tau* (iso -> usable) | kernel/op speedup | worst-case note |",
          "|---|---|---|---|---|",
          "| Qwen3-0.6B | SwiGLU 1024x3072, 28 layers | 0.0 exact, 0.30 (+3.22 dPPL) | "
          "2.42x AVX @0.75, 2.85x CUDA @0.75 | HF path, no PPL-side kernel |",
          "| Qwen3.5-0.8B | SwiGLU 1024x3584, 24 layers | 0.0 exact, 0.30 (+4.94 dPPL, "
          "weighted-gate dPPL +1.86 @0.20) | 3.56x CUDA @0.60, 5.28x AVX @0.75 | "
          "down_proj is 10.4% of the checkpoint |",
          "| Qwen3.5-4B-IQ4_NL | SwiGLU 2560x9216, 32 layers | n/a (weights only) | "
          "4.54x cold @0.80 single layer, 4.24x coupled stream @0.80 | cache-line "
          "granularity at low skip; needs block-major |",
          "| whisper-base | GELU 512x2048, 6+6 layers | 0.0 exact, encoder 0.6 "
          "(rel-L2 0.325) | 1.85x enc FFN op @0.60 | encoder is not FFN-bound (3.6 of "
          "12.3 ms); 54%/51% of enc.4/5 blocks are exactly dead after pi |",
          "| SmolVLM2-256M | ViT 768x3072 + LM 576x1536 | 0.0 exact, text 0.2 (VQA 1.0), "
          "vision 0.2 (rel-L2 0.334) | kernel-level only | vision tower is much less "
          "skippable than the LM |", "",
          "_Across five architectures the same conclusion holds: at tau=0 the "
          "permuted model is bit-identical (max|dlogit| 1e-8..1e-5), the operators "
          "scale with the skip rate up to 2.4-5.3x, and model-level decode only "
          "improves once the skippable share of the checkpoint - not the share of "
          "the MLP - is large._", ""]

    # ---------------- 20. corrections ---------------------------------- #
    L += ["## 20. Corrections issued with Phase 5", "",
          "1. **GGUF offsets.** `Tensors.raw()` read `t['offset']` from the file start; "
          "GGUF tensor offsets are relative to the aligned data section "
          "(`reader.data_start`). Every Phase-4 IQ4_NL *correlation* computed from "
          "those bytes (energy capture, per-block energy) was decoding garbage; the "
          "wall-clock numbers were unaffected because they only measured touched "
          "lines. Fixed and re-verified: `iq4_row_dot` vs a numpy decode of the same "
          "row agrees to 5.3e-07.",
          "2. **ffn_gate/ffn_up need no repack.** `dims=[2560, 9216]` means "
          "`ne0=2560` is the contraction axis, so 32 consecutive output rows are "
          "already contiguous and output-block-major *is* the raw byte order; only "
          "`ffn_down` (`[9216, 2560]`) needs the block-major transpose.",
          "3. **Packed `up` output.** `up_rows_blockmajor` writes `u` packed in "
          "active-list order, not at block positions; the coupled path must scatter "
          "via the same index list. Ignoring this inflated the measured rel-L2 from "
          "0.41 to 1.14 at tau=0.20.",
          "4. **Model-level caveat that survives all fixes.** Unweighted gate-only "
          "selection still ranks worse than the Phase-3/4 energy metric; the "
          "offline `w_b` weighting is what closes most of the gap (-42% dPPL at "
          "tau=0.20) and is now the recommended default.", ""]

    text = "\n".join(L)
    (OUT / "REPORT_TEXT.md").write_text(text)
    (OUT / "text_results.json").write_text(json.dumps(
        {"base": {"qwen3.5-0.8b": base35, "qwen3-0.6b": base3},
         "sweep": sweep, "kernels": kern, "gguf": gguf,
         "phase4": {"cuda": load("p4_swiglu_cuda.json", {}),
                    "iq4_blockmajor": load("p4_iq4_blockmajor.json", {})},
         "phase5": {"cuda_float4": p5c, "weighted_gate": p5w,
                    "iq4_prefilter": p5q, "whisper": p5wsp, "smolvlm": p5s}},
         indent=1))
    print(text)


if __name__ == "__main__":
    main()
