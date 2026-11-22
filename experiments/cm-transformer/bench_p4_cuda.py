#!/usr/bin/env python3
"""Phase 4 / Task 1 (CUDA): fused zero-copy SwiGLU coupled skip on RTX 5070."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

from common import MODEL_DIR, dump_json
from model_ref import TextBackbone, Weights

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.2,0.4,0.6")
    ap.add_argument("--out", default="results/p4_swiglu_cuda.json")
    a = ap.parse_args()
    rates = [float(x) for x in a.rates.split(",")]
    mod = load(name="swiglu_cuda", sources=["swiglu_cuda.cu"],
               extra_cuda_cflags=["-O3", "--use_fast_math",
                                  "-gencode=arch=compute_120,code=sm_120",
                                  "-DGLOG_USE_GLOG_EXPORT"],
               extra_cflags=["-DGLOG_USE_GLOG_EXPORT"],
               build_directory=str(RESULTS / "build"), verbose=False)

    tb = TextBackbone(Weights(torch.float32, "cuda"))
    c = tb.cfg
    H, D, O, L = c.inter, c.hidden, c.hidden, c.layers
    dev = "cuda"
    p = f"{tb.prefix}layers.11.mlp."
    Wg = tb.w[p + "gate_proj.weight"].contiguous()
    Wu = tb.w[p + "up_proj.weight"].contiguous()
    Wd = tb.w[p + "down_proj.weight"].contiguous()
    x = torch.randn(D, device=dev) * 0.5
    g_buf = torch.zeros(H, device=dev)
    idx_buf = torch.zeros(H // 32, dtype=torch.int32, device=dev)
    k_buf = torch.zeros(1, dtype=torch.int32, device=dev)
    h_buf = torch.zeros(H, device=dev)
    out = torch.zeros(O, device=dev)

    def fused(rate):
        K = mod.swiglu_coupled(x, Wg, Wu, Wd, g_buf, idx_buf, k_buf, h_buf, out, rate)
        return out, K

    def dense_torch():
        return torch.nn.functional.linear(
            torch.nn.functional.silu(x @ Wg.T) * (x @ Wu.T), Wd)

    def bench(fn, reps=50):
        for _ in range(5): fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(reps): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / reps

    ref = dense_torch()
    o0, _ = fused(0.0)
    iso = float((o0 - ref).abs().max())
    rows = [{"rate": -1.0, "ms": bench(dense_torch), "speedup": 1.0, "kept": H // 32,
             "mode": "torch dense (reference)"}]
    dm = bench(lambda: fused(0.0))
    rows.append({"rate": 0.0, "ms": dm, "speedup": rows[0]["ms"] / dm, "kept": H // 32,
                 "mode": "fused dense (no skip)"})
    for r in rates:
        o, K = fused(r)
        ms = bench(lambda r=r: fused(r))
        rows.append({"rate": r, "ms": ms, "speedup": rows[0]["ms"] / ms, "kept": int(K),
                     "max_abs": float((o - ref).abs().max()),
                     "rel_l2": float((o - ref).norm() / ref.norm()),
                     "mode": "coupled skip"})
    print(f"CUDA single layer: torch {rows[0]['ms']:.3f} ms, fused dense {dm:.3f} ms, "
          f"iso {iso:.2e}")
    for r in rows[2:]:
        print(f"  tau={r['rate']:.2f}: {r['ms']:.3f} ms ({r['speedup']:.2f}x vs torch, "
              f"{dm / r['ms']:.2f}x vs fused dense) kept={r['kept']}/112")

    # 24-layer MLP-only chain (same weights, isolates the MLP path)
    chain = []
    for r in [0.0] + rates:
        for _ in range(3): fused(r)
        torch.cuda.synchronize()
        import time as _t
        t0 = _t.perf_counter()
        for _ in range(20):
            for _ in range(L): fused(r)
        torch.cuda.synchronize()
        chain.append({"rate": r, "ms_per_token": (_t.perf_counter() - t0) / 20 * 1e3})
    base = chain[0]["ms_per_token"]
    for r in chain: r["speedup"] = base / r["ms_per_token"]
    print("24-layer MLP chain:", [(round(r['rate'], 2), round(r['ms_per_token'], 2)) for r in chain])

    dump_json({"model": "Qwen3.5-0.8B", "backend": "cuda", "single_layer": rows,
               "iso_vs_torch_max_abs": iso, "mlp_chain_24": chain,
               "note": "zero weight copy; gate dense, up rows + down cols share one idx"}, HERE / a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
