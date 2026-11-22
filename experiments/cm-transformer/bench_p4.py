#!/usr/bin/env python3
"""Phase 4 / Task 1: fused zero-copy SwiGLU coupled skipping on AVX-512.

Measures, on real Qwen3.5-0.8B fp32 weights:
  * single-layer MLP (gate + up + down) at T=1 and T=64,
  * end-to-end decode ms/token with the fused op wired into the 24 layers,
  * numerics vs the torch dense path (permutation-isomorphic baseline).

The fused kernel never copies weights: `up_proj` active rows are read in place,
`down_proj` active columns are read in place, and both share the single
compacted block list produced from `g = SiLU(W_gate x)`.

Usage: python bench_p4.py [--rates 0.2,0.4,0.6] [--threads 16] [--out results/p4_swiglu_cpu.json]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np
import torch

import common
from model_ref import TextBackbone, Weights
from common import MODEL_DIR, dump_json

RESULTS = Path(__file__).resolve().parent / "results"

HERE = Path(__file__).resolve().parent
LIB = RESULTS / "build" / "swiglu_avx512.so"

f32p = ctypes.POINTER(ctypes.c_float)
i32p = ctypes.POINTER(ctypes.c_int)


def load_lib():
    lib = ctypes.CDLL(str(LIB))
    lib.swiglu_coupled_f32.restype = ctypes.c_int
    lib.swiglu_coupled_f32.argtypes = [
        f32p, f32p, f32p, f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_float, ctypes.c_float, f32p, i32p, f32p, f32p]
    lib.swiglu_dense_f32.restype = ctypes.c_int
    lib.swiglu_dense_f32.argtypes = [
        f32p, f32p, f32p, f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, f32p]
    return lib


def ptr(a):
    a = np.ascontiguousarray(a)
    return a.ctypes.data_as(f32p), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.2,0.4,0.6")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--e2e-tokens", type=int, default=12)
    ap.add_argument("--out", default="results/p4_swiglu_cpu.json")
    args = ap.parse_args()
    rates = [float(x) for x in args.rates.split(",")]
    lib = load_lib()

    tb = TextBackbone(Weights(torch.float32, "cpu"))
    c = tb.cfg
    H, D, O = c.inter, c.hidden, c.hidden
    layer = 11
    p = f"{tb.prefix}layers.{layer}.mlp."
    Wg = tb.w[p + "gate_proj.weight"].numpy().astype(np.float32)
    Wu = tb.w[p + "up_proj.weight"].numpy().astype(np.float32)
    Wd = tb.w[p + "down_proj.weight"].numpy().astype(np.float32)
    print(f"layer {layer}: gate {Wg.shape} up {Wu.shape} down {Wd.shape}")

    # a REAL MLP input, captured from a real prompt (random N(0,0.5) inputs have
    # no heavy tail, so any top-k choice looks catastrophic on them)
    tok0 = common.build_tokenizer(MODEL_DIR)
    cap = {}
    ids0 = torch.tensor(tok0.encode("The capital of France is Paris and the Eiffel Tower")[:16])
    tb.forward(ids0, capture=cap)
    x = cap["mlp_in"][layer].detach().to(torch.float32).numpy()[-1].copy()
    torch_mlp = None
    out = np.zeros(O, np.float32)
    g_buf = np.zeros(H, np.float32)
    idx_buf = np.zeros(H // 32, np.int32)
    h_buf = np.zeros(H, np.float32)
    alpha = ctypes.c_float(1.0)

    xp, _xa = ptr(x)
    gbp, _gb = ptr(g_buf)
    hbp, _hb = ptr(h_buf)
    obp, _ob = ptr(out)
    ibp = idx_buf.ctypes.data_as(i32p)

    def dense():
        lib.swiglu_dense_f32(*ptr(Wg)[0:1], *ptr(Wu)[0:1], *ptr(Wd)[0:1],
                             xp, obp, D, H, O, gbp)
        return out.copy()

    def coupled(rate):
        K = lib.swiglu_coupled_f32(xp, *ptr(Wg)[0:1], *ptr(Wu)[0:1], *ptr(Wd)[0:1],
                                   obp, D, H, O, 32, rate, 1.5, gbp, ibp, hbp,
                                   ctypes.byref(alpha))
        return out.copy(), K

    ref = dense()
    tor = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.from_numpy(x) @ torch.from_numpy(Wg).T)
        * (torch.from_numpy(x) @ torch.from_numpy(Wu).T),
        torch.from_numpy(Wd))
    iso = float(np.abs(ref - tor.numpy()).max())

    Wgt = torch.from_numpy(Wg); Wut = torch.from_numpy(Wu); Wdt = torch.from_numpy(Wd)
    xt = torch.from_numpy(x)

    def torch_mlp():
        return torch.nn.functional.linear(
            torch.nn.functional.silu(xt @ Wgt.T) * (xt @ Wut.T), Wdt)

    def bench(fn, reps=20):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps * 1e3

    rows = []
    t_ms = bench(torch_mlp)
    rows.append({"rate": -1.0, "ms": t_ms, "speedup": 1.0, "kept_blocks": H // 32,
                 "max_abs": 0.0, "mode": "torch dense (reference)"})
    d_ms = bench(dense)
    rows.append({"rate": 0.0, "ms": d_ms, "speedup": 1.0, "kept_blocks": H // 32,
                 "max_abs": iso, "mode": "dense (fused, no skip)"})
    dref = ref
    for r in rates:
        o, K = coupled(r)
        ms = bench(lambda r=r: coupled(r))
        rel = float(np.linalg.norm(o - dref) / (np.linalg.norm(dref) + 1e-9))
        rows.append({"rate": r, "ms": ms, "speedup": d_ms / ms, "kept_blocks": int(K),
                     "max_abs": float(np.abs(o - dref).max()), "rel_l2": rel,
                     "mode": "coupled skip"})
    print(f"single-layer T=1 dense {d_ms:.3f} ms (iso vs torch {iso:.2e})")
    for r in rows:
        r.setdefault("rel_l2", 0.0)
        print(f"  coupled tau={r['rate']:.2f}: {r['ms']:.3f} ms "
              f"({r['speedup']:.2f}x) kept={r['kept_blocks']}/112 "
              f"max|d|={r['max_abs']:.3f} relL2={r['rel_l2']:.4f}")

    # ---- T=64: batch of 64 tokens through the same fused entry point ------
    X = np.random.RandomState(1).randn(64, D).astype(np.float32) * 0.5
    out64 = np.zeros((64, O), np.float32)

    def dense64():
        for t in range(64):
            xp_t, _ = ptr(X[t])
            lib.swiglu_dense_f32(*ptr(Wg)[0:1], *ptr(Wu)[0:1], *ptr(Wd)[0:1],
                                 xp_t, ptr(out64[t])[0], D, H, O, gbp)

    def coupled64(rate):
        for t in range(64):
            xp_t, _ = ptr(X[t])
            lib.swiglu_coupled_f32(xp_t, *ptr(Wg)[0:1], *ptr(Wu)[0:1], *ptr(Wd)[0:1],
                                   ptr(out64[t])[0], D, H, O, 32, rate, 1.5, gbp,
                                   ibp, hbp, ctypes.byref(alpha))

    t_rows = []
    dm = bench(dense64, reps=3) / 64
    t_rows.append({"T": 64, "rate": 0.0, "ms_per_token": dm, "speedup": 1.0})
    for r in rates:
        m = bench(lambda r=r: coupled64(r), reps=3) / 64
        t_rows.append({"T": 64, "rate": r, "ms_per_token": m, "speedup": dm / m})

    # ---- E2E decode: replace every MLP with the fused op ------------------
    weights = {}
    for i in range(c.layers):
        pp = f"{tb.prefix}layers.{i}.mlp."
        weights[i] = (ptr(tb.w[pp + "gate_proj.weight"].numpy().astype(np.float32))[0],
                      ptr(tb.w[pp + "up_proj.weight"].numpy().astype(np.float32))[0],
                      ptr(tb.w[pp + "down_proj.weight"].numpy().astype(np.float32))[0])
    scratch = {"g": np.zeros(H, np.float32), "h": np.zeros(H, np.float32),
               "idx": np.zeros(H // 32, np.int32), "o": np.zeros(O, np.float32)}
    keep = np.ascontiguousarray(scratch["g"])
    sc_g = keep.ctypes.data_as(f32p)
    sc_h = scratch["h"].ctypes.data_as(f32p)
    sc_idx = scratch["idx"].ctypes.data_as(i32p)
    sc_o = scratch["o"].ctypes.data_as(f32p)
    state = {"rate": 0.0, "kept": 0}

    def fused_mlp(self, x, i, W=None, keep=None, skip_mask=None, skip_rate=None,
                  energy_out=None, skip_scale=None, skip_theta=None,
                  norm_gamma=None, coupled=False):
        xv = x.detach().to(torch.float32).numpy()
        one = xv.ndim == 1
        if one:
            xv = xv[None]
        outs = []
        for t in range(xv.shape[0]):
            xt, _ = ptr(xv[t])
            wg, wu, wd = weights[i]
            if state["rate"] <= 0.0:
                lib.swiglu_dense_f32(wg, wu, wd, xt, sc_o, D, H, O, sc_g)
                state["kept"] = H // 32
            else:
                state["kept"] = lib.swiglu_coupled_f32(
                    xt, wg, wu, wd, sc_o, D, H, O, 32, state["rate"], 1.5,
                    sc_g, sc_idx, sc_h, None)
            outs.append(torch.from_numpy(scratch["o"].copy()))
        out = torch.stack(outs)
        return out[0] if one else out

    import types
    tb.mlp = types.MethodType(fused_mlp, tb)

    toks = tok0.encode("The capital of France is Paris and the")
    ids = torch.tensor(toks[:8])
    e2e = {}
    # torch baseline first, with the same incremental-decode harness
    tb_ref = TextBackbone(Weights(torch.float32, "cpu"))
    import model_ref as _mr2
    st0 = _mr2.new_state()
    tb_ref.forward(ids, state=st0, pos0=0)
    pos = len(toks[:8]); cur = torch.tensor([toks[pos - 1]])
    t0 = time.perf_counter()
    for _ in range(args.e2e_tokens):
        tb_ref.forward(cur, state=st0, pos0=pos); pos += 1
    ms_torch = (time.perf_counter() - t0) / args.e2e_tokens * 1e3
    print(f"E2E torch baseline: {ms_torch:.1f} ms/token ({1e3 / ms_torch:.3f} tok/s)")
    e2e["torch_baseline"] = {"ms_per_token": ms_torch, "tok_s": 1e3 / ms_torch}
    for rate in [0.0] + rates:
        state["rate"] = rate
        import model_ref as _mr
        st = _mr.new_state()
        tb.forward(ids, state=st, pos0=0)  # prefill
        pos = len(toks[:8])
        cur = torch.tensor([toks[pos - 1]])
        t0 = time.perf_counter()
        for _ in range(args.e2e_tokens):
            o = tb.forward(cur, state=st, pos0=pos)
            pos += 1
        ms = (time.perf_counter() - t0) / args.e2e_tokens * 1e3
        e2e[f"rate_{rate:.2f}"] = {"ms_per_token": ms, "tok_s": 1e3 / ms,
                                   "kept_blocks": state["kept"]}
        print(f"E2E tau={rate:.2f}: {ms:.1f} ms/token ({1e3 / ms:.2f} tok/s) "
              f"kept={state['kept']}/112")

    for k, v in e2e.items():
        if isinstance(v, dict) and "ms_per_token" in v:
            v["speedup_vs_torch"] = ms_torch / v["ms_per_token"]
    base = e2e["rate_0.00"]["ms_per_token"]
    for k, v in e2e.items():
        v["speedup"] = base / v["ms_per_token"]

    res = {"model": "Qwen3.5-0.8B", "layer": layer, "threads": args.threads,
           "iso_vs_torch_max_abs": iso, "single_layer": rows,
           "t64_per_token": t_rows, "e2e_decode": e2e,
           "note": "fused swiglu_coupled_f32: gate dense, shared compacted block "
                   "list, up_proj active rows + down_proj active columns read in "
                   "place (zero weight copy)"}
    dump_json(res, HERE / args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
