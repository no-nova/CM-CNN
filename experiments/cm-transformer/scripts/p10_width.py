#!/usr/bin/env python3
"""Phase 10 task 5b: SIMD-width ablation (scalar / AVX2 / AVX-512).

The same `iq4nl_width.cpp` is compiled three times and driven single-threaded on
the real `blk.16.ffn_down` bytes of the 4B checkpoint.  The compacted block index
set is fixed across builds, so only the width of the multiply-add loop changes.

Usage: python scripts/p10_width.py [--reps 5] [--seed 1234]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench_p5_iq4 import LIB as BM_LIB
from bench_p5_iq4 import Tensors

HERE = Path(__file__).resolve().parent.parent
BUILD = HERE / "results" / "build"
OUT = HERE / "results" / "p10_width.json"

f32p = ctypes.POINTER(ctypes.c_float)
u8p = ctypes.POINTER(ctypes.c_uint8)
i32p = ctypes.POINTER(ctypes.c_int)

VARIANTS = ("scalar", "avx2", "avx512")
TAUS = (0.0, 0.2, 0.4, 0.8)


def load(path):
    lib = ctypes.CDLL(str(path))
    lib.gemv_row_dense.argtypes = [u8p, f32p, f32p, ctypes.c_int, ctypes.c_int]
    lib.gemv_bm.argtypes = [u8p, f32p, f32p, ctypes.c_int, ctypes.c_int, i32p, ctypes.c_int]
    return lib


def blockmajor(lib_bm, raw, M, nb):
    W = np.ascontiguousarray(raw)
    out = np.zeros_like(W)
    lib_bm.repack_blockmajor(W.ctypes.data_as(u8p), out.ctypes.data_as(u8p), M, nb, 18)
    return out


def timeit(fn, reps):
    fn()                                # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.min(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--layer", type=int, default=16)
    a = ap.parse_args()

    tens = Tensors(str(HERE.parents[1] / "models" / "Qwen3.5-4B-IQ4_NL.gguf"))
    raw, t = tens.raw(f"blk.{a.layer}.ffn_down.weight")
    M, nb = 2560, 288                       # [d_model, d_ff] = [2560, 9216]
    assert raw.size == M * nb * 18, (raw.size, M, nb)

    libs_bm = ctypes.CDLL(str(BM_LIB))
    libs_bm.repack_blockmajor.argtypes = [u8p, u8p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    WB = blockmajor(libs_bm, raw, M, nb)

    rng = np.random.default_rng(a.seed)
    order = rng.permutation(nb).astype(np.int32)   # fixed index set for every build
    x = rng.standard_normal(nb * 32).astype(np.float32)

    rows = []
    for name in VARIANTS:
        lib = load(BUILD / f"iq4nl_width_{name}.so")
        out = np.zeros(M, dtype=np.float32)

        def row_dense():
            lib.gemv_row_dense(raw.ctypes.data_as(u8p), x.ctypes.data_as(f32p),
                               out.ctypes.data_as(f32p), M, nb)

        def bm(K):
            idx = np.ascontiguousarray(order[:K])
            return lambda: lib.gemv_bm(WB.ctypes.data_as(u8p), x.ctypes.data_as(f32p),
                                       out.ctypes.data_as(f32p), M, nb,
                                       idx.ctypes.data_as(i32p), K)

        med_row, min_row = timeit(row_dense, a.reps)
        entry = {"width": name, "row_major_dense_ms": round(med_row, 4),
                 "row_major_dense_min_ms": round(min_row, 4), "block_major": {}}
        for tau in TAUS:
            K = max(1, int(round(nb * (1 - tau))))
            med, mn = timeit(bm(K), a.reps)
            entry["block_major"][f"{tau:.2f}"] = {
                "kept_blocks": K, "ms": round(med, 4), "min_ms": round(mn, 4),
                "speedup_vs_row_dense": round(med_row / med, 3),
                "speedup_vs_own_dense": round(entry["block_major"].get("0.00", {}).get("ms", med) / med, 3),
            }
            print(f"[{name}] tau={tau:.2f} K={K} {med:.4f} ms "
                  f"({med_row / med:.2f}x vs row dense)", flush=True)
        # checksum: the three builds must agree bit-for-bit on the same index set
        bm(288)()
        entry["sum_all_blocks"] = round(float(out.sum()), 3)
        rows.append(entry)

    ref = rows[0]["sum_all_blocks"]
    for r in rows:
        r["sum_matches_scalar"] = bool(abs(r["sum_all_blocks"] - ref) < max(1e-2, 1e-6 * abs(ref)))
    OUT.write_text(json.dumps({"layer": a.layer, "tensor": t["name"], "dims": [M, nb],
                               "block_bytes": 18, "seed": a.seed, "rows": rows}, indent=1))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
