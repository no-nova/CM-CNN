#!/usr/bin/env python3
"""Phase 4 / Task 2: block-major IQ4_NL with 8-row blocking + coupled FFN.

* Dense cold baseline of the 8-row kernel vs the 4-row one and vs the row-major
  reference measured in Phase 3 (0.930 ms single layer / 35.3 ms for 32 layers).
* All 32 layers' `ffn_down` repacked block-major once, then streamed from DRAM.
* Coupled IQ4_NL FFN: gate (dense) -> active blocks -> `ffn_up` active row
  blocks -> `ffn_down` active column blocks, all sharing one compacted idx.

Usage: python bench_p4_iq4.py [--threads 32] [--out results/p4_iq4_blockmajor.json]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time
from pathlib import Path

import numpy as np

from gguf_iq4nl import GGUFReader, IQ4_NL_BLOCK_BYTES

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GGUF = ROOT / "models" / "Qwen3.5-4B-IQ4_NL.gguf"
LIB = HERE / "results" / "build" / "iq4nl_blockmajor.so"

f32p = ctypes.POINTER(ctypes.c_float)
u8p = ctypes.POINTER(ctypes.c_uint8)
i32p = ctypes.POINTER(ctypes.c_int)


def load_lib():
    lib = ctypes.CDLL(str(LIB))
    for fn, argtypes in (
        ("repack_blockmajor", [u8p, u8p, ctypes.c_int, ctypes.c_int, ctypes.c_int]),
        ("gemv_blockmajor", [u8p, f32p, f32p, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, i32p, ctypes.c_int]),
        ("gemv_blockmajor8", [u8p, f32p, f32p, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, i32p, ctypes.c_int]),
        ("gemv_blockmajor8_mt", [u8p, f32p, f32p, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_int, i32p, ctypes.c_int, ctypes.c_int]),
        ("up_rows_blockmajor", [u8p, f32p, f32p, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, i32p, ctypes.c_int, ctypes.c_int]),
    ):
        getattr(lib, fn).argtypes = argtypes
    lib.repack_blockmajor.restype = None
    lib.gemv_blockmajor.restype = None
    lib.gemv_blockmajor8.restype = None
    lib.gemv_blockmajor8_mt.restype = None
    lib.up_rows_blockmajor.restype = None
    return lib


class Tensors:
    """mmap-backed access to one GGUF tensor's raw bytes."""

    def __init__(self, path):
        self.r = GGUFReader(path)
        self.f = open(path, "rb")
        self.tmap = {t["name"]: t for t in self.r.tensors}

    def raw(self, name):
        t = self.tmap[name]
        n_elems = int(np.prod(t["dims"]))
        nbytes = n_elems // 32 * IQ4_NL_BLOCK_BYTES
        self.f.seek(t["offset"])
        return np.frombuffer(self.f.read(nbytes), dtype=np.uint8).copy(), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--rates", default="0.2,0.4,0.6,0.8")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--out", default="results/p4_iq4_blockmajor.json")
    args = ap.parse_args()
    rates = [float(x) for x in args.rates.split(",")]
    lib = load_lib()
    tt = Tensors(GGUF)
    M, nb = 2560, 288           # down: [M=2560 rows, 9216 in = 288 blocks]
    blk = IQ4_NL_BLOCK_BYTES
    print(f"ffn_down {M} rows x {nb} blocks, block-major {M*blk} B/block")

    # ---------- single layer, cold --------------------------------------- #
    raw, _t = tt.raw("blk.16.ffn_down.weight")
    wb = np.empty(nb * M * blk, dtype=np.uint8)
    t0 = time.perf_counter()
    lib.repack_blockmajor(raw.ctypes.data_as(u8p), wb.ctypes.data_as(u8p), M, nb, blk)
    repack_ms = (time.perf_counter() - t0) * 1e3

    x = np.random.RandomState(0).randn(nb * 32).astype(np.float32) * 0.3
    out = np.zeros(M, np.float32)
    idx = np.zeros(nb, np.int32)
    evict = np.empty(24 * 1024 * 1024, np.float32)  # 96 MB L3 eviction buffer
    xp = x.ctypes.data_as(f32p)
    op = out.ctypes.data_as(f32p)
    wbp = wb.ctypes.data_as(u8p)

    def line_bytes(idx_arr):
        return int(len(idx_arr)) * M * blk

    def cold(fn, reps=5):
        best = 1e9
        for _ in range(reps):
            evict[:] = 1.0
            t = time.perf_counter()
            fn()
            best = min(best, (time.perf_counter() - t) * 1e3)
        return best

    rows = []
    for skip in [0.0] + rates:
        K = max(1, int(round(nb * (1 - skip))))
        idx_arr = np.arange(K, dtype=np.int32)
        ip = idx_arr.ctypes.data_as(i32p)
        t4 = cold(lambda: lib.gemv_blockmajor(wbp, xp, op, M, nb, blk, ip, K))
        t8 = cold(lambda: lib.gemv_blockmajor8(wbp, xp, op, M, nb, blk, ip, K))
        tmt = cold(lambda: lib.gemv_blockmajor8_mt(wbp, xp, op, M, nb, blk, ip, K,
                                                   args.threads))
        rows.append({"skip": skip, "kept_blocks": K, "line_bytes": line_bytes(idx_arr),
                     "ms_st4": t4, "ms_st8": t8, "ms_mt": tmt,
                     "ref_rowmajor_ms": 0.930})
        print(f"skip {skip:.2f}: 4-row {t4:.3f} | 8-row {t8:.3f} | "
              f"8-row mt{args.threads} {tmt:.3f} ms  lines {line_bytes(idx_arr)/1e6:.2f} MB")

    dense_mt = rows[0]["ms_mt"]
    for r in rows:
        r["speedup_mt"] = dense_mt / r["ms_mt"]
        r["speedup_vs_rowmajor_dense"] = 0.930 / r["ms_mt"]

    # ---------- all 32 layers streamed, coupled FFN ----------------------- #
    down_bm, up_bm = [], []
    t0 = time.perf_counter()
    for L in range(args.layers):
        raw, _ = tt.raw(f"blk.{L}.ffn_down.weight")
        b = np.empty(nb * M * blk, dtype=np.uint8)
        lib.repack_blockmajor(raw.ctypes.data_as(u8p), b.ctypes.data_as(u8p), M, nb, blk)
        down_bm.append(b)
        raw, _ = tt.raw(f"blk.{L}.ffn_up.weight")
        b2 = np.empty(nb * 32 * 80 * blk, dtype=np.uint8)  # [nb, 32 rows x 80 blocks]
        lib.repack_blockmajor(raw.ctypes.data_as(u8p), b2.ctypes.data_as(u8p),
                             32 * 80, nb, blk)
        up_bm.append(b2)
    repack_all_s = time.perf_counter() - t0
    print(f"repacked {len(down_bm)} layers (down+up) in {repack_all_s:.1f} s, "
          f"{(sum(a.nbytes for a in down_bm) + sum(a.nbytes for a in up_bm))/1e6:.0f} MB")

    u = np.zeros(nb * 32, np.float32)
    g = np.zeros(nb * 32, np.float32)
    xin = np.random.RandomState(1).randn(2560).astype(np.float32) * 0.2
    xinp = xin.ctypes.data_as(f32p)
    up_p = u.ctypes.data_as(f32p)
    g_p = g.ctypes.data_as(f32p)

    stream = []
    for skip in [0.0] + rates:
        K = max(1, int(round(nb * (1 - skip))))
        idx_arr = np.sort(np.random.RandomState(7).choice(nb, K, replace=False)).astype(np.int32)
        ip = idx_arr.ctypes.data_as(i32p)
        t0 = time.perf_counter()
        for L in range(args.layers):
            d = down_bm[L].ctypes.data_as(u8p)
            up = up_bm[L].ctypes.data_as(u8p)
            # gate: dense (all blocks) on the packed up-layout of ffn_gate
            lib.up_rows_blockmajor(up, xinp, g_p, 9216, nb, 2560,
                                   np.arange(nb, dtype=np.int32).ctypes.data_as(i32p),
                                   nb, args.threads)
            lib.up_rows_blockmajor(up, xinp, up_p, 9216, nb, 2560, ip, K, args.threads)
            lib.gemv_blockmajor8_mt(d, xinp, op, M, nb, blk, ip, K, args.threads)
        ms = (time.perf_counter() - t0) * 1e3
        bytes_tok = (nb * 32 * 80 * blk) + 2 * K * 32 * 80 * blk
        stream.append({"skip": skip, "ms_per_token": ms, "kept_blocks": K,
                       "bytes_per_token": bytes_tok})
        print(f"coupled FFN stream skip {skip:.2f}: {ms:.1f} ms/token, "
              f"{bytes_tok/1e6:.0f} MB read")
    base = stream[0]["ms_per_token"]
    for r in stream:
        r["speedup"] = base / r["ms_per_token"]

    res = {"dense_baselines": {"rowmajor_single_layer_ms": 0.930,
                               "rowmajor_32_layer_ms": 35.3},
           "repack_ms_per_tensor": repack_ms,
           "repack_all_layers_s": repack_all_s,
           "threads": args.threads, "single_layer": rows,
           "coupled_ffn_stream": stream,
           "note": "gate is dense by construction; up and down share one idx "
                   "produced from the gate output, both read in place"}
    (HERE / args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
