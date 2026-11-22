#!/usr/bin/env python3
"""Phase 5 / Task 2: break the 2.14x Amdahl ceiling of the coupled IQ4_NL FFN.

The coupled FFN was capped because `ffn_gate` stayed dense.  Here a cheap
pre-filter reads only the first `r` rows of every 32-row block-major gate block,
ranks blocks by that partial energy, and finishes the remaining 32-r rows only
for the K selected blocks.  Gate row traffic therefore becomes
nb*r + K*(32-r) instead of nb*32, and the same idx drives ffn_up + ffn_down.

Usage: python bench_p5_iq4.py [--threads 32] [--rs 2,4,8]
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
LIB = HERE / "results" / "build" / "iq4nl_bm_v2.so"

f32p = ctypes.POINTER(ctypes.c_float)
u8p = ctypes.POINTER(ctypes.c_uint8)
i32p = ctypes.POINTER(ctypes.c_int)


def load_lib():
    lib = ctypes.CDLL(str(LIB))
    lib.repack_blockmajor.argtypes = [u8p, u8p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.gemv_blockmajor8_mt.argtypes = [u8p, f32p, f32p, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_int, i32p, ctypes.c_int, ctypes.c_int]
    lib.up_rows_blockmajor.argtypes = [u8p, f32p, f32p, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, i32p, ctypes.c_int, ctypes.c_int]
    lib.gate_prefilter_blockmajor.argtypes = [u8p, f32p, f32p, i32p, i32p,
                                              ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                              ctypes.c_float, f32p, ctypes.c_int]
    lib.gate_prefilter_blockmajor.restype = None
    return lib


class Tensors:
    def __init__(self, path):
        self.r = GGUFReader(path)
        self.f = open(path, "rb")
        self.tmap = {t["name"]: t for t in self.r.tensors}

    def raw(self, name):
        t = self.tmap[name]
        n = int(np.prod(t["dims"])) // 32 * IQ4_NL_BLOCK_BYTES
        # GGUF tensor offsets are relative to the (aligned) data section, not the
        # file start -- forgetting data_start silently decodes garbage.
        self.f.seek(self.r.data_start + t["offset"])
        return np.frombuffer(self.f.read(n), dtype=np.uint8).copy(), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--rates", default="0.2,0.4,0.6,0.8")
    ap.add_argument("--rs", default="2,4,8")
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--out", default="results/p5_iq4_prefilter.json")
    a = ap.parse_args()
    rates = [float(x) for x in a.rates.split(",")]
    rs = [int(x) for x in a.rs.split(",")]
    lib = load_lib()
    tt = Tensors(GGUF)
    M, nb, blk = 2560, 288, IQ4_NL_BLOCK_BYTES
    H = 9216
    print(f"ffn: {H} inter rows = {nb} blocks of 32, down {M} rows, block {M*blk} B")

    # repack all layers (gate, up, down) block-major, once
    t0 = time.perf_counter()
    G, U, D = [], [], []
    for L in range(a.layers):
        # ffn_gate / ffn_up are [ne0=2560, ne1=9216]: 32 consecutive output rows are
        # already contiguous, so output-block-major == raw row-major (no repack).
        for nm, acc in (("ffn_gate", G), ("ffn_up", U)):
            raw, _ = tt.raw(f"blk.{L}.{nm}.weight")
            acc.append(np.ascontiguousarray(raw))
        # ffn_down is [ne0=9216, ne1=2560]: needs the real block-major transpose.
        raw, _ = tt.raw(f"blk.{L}.ffn_down.weight")
        b = np.empty(nb * M * blk, dtype=np.uint8)
        lib.repack_blockmajor(raw.ctypes.data_as(u8p), b.ctypes.data_as(u8p), M, nb, blk)
        D.append(b)
    repack_s = time.perf_counter() - t0
    tot_mb = sum(x.nbytes for acc in (G, U, D) for x in acc) / 1e6
    print(f"repacked {3*a.layers} tensors in {repack_s:.1f} s, {tot_mb:.0f} MB")

    x = np.random.RandomState(1).randn(2560).astype(np.float32) * 0.2
    g = np.zeros(nb * 32, np.float32)
    u = np.zeros(nb * 32, np.float32)
    h = np.zeros(nb * 32, np.float32)
    out = np.zeros(M, np.float32)
    idx = np.zeros(nb, np.int32)
    kp = np.zeros(1, np.int32)
    xp = x.ctypes.data_as(f32p)
    gp = g.ctypes.data_as(f32p)
    up = u.ctypes.data_as(f32p)
    hp = h.ctypes.data_as(f32p)
    op = out.ctypes.data_as(f32p)
    ip = idx.ctypes.data_as(i32p)
    kpp = kp.ctypes.data_as(i32p)
    all_idx = np.arange(nb, dtype=np.int32)
    all_ip = all_idx.ctypes.data_as(i32p)

    def dense_ref(gl, ul, dl):
        lib.up_rows_blockmajor(gl, xp, gp, H, nb, 2560, all_ip, nb, a.threads)
        lib.up_rows_blockmajor(ul, xp, up, H, nb, 2560, all_ip, nb, a.threads)
        gs = g * np.reciprocal(1.0 + np.exp(-np.clip(g, -60, 60)))   # stable silu
        np.multiply(gs, u, out=g)
        lib.gemv_blockmajor8_mt(dl, gp, op, M, nb, blk, all_ip, nb, a.threads)
        return out.copy()

    def coupled(L, rate, r):
        """prefilter -> up(active) -> gate*up -> down(active), one idx shared."""
        if r >= 32:
            lib.up_rows_blockmajor(G[L].ctypes.data_as(u8p), xp, gp, H, nb, 2560,
                                   all_ip, nb, a.threads)
            idx[:] = all_idx
            K = nb
        else:
            lib.gate_prefilter_blockmajor(G[L].ctypes.data_as(u8p), xp, gp, ip, kpp,
                                          nb, 2560, r, rate, None, a.threads)
            K = int(kp[0])
        lib.up_rows_blockmajor(U[L].ctypes.data_as(u8p), xp, up, H, nb, 2560,
                               ip, K, a.threads)
        # h = silu(g) * u on the active blocks only.  up_rows_blockmajor writes u
        # PACKED in active-list order (u[j] <-> idx[j>>5], lane j&31), while g is
        # block-indexed, so scatter through the same index list.
        sel = (idx[:K, None] * 32 + np.arange(32)[None, :]).ravel()
        sub = g[sel]
        h[sel] = (sub * np.reciprocal(1.0 + np.exp(-np.clip(sub, -60, 60)))) * u[:K * 32]
        lib.gemv_blockmajor8_mt(D[L].ctypes.data_as(u8p), hp, op, M, nb, blk,
                                ip, K, a.threads)
        return K

    ref_out = dense_ref(G[0].ctypes.data_as(u8p), U[0].ctypes.data_as(u8p),
                        D[0].ctypes.data_as(u8p))
    rn = float(np.linalg.norm(ref_out))

    # ---------------- single layer --------------------------------------- #
    single = []
    for rate in [0.0] + rates:
        for r in rs:
            K = coupled(0, rate, r if rate > 0 else 32)
            e = float(np.linalg.norm(out - ref_out) / rn)
            reps = 20
            for _ in range(3):
                coupled(0, rate, r if rate > 0 else 32)
            t = time.perf_counter()
            for _ in range(reps):
                coupled(0, rate, r if rate > 0 else 32)
            ms = (time.perf_counter() - t) / reps * 1e3
            single.append({"rate": rate, "r": r if rate > 0 else 32, "kept": K,
                           "ms": ms, "rel_l2": e})
            print(f"  single  tau {rate:.2f} r={r:2d} K={K:3d}  {ms:6.3f} ms  relL2 {e:.4f}")

    # ---------------- 32-layer stream ------------------------------------ #
    stream = []
    for rate in [0.0] + rates:
        for r in rs:
            rounds = []
            Ks = [0]
            for rep in range(5):
                t = time.perf_counter()
                Ks = [coupled(L, rate, r if rate > 0 else 32) for L in range(a.layers)]
                rounds.append((time.perf_counter() - t) * 1e3)
            ms = float(np.median(rounds))
            K = Ks[-1]
            gate_rows = nb * (r if rate > 0 else 32) + (K * (32 - (r if rate > 0 else 32))
                                                        if rate > 0 else 0)
            mb = (gate_rows * 80 * blk + K * 32 * 80 * blk + K * M * blk) / 1e6
            stream.append({"rate": rate, "r": r if rate > 0 else 32, "kept": K,
                           "ms_per_token": ms, "mb_per_token": mb})
            print(f"  stream  tau {rate:.2f} r={r:2d} K={K:3d}  {ms:6.1f} ms/tok  {mb:5.1f} MB")

    d = stream[0]["ms_per_token"]
    for s in stream:
        s["speedup"] = d / s["ms_per_token"]
    res = {"model": str(GGUF), "threads": a.threads, "repack_s": repack_s,
           "repacked_mb": tot_mb,
           "dense_reference": {"ms_per_token": d, "mb_per_token": stream[0]["mb_per_token"]},
           "single_layer": single, "stream_all_layers": stream,
           "note": "gate pre-filter reads only r rows per block, then finishes the "
                   "remaining 32-r rows of the K selected blocks; up and down reuse idx"}
    (HERE / a.out).write_text(json.dumps(res, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
