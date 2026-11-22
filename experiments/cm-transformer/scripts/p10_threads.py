"""Phase 10 (Q7): does the block-major gain survive other parallelism widths?

Sweeps the OpenMP thread count of the 8-row block-major kernel on a single
`ffn_down` tensor at several block budgets.  The row-major reference is the
measured 0.930 ms dense figure from `results/p4_iq4_blockmajor.json`, which uses
the same dot-product formulation with a different memory layout; the point here is
the shape of the curve, not a new baseline.

SIMD width is not swept: the source is compiled with AVX-512 intrinsics
throughout, and building an AVX2/scalar fallback was left out of scope.  The
report says so rather than implying a width study.
"""
from __future__ import annotations

import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import bench_p4_iq4 as B4  # noqa: E402

RES = EXP / "results"
TENSOR = "blk.16.ffn_down.weight"
SKIPS = (0.0, 0.2, 0.4, 0.6, 0.8)
THREADS = (1, 4, 8, 16, 32)


def log(*a):
    print(*a, flush=True)


def main():
    lib = ctypes.CDLL(str(RES / "build" / "iq4nl_bm_v2.so"))
    lib.gemv_blockmajor8_mt.restype = None
    lib.gemv_blockmajor8_mt.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                        ctypes.c_int]
    t = B4.Tensors(B4.GGUF)
    ent = t.tmap[TENSOR]
    rows, cols = int(ent["dims"][1]), int(ent["dims"][0])
    ent = t.tmap[TENSOR]
    nbytes = int(np.prod(ent["dims"])) // 32 * 18
    with open(B4.GGUF, "rb") as f:      # GGUF offsets are relative to data_start
        f.seek(t.r.data_start + int(ent["offset"]))
        raw = np.frombuffer(f.read(nbytes), np.uint8).copy()
    nb = cols // 32
    blk = 18
    wb = np.zeros(rows * nb * blk, np.uint8)
    lib.repack_blockmajor(raw.ctypes.data_as(ctypes.c_void_p),
                          wb.ctypes.data_as(ctypes.c_void_p), rows, nb, blk)
    a = np.random.RandomState(0).randn(cols).astype(np.float32)
    a /= np.linalg.norm(a)
    out = np.zeros(rows, np.float32)
    evict = np.zeros(96 * 1024 * 1024 // 8, np.uint8)
    res = {"tensor": TENSOR, "rows": rows, "nb_blocks": nb, "skips": list(SKIPS),
           "threads": list(THREADS), "rows_table": [],
           "note": "row-major dense reference from p4_iq4_blockmajor.json: 0.930 ms"}
    for skip in SKIPS:
        keep = max(1, int(round(nb * (1 - skip))))
        idx = np.arange(keep, dtype=np.int32)
        line_bytes = keep * rows * blk + keep * ((rows * blk) // 64) * 0
        lines = len(np.unique(((np.arange(keep)[:, None] * (rows * blk)
                                + np.arange(rows * blk)[None, :]) // 64).ravel()))
        line_bytes = lines * 64
        row = {"skip": skip, "kept": keep, "line_bytes": int(line_bytes), "ms": {}}
        for n in THREADS:
            best = 1e9
            for _ in range(5):
                evict[::64] = np.uint8(np.random.randint(0, 255))
                t0 = time.perf_counter()
                lib.gemv_blockmajor8_mt(wb.ctypes.data_as(ctypes.c_void_p),
                                        a.ctypes.data_as(ctypes.c_void_p),
                                        out.ctypes.data_as(ctypes.c_void_p),
                                        rows, nb, blk, idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                        keep, n)
                best = min(best, (time.perf_counter() - t0) * 1e3)
            row["ms"][str(n)] = round(best, 4)
            row["gbps"] = row.get("gbps", {})
            row["gbps"][str(n)] = round(line_bytes / (best * 1e-3) / 1e9, 2)
        res["rows_table"].append(row)
        log(f"[threads] skip {skip:.1f} keep {keep}/{nb} " +
            " ".join(f"{n}t={row['ms'][str(n)]:.3f}ms" for n in THREADS))
    import os
    policy = "passive" if os.environ.get("OMP_WAIT_POLICY") == "passive" else "active"
    res["wait_policy"] = policy
    out = RES / f"p10_threads_{policy}.json"
    out.write_text(json.dumps(res, indent=1))
    log("wrote", out.name)


if __name__ == "__main__":
    main()
