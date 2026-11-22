"""Defect-1 fix benchmark: row-major vs block-major IQ4_NL GEMV on blk.16 (and
all 32 ffn_down tensors), cold cache, G=1 block granularity."""
import ctypes, json, mmap, os, subprocess, time
from pathlib import Path
import numpy as np

import gguf_skip_bench as B

HERE = Path(__file__).parent
BUILD = HERE / "results/build"
SO = BUILD / "iq4nl_bm.so"
OUT = HERE / "results/gguf_blockmajor.json"
BLK = B.IQ4_NL_BLOCK_BYTES
KV = B.KVALUES_IQ4NL


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    subprocess.run(["g++", "-O3", "-fopenmp", "-mavx512f", "-mavx512bw", "-mfma",
                    "-mf16c", "-march=native", "-shared", "-fPIC", "-o", str(SO),
                    str(HERE / "iq4nl_blockmajor.cpp")], check=True)
    return ctypes.CDLL(str(SO))


def decompress(w, M, nb):
    W = np.frombuffer(w, dtype=np.uint8)[: M * nb * BLK].reshape(M, nb, BLK)
    d = W[:, :, 0:2].copy().view(np.float16).astype(np.float32)[:, :, 0]
    packed = W[:, :, 2:]
    lo = (packed & 0x0F).astype(np.int64)
    hi = ((packed >> 4) & 0x0F).astype(np.int64)
    q = np.concatenate([lo, hi], axis=2).astype(np.int64)
    tab = np.asarray(KV, dtype=np.float32).reshape(-1)
    return d[:, :, None] * tab[q]


def timeit(fn, reps=5, evict=None, warm=2):
    for _ in range(warm):
        fn()
    best = 1e9
    for _ in range(reps):
        if evict is not None:
            evict()
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def main():
    try:
        os.sched_setaffinity(0, {4})
    except OSError:
        pass
    lib = build()
    P = ctypes.POINTER(ctypes.c_float)
    U = ctypes.POINTER(ctypes.c_ubyte)
    I = ctypes.POINTER(ctypes.c_int)
    lib.repack_blockmajor.argtypes = [U, U, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.gemv_blockmajor.argtypes = [U, P, P, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, I, ctypes.c_int]
    lib.gemv_blockmajor_mt.argtypes = lib.gemv_blockmajor.argtypes
    k = B.Kernel(B.SO)
    ev = B.Evictor(k)
    fh = open(B.MODEL, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
    base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    out = {"layout": "block-major [nb, M*18]", "line_bytes": 64,
           "block_bytes": BLK, "row_major_from": "results/gguf_groups.json",
           "single_layer": {}, "stream_all_layers": {}}
    with B.GGUFReader(B.MODEL) as g:
        arch = g.kv["general.architecture"]
        n_ff = int(g.kv[f"{arch}.feed_forward_length"])
        ffn = [t for t in g.tensors if t["name"].endswith("ffn_down.weight")]
        ffn.sort(key=lambda t: int(t["name"].split(".")[1]))
        t16 = [t for t in ffn if t["name"] == "blk.16.ffn_down.weight"][0]
        M = int(t16["dims"][1]); nb = int(t16["dims"][0]) // B.IQ4_NL_BLOCK_ELEMS
        Wp = base + g.data_start + t16["offset"]
        a, astats = B.synthetic_activation(nb)
        energy = B.block_energies(a, nb)
        # repack once (offline cost, reported for honesty)
        WB = np.zeros(nb * M * BLK, dtype=np.uint8)
        t0 = time.perf_counter()
        lib.repack_blockmajor(ctypes.cast(Wp, U), WB.ctypes.data_as(U), M, nb, BLK)
        repack_ms = (time.perf_counter() - t0) * 1e3
        # numerics: compare full (K=nb) block-major vs numpy dequant
        idx_all = np.arange(nb, dtype=np.int32)
        o = np.zeros(M, dtype=np.float32)
        lib.gemv_blockmajor(WB.ctypes.data_as(U), a.ctypes.data_as(P),
                            o.ctypes.data_as(P), M, nb, BLK,
                            idx_all.ctypes.data_as(I), nb)
        Wd = decompress(mm[g.data_start + t16["offset"]:], M, nb)   # [M, nb, 32]
        ref = np.einsum("kb,mkb->m", a.reshape(nb, 32), Wd).astype(np.float32)
        out["single_layer"] = {
            "layer": "blk.16.ffn_down.weight", "rows": M, "blocks": nb,
            "activation": astats, "repack_ms": repack_ms,
            "max_abs_err_full": float(np.abs(o - ref).max()),
            "rel_l2_full": float(np.linalg.norm(o - ref) / (np.linalg.norm(ref) + 1e-9)),
            "rows_table": []}
        print(f"repack {repack_ms:.1f} ms, max|err| {np.abs(o - ref).max():.2e}", flush=True)
        dense = timeit(lambda: lib.gemv_blockmajor(WB.ctypes.data_as(U),
                                                   a.ctypes.data_as(P),
                                                   o.ctypes.data_as(P), M, nb, BLK,
                                                   idx_all.ctypes.data_as(I), nb),
                       5, ev)
        out["single_layer"]["dense_ms_cold"] = dense
        for skip in (0.2, 0.4, 0.6, 0.8):
            K = max(1, int(round((1 - skip) * nb)))
            idx = np.sort(np.argsort(-energy)[:K]).astype(np.int32)
            t = timeit(lambda: lib.gemv_blockmajor(WB.ctypes.data_as(U),
                                                   a.ctypes.data_as(P),
                                                   o.ctypes.data_as(P), M, nb, BLK,
                                                   idx.ctypes.data_as(I), K), 5, ev)
            lines = K * M * BLK
            ecap = float(np.sort(energy)[::-1][:K].sum() / energy.sum())
            out["single_layer"]["rows_table"].append({
                "G": 1, "skip": skip, "blocks_active": K,
                "line_bytes": lines, "ms_cold": t, "speedup_cold": dense / t,
                "energy_captured": ecap})
            print(f"L16 skip={skip:.2f} K={K} cold {t:.3f} ms ({dense / t:.2f}x) "
                  f"lines {lines/1e6:.2f} MB E={ecap:.4f}", flush=True)
        # all 32 layers
        acts, offs = [], []
        yy = np.zeros(M, dtype=np.float32)
        WBbuf = np.zeros(nb * M * BLK, dtype=np.uint8)
        dense_t, rows = 0.0, []
        packs = []
        for t_ in ffn:
            nb_ = int(t_["dims"][0]) // B.IQ4_NL_BLOCK_ELEMS
            aa, _ = B.synthetic_activation(nb_, seed=7 + int(t_["name"].split(".")[1]))
            buf = np.zeros(nb_ * M * BLK, dtype=np.uint8)
            lib.repack_blockmajor(ctypes.cast(base + g.data_start + t_["offset"], U),
                                  buf.ctypes.data_as(U), M, nb_, BLK)
            packs.append((aa, nb_, buf, B.block_energies(aa, nb_)))
        for aa, nb_, buf, e in packs:
            idx_all_ = np.arange(nb_, dtype=np.int32)
            dense_t += timeit(lambda: lib.gemv_blockmajor(
                buf.ctypes.data_as(U), aa.ctypes.data_as(P), yy.ctypes.data_as(P),
                M, nb_, BLK, idx_all_.ctypes.data_as(I), nb_), 3, ev)
        out["stream_all_layers"]["dense_ms_per_token"] = dense_t
        for skip in (0.2, 0.4, 0.6, 0.8):
            tot, lines, ecaps = 0.0, 0, []
            for aa, nb_, buf, e in packs:
                K = max(1, int(round((1 - skip) * nb_)))
                idx = np.sort(np.argsort(-e)[:K]).astype(np.int32)
                tot += timeit(lambda: lib.gemv_blockmajor(
                    buf.ctypes.data_as(U), aa.ctypes.data_as(P), yy.ctypes.data_as(P),
                    M, nb_, BLK, idx.ctypes.data_as(I), K), 3, ev)
                lines += K * M * BLK
                ecaps.append(float(np.sort(e)[::-1][:K].sum() / e.sum()))
            rows.append({"G": 1, "skip": skip, "ms_per_token": tot,
                         "speedup": dense_t / tot, "line_bytes_per_token": lines,
                         "energy_captured": float(np.mean(ecaps))})
            print(f"all32 skip={skip:.2f} {tot:.1f} ms ({dense_t / tot:.2f}x) "
                  f"lines {lines/1e6:.0f} MB", flush=True)
        out["stream_all_layers"]["rows_table"] = rows
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
