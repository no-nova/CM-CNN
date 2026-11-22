"""Super-block (cache-line granularity) gating for the real IQ4_NL ffn_down.

An 18-byte IQ4_NL block lives inside a 64-byte cache line, so skipping single
blocks does not reduce DRAM traffic until few enough blocks survive to drop
whole lines.  Grouping G consecutive blocks (G=2 -> 36 B, G=4 -> 72 B) and
skipping whole groups restores line-granular savings; this script measures the
cold-cache cost of both schemes on blk.16 and on all 32 layers.
"""
import ctypes, json, mmap, sys, time
from pathlib import Path
import numpy as np

import gguf_skip_bench as B

HERE = Path(__file__).parent
OUT = HERE / "results/gguf_groups.json"
LINE = 64
BLK = B.IQ4_NL_BLOCK_BYTES          # 18 B
G_SKIPS = (0.2, 0.4, 0.6, 0.8)


def group_select(energy, G, skip):
    """Pick whole groups of G blocks; returns sorted active block ids."""
    nb = energy.size
    ng = nb // G
    ge = energy[:ng * G].reshape(ng, G).sum(1)
    n_keep_groups = max(1, int(round((1.0 - skip) * ng)))
    keep = np.argsort(-ge)[:n_keep_groups]
    blocks = (keep[:, None] * G + np.arange(G)[None, :]).ravel()
    return np.sort(blocks).astype(np.int32), ge


def line_bytes(active):
    """Distinct 64-byte lines touched by these 18-byte blocks in one row."""
    starts = BLK * active.astype(np.int64)
    idx = np.concatenate([starts // LINE, (starts + BLK - 1) // LINE])
    return int(np.unique(idx).size * LINE)


def timeit(fn, reps, evict=None, warm=2):
    for _ in range(warm):
        fn()
    best = float("inf")
    for _ in range(reps):
        if evict is not None:
            evict()
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def main():
    try:
        import os
        os.sched_setaffinity(0, {4})
    except OSError:
        pass
    B.build_kernel()
    k = B.Kernel(B.SO)
    ev = B.Evictor(k)
    fh = open(B.MODEL, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
    base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    out = {"note": ("measurement: single-threaded CPU pinned to core 4, weights "
                    "mmap'd read-only, cold = 96 MB scratch read between reps "
                    "(untimed), best of 5 reps; G=1 is the block-granular "
                    "scheme, G=2/G=4 skip whole 36 B / 72 B groups. "
                    "lines* = distinct 64-byte lines touched, i.e. real DRAM "
                    "traffic under a line-granular memory system."),
           "line_bytes": 64, "block_bytes": BLK}

    with B.GGUFReader(B.MODEL) as g:
        arch = g.kv["general.architecture"]
        n_layers = int(g.kv[f"{arch}.block_count"])
        n_ff = int(g.kv[f"{arch}.feed_forward_length"])
        nblk_all = n_ff // B.IQ4_NL_BLOCK_ELEMS
        ffn = [t for t in g.tensors if t["name"].endswith("ffn_down.weight")]
        ffn.sort(key=lambda t: int(t["name"].split(".")[1]))
        tinfo = [t for t in ffn if t["name"] == "blk.16.ffn_down.weight"][0]
        rows = int(tinfo["dims"][1])
        nblk = int(tinfo["dims"][0]) // B.IQ4_NL_BLOCK_ELEMS
        Wp = base + g.data_start + tinfo["offset"]
        a, astats = B.synthetic_activation(nblk)
        energy = B.block_energies(a, nblk)
        y = np.zeros(rows, dtype=np.float32)
        out["activation"] = astats

        dense_cold = timeit(lambda: k.matvec_dense(Wp, a.ctypes.data,
                                                   y.ctypes.data, rows, nblk),
                            5, ev)
        dense_hot = timeit(lambda: k.matvec_dense(Wp, a.ctypes.data,
                                                  y.ctypes.data, rows, nblk), 9)
        out["single_layer"] = {
            "layer": "blk.16.ffn_down.weight", "rows": rows, "blocks": nblk,
            "dense_ms_cold": dense_cold, "dense_ms_hot": dense_hot,
            "dense_bytes": rows * nblk * BLK, "rows_table": []}
        for G in (1, 2, 4):
            for skip in G_SKIPS:
                act, ge = group_select(energy, G, skip)
                nb_ = act.size
                ngrp = max(1, nblk // G)
                nkeep = int(round((1.0 - skip) * ngrp))
                ecap = float(np.sort(ge)[::-1][:nkeep].sum() / ge.sum())
                t_cold = timeit(lambda: k.matvec(Wp, a.ctypes.data, y.ctypes.data,
                                                 rows, nblk, act), 5, ev)
                t_hot = timeit(lambda: k.matvec(Wp, a.ctypes.data, y.ctypes.data,
                                                rows, nblk, act), 9)
                lb = line_bytes(act)
                out["single_layer"]["rows_table"].append({
                    "G": G, "skip": skip, "blocks_active": int(nb_),
                    "modeled_bytes": rows * int(nb_) * BLK,
                    "line_bytes": rows * lb,
                    "line_ratio": rows * lb / (rows * nblk * BLK),
                    "ms_cold": t_cold, "ms_hot": t_hot,
                    "speedup_cold": dense_cold / t_cold,
                    "speedup_hot": dense_hot / t_hot,
                    "energy_captured": ecap,
                    "rel_l2": None})
                print(f"L16 G={G} skip={skip:.2f} cold {t_cold:.3f} ms "
                      f"({dense_cold / t_cold:.2f}x) lines {lb}B/row "
                      f"E={ecap:.4f}", flush=True)

        # ---- all 32 layers streamed from DRAM per token
        stream = {"rows_table": []}
        dense_t = 0.0
        acts = []
        for t in ffn:
            nb = int(t["dims"][0]) // B.IQ4_NL_BLOCK_ELEMS
            aa, _ = B.synthetic_activation(nb, seed=7 + int(t["name"].split(".")[1]))
            acts.append((t, nb, aa, base + g.data_start + t["offset"]))
        yy = np.zeros(rows, dtype=np.float32)
        for t, nb, aa, wp in acts:
            dense_t += timeit(lambda: k.matvec_dense(wp, aa.ctypes.data,
                                                     yy.ctypes.data, rows, nb), 3, ev)
        stream["dense_ms_per_token"] = dense_t
        for G in (1, 2, 4):
            for skip in G_SKIPS:
                tot_t, tot_lines, tot_modeled, ecaps = 0.0, 0, 0, []
                for t, nb, aa, wp in acts:
                    e = B.block_energies(aa, nb)
                    act, ge = group_select(e, G, skip)
                    tot_t += timeit(lambda: k.matvec(wp, aa.ctypes.data,
                                                     yy.ctypes.data, rows, nb, act),
                                    3, ev)
                    lb = line_bytes(act)
                    tot_lines += rows * lb
                    tot_modeled += rows * act.size * BLK
                ng = nb // G
                nk = int(round((1 - skip) * ng))
                ecaps.append(float(np.sort(ge)[::-1][:nk].sum() / ge.sum()))
                stream["rows_table"].append({
                    "G": G, "skip": skip, "ms_per_token": tot_t,
                    "speedup": dense_t / tot_t,
                    "modeled_bytes_per_token": tot_modeled,
                    "line_bytes_per_token": tot_lines,
                    "line_ratio": tot_lines / (dense_t and (rows * nblk_all * BLK * len(acts))),
                    "energy_captured": float(np.mean(ecaps))})
                print(f"all32 G={G} skip={skip:.2f} {tot_t:.1f} ms "
                      f"({dense_t / tot_t:.2f}x) lines {tot_lines/1e6:.1f} MB", flush=True)
        out["stream_all_layers"] = stream

    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
