#!/usr/bin/env python3
"""Dense-vs-skip benchmark of a real IQ4_NL ffn_down weight on CPU (AVX-512).

CPU ONLY: no torch, no CUDA, no GPU allocation of any kind. The GGUF is opened
read-only and memory-mapped (MAP_PRIVATE); not a single byte is written to it.

Kernel: iq4nl_skip.cpp -> iq4nl_skip.so, built with
  -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma -mf16c -march=native
  -shared -fPIC
Weight bytes are used exactly as stored in the GGUF (rows x nblocks x 18 B).
The matvec inner loop walks a compacted list of active 32-weight block indices,
so inactive blocks are never addressed: no bytes read, no dequant, no FMA.

Per configuration three numbers are measured:
  * matvec hot / cold  - the real kernel, dense vs skipping
  * touch cold         - reads only the active 18-byte blocks, no math, so it
                         isolates pure memory traffic
  * cache-line accounting - modeled 18-byte bytes vs the 64-byte lines those
                         blocks actually occupy (18 B < 64 B line, so line
                         granularity caps the achievable DRAM saving)

Usage: python gguf_skip_bench.py [--layer 16] [--quick]
Writes results/gguf_iq4nl.json.
"""
from __future__ import annotations

import ctypes
import json
import mmap
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gguf_iq4nl import (  # noqa: E402
    GGUFReader, KVALUES_IQ4NL, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMS,
    dequant_iq4nl_np,
)

MODEL = os.environ.get("GGUF_PATH",
         str(Path(__file__).resolve().parents[2] / "models/Qwen3.5-4B-IQ4_NL.gguf"))
SO = HERE / "iq4nl_skip.so"
CPP = HERE / "iq4nl_skip.cpp"
OUT = HERE / "results" / "gguf_iq4nl.json"
LINE = 64
CXXFLAGS = ("-O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma -mf16c "
            "-march=native -shared -fPIC")


# ---------------------------------------------------------------------------
# kernel binding (everything addressed by raw pointer)
# ---------------------------------------------------------------------------
def build_kernel() -> None:
    if SO.exists() and SO.stat().st_mtime > CPP.stat().st_mtime:
        return
    cmd = f"g++ {CXXFLAGS} -o {SO} {CPP}".split()
    print("[build]", " ".join(cmd))
    subprocess.run(cmd, check=True)


class Kernel:
    def __init__(self, path: Path):
        lib = ctypes.CDLL(str(path))
        V, I = ctypes.c_void_p, ctypes.c_int
        lib.iq4nl_matvec.argtypes = [V, V, V, I, I, V, I]
        lib.iq4nl_matvec.restype = None
        lib.iq4nl_matvec_dense.argtypes = [V, V, V, I, I]
        lib.iq4nl_matvec_dense.restype = None
        lib.iq4nl_dequant.argtypes = [V, V, I]
        lib.iq4nl_dequant.restype = None
        lib.stream_read_sum.argtypes = [V, ctypes.c_size_t]
        lib.stream_read_sum.restype = ctypes.c_double
        lib.iq4nl_touch.argtypes = [V, I, I, V, I]
        lib.iq4nl_touch.restype = ctypes.c_double
        self.lib = lib

    def matvec(self, Wp, ap, yp, rows, nblk, active):
        self.lib.iq4nl_matvec(Wp, ap, yp, rows, nblk, active.ctypes.data,
                              active.size)

    def matvec_dense(self, Wp, ap, yp, rows, nblk):
        self.lib.iq4nl_matvec_dense(Wp, ap, yp, rows, nblk)

    def dequant(self, qp, outp, nblk):
        self.lib.iq4nl_dequant(qp, outp, nblk)

    def touch(self, Wp, rows, nblk, active) -> float:
        return self.lib.iq4nl_touch(Wp, rows, nblk, active.ctypes.data,
                                    active.size)

    def stream_read(self, addr: int, nbytes: int) -> float:
        return self.lib.stream_read_sum(addr, nbytes)


# ---------------------------------------------------------------------------
# activation model
# ---------------------------------------------------------------------------
def block_energies(a: np.ndarray, nblk: int) -> np.ndarray:
    return (a.astype(np.float64).reshape(nblk, IQ4_NL_BLOCK_ELEMS) ** 2).sum(1)


def synthetic_activation(nblk: int, seed: int = 7) -> tuple[np.ndarray, dict]:
    """Realistic decode-time activation vector for one ffn_down input.

    Model (as documented in the JSON report):
      1. bulk:       x ~ N(0,1) with p=0.90, N(0,6) with p=0.10  (gaussian-ish
                     core plus heavy tail / outlier channels)
      2. spikes:     1 element in 1024 gets +-12 (attention-sink style)
      3. block gain: every 32-element block is scaled by lognormal(0, 0.9) to
                     reproduce the per-block energy structure real activations
                     show (a minority of 32-wide groups carries most energy)
      4. rescale to std 0.047 = measured hidden-state std of Qwen3.5-0.8B
         (results/p1_activations.pt, 174 tokens x 3584 dims).

    Ground truth for (3): those real activations have kurtosis 130-590 and their
    top 20% of 32-blocks carry 56-80% of the energy.
    """
    rng = np.random.default_rng(seed)
    n = nblk * IQ4_NL_BLOCK_ELEMS
    x = rng.standard_normal(n)
    x[rng.random(n) < 0.10] *= 6.0
    spikes = rng.choice(n, size=max(8, n // 1024), replace=False)
    x[spikes] += 12.0 * rng.choice([-1.0, 1.0], size=spikes.size)
    x = x.reshape(nblk, IQ4_NL_BLOCK_ELEMS)
    x *= rng.lognormal(0.0, 0.9, size=(nblk, 1))
    x = x.ravel()
    x *= 0.047 / x.std()
    a = x.astype(np.float32)
    z = a.astype(np.float64)
    zc = z - z.mean()
    e = block_energies(a, nblk)
    stats = {
        "model": "0.90*N(0,1)+0.10*N(0,6); +-12 spikes on 1/1024 elements; "
                 "per-32-block lognormal(0,0.9) gain; rescaled to std=0.047",
        "n": int(n), "std": float(a.std()), "abs_max": float(np.abs(a).max()),
        "kurtosis": float((zc ** 4).mean() / (zc ** 2).mean() ** 2),
        "n_spikes": int(spikes.size),
        "block_energy_cv": float(e.std() / e.mean()),
        "real_reference": "Qwen3.5-0.8B hidden states (results/p1_activations.pt): "
                          "std 0.037-0.047, kurtosis 130-590, top-20% of blocks "
                          "carry 56-80% of energy",
    }
    return a, stats


def mask_top_energy(energy: np.ndarray, keep: float) -> np.ndarray:
    n_active = max(1, int(round(keep * energy.size)))
    return np.sort(np.argsort(-energy)[:n_active]).astype(np.int32)


def line_bytes_per_row(active: np.ndarray) -> int:
    """Distinct 64-byte cache lines touched by these blocks in one row."""
    starts = 18 * active.astype(np.int64)
    idx = np.concatenate([starts // LINE, (starts + 17) // LINE])
    return int(np.unique(idx).size * LINE)


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------
class Evictor:
    """Cold-cache eviction by capacity (clflushopt was a no-op on this VM)."""

    def __init__(self, k: Kernel, nbytes: int = 96 * 1024 * 1024):
        self.k = k
        self.buf = np.ones(nbytes, dtype=np.uint8)
        self.buf[:] = 1
        self.addr = self.buf.ctypes.data
        self.n = nbytes

    def __call__(self):
        self.k.stream_read(self.addr, self.n)


def time_call(fn, reps: int, evict=None):
    times, out = [], None
    for _ in range(reps):
        if evict is not None:
            evict()
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)      # ms
    return float(np.median(times)), float(np.min(times)), times, out


def cpu_info() -> dict:
    txt = Path("/proc/cpuinfo").read_text()
    model = next((l.split(":", 1)[1].strip() for l in txt.splitlines()
                  if l.startswith("model name")), "")
    flags = next((l.split(":", 1)[1].strip() for l in txt.splitlines()
                  if l.startswith("flags")), "").split()
    mhz = [float(l.split(":")[1]) for l in txt.splitlines()
           if l.startswith("cpu MHz")]
    want = ["avx2", "fma", "avx512f", "avx512bw", "avx512vl", "avx512dq",
            "avx512_vnni", "avx512_bf16", "avx512vbmi", "avx_vnni"]
    return {
        "model": model, "logical_cpus": os.cpu_count(),
        "isa": {k: (k in flags) for k in want},
        "mhz_sample": {"min": min(mhz), "max": max(mhz)} if mhz else None,
        "python": sys.version.split()[0], "numpy": np.__version__,
        "compiler": subprocess.run(["g++", "--version"], capture_output=True,
                                   text=True).stdout.splitlines()[0],
        "cxxflags": CXXFLAGS, "platform": platform.platform(),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv):
    layer = 16
    quick = "--quick" in argv
    stream_only = "--stream-only" in argv
    if "--layer" in argv:
        layer = int(argv[argv.index("--layer") + 1])

    try:                                   # CPU only, single core
        os.sched_setaffinity(0, {4})
    except OSError:
        pass

    build_kernel()
    k = Kernel(SO)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    evict = Evictor(k)

    report: dict = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_used": False, "model_file_written": False,
        "runtime": "single-threaded CPU, pinned to core 4; weights mmap'd "
                   "read-only (MAP_PRIVATE); no torch/CUDA imported; "
                   "timings are best-of-N (min) wall clock",
        "loadavg_at_start": Path("/proc/loadavg").read_text().split()[:3],
    }
    report["cpu"] = cpu_info()

    fh = open(MODEL, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
    file_addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))

    with GGUFReader(MODEL) as g:
        arch = g.kv["general.architecture"]
        n_layers = int(g.kv[f"{arch}.block_count"])
        n_embd = int(g.kv[f"{arch}.embedding_length"])
        n_ff = int(g.kv[f"{arch}.feed_forward_length"])
        data_base = file_addr + g.data_start

        ffn_down = [t for t in g.tensors if t["name"].endswith("ffn_down.weight")]
        ffn_down.sort(key=lambda t: int(t["name"].split(".")[1]))

        if stream_only:            # re-measure just the 32-layer stream
            out = json.loads(OUT.read_text())
            nb = n_ff // IQ4_NL_BLOCK_ELEMS
            a_s, _ = synthetic_activation(nb, seed=11)
            out["stream_all_layers"] = bench_stream(
                g, k, ffn_down, block_energies(a_s, nb), evict, data_base, quick)
            out["stream_remeasured"] = time.strftime("%Y-%m-%d %H:%M:%S")
            out["headline"] = _headline(out)
            OUT.write_text(json.dumps(out, indent=1))
            summarize(out)
            print(f"\nre-measured stream_all_layers, wrote {OUT}")
            return out

        classes: dict[str, dict] = {}
        for t in g.tensors:
            cls = t["name"]
            for i in range(n_layers):
                cls = cls.replace(f"blk.{i}.", "blk.*.")
            d = classes.setdefault(cls, {"types": set(), "count": 0,
                                         "dims": tuple(t["dims"])})
            d["types"].add(t["type"])
            d["count"] += 1
        report["model"] = {
            "path": MODEL, "size_bytes": g.file_size,
            "gguf_version": g.version, "tensor_count": g.tensor_count,
            "kv_count": g.kv_count, "alignment": g.alignment,
            "data_start": g.data_start, "arch": arch,
            "name": g.kv.get("general.name"), "n_layers": n_layers,
            "n_embd": n_embd, "n_ff": n_ff,
            "quant_types_per_class": {
                c: {"types": sorted(v["types"]), "count": v["count"],
                    "dims": list(v["dims"])} for c, v in sorted(classes.items())},
            "tensor_type_histogram": _hist(g.tensors),
        }
        report["ffn_down_tensors"] = [
            {"name": t["name"], "dims": t["dims"], "type": t["type"],
             "n_elems": int(np.prod(t["dims"])),
             "n_bytes": g.tensor_layout(t)["expected_bytes"],
             "n_blocks": g.tensor_layout(t)["n_blocks"],
             "file_offset": g.data_start + t["offset"]}
            for t in ffn_down]

        # ---------- memory hierarchy probe ------------------------------
        t_hot13, _, _, _ = time_call(
            lambda: k.stream_read(data_base + 1024 * 1024, 13 * 1024 * 1024),
            5)
        big = np.ones(384 * 1024 * 1024, dtype=np.uint8)
        big[:] = 2
        t_dram, _, _, _ = time_call(
            lambda: k.stream_read(big.ctypes.data, big.nbytes), 3, evict)
        big = None
        report["memory_hierarchy"] = {
            "l3_read_gbs_13MB_hot": 13 / t_hot13,
            "dram_read_gbs_384MB_cold": 384 / t_dram,
            "note": "cold = pages evicted by reading a 96 MB scratch buffer "
                    "(CLFLUSHOPT measured as a no-op on this VM)",
        }

        # ---------- layer layout verification ---------------------------
        tinfo = [t for t in ffn_down
                 if t["name"] == f"blk.{layer}.ffn_down.weight"][0]
        lay = g.tensor_layout(tinfo)
        rows = int(tinfo["dims"][1])
        nblk = int(tinfo["dims"][0]) // IQ4_NL_BLOCK_ELEMS
        nbytes = lay["expected_bytes"]
        Wp = data_base + tinfo["offset"]
        assert nbytes == rows * nblk * IQ4_NL_BLOCK_BYTES
        Wraw = np.frombuffer(mm, dtype=np.uint8, count=nbytes,
                             offset=g.data_start + tinfo["offset"])
        report["iq4nl_layout"] = layout_verification(g, tinfo, lay, Wraw, rows,
                                                    nblk, k, Wp)

        # ---------- activation ------------------------------------------
        a, astats = synthetic_activation(nblk)
        energy = block_energies(a, nblk)
        report["activation"] = astats
        report["activation"].update({
            "layer": tinfo["name"], "in_features": nblk * 32,
            "out_features": rows, "total_energy": float(energy.sum()),
            "energy_capture_by_keep": {
                f"{keep:.1f}": float(np.sort(energy)[::-1][
                    :int(round(keep * nblk))].sum() / energy.sum())
                for keep in (0.8, 0.6, 0.4, 0.2)},
        })

        # ---------- kernel validation ------------------------------------
        y_dense = np.zeros(rows, dtype=np.float32)
        k.matvec_dense(Wp, a.ctypes.data, y_dense.ctypes.data, rows, nblk)
        y_idx = np.zeros(rows, dtype=np.float32)
        k.matvec(Wp, a.ctypes.data, y_idx.ctypes.data, rows, nblk,
                 np.arange(nblk, dtype=np.int32))
        report["kernel_validation"] = {
            "dense_vs_indexed_max_abs": float(np.abs(y_dense - y_idx).max()),
            "dense_vs_numpy_reference": _numpy_reference(k, Wraw, Wp, a, rows,
                                                         nblk),
            "y_dense_abs_max": float(np.abs(y_dense).max()),
            "y_dense_rms": float(np.sqrt((y_dense.astype(np.float64) ** 2).mean())),
        }

        # ---------- benchmark: one layer --------------------------------
        reps_hot = 7 if quick else 15
        reps_cold = 5 if quick else 9
        dense_bytes = rows * nblk * IQ4_NL_BLOCK_BYTES
        line_bytes_dense = rows * line_bytes_per_row(np.arange(nblk))
        t_dense_hot = _med(lambda: k.matvec_dense(Wp, a.ctypes.data,
                                                 y_dense.ctypes.data, rows, nblk),
                           reps_hot)
        t_dense_cold = _med(lambda: k.matvec_dense(Wp, a.ctypes.data,
                                                  y_dense.ctypes.data, rows, nblk),
                            reps_cold, evict)
        t_dense_touch = _med(lambda: k.touch(Wp, rows, nblk,
                                             np.arange(nblk, dtype=np.int32)),
                             reps_cold, evict)

        # Interleave the configurations and repeat the whole sweep `rounds`
        # times, keeping the minimum per cell: the box is shared (loadavg ~3.5)
        # and background load both drifts and inflates timings.
        skips = (0.0, 0.2, 0.4, 0.6, 0.8)
        rounds = 1 if quick else 3
        masks = {s: mask_top_energy(energy, 1.0 - s) for s in skips}
        conts = {s: np.arange(max(1, int(round((1.0 - s) * nblk))),
                              dtype=np.int32) for s in skips}
        # random mask with the same block count: isolates "scattered" from
        # "energy-selected", so the scatter penalty can be attributed.
        _rng = np.random.default_rng(1234)
        rnds = {s: np.sort(_rng.permutation(nblk)[: masks[s].size]).astype(np.int32)
                for s in skips}
        best: dict = {s: {} for s in skips}
        y_hot_out: dict = {}
        y_cold_out: dict = {}
        for rnd in range(rounds):
            for skip in skips:
                active = masks[skip]
                y = np.zeros(rows, dtype=np.float32)
                b = best[skip]
                b["hot"] = min(b.get("hot", 1e9), _med(
                    lambda: k.matvec(Wp, a.ctypes.data, y.ctypes.data, rows,
                                     nblk, active), reps_hot))
                if rnd == 0:
                    y_hot_out[skip] = y.copy()
                b["cold"] = min(b.get("cold", 1e9), _med(
                    lambda: k.matvec(Wp, a.ctypes.data, y.ctypes.data, rows,
                                     nblk, active), reps_cold, evict))
                y_cold_out[skip] = y.copy()
                b["touch"] = min(b.get("touch", 1e9), _med(
                    lambda: k.touch(Wp, rows, nblk, active), reps_cold, evict))
                cont = conts[skip]
                b["cont"] = min(b.get("cont", 1e9), _med(
                    lambda: k.matvec(Wp, a.ctypes.data, y.ctypes.data, rows,
                                     nblk, cont), reps_cold, evict))
                b["cont_touch"] = min(b.get("cont_touch", 1e9), _med(
                    lambda: k.touch(Wp, rows, nblk, cont), reps_cold, evict))
                rmask = rnds[skip]
                b["rnd_cold"] = min(b.get("rnd_cold", 1e9), _med(
                    lambda: k.matvec(Wp, a.ctypes.data, y.ctypes.data, rows,
                                     nblk, rmask), reps_cold, evict))
                b["rnd_touch"] = min(b.get("rnd_touch", 1e9), _med(
                    lambda: k.touch(Wp, rows, nblk, rmask), reps_cold, evict))

        table, aligned = [], []
        rnd_tbl: list = []
        for skip in skips:
            keep = 1.0 - skip
            active = masks[skip]
            nact = int(active.size)
            modeled = nact * IQ4_NL_BLOCK_BYTES * rows
            lines = rows * line_bytes_per_row(active)
            t_hot = best[skip]["hot"]
            t_cold = best[skip]["cold"]
            t_touch = best[skip]["touch"]
            y_cold = y_cold_out[skip]
            err = y_cold.astype(np.float64) - y_dense.astype(np.float64)
            table.append({
                "skip_rate": skip, "keep_rate": round(keep, 6),
                "n_active_blocks": nact, "n_blocks": nblk,
                "modeled_bytes": int(modeled),
                "cache_line_bytes": int(lines),
                "modeled_byte_ratio": modeled / dense_bytes,
                "cache_line_byte_ratio": lines / line_bytes_dense,
                "lines_touched_per_row": line_bytes_per_row(active) // LINE,
                "ms_per_token_hot": t_hot,
                "ms_per_token_cold": t_cold,
                "gbs_modeled_hot": modeled / (t_hot * 1e6),
                "gbs_modeled_cold": modeled / (t_cold * 1e6),
                "gbs_cacheline_cold": lines / (t_cold * 1e6),
                "ms_touch_cold": t_touch,
                "gbs_touch_modeled": modeled / (t_touch * 1e6),
                "gbs_touch_cacheline": lines / (t_touch * 1e6),
                "hot_over_cold": t_hot / t_cold,
                "max_abs_err": float(np.abs(err).max()),
                "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(y_dense)),
                "energy_captured": float(energy[active].sum() / energy.sum()),
                "hot_cold_outputs_identical": bool(
                    np.array_equal(y_hot_out[skip], y_cold)),
            })

            # control: contiguous (line-friendly) mask, not activation-driven;
            # used only to check that time follows bytes when the kept blocks
            # are packed contiguously instead of scattered.
            cont = conts[skip]
            clines = rows * line_bytes_per_row(cont)
            t_cont = best[skip]["cont"]
            aligned.append({
                "keep_rate": round(keep, 6), "n_active_blocks": int(cont.size),
                "modeled_bytes": int(cont.size * 18 * rows),
                "cache_line_bytes": int(clines),
                "modeled_byte_ratio": cont.size / nblk,
                "cache_line_byte_ratio": clines / line_bytes_dense,
                "ms_per_token_cold": t_cont,
                "gbs_modeled_cold": cont.size * 18 * rows / (t_cont * 1e6),
                "gbs_cacheline_cold": clines / (t_cont * 1e6),
                "ms_touch_cold": best[skip]["cont_touch"],
                "gbs_touch_modeled": (cont.size * 18 * rows
                                      / (best[skip]["cont_touch"] * 1e6)),
            })
            rmask = rnds[skip]
            rlines = rows * line_bytes_per_row(rmask)
            t_rnd = best[skip]["rnd_cold"]
            rnd_tbl.append({
                "keep_rate": round(keep, 6), "n_active_blocks": int(rmask.size),
                "random_mask_seed": 1234,
                "modeled_bytes": int(rmask.size * 18 * rows),
                "cache_line_bytes": int(rlines),
                "cache_line_byte_ratio": rlines / line_bytes_dense,
                "ms_per_token_cold": t_rnd,
                "ms_touch_cold": best[skip]["rnd_touch"],
                "gbs_modeled_cold": rmask.size * 18 * rows / (t_rnd * 1e6),
                "gbs_touch_modeled": (rmask.size * 18 * rows
                                      / (best[skip]["rnd_touch"] * 1e6)),
            })

        base_hot = table[0]["ms_per_token_hot"]
        base_cold = table[0]["ms_per_token_cold"]
        base_touch = table[0]["ms_touch_cold"]
        base_cont = aligned[0]["ms_per_token_cold"]
        for r in table:
            r["speedup_hot"] = base_hot / r["ms_per_token_hot"]
            r["speedup_cold"] = base_cold / r["ms_per_token_cold"]
            r["touch_speedup"] = base_touch / r["ms_touch_cold"]
        for r in aligned:
            r["speedup_cold"] = base_cont / r["ms_per_token_cold"]
        for r in rnd_tbl:
            r["speedup_cold"] = base_cont / r["ms_per_token_cold"]
            r["touch_ratio"] = r["ms_touch_cold"] / table[0]["ms_touch_cold"]

        dense_specialized = {
            "ms_per_token_hot": t_dense_hot,
            "ms_per_token_cold": t_dense_cold,
            "ms_touch_cold": t_dense_touch,
            "gbs_hot": dense_bytes / (t_dense_hot * 1e6),
            "gbs_cold": dense_bytes / (t_dense_cold * 1e6),
            "gbs_touch_cold": dense_bytes / (t_dense_touch * 1e6),
        }
        report["benchmark"] = {
            "layer": tinfo["name"], "layer_index": layer,
            "note": ("per-token = one ffn_down matvec for a single decode token "
                     f"(in={nblk * 32}, out={rows}). hot = weights resident in "
                     "L3 (13.3 MB < 32 MB); cold = L3 evicted by reading a "
                     "96 MB scratch buffer between reps (untimed). modeled_bytes"
                     " = n_active*18*rows (the task's definition); cache_line_"
                     "bytes = the 64-byte lines those 18-byte blocks actually "
                     "occupy. 'touch' = traffic-only kernel (no dequant, no FMA). "
                     f"Each cell is the best sample over {rounds} interleaved "
                     f"rounds x {reps_cold} reps (cold) / {reps_hot} reps (hot)."),
            "rows": rows, "in_features": nblk * 32, "n_blocks": nblk,
            "dense_bytes": int(dense_bytes),
            "dense_cache_line_bytes": int(line_bytes_dense),
            "dense_specialized": dense_specialized,
            "table": table,
            "contiguous_mask_control": aligned,
            "random_mask_control": rnd_tbl,
            "interpretation": _interpretation(table, aligned, rnd_tbl),
        }

        report["stream_all_layers"] = bench_stream(g, k, ffn_down, energy, evict,
                                                  data_base, quick)
        report["iq4nl_layout"]["scale_stats"] = scale_stats(g, mm, ffn_down)
        report["headline"] = _headline(report)

    OUT.write_text(json.dumps(report, indent=1))
    try:                       # numpy views still reference the mmap buffer
        mm.close()
    except BufferError:
        pass
    fh.close()
    summarize(report)
    print(f"\nwrote {OUT}")
    return report


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _med(fn, reps, evict=None):
    """Best-of-N wall time in ms. Contention on this shared box (loadavg ~3.5)
    only ever adds time, so the minimum is the honest estimate of the kernel."""
    return time_call(fn, reps, evict)[1]


def _headline(report: dict) -> dict:
    """One-screen distillation computed from the report itself."""
    b = report["benchmark"]
    row = {t["skip_rate"]: t for t in b["table"]}
    cont = {round(t["keep_rate"], 1): t for t in b["contiguous_mask_control"]}
    st = {t["skip_rate"]: t for t in report["stream_all_layers"]["table"]}
    d, s = row[0.0], row[0.8]
    return {
        "layer": b["layer"], "skip_rate": 0.8,
        "bytes_addressed_ratio": round(s["modeled_byte_ratio"], 4),
        "cache_lines_fetched_ratio": round(s["cache_line_byte_ratio"], 4),
        "measured_traffic_time_ratio_touch": round(
            s["ms_touch_cold"] / d["ms_touch_cold"], 3),
        "ms_per_token_cold": {"dense": round(d["ms_per_token_cold"], 3),
                              "skip80": round(s["ms_per_token_cold"], 3)},
        "speedup_cold": round(s["speedup_cold"], 2),
        "ms_per_token_hot": {"dense": round(d["ms_per_token_hot"], 3),
                             "skip80": round(s["ms_per_token_hot"], 3)},
        "speedup_hot": round(s["speedup_hot"], 2),
        "gbs_modeled_cold_dense": round(d["gbs_modeled_cold"], 2),
        "gbs_modeled_cold_skip80": round(s["gbs_modeled_cold"], 2),
        "gbs_cacheline_cold_skip80": round(s["gbs_cacheline_cold"], 2),
        "contiguous_packing_20pct_keep_speedup": round(
            cont[0.2]["speedup_cold"], 2),
        "stream32layers_ms_per_token": {
            "dense": round(st[0.0]["ms_per_token"], 2),
            "skip80": round(st[0.8]["ms_per_token"], 2),
            "speedup": round(st[0.8]["speedup"], 2)},
    }


def _interpretation(table, aligned, rnd_tbl) -> dict:
    """Narrative built from the measured numbers (no hard-coded timings)."""
    row = {t["skip_rate"]: t for t in table}
    cont = {round(t["keep_rate"], 1): t for t in aligned}
    rnd = {round(t["keep_rate"], 1): t for t in rnd_tbl}
    d, s80 = row[0.0], row[0.8]
    tr = lambda r: r["ms_touch_cold"] / d["ms_touch_cold"]  # noqa: E731
    return {
        "traffic_granularity": (
            "18 B blocks are smaller than a 64 B cache line (288 blocks x 18 B "
            "= 5184 B = exactly 81 lines per row), so a scattered block mask "
            "keeps most lines alive. Measured traffic (touch kernel) sits "
            "between the two bounds: at 80% skip the modeled byte ratio is "
            f"{s80['modeled_byte_ratio']:.2f}x, the 64 B-line ratio is "
            f"{s80['cache_line_byte_ratio']:.2f}x and the measured touch-time "
            f"ratio is {tr(s80):.2f}x (consistent with 32 B sector granularity); "
            f"at 20% skip the touch ratio {tr(row[0.2]):.2f}x matches the byte "
            f"ratio {row[0.2]['modeled_byte_ratio']:.2f}x. The 18 B byte ratio "
            "is the optimistic bound, the 64 B-line ratio the pessimistic one."),
        "scatter_penalty": (
            "Same number of kept blocks, three patterns (cold, best-of-N): "
            f"contiguous 58/288 -> touch {cont[0.2]['ms_touch_cold']:.3f} ms "
            f"({cont[0.2]['gbs_touch_modeled']:.1f} GB/s), random 58/288 -> "
            f"{rnd[0.2]['ms_touch_cold']:.3f} ms "
            f"({rnd[0.2]['gbs_touch_modeled']:.1f} GB/s), top-energy 58/288 -> "
            f"{s80['ms_touch_cold']:.3f} ms "
            f"({s80['gbs_touch_modeled']:.1f} GB/s) vs dense "
            f"{d['ms_touch_cold']:.3f} ms ({d['gbs_touch_modeled']:.1f} GB/s). "
            "A lone 18 B block still costs a full line fill and breaks "
            "sequential prefetch, so the same 58 kept blocks (2.67 MB "
            "addressed either way) give "
            f"{cont[0.2]['speedup_cold']:.2f}x when packed contiguously but "
            f"only {s80['speedup_cold']:.2f}x when they are the scattered "
            "top-energy blocks."),
        "compute": (
            f"The dense kernel is compute-bound: {d['ms_per_token_hot']:.3f} ms "
            f"hot vs {d['ms_touch_cold']:.3f} ms pure-traffic touch for the "
            "same bytes, so skipping removes work as well as traffic. Hot "
            f"(L3-resident) time scales with kept blocks "
            f"({s80['speedup_hot']:.2f}x at 80% skip); cold/DRAM time scales "
            f"with cache lines ({s80['speedup_cold']:.2f}x)."),
        "accuracy": (
            "keep = top-energy 32-blocks of the synthetic activation; rel_l2 "
            f"{row[0.2]['rel_l2']:.3f} at 20% skip "
            f"({row[0.2]['energy_captured']:.3f} of block energy kept) up to "
            f"{s80['rel_l2']:.3f} at 80% skip "
            f"({s80['energy_captured']:.3f} kept), so the 80% row is a "
            "traffic/compute bound, not a usable quality point."),
    }


def _hist(tensors) -> dict:
    h: dict[str, int] = {}
    for t in tensors:
        h[t["type"]] = h.get(t["type"], 0) + 1
    return h


def _numpy_reference(k, Wraw, Wp, a, rows, nblk, n_check=8) -> dict:
    blocks = Wraw[: n_check * nblk * 18].reshape(n_check, nblk, 18)
    Wd = dequant_iq4nl_np(np.ascontiguousarray(blocks.reshape(-1, 18)))
    ref = Wd.reshape(n_check, -1).astype(np.float64) @ a.astype(np.float64)
    y = np.zeros(rows, dtype=np.float32)
    k.matvec_dense(Wp, a.ctypes.data, y.ctypes.data, rows, nblk)
    err = np.abs(y[:n_check].astype(np.float64) - ref)
    return {"rows_checked": n_check, "max_abs_err": float(err.max()),
            "rel_l2": float(np.linalg.norm(y[:n_check] - ref) /
                            np.linalg.norm(ref)),
            "ref_abs_max": float(np.abs(ref).max())}


def layout_verification(g, tinfo, lay, Wraw, rows, nblk, k, Wp) -> dict:
    nb_tot = Wraw.size // IQ4_NL_BLOCK_BYTES
    d = np.frombuffer(np.ascontiguousarray(
        Wraw[: nb_tot * 18].reshape(-1, 18)[:, :2]).tobytes(),
        dtype="<f2").astype(np.float64)
    kv = np.asarray(KVALUES_IQ4NL, dtype=np.float64)
    sample = np.ascontiguousarray(
        Wraw[: min(nb_tot, 200000) * 18]).reshape(-1, 18)
    lo = (sample[:, 2:] & 0x0F).astype(np.int64)
    hi = (sample[:, 2:] >> 4).astype(np.int64)
    ds = d[: sample.shape[0]]
    vals = np.concatenate([kv[lo] * ds[:, None], kv[hi] * ds[:, None]], axis=1)

    nchk = 64
    out = np.zeros(nchk * 32, dtype=np.float32)
    k.dequant(Wp, out.ctypes.data, nchk)
    ref = dequant_iq4nl_np(Wraw[: nchk * 18].ravel()).ravel()
    ratio = vals / ds[:, None]
    dist = np.abs(ratio[..., None] - kv[None, None, :]).min(axis=2)

    return {
        "tensor": tinfo["name"], "ggml_type_id": tinfo["type_id"],
        "ggml_type": tinfo["type"], "dims_in_out": tinfo["dims"],
        "in_features": int(tinfo["dims"][0]),
        "out_features": int(tinfo["dims"][1]),
        "n_elems": lay["n_elems"], "n_bytes": lay["expected_bytes"],
        "bytes_exact_in_file": lay["expected_bytes"] == Wraw.size,
        "block_elems": IQ4_NL_BLOCK_ELEMS, "block_bytes": IQ4_NL_BLOCK_BYTES,
        "n_blocks": lay["n_blocks"], "n_blocks_per_row": nblk,
        "bytes_per_row": nblk * IQ4_NL_BLOCK_BYTES,
        "bytes_per_row_is_64B_multiple": (nblk * 18) % LINE == 0,
        "row_major_inner_dim_contiguous": True,
        "nibble_layout": "element i (0..15) = low nibble of byte i; "
                         "element 16+i = high nibble of byte i",
        "bits_per_weight": 4.5,
        "dequant_formula": "value = d * kvalues_iq4nl[nibble]",
        "cpp_decode_vs_numpy_max_abs": float(np.abs(out - ref).max()),
        "decoded_values_are_d_times_table_entry": bool(dist.max() == 0.0),
        "max_dist_to_table_entry": float(dist.max()),
        "max_abs_value_over_d_times_127": float(
            (np.abs(vals).max(axis=1) / (np.abs(ds) * 127.0)).max()),
        "frac_blocks_all_values_within_table_range": float(
            (np.abs(vals).max(axis=1) <= np.abs(ds) * 127.0 + 1e-12).mean()),
        "scale_sign_positive_frac": float((ds > 0).mean()),
        "scale_abs_median": float(np.median(np.abs(ds))),
        "scale_abs_min": float(np.abs(ds).min()),
        "scale_abs_max": float(np.abs(ds).max()),
        "note_scale_sign": (
            "GGML IQ4_NL stores one fp16 scale per 32-weight block and dequant "
            "is a plain multiply, so d<0 is legal (the reference quantizer "
            "derives d from a signed block extreme). |d| is tightly bounded, no "
            "NaN/Inf occurs and every |value| <= |d|*127, which confirms the "
            "18-byte block alignment."),
    }


def scale_stats(g, mm, ffn_down) -> dict:
    out = {}
    for t in (ffn_down[:2] + ffn_down[len(ffn_down) // 2:len(ffn_down) // 2 + 1]
              + ffn_down[-1:]):
        nbytes = min(g.tensor_layout(t)["expected_bytes"], 18 * 100000)
        raw = np.frombuffer(mm, dtype=np.uint8, count=nbytes,
                            offset=g.data_start + t["offset"])
        b = np.ascontiguousarray(raw[: raw.size // 18 * 18]).reshape(-1, 18)
        d = np.frombuffer(np.ascontiguousarray(b[:, :2]).tobytes(),
                          dtype="<f2").astype(np.float64)
        out[t["name"]] = {"blocks_sampled": int(d.size),
                          "abs_median": float(np.median(np.abs(d))),
                          "abs_max": float(np.abs(d).max()),
                          "frac_negative": float((d < 0).mean())}
    return out


def bench_stream(g, k, ffn_down, energy, evict, data_base, quick) -> dict:
    """One decode token through all 32 ffn_down layers (425 MB > L3)."""
    n_layers = len(ffn_down)
    total = sum(g.tensor_layout(t)["expected_bytes"] for t in ffn_down)
    ptrs = [(data_base + t["offset"], g.tensor_layout(t)["expected_bytes"])
            for t in ffn_down]
    rows = int(ffn_down[0]["dims"][1])
    nblk = int(ffn_down[0]["dims"][0]) // IQ4_NL_BLOCK_ELEMS
    a, _ = synthetic_activation(nblk, seed=11)
    y = np.zeros(rows, dtype=np.float32)
    reps = 3 if quick else 5
    rounds = 1 if quick else 3

    def run(active, touch=False):
        for p, _n in ptrs:
            if touch:
                k.touch(p, rows, nblk, active)
            else:
                k.matvec(p, a.ctypes.data, y.ctypes.data, rows, nblk, active)

    skips = (0.0, 0.2, 0.4, 0.6, 0.8)
    masks = {s: mask_top_energy(energy, 1.0 - s) for s in skips}
    best = {s: {} for s in skips}
    for _rnd in range(rounds):            # interleave, keep the best per cell
        for skip in skips:
            active = masks[skip]
            b = best[skip]
            b["matvec"] = min(b.get("matvec", 1e9),
                              _med(lambda: run(active), reps, evict))
            b["touch"] = min(b.get("touch", 1e9),
                             _med(lambda: run(active, True), reps, evict))

    table = []
    for skip in skips:
        active = masks[skip]
        modeled = int(active.size * 18 * rows * n_layers)
        lines = int(line_bytes_per_row(active) * rows * n_layers)
        t = best[skip]["matvec"]
        t_touch = best[skip]["touch"]
        table.append({
            "skip_rate": skip, "n_active_blocks": int(active.size),
            "modeled_bytes_per_token": modeled,
            "cache_line_bytes_per_token": lines,
            "ms_per_token": t, "ms_touch": t_touch,
            "gbs_modeled": modeled / (t * 1e6),
            "gbs_cacheline": lines / (t * 1e6),
            "gbs_touch_modeled": modeled / (t_touch * 1e6),
            "gbs_touch_cacheline": lines / (t_touch * 1e6),
        })
    base, base_touch = table[0]["ms_per_token"], table[0]["ms_touch"]
    for r in table:
        r["speedup"] = base / r["ms_per_token"]
        r["touch_speedup"] = base_touch / r["ms_touch"]
        r["modeled_byte_ratio"] = (r["modeled_bytes_per_token"]
                                   / table[0]["modeled_bytes_per_token"])
        r["cache_line_byte_ratio"] = (r["cache_line_bytes_per_token"]
                                      / table[0]["cache_line_bytes_per_token"])
    return {"layers": n_layers, "dense_bytes_per_token": int(total),
            "rounds": rounds, "reps": reps,
            "rows": rows, "n_blocks": nblk, "reps": reps,
            "note": "all 32 ffn_down tensors (mmap, no copy) streamed from DRAM "
                    "per token, L3 evicted by capacity between reps",
            "table": table}


def summarize(rep: dict) -> None:
    b = rep["benchmark"]
    mh = rep["memory_hierarchy"]
    print(f"\n=== IQ4_NL 32-block skipping: {b['layer']} "
          f"({b['rows']} x {b['in_features']}, {b['n_blocks']} blocks/row) ===")
    print(f"memory: L3 read {mh['l3_read_gbs_13MB_hot']:.0f} GB/s, "
          f"DRAM read {mh['dram_read_gbs_384MB_cold']:.1f} GB/s (single core)")
    print(f"{'skip':>5} {'blk':>4} {'MB':>6} {'MBline':>7} {'ms hot':>7} "
          f"{'ms cold':>8} {'GB/s':>6} {'lnGB/s':>7} {'spd':>5} | "
          f"{'touch':>6} {'tSpd':>5} | {'max|e|':>7} {'relL2':>7}")
    for r in b["table"]:
        print(f"{r['skip_rate']:5.1f} {r['n_active_blocks']:4d} "
              f"{r['modeled_bytes']/1e6:6.2f} {r['cache_line_bytes']/1e6:7.2f} "
              f"{r['ms_per_token_hot']:7.3f} {r['ms_per_token_cold']:8.3f} "
              f"{r['gbs_modeled_cold']:6.2f} {r['gbs_cacheline_cold']:7.2f} "
              f"{r['speedup_cold']:5.2f} | {r['ms_touch_cold']:6.3f} "
              f"{r['touch_speedup']:5.2f} | {r['max_abs_err']:7.4f} "
              f"{r['rel_l2']:7.4f}")
    print("\ncontrols (cold, same n_active_blocks as the row above):")
    print(f"  {'pattern':>8} {'keep':>5} {'MB':>6} {'MBline':>7} {'ms':>7} "
          f"{'touch ms':>9} {'touch GB/s':>11} {'spd':>5}")
    for label, pats in (("contig", b["contiguous_mask_control"]),
                        ("random", b["random_mask_control"])):
        for r in pats:
            print(f"  {label:>8} {r['keep_rate']:5.1f} "
                  f"{r['modeled_bytes']/1e6:6.2f} "
                  f"{r['cache_line_bytes']/1e6:7.2f} "
                  f"{r['ms_per_token_cold']:7.3f} {r['ms_touch_cold']:9.3f} "
                  f"{r['gbs_touch_modeled']:11.2f} {r['speedup_cold']:5.2f}")
    s = rep.get("stream_all_layers", {})
    if not s.get("skipped"):
        print(f"\n=== all {s['layers']} ffn_down layers per token "
              f"({s['dense_bytes_per_token']/1e6:.0f} MB) ===")
        print(f"{'skip':>5} {'MB':>7} {'ms/token':>9} {'GB/s':>6} {'speedup':>8}"
              f" | {'touch':>7} {'GB/s':>6} {'spd':>5}")
        for r in s["table"]:
            print(f"{r['skip_rate']:5.1f} "
                  f"{r['modeled_bytes_per_token']/1e6:7.2f} "
                  f"{r['ms_per_token']:9.2f} {r['gbs_modeled']:6.2f} "
                  f"{r['speedup']:8.2f} | {r['ms_touch']:7.2f} "
                  f"{r['gbs_touch_modeled']:6.2f} {r['touch_speedup']:5.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
