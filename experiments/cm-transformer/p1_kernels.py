"""Step 2 kernels: 32-lane block-skipping down_proj on RTX 5070 (CUDA) and
Zen 5 (AVX-512), measured on the real Qwen3.5-0.8B FFN activations.

The kernel cost depends only on the *skip rate* (how many 32-lane tiles are
dropped), not on which permutation produced the grouping; the grouping decides
fidelity, measured in p1_ffn.py.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import time

import numpy as np
import torch

from common import EXP_DIR, RESULTS_DIR, dump_json, log, LANE
from model_ref import block_energy, keep_mask_from_energy, apply_block_mask

RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]
CUDA_SRC = EXP_DIR / "lanemlp_cuda.cu"
AVX_SRC = EXP_DIR / "lanemlp_avx512.cpp"
BUILD = RESULTS_DIR / "build"


# --------------------------------------------------------------------------- #
def build_cuda():
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("CUDA_HOME", "/opt/cuda")
    BUILD.mkdir(parents=True, exist_ok=True)
    return load(name="lanemlp_cuda", sources=[str(CUDA_SRC)],
                extra_cuda_cflags=["-O3", "-DGLOG_USE_GLOG_EXPORT"],
                extra_cflags=["-DGLOG_USE_GLOG_EXPORT"],
                build_directory=str(BUILD), verbose=False)


def build_avx512():
    BUILD.mkdir(parents=True, exist_ok=True)
    so = BUILD / "liblanemlp_avx512.so"
    src_mtime = AVX_SRC.stat().st_mtime
    if not so.exists() or so.stat().st_mtime < src_mtime:
        cmd = ["g++", "-O3", "-march=native", "-fPIC", "-shared", "-fno-math-errno",
               str(AVX_SRC), "-o", str(so)]
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so))
    f = lib.down_skip_avx512
    f.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_void_p, ctypes.c_void_p,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_int]
    f.restype = None
    return f


def cuda_gpu_time(fn, *args, iters=300):
    return float(fn(*args, iters))


def masks_for(a: torch.Tensor, rate: float):
    """Per-token 32-lane block masks from the post-activation block energy."""
    if rate <= 0:
        return torch.ones(a.shape[0], a.shape[1] // LANE, dtype=torch.uint8, device=a.device)
    e = block_energy(a)
    return keep_mask_from_energy(e, rate).to(torch.uint8)


# --------------------------------------------------------------------------- #
def main():
    torch.manual_seed(0)
    acts = torch.load(RESULTS_DIR / "p1_activations.pt")["eval_acts"]
    from common import Weights
    w = Weights(torch.float32, "cpu")
    down = {i: w[f"model.language_model.layers.{i}.mlp.down_proj.weight"] for i in range(24)}
    layers = sorted(acts)

    out = {"lane": LANE, "n_blocks": 112, "rates": RATES,
           "layers": len(layers), "tokens_per_layer": int(acts[layers[0]].shape[0]),
           "mac_per_token_down_proj": 1024 * 3584,
           "cuda": {}, "avx512": {}}

    # ---------------- CUDA -------------------------------------------------- #
    mod = build_cuda()
    log("CUDA extension ready; kernel variants: down_skip4 (dense-optimal), down_skip_v6 (skip-scalable)")
    a_gpu = {i: acts[i].to("cuda").contiguous() for i in layers}
    w_gpu = {i: down[i].to("cuda").contiguous() for i in layers}

    # correctness on real weights/activations
    a0, w0 = a_gpu[layers[0]], w_gpu[layers[0]]
    m0 = masks_for(a0, 0.5)
    ref = apply_block_mask(a0, m0.bool()) @ w0.T
    got6 = mod.down_skip_v6(a0, w0, m0)
    got4 = mod.down_skip4(a0, w0, m0)
    out["cuda"]["correctness_layer0_rate50"] = {
        "max_abs_v6": float((got6 - ref).abs().max()),
        "max_abs_v4": float((got4 - ref).abs().max()),
        "max_abs_dense_v6": float((mod.down_skip_v6(a0, w0, masks_for(a0, 0.0)) - a0 @ w0.T).abs().max()),
    }
    log("CUDA correctness:", out["cuda"]["correctness_layer0_rate50"])

    for variant in ("v6", "v4"):
        bench = getattr(mod, f"bench_down_skip_{variant}" if variant == "v6" else "bench_down_skip4")
        rows = []
        for rate in RATES:
            tot = {1: 0.0, 4: 0.0}
            for T in (1, 4):
                for i in layers:
                    a = a_gpu[i][:T]
                    m = masks_for(a, rate)
                    tot[T] += cuda_gpu_time(bench, a, w_gpu[i], m)
            rows.append({"rate": rate,
                         "ms_per_token_all24_T1": tot[1] / 24,   # per (token, layer-block) normalised below
                         "sum_ms_all24_layers_T1": tot[1],
                         "sum_ms_all24_layers_T4": tot[4],
                         "total_ms_per_token_all_layers": tot[1]})
        base1 = rows[0]["sum_ms_all24_layers_T1"]
        for r in rows:
            r["speedup_vs_dense"] = base1 / r["sum_ms_all24_layers_T1"]
            r["GFLOPs"] = (24 * 1024 * 3584 * 2) / (r["sum_ms_all24_layers_T1"] * 1e6)
            r["speedup_vs_dense_T4"] = rows[0]["sum_ms_all24_layers_T4"] / r["sum_ms_all24_layers_T4"]
        out["cuda"][variant] = rows
        for r in rows:
            log(f"CUDA {variant} rate={r['rate']:.2f}: 24-layer down_proj "
                f"{r['sum_ms_all24_layers_T1']:.4f} ms/token  "
                f"speedup {r['speedup_vs_dense']:.3f}x  {r['GFLOPs']:.1f} GFLOP/s")
    torch.cuda.empty_cache()

    # ---------------- AVX-512 ----------------------------------------------- #
    f = build_avx512()
    a_cpu = {i: acts[i].contiguous() for i in layers}
    w_cpu = {i: down[i].contiguous() for i in layers}
    out_buf = torch.empty(a_cpu[layers[0]].shape[0], 1024)

    def run_avx(a, w, m, out_buf):
        f(a.data_ptr(), w.data_ptr(), m.data_ptr(), out_buf.data_ptr(),
          a.shape[0], a.shape[1], w.shape[0])

    m0 = masks_for(a_cpu[layers[0]], 0.5)
    out_buf_ = torch.empty(a_cpu[layers[0]].shape[0], 1024)
    run_avx(a_cpu[layers[0]], w_cpu[layers[0]], m0, out_buf_)
    out["avx512"]["correctness_layer0_rate50"] = {
        "max_abs": float((out_buf_ - ref.cpu()).abs().max())}
    log("AVX-512 correctness:", out["avx512"]["correctness_layer0_rate50"])

    rows = []
    for rate in RATES:
        reps = 7
        t0 = time.perf_counter()
        for _ in range(reps):
            for i in layers:
                a = a_cpu[i][:1]
                run_avx(a, w_cpu[i], masks_for(a, rate), out_buf[:1])
        ms1 = (time.perf_counter() - t0) / reps * 1e3
        t0 = time.perf_counter()
        for _ in range(reps):
            for i in layers:
                a = a_cpu[i]
                run_avx(a, w_cpu[i], masks_for(a, rate), out_buf)
        msT = (time.perf_counter() - t0) / reps * 1e3
        rows.append({"rate": rate, "sum_ms_all24_layers_T1": ms1,
                     "sum_ms_all24_layers_Tfull": msT,
                     "tokens_full": int(a_cpu[layers[0]].shape[0])})
    base = rows[0]["sum_ms_all24_layers_T1"]
    for r in rows:
        r["speedup_vs_dense"] = base / r["sum_ms_all24_layers_T1"]
        r["GFLOPs"] = (24 * 1024 * 3584 * 2) / (r["sum_ms_all24_layers_T1"] * 1e6)
        r["speedup_vs_dense_full"] = rows[0]["sum_ms_all24_layers_Tfull"] / r["sum_ms_all24_layers_Tfull"]
    out["avx512"]["sweep"] = rows
    for r in rows:
        log(f"AVX512 rate={r['rate']:.2f}: 24-layer down_proj {r['sum_ms_all24_layers_T1']:.3f} ms/token "
            f"speedup {r['speedup_vs_dense']:.2f}x  {r['GFLOPs']:.1f} GFLOP/s")

    dump_json(out, RESULTS_DIR / "p1_kernels.json")
    log("wrote", RESULTS_DIR / "p1_kernels.json")
    return out


if __name__ == "__main__":
    main()
