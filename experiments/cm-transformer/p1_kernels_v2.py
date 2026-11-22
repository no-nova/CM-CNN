"""Step 3: kernel-level and end-to-end speedups for P1 block skipping.

Covers the two things the earlier run got wrong:

  * the warp-per-row CUDA kernel (v4, the 760 GFLOP/s dense-optimal layout) was
    mask-driven, so skipping only saved a branch and never scaled; `down_skip4_list`
    now consumes a per-token compacted active-block index list and scales as
    1/(1 - skip).
  * the AVX-512 kernel had the same problem; `down_skip_avx512_list` fixes it.

Measurements:
  1. micro-benchmark: dense vs list-driven skipping, T=1 (decode) and T=64
     (prefill), real Qwen3.5-0.8B down_proj weights and real activations.
  2. end-to-end: greedy decode ms/token and prefill tok/s with the kernel wired
     into the MLP of the reference model, GPU (CUDA) and CPU (AVX-512).

    python3 p1_kernels_v2.py --modules gpu,cpu
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import LANE, MODEL_DIR, Weights, build_tokenizer, dump_json, log
from model_ref import TextBackbone, block_energy

EXP = Path(__file__).resolve().parent
RES = EXP / "results"
BUILD = RES / "build"
BUILD.mkdir(parents=True, exist_ok=True)
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75]


# --------------------------------------------------------------------------- #
# build helpers
# --------------------------------------------------------------------------- #
def build_cuda():
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    os.environ.setdefault("CUDA_HOME", "/opt/cuda")
    return load(name="lanemlp_cuda_v7", sources=[str(EXP / "lanemlp_cuda.cu")],
                extra_cuda_cflags=["-O3", "--use_fast_math", "-DGLOG_USE_GLOG_EXPORT"],
                extra_cflags=["-DGLOG_USE_GLOG_EXPORT"],
                build_directory=str(BUILD), verbose=False)


def build_avx(tmp=True):
    src = BUILD / ("lanemlp_avx512_v2.so" if tmp else "lanemlp_avx512.so")
    subprocess.run(["g++", "-O3", "-fopenmp", "-mavx512f", "-mavx512bw", "-mfma",
                    "-march=native", "-shared", "-fPIC", "-o", str(src),
                    str(EXP / "lanemlp_avx512.cpp")], check=True)
    lib = ctypes.CDLL(str(src))
    P = ctypes.POINTER(ctypes.c_float)
    I = ctypes.POINTER(ctypes.c_int)
    lib.down_skip_avx512.argtypes = [P, P, ctypes.POINTER(ctypes.c_ubyte), P,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.down_skip_avx512_list.argtypes = [P, P, I, P, ctypes.c_int, ctypes.c_int,
                                          ctypes.c_int, ctypes.c_int]
    lib.down_skip_avx512_auto.argtypes = [P, P, ctypes.POINTER(ctypes.c_ubyte), P,
                                          ctypes.c_int, ctypes.c_int, ctypes.c_int]
    # Defect-2 fix: parallel over output rows (works at T=1 as well).
    lib.down_skip_avx512_auto2.argtypes = [P, P, ctypes.POINTER(ctypes.c_ubyte), P,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int]
    return lib


def idx_from_keep(keep: torch.Tensor):
    """[B] boolean keep -> [K] compacted active block indices (padded with -1)."""
    n = int(keep.sum())
    idx = torch.full((n,), -1, dtype=torch.int32, device=keep.device)
    pos = keep.nonzero().flatten().to(torch.int32)
    idx[:pos.numel()] = pos[:n]
    return idx


def batch_idx(keep2d: torch.Tensor):
    """[T,B] -> [T,K] padded list of active blocks."""
    T = keep2d.shape[0]
    K = int(keep2d.sum(-1).max())
    out = torch.full((T, K), -1, dtype=torch.int32, device=keep2d.device)
    for t in range(T):
        pos = keep2d[t].nonzero().flatten().to(torch.int32)
        out[t, :pos.numel()] = pos
    return out


# --------------------------------------------------------------------------- #
# micro-benchmark
# --------------------------------------------------------------------------- #
def micro(mod_gpu, lib_cpu, layer=11, T_list=(1, 64), rates=RATES):
    tb = TextBackbone(Weights(torch.float32, "cuda" if mod_gpu else "cpu"))
    p = f"{tb.prefix}layers.{layer}.mlp."
    Wd = tb.w[p + "down_proj.weight"].detach().contiguous()     # [H, I]
    Wg, Wu = tb.w[p + "gate_proj.weight"], tb.w[p + "up_proj.weight"]
    ids = torch.tensor(build_tokenizer(MODEL_DIR)(
        "The history of the Roman Empire spans more than a thousand years, and its "
        "influence on law, language and architecture can still be traced today. "
        "In computer science, an algorithm is a finite sequence of well-defined "
        "instructions, typically to solve a class of problems. "
        "A compiler translates source code into machine instructions, and a "
        "kernel is the part of an operating system that talks to the hardware. "
        "Measuring latency requires care: warm caches, pinned clocks and many "
        "repetitions are needed before a number can be trusted.", add_special_tokens=False)["input_ids"])
    A_all = []
    for i in (layer,):
        cap = {}
        tb.forward(ids.to(tb.w.device), capture=cap, logits_positions=[0])
        x = cap["mlp_in"][i][: max(T_list)]
        A_all.append((F.silu(F.linear(x, Wg)) * F.linear(x, Wu)).detach().contiguous())
    A = A_all[0]
    out = {}
    scale = Wd.float().norm(dim=0)
    for T in T_list:
        T = min(T, A.shape[0])
        a = A[:T].contiguous()
        e = block_energy(a * scale)                     # [T, B]
        row = {}
        ref = a @ Wd.t()
        for r in rates:
            keep = torch.zeros_like(e, dtype=torch.bool)
            if r < 1.0:
                nk = max(1, int(round(e.shape[-1] * (1 - r))))
                keep.scatter_(1, e.argsort(-1)[:, :nk], True)
            entry = {"active_blocks": float(keep.sum(-1).float().mean()),
                     "skip_rate": r}
            if mod_gpu:
                idx = batch_idx(keep)
                y_new = mod_gpu.down_skip4_list(a, Wd, idx)
                y_mask = mod_gpu.down_skip4(a, Wd, keep.to(torch.uint8))
                y_v6 = mod_gpu.down_skip_v6(a, Wd, keep.to(torch.uint8))
                err = float((y_new - y_mask).abs().max())
                it = max(4, 2000 // T)
                dense_ms = mod_gpu.bench_down_skip4(a, Wd, torch.ones_like(keep, dtype=torch.uint8), it)
                v4_ms = mod_gpu.bench_down_skip4(a, Wd, keep.to(torch.uint8), it)
                list_ms = mod_gpu.bench_down_skip4_list(a, Wd, idx, it)
                v6_ms = mod_gpu.bench_down_skip_v6(a, Wd, keep.to(torch.uint8), it)
                flop = 2.0 * T * Wd.shape[0] * Wd.shape[1] * (1 - r)
                entry.update({"cuda_dense_ms": dense_ms, "cuda_v4_mask_ms": v4_ms,
                              "cuda_v7_list_ms": list_ms, "cuda_v6_ms": v6_ms,
                              "cuda_v7_speedup": dense_ms / list_ms if list_ms else None,
                              "cuda_v4_speedup": dense_ms / v4_ms if v4_ms else None,
                              "cuda_v7_GFLOPS": flop / (list_ms * 1e-3) / 1e9 if list_ms else None,
                              "list_vs_mask_max_abs": err})
            if lib_cpu is not None:
                ad = a.detach().cpu().contiguous().numpy()
                wd = Wd.detach().cpu().contiguous().numpy()
                m = keep.to(torch.uint8).cpu().numpy()
                idx = batch_idx(keep).cpu().numpy().astype(np.int32)
                o1 = np.zeros((T, Wd.shape[0]), dtype=np.float32)
                o2 = np.zeros((T, Wd.shape[0]), dtype=np.float32)
                lib_cpu.down_skip_avx512(ad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         wd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         m.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                         o1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                         T, Wd.shape[1], Wd.shape[0])
                for t in range(T):
                    lib_cpu.down_skip_avx512_list(
                        ad[t].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        wd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        idx[t].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        o2[t].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        1, Wd.shape[1], Wd.shape[0], int(keep[t].sum()))
                it = max(20, 400 // T)
                ap_ = ad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                wp_ = wd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                mp_ = m.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
                op_ = o1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                fn = lambda: lib_cpu.down_skip_avx512(ap_, wp_, mp_, op_, T,
                                                     Wd.shape[1], Wd.shape[0])
                fn()
                t0 = time.perf_counter()
                for _ in range(it):
                    fn()
                tm = {"mask": (time.perf_counter() - t0) / it * 1e3}
                tk = 0.0
                for t in range(T):
                    k = int(keep[t].sum())
                    t0 = time.perf_counter()
                    for _ in range(it):
                        lib_cpu.down_skip_avx512_list(
                            ad[t].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            wd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            idx[t].ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                            o2[t].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            1, Wd.shape[1], Wd.shape[0], k)
                    tk += (time.perf_counter() - t0) / it * 1e3
                entry.update({"cpu_mask_ms": tm["mask"], "cpu_list_ms": tk,
                              "cpu_list_speedup_vs_mask": tm["mask"] / tk if tk else None,
                              "cpu_list_max_abs_vs_mask": float(np.abs(o2 - o1).max()),
                              "cpu_bytes_MB": float(keep.sum() * LANE * 4 * Wd.shape[0] / 1e6)})
            row[f"{r:.2f}"] = entry
        out[f"T{T}"] = row
    del tb
    if mod_gpu:
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# end-to-end decode
# --------------------------------------------------------------------------- #
class KernelDown:
    """down_proj replacement running the real block-skipping kernel."""

    def __init__(self, gpu_ext=None, cpu_lib=None, quiet=True):
        self.gpu, self.cpu = gpu_ext, cpu_lib
        self.calls = 0
        self.active = 0
        self.total = 0
        self._acc = None            # device-side counter: never syncs per call

    def __call__(self, a, W, keep):
        self.calls += 1
        one = a.dim() == 1
        if self.gpu is not None:
            # keep can be [B] (single token) or [T, B]
            m = keep if not one else keep[None]
            if self._acc is None:
                self._acc = torch.zeros((), device=m.device)
            self._acc += m.sum().to(self._acc.dtype)
            self.total += m.numel()
            out = self.gpu.down_skip4_auto(a[None] if one else a, W,
                                           m.to(torch.uint8))
            return out[0] if one else out
        a2 = a.detach().cpu().contiguous().numpy()
        w = W.detach().cpu().contiguous().numpy()
        k = keep.detach().cpu().reshape(-1, keep.shape[-1]).numpy()
        self.active += int(k.sum())
        self.total += k.size
        o = np.zeros((a2.shape[0], w.shape[0]), dtype=np.float32)
        # Defect-2: row-parallel entry point (works at T=1); the token-parallel
        # one idles 31/32 cores during decode.
        self.cpu.down_skip_avx512_auto2(
            a2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            a2.shape[0], w.shape[1], w.shape[0])
        return torch.from_numpy(o).reshape(a.shape[:-1] + (o.shape[-1],))

    def skip_rate(self):
        if self._acc is not None:
            self.active = int(self._acc.item())
        return self.active / max(self.total, 1)


@torch.no_grad()
def e2e(mod_gpu, lib_cpu, device="cuda", rate=0.2, n_prefill=128, n_decode=24,
        rates=(0.2, 0.4, 0.6, 0.75, 0.85)):
    tb = TextBackbone(Weights(torch.float32, device))
    tok = build_tokenizer(MODEL_DIR)
    prompt = ("The history of the Roman Empire spans more than a thousand years. "
              "Its influence on law, language and architecture can still be traced. " * 4)
    ids = torch.tensor(tok(prompt, add_special_tokens=False)["input_ids"][:n_prefill],
                       device=device)
    scale = {i: tb.w[f"{tb.prefix}layers.{i}.mlp.down_proj.weight"].detach().float()
             .norm(dim=0) for i in range(tb.cfg.layers)}
    from model_ref import new_state

    def run(rate_, kernel, reps=3):
        tb.down_fn = kernel
        st = new_state()
        t0 = time.perf_counter()
        o = tb.forward(ids, skip_rate=rate_, skip_scale=scale, state=st, pos0=0)
        prefill = time.perf_counter() - t0
        cur = int(o["logits"][-1].argmax())
        lat = []
        for k in range(n_decode):
            t1 = time.perf_counter()
            o = tb.forward(torch.tensor([cur], device=device), skip_rate=rate_,
                           skip_scale=scale, state=st, pos0=ids.numel() + k)
            cur = int(o["logits"][-1].argmax())
            lat.append((time.perf_counter() - t1) * 1e3)
        tb.down_fn = None
        return {"prefill_ms": prefill * 1e3,
                "prefill_tok_s": ids.numel() / prefill,
                "decode_ms_per_tok": float(np.mean(lat[2:])),
                "decode_tok_s": 1000.0 / float(np.mean(lat[2:])),
                "decode_ms_std": float(np.std(lat[2:]))}

    out = {}
    run(0.0, None)                       # warm-up: first call pays JIT/cuBLAS init
    base = run(0.0, None)
    out["dense"] = base
    kd0 = KernelDown(gpu_ext=mod_gpu if device == "cuda" else None,
                     cpu_lib=lib_cpu if device != "cuda" else None)
    run(0.0, kd0)                        # warm the kernel path too
    ok = run(0.0, kd0)
    ok["actual_skip_rate"] = kd0.skip_rate()
    out["kernel_dense"] = ok
    for r in rates:
        kd = KernelDown(gpu_ext=mod_gpu if device == "cuda" else None,
                        cpu_lib=lib_cpu if device != "cuda" else None)
        m = run(r, kd)
        m["actual_skip_rate"] = kd.skip_rate()
        m["decode_speedup"] = base["decode_ms_per_tok"] / m["decode_ms_per_tok"]
        m["prefill_speedup"] = base["prefill_ms"] / m["prefill_ms"]
        m["decode_speedup_vs_kernel"] = ok["decode_ms_per_tok"] / m["decode_ms_per_tok"]
        m["prefill_speedup_vs_kernel"] = ok["prefill_ms"] / m["prefill_ms"]
        out[f"skip_{r:.2f}"] = m
        log(f"  rate {r:.2f}: decode {m['decode_ms_per_tok']:.1f} ms/tok "
            f"({m['decode_speedup']:.2f}x) prefill {m['prefill_speedup']:.2f}x "
            f"skip={m['actual_skip_rate']:.3f}")
    del tb
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="gpu,cpu")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()
    mods = args.modules.split(",")
    gpu = build_cuda() if "gpu" in mods else None
    cpu = build_avx() if "cpu" in mods else None
    log("modules built")
    res = {"layer": args.layer, "lane": LANE, "rates": RATES}
    if gpu is not None:
        res["micro"] = micro(gpu, cpu)
        log("micro done")
        res["e2e_gpu"] = e2e(gpu, cpu, device="cuda")
        log("e2e gpu done")
    if cpu is not None:
        res["e2e_cpu"] = e2e(None, cpu, device="cpu", n_prefill=64, n_decode=8)
        log("e2e cpu done")
    dump_json(res, RES / f"p1_kernels_{args.tag}.json")
    log("wrote", RES / f"p1_kernels_{args.tag}.json")


if __name__ == "__main__":
    main()
