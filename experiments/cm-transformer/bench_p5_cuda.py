"""Phase 5, Task 1: CUDA fused SwiGLU with float4 loads and 2 rows/warp.

Measures (a) the operator against cuBLAS, (b) the offline importance-weighted
gate  S_b = w_b * ||g_b||  with  w_b = RMS(u_b) * ||W_down[:, b]||_F  versus the
unweighted gate ||g_b||, both by single-layer rel-L2 and by 24-layer MLP-chain
time.  Writes results/p5_cuda.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("GLOG_minloglevel", "2")

from common import LANE, dump_json  # noqa: E402
from model_ref import TextBackbone  # noqa: E402
from common import Weights  # noqa: E402
from model_ref import new_state  # noqa: E402

MODEL_DIR = HERE.parents[1] / "models" / "Qwen3.5-0.8B"
RESULTS = HERE / "results"
torch.manual_seed(0)


def load_cuda():
    from torch.utils.cpp_extension import load
    flags = ["-DGLOG_USE_GLOG_EXPORT", "-O3", "--use_fast_math"]
    return load(name="swiglu_cuda_p5", sources=[str(HERE / "swiglu_cuda.cu")],
                extra_cuda_cflags=flags, extra_cflags=flags,
                extra_ldflags=["-L/opt/cuda/lib64", "-lcudart"], verbose=False)


def capture_layer(dev, layer=11):
    tb = TextBackbone(Weights(torch.float32, dev))
    st = new_state()
    cap = {}
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=dev)
    with torch.no_grad():
        tb.forward(ids, state=st, capture=cap)
    return tb, cap["mlp_in"][layer].reshape(-1, tb.cfg.hidden)


def block_weights(x, Wg, Wu, Wd):
    """w_b = RMS(u_b) * ||W_down[:, b]||_F, one float per 32-lane block."""
    H = Wu.shape[0]
    with torch.no_grad():
        u = x @ Wu.t()                       # [T, H]
    nb = H // LANE
    ub = u.reshape(-1, nb, LANE)
    rms = (ub.pow(2).mean(-1)).sqrt().mean(0)          # [nb]
    wd = Wd.reshape(Wd.shape[0], nb, LANE)
    nrm = wd.pow(2).sum((0, 2)).sqrt()                 # [nb]
    w = rms * nrm
    return (w / w.mean()).clamp(min=1e-3)


def bench(fn, dev="cuda", warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lib = load_cuda()
    tb, x = capture_layer(dev)
    D, H = tb.cfg.hidden, tb.cfg.inter
    p0 = f"{tb.prefix}layers.11.mlp."
    Wg = tb.w[p0 + "gate_proj.weight"].detach().contiguous()
    Wu = tb.w[p0 + "up_proj.weight"].detach().contiguous()
    Wd = tb.w[p0 + "down_proj.weight"].detach().contiguous()
    O = Wd.shape[0]
    nb = H // LANE
    g_buf = torch.empty(H, device=dev)
    idx_buf = torch.empty(nb, device=dev, dtype=torch.int32)
    k_buf = torch.zeros(1, device=dev, dtype=torch.int32)
    h_buf = torch.empty(H, device=dev)
    out = torch.empty(O, device=dev)
    wb = block_weights(x, Wg, Wu, Wd).contiguous()
    wb2 = (wb * wb).contiguous()

    xv = x[-1].contiguous()

    def torch_mlp():
        g = F.silu(F.linear(xv, Wg)); u = F.linear(xv, Wu)
        return F.linear(g * u, Wd)

    def fused(rate, w2=None):
        return lib.swiglu_coupled(xv, Wg, Wu, Wd, g_buf, idx_buf, k_buf, h_buf, out,
                                  float(rate), w2)

    ref = torch_mlp()
    rows = []
    # isomorphism at tau=0
    out0 = torch.empty_like(out)
    lib.swiglu_coupled(xv, Wg, Wu, Wd, g_buf, idx_buf, k_buf, h_buf, out0, 0.0, None)
    iso = float((out0 - ref).abs().max())

    t_torch = bench(torch_mlp)
    t_dense = bench(lambda: fused(0.0))
    for rate in (0.2, 0.4, 0.6, 0.75, 0.85):
        for tag, w2 in (("plain", None), ("weighted", wb2)):
            o = torch.empty_like(out)
            lib.swiglu_coupled(xv, Wg, Wu, Wd, g_buf, idx_buf, k_buf, h_buf, o, float(rate), w2)
            rel = float((o - ref).norm() / ref.norm())
            rel_cos = float(F.cosine_similarity(o, ref, dim=0))
            ms = bench(lambda: fused(rate, w2))
            rows.append({"rate": rate, "gate": tag, "ms": ms, "speedup_vs_torch": t_torch / ms,
                         "speedup_vs_dense_kernel": t_dense / ms,
                         "rel_l2": rel, "cosine": rel_cos,
                         "kept": float(1.0 - rate)})
    # 24-layer MLP-only chain (isolate the operator from attention)
    W = []
    for b in range(tb.cfg.layers):
        q = f"{tb.prefix}layers.{b}.mlp."
        W.append((F.silu, tb.w[q + "gate_proj.weight"].detach().contiguous(),
                  tb.w[q + "up_proj.weight"].detach().contiguous(),
                  tb.w[q + "down_proj.weight"].detach().contiguous()))
    h = x[-1].contiguous().clone()
    outd = torch.empty_like(h)

    def chain(rate, w2=None):
        nonlocal h
        v = h
        for (_, wg, wu, wd) in W:
            lib.swiglu_coupled(v, wg, wu, wd, g_buf, idx_buf, k_buf, h_buf, v, float(rate), w2)
        return v

    def chain_torch():
        v = h
        for (_, wg, wu, wd) in W:
            v = F.linear(F.silu(F.linear(v, wg)) * F.linear(v, wu), wd)
        return v

    t_ct = bench(chain_torch, reps=10)
    chain_rows = []
    for rate in (0.0, 0.2, 0.4, 0.6, 0.75):
        ms = bench(lambda: chain(rate), reps=10)
        chain_rows.append({"rate": rate, "ms": ms, "speedup_vs_torch_chain": t_ct / ms})

    res = {"device": dev, "layer": 11, "d_model": D, "intermediate": H, "n_blocks": nb,
           "isomorphism_tau0_max_abs": iso,
           "operator": {"torch_ms": t_torch, "fused_dense_ms": t_dense,
                        "dense_vs_torch": t_torch / t_dense},
           "rows": rows, "chain_24_layers": {"torch_ms": t_ct, "rows": chain_rows},
           "block_weight_stats": {"min": float(wb.min()), "max": float(wb.max()),
                                  "std": float(wb.std())}}
    dump_json(res, RESULTS / "p5_cuda.json")
    print(f"iso {iso:.2e}  torch {t_torch*1e3:.1f}us  fused-dense {t_dense*1e3:.1f}us "
          f"({t_torch/t_dense:.2f}x)")
    for r in rows:
        print(f"  tau {r['rate']:.2f} {r['gate']:8s} {r['ms']*1e3:6.1f}us  "
              f"{r['speedup_vs_torch']:5.2f}x vs torch  {r['speedup_vs_dense_kernel']:5.2f}x vs dense-kernel  "
              f"relL2 {r['rel_l2']:.4f}  cos {r['cosine']:.5f}")
    print(f"  24-layer chain: torch {t_ct*1e3:.0f}us")
    for r in chain_rows:
        print(f"    tau {r['rate']:.2f}  {r['ms']*1e3:6.1f}us  {r['speedup_vs_torch_chain']:5.2f}x")


if __name__ == "__main__":
    main()
