"""Step 2 (P1): 32-lane FFN co-activation permutation + structured block skipping.

M2 == M0 at tau=0 (functional isomorphism, zero retraining), then P1-clustered vs
P1-natural vs P1-random at matched 32-lane block skip rates.

The learned permutation is computed offline on calibration prompts from the
per-neuron post-activation profile  a = silu(gate(x)) * up(x)  and groups
frequently co-activating neurons into contiguous 32-lane blocks (1 CUDA warp /
2 AVX-512 ZMM registers).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from common import (Weights, EXP_DIR, RESULTS_DIR, bench, dump_json, log,
                    set_deterministic, LANE)
from model_ref import (TextBackbone, TextCfg, apply_block_mask, block_energy,
                       keep_mask_from_energy, permute_mlp)
from video_data import CALIB_PROMPTS, EVAL_PROMPTS, encode_prompts

LAYERS = 24
INTER = 3584
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


# --------------------------------------------------------------------------- #
def activation_profiles(tb: TextBackbone, prompts, device="cuda"):
    """Per-layer post-activation matrices a = silu(gate)*up on calibration data."""
    acts = {i: [] for i in range(LAYERS)}
    for ids in prompts:
        capture = {}
        with torch.no_grad():
            tb.forward(ids.to(device), capture=capture, logits_positions=[-1])
        for i in range(LAYERS):
            x = capture["mlp_in"][i]
            p = f"{tb.prefix}layers.{i}.mlp."
            a = F.silu(F.linear(x, tb.w[p + "gate_proj.weight"])) * \
                F.linear(x, tb.w[p + "up_proj.weight"])
            acts[i].append(a.float().cpu())
    return {i: torch.cat(v, 0) for i, v in acts.items()}


def cluster_blocks(A: torch.Tensor, lane: int = LANE, seed: int = 0,
                   max_tokens: int = 384):
    """Greedy co-activation clustering into contiguous `lane`-neuron blocks.

    A: [N, I] post-activation matrix.  Returns perm [I] (new position -> old
    neuron), block order sorted by descending mean block energy.
    """
    torch.manual_seed(seed)
    if A.shape[0] > max_tokens:
        idx = torch.linspace(0, A.shape[0] - 1, max_tokens).long()
        A = A[idx]
    X = A.t().float()                                  # [I, N]
    mag = X.abs()
    energy = mag.mean(1)                               # [I]
    # L2-normalise each neuron's co-activation profile
    P = mag / mag.norm(dim=1, keepdim=True).clamp_min(1e-8)
    I = P.shape[0]
    nb = I // lane
    assert nb * lane == I, (I, lane)
    dev = P.device
    assigned = torch.full((I,), -1, dtype=torch.long, device=dev)
    blocks = []
    order = torch.argsort(energy, descending=True)
    for b in range(nb):
        free = assigned < 0
        # seed = highest-energy unassigned neuron
        seed_i = int(order[free[order]][0])
        members = [seed_i]
        assigned[seed_i] = b
        c = P[seed_i].clone()
        for _ in range(lane - 1):
            score = P @ (c / c.norm().clamp_min(1e-8))
            score[assigned >= 0] = -2.0
            j = int(torch.argmax(score))
            members.append(j)
            assigned[j] = b
            c += P[j]
        blocks.append(members)
    # order blocks by descending mean energy (stable, deterministic)
    blocks.sort(key=lambda m: -float(energy[m].mean()))
    perm = torch.tensor([i for m in blocks for i in m], dtype=torch.long)
    assert perm.numel() == I and torch.unique(perm).numel() == I
    return perm, {"energy": energy, "blocks": blocks}


def grouping_stats(perm: torch.Tensor, P: torch.Tensor, lane: int = LANE):
    """Mean intra-block cosine (co-activation coherence) of a 32-lane grouping."""
    Q = P[perm].reshape(-1, lane, P.shape[1])
    gram = torch.einsum("bld,bmd->blm", Q, Q)
    n, nb = lane, Q.shape[0]
    intra = (gram.sum() - nb * n) / (nb * n * (n - 1))
    return float(intra)


def default_weights(tb: TextBackbone, layer: int):
    p = f"{tb.prefix}layers.{layer}.mlp."
    return (tb.w[p + "gate_proj.weight"], tb.w[p + "up_proj.weight"],
            tb.w[p + "down_proj.weight"])


def make_override(tb, perm_map):
    out = {}
    for i in range(LAYERS):
        out[i] = permute_mlp(default_weights(tb, i), perm_map[i].to(tb.w.device))
    return out


# --------------------------------------------------------------------------- #
def logit_metrics(ref: torch.Tensor, test: torch.Tensor) -> dict:
    d = (test.float() - ref.float())
    pr, pt = F.softmax(ref.float(), -1), F.softmax(test.float(), -1)
    kl = (pr * (pr.clamp_min(1e-12).log() - pt.clamp_min(1e-12).log())).sum(-1)
    cos = F.cosine_similarity(ref.float().flatten(), test.float().flatten(), dim=0)
    return {
        "logitRmse": float(torch.sqrt((d ** 2).mean())),
        "max_abs_dlogit": float(d.abs().max()),
        "kl": float(kl.mean()),
        "top1_agreement": float((ref.argmax(-1) == test.argmax(-1)).float().mean()),
        "cosine": float(cos),
    }


def run_p1(device="cuda", dtype=torch.float32, rates=RATES, do_kernels=True, max_len=128):
    set_deterministic(0)
    tb = TextBackbone(Weights(dtype, device))
    calib = encode_prompts(CALIB_PROMPTS, max_len=max_len, device=device)
    evals = encode_prompts(EVAL_PROMPTS, max_len=max_len, device=device)

    # ---- M0 reference logits ------------------------------------------- #
    with torch.no_grad():
        ref = [tb.forward(p, logits_positions=[-1])["logits"].float().cpu() for p in evals]
    ref_stack = torch.cat(ref, 0)

    # ---- calibration activations & clustered permutation --------------- #
    log("P1: capturing calibration activations ...")
    acts = activation_profiles(tb, calib, device)
    # profiles for coherence stats (subsampled, normalised)
    prof = {i: (acts[i][:256].t().abs()) for i in range(LAYERS)}
    for i in prof:
        prof[i] = prof[i] / prof[i].norm(dim=1, keepdim=True).clamp_min(1e-8)

    perms, cluster_meta = {}, {}
    rng = torch.Generator().manual_seed(1234)
    perm_random = {i: torch.randperm(INTER, generator=rng) for i in range(LAYERS)}
    for i in range(LAYERS):
        p, meta = cluster_blocks(acts[i])
        perms[i] = {"clustered": p}
        cluster_meta[i] = {
            "intra_block_cosine_clustered": grouping_stats(p, prof[i]),
            "intra_block_cosine_natural": grouping_stats(torch.arange(INTER), prof[i]),
            "intra_block_cosine_random": grouping_stats(perm_random[i], prof[i]),
            "mean_energy": float(meta["energy"].mean()),
        }
    log("P1: clustered permutation done; coherence",
        np.mean([v["intra_block_cosine_clustered"] for v in cluster_meta.values()]).round(4),
        "vs natural",
        np.mean([v["intra_block_cosine_natural"] for v in cluster_meta.values()]).round(4))

    variants = {
        "clustered": {i: perms[i]["clustered"] for i in range(LAYERS)},
        "natural": {i: torch.arange(INTER) for i in range(LAYERS)},
        "random": perm_random,
    }

    # ---- isomorphism at tau = 0 --------------------------------------- #
    iso = {}
    for name in ("clustered", "natural"):
        ov = make_override(tb, variants[name])
        with torch.no_grad():
            lg = [tb.forward(p, mlp_override=ov, skip_rate=0.0,
                             logits_positions=[-1])["logits"].float().cpu() for p in evals]
        iso[name] = logit_metrics(ref_stack, torch.cat(lg, 0))
        log(f"P1 isomorphism {name}: max|dlogit|={iso[name]['max_abs_dlogit']:.3e} "
            f"top1={iso[name]['top1_agreement']:.3f}")
        del ov
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- block-skip sweep --------------------------------------------- #
    sweep = {}
    for name, perm_map in variants.items():
        ov = make_override(tb, perm_map)
        rows = []
        for rate in rates:
            e_out = {}
            with torch.no_grad():
                lg, energy_kept, energy_tot = [], 0.0, 0.0
                for p in evals:
                    e_out.clear()
                    o = tb.forward(p, mlp_override=ov, skip_rate=rate,
                                   energy_out=e_out, logits_positions=[-1])
                    lg.append(o["logits"].float().cpu())
                    # energy retained under this mask (top-k block energy sum)
                    for i, e in e_out.items():
                        B = e.shape[-1]
                        nk = max(1, int(round(B * (1.0 - rate))))
                        energy_kept += float(e.topk(nk, dim=-1).values.sum())
                        energy_tot += float(e.sum())
            m = logit_metrics(ref_stack, torch.cat(lg, 0))
            m.update({"rate": rate,
                      "energy_retained": (energy_kept / energy_tot) if energy_tot else 1.0,
                      "mac_reduction_down_proj": rate,
                      "mac_reduction_ffn_total": rate / 3.0})
            rows.append(m)
            log(f"P1 {name:9s} rate={rate:.1f} KL={m['kl']:.4e} "
                f"top1={m['top1_agreement']:.3f} rmse={m['logitRmse']:.4e} "
                f"Ekeep={m['energy_retained']:.3f}")
        sweep[name] = rows
        del ov
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- save activations for kernel benchmarks ----------------------- #
    torch.save({"acts": {i: acts[i] for i in range(LAYERS)},
                "eval_acts": capture_eval_acts(tb, evals, device)},
               RESULTS_DIR / "p1_activations.pt")

    result = {
        "device": str(device), "dtype": str(dtype).replace("torch.", ""),
        "lane": LANE, "n_blocks": INTER // LANE, "layers": LAYERS,
        "calibration": {"n_prompts": len(calib), "tokens": int(acts[0].shape[0]),
                        "prompts": CALIB_PROMPTS},
        "eval": {"n_prompts": len(evals), "prompts": EVAL_PROMPTS},
        "max_len": max_len,
        "clustering": cluster_meta,
        "isomorphism_tau0": iso,
        "sweep": sweep,
    }
    del tb
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def capture_eval_acts(tb, evals, device):
    """Post-activations of every layer on the first eval prompt (for kernels)."""
    out = {}
    with torch.no_grad():
        capture = {}
        tb.forward(evals[0], capture=capture, logits_positions=[-1])
        for i in range(LAYERS):
            x = capture["mlp_in"][i]
            p = f"{tb.prefix}layers.{i}.mlp."
            a = F.silu(F.linear(x, tb.w[p + "gate_proj.weight"])) * \
                F.linear(x, tb.w[p + "up_proj.weight"])
            out[i] = a.float().cpu()
    return out


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    dtype = torch.float32 if "--fp32" in sys.argv else torch.bfloat16
    ml = 32 if "--short" in sys.argv else 128
    res = run_p1(device=device, dtype=dtype, max_len=ml)
    dump_json(res, RESULTS_DIR / f"p1_{device}_{str(dtype).replace('torch.','')}.json")
