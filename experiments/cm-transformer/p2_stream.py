"""Step 3 (P2): zero-halo (L_halo = 0) temporal token compaction on the
Qwen3.5-0.8B vision encoder, V1-V6 real video.

Patch embedding is a 16x16 stride-16 convolution: every output patch token
depends on exactly one 16x16 input tile, so a dirty-token mask needs *zero*
spatial halo dilation.  Dirty tokens only are gathered for QKV/MLP, the
persistent per-layer K/V cache is updated in place at those indices, and
attention is computed against the full cached K/V.

Guards: scene cut -> flush the whole frame; high motion (s_hat > tau_flush) ->
flush.  A flushed frame is bit-identical to the dense M0 forward.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (RESULTS_DIR, PAPER1_RESULTS, REGIMES, CUT_FRAMES, dump_json,
                    log, set_deterministic)
from model_ref import VisionEncoder, VisionCfg, gelu_tanh, layer_norm, silu
from video_data import load_all, to_single_steps
from common import Weights

PATCH = 16
GRID = 14                      # 224 / 16
NTOK = GRID * GRID             # 196 patch tokens per frame
DENSE_MAC = {                  # MACs per frame (single-image streaming step)
    "patch_embed": NTOK * 768 * 3 * 2 * PATCH * PATCH,
    "qkv": NTOK * 768 * 2304,
    "attn_qk_av": 2 * 12 * NTOK * NTOK * 64,
    "proj": NTOK * 768 * 768,
    "mlp": 2 * NTOK * 768 * 3072,
    "merger": (GRID // 2) ** 2 * (3072 * 3072 + 1024 * 3072),
}
DENSE_TOTAL = DENSE_MAC["patch_embed"] + 12 * (DENSE_MAC["qkv"] + DENSE_MAC["attn_qk_av"] +
                                               DENSE_MAC["proj"] + DENSE_MAC["mlp"]) + DENSE_MAC["merger"]

MLP_SHARE = DENSE_MAC["mlp"] / (DENSE_MAC["qkv"] + DENSE_MAC["attn_qk_av"] +
                                DENSE_MAC["proj"] + DENSE_MAC["mlp"])


# --------------------------------------------------------------------------- #
def per_patch_rel_l2(prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    """[3,H,W] x2 -> [196] relative L2 per 16x16 patch."""
    a = prev.unfold(1, PATCH, PATCH).unfold(2, PATCH, PATCH).permute(1, 2, 0, 3, 4).reshape(NTOK, -1)
    b = cur.unfold(1, PATCH, PATCH).unfold(2, PATCH, PATCH).permute(1, 2, 0, 3, 4).reshape(NTOK, -1)
    return (b - a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-6)


def dilate(mask: torch.Tensor, steps: int) -> torch.Tensor:
    """3x3 dilation of a 14x14 patch mask, `steps` times (one 3x3 stage)."""
    m = mask.reshape(GRID, GRID).float()[None, None]
    for _ in range(steps):
        m = F.max_pool2d(F.pad(m, (1, 1, 1, 1), value=0.0), 3, stride=1)
    return m.reshape(-1) > 0.5


def apply_age_guard(dirty, max_age: int):
    """Recompute a token anyway once it has been clean for max_age frames.

    Paper 1's CNNs recompute clean tiles exactly, so staleness never
    accumulates; a transformer's residual stream does drift, hence the guard.
    """
    dev = dirty[0].device
    out, age = [], torch.zeros(NTOK, dtype=torch.long, device=dev)
    for d in dirty:
        age = torch.where(d, torch.zeros_like(age), age + 1)
        d2 = d | (age > max_age)
        age = torch.where(d2, torch.zeros_like(age), age)
        out.append(d2)
    return out


def apply_drift_guard(frames, dirty, budget: float = 0.15):
    """Refresh a token once its cumulative pixel drift since the last refresh
    exceeds `budget` (L2 of per-patch relative change).

    The dirty-count guard alone is not enough: on V1 ~94% of patches are
    "clean" yet still change ~10% per frame, so a stale state accumulates error.
    """
    drift = torch.zeros(NTOK, device=frames.device)
    out = []
    for t, d in enumerate(dirty):
        if t > 0:
            drift = drift + per_patch_rel_l2(frames[t - 1], frames[t])
        d2 = d | (drift > budget)
        drift = torch.where(d2, torch.zeros_like(drift), drift)
        out.append(d2)
    return out


def detect_dirty(frames: torch.Tensor, tau_floor=0.02, k_mad=4.0, tau_flush_floor=1.0):
    """Causal dirty-token detection + flush guard for a 36-frame regime."""
    n = frames.shape[0]
    dirty, scores, flush = [], [], []
    hist = []
    for t in range(n):
        if t == 0:
            d = torch.ones(NTOK, dtype=torch.bool, device=frames.device)
            s_hat = 0.0
            fl = True
        else:
            rel = per_patch_rel_l2(frames[t - 1], frames[t])
            s_hat = float(rel.mean())
            tau = max(tau_floor, float(rel.median() + k_mad * (rel - rel.median()).abs().median()))
            tau_flush = max(tau_flush_floor, 4.0 * float(np.median(hist[-5:])) if hist else tau_flush_floor)
            fl = s_hat > tau_flush
            d = (torch.ones(NTOK, dtype=torch.bool, device=frames.device) if fl
                 else (rel > tau))
        hist.append(s_hat)
        dirty.append(d)
        scores.append(s_hat)
        flush.append(bool(fl))
    return dirty, scores, flush


# --------------------------------------------------------------------------- #
class StreamingVision:
    """Vision encoder with per-layer persistent K/V cache and dirty-token gather."""

    def __init__(self, enc: VisionEncoder):
        self.enc = enc
        self.c = enc.cfg
        self.N = NTOK
        self.reset()

    def reset(self):
        self.h = None                       # [N,768] last frame token states
        self.K = [None] * self.c.depth      # [N,768]
        self.V = [None] * self.c.depth
        self.groups = None                  # [49,4] token indices per merge group

    def embed_dirty(self, frame: torch.Tensor, idx: torch.Tensor):
        # frame: [2,3,H,W] (single displayed frame replicated over the temporal slot)
        """frame [2,3,H,W]; recompute patch tokens only at idx (zero halo)."""
        c = self.c
        p = self.enc.patchify(frame[None])                 # [196,1536]
        w = self.enc.w
        h = F.linear(p[idx], w[self.enc.prefix + "patch_embed.proj.weight"].reshape(c.hidden, -1),
                     w[self.enc.prefix + "patch_embed.proj.bias"])
        pos = self.enc.pos_embed(GRID, GRID, h.device, h.dtype)
        return h + pos[idx]

    def step(self, frame: torch.Tensor, dirty: torch.Tensor, mlp_override=None,
             skip_rate=0.0):
        """One streaming frame. Returns merged features for this frame."""
        c = self.c
        w = self.enc.w
        dirty = dirty.to(frame.device)
        idx = torch.nonzero(dirty, as_tuple=False).flatten()
        if self.h is None:
            self.h = torch.zeros(self.N, c.hidden, dtype=torch.float32, device=frame.device)
        h = self.h                                    # cache tensor, updated in place
        hd = self.embed_dirty(frame, idx)            # [k,768]
        for i in range(c.depth):
            p = f"{self.enc.prefix}blocks.{i}."
            x = layer_norm(hd, w[p + "norm1.weight"], w[p + "norm1.bias"], c.eps)
            qkv = F.linear(x, w[p + "attn.qkv.weight"], w[p + "attn.qkv.bias"])
            q, k, v = qkv.reshape(idx.numel(), 3, c.heads, c.hidden // c.heads).permute(1, 0, 2, 3)
            # axial 2-D RoPE at the *absolute* positions of the dirty tokens
            hdd = c.hidden // c.heads
            half = hdd // 2
            cos, sin = self.enc.vision_rope(GRID, GRID, q.device, q.dtype)
            cos_i = cos[idx][:, None, :]      # [k,1,half] broadcast over heads
            sin_i = sin[idx][:, None, :]

            def rot(t):
                t1, t2 = t[..., :half], t[..., half:]
                return torch.cat([t1 * cos_i - t2 * sin_i, t2 * cos_i + t1 * sin_i], dim=-1)

            q, k = rot(q), rot(k)
            # in-place cache update at the dirty indices, clean rows keep last-frame K/V
            if self.K[i] is None:
                self.K[i] = k.new_zeros(self.N, c.heads, k.shape[-1])
                self.V[i] = v.new_zeros(self.N, c.heads, v.shape[-1])
            self.K[i].index_copy_(0, idx, k)
            self.V[i].index_copy_(0, idx, v)
            qq = q.transpose(0, 1)                                    # [heads,k,64]
            kk = self.K[i].permute(1, 0, 2)                           # [heads,N,64]
            vv = self.V[i].permute(1, 0, 2)
            o = F.scaled_dot_product_attention(qq.unsqueeze(0), kk.unsqueeze(0),
                                               vv.unsqueeze(0))[0]
            o = o.transpose(0, 1).reshape(idx.numel(), c.hidden)
            hd = hd + F.linear(o, w[p + "attn.proj.weight"], w[p + "attn.proj.bias"])
            x = layer_norm(hd, w[p + "norm2.weight"], w[p + "norm2.bias"], c.eps)
            if mlp_override is not None and i in mlp_override:
                gw, gb = mlp_override[i][0], mlp_override[i][1]
                a = gelu_tanh(F.linear(x, gw, gb))
            else:
                a = gelu_tanh(F.linear(x, w[p + "mlp.linear_fc1.weight"],
                                       w[p + "mlp.linear_fc1.bias"]))
            if skip_rate > 0:
                from model_ref import apply_block_mask, block_energy, keep_mask_from_energy
                a = apply_block_mask(a, keep_mask_from_energy(block_energy(a), skip_rate))
            if mlp_override is not None and i in mlp_override:
                hd = hd + F.linear(a, mlp_override[i][2], w[p + "mlp.linear_fc2.bias"])
            else:
                hd = hd + F.linear(a, w[p + "mlp.linear_fc2.weight"],
                                   w[p + "mlp.linear_fc2.bias"])
        h.index_copy_(0, idx, hd)
        self.h = h
        merged = self.enc.merger(h[None], (1, GRID, GRID))   # [49,1024]
        return merged, h


# --------------------------------------------------------------------------- #
def p1_vision_perms(enc: VisionEncoder, calib_frames: torch.Tensor, lane=32,
                    layers=(0, 3, 6, 9, 11), seed=1234):
    """32-lane co-activation permutations for the vision MLPs (fc1/fc2).

    Returns {variant: {layer: (w1_perm, b1_perm, w2_perm)}} for
    clustered / natural / random orderings.
    """
    from p1_ffn import cluster_blocks
    c = enc.cfg
    w = enc.w
    acts = {i: [] for i in layers}
    calib_frames = calib_frames.to(w.device, dtype=w.dtype)
    for t in range(calib_frames.shape[0]):
        capture = {}
        with torch.no_grad():
            enc.forward(calib_frames[t][None], capture=capture)
        for i in layers:
            x = layer_norm(capture["block_in"][i], w[f"{enc.prefix}blocks.{i}.norm2.weight"],
                           w[f"{enc.prefix}blocks.{i}.norm2.bias"], c.eps)
            a = gelu_tanh(F.linear(x, w[f"{enc.prefix}blocks.{i}.mlp.linear_fc1.weight"],
                                   w[f"{enc.prefix}blocks.{i}.mlp.linear_fc1.bias"]))
            acts[i].append(a.float().cpu())
    g = torch.Generator().manual_seed(seed)
    out = {"clustered": {}, "natural": {}, "random": {}}
    for i in layers:
        A = torch.cat(acts[i], 0)
        I = A.shape[1]
        perms = {
            "clustered": cluster_blocks(A, lane=lane)[0],
            "natural": torch.arange(I),
            "random": torch.randperm(I, generator=g),
        }
        w1 = w[f"{enc.prefix}blocks.{i}.mlp.linear_fc1.weight"]
        b1 = w[f"{enc.prefix}blocks.{i}.mlp.linear_fc1.bias"]
        w2 = w[f"{enc.prefix}blocks.{i}.mlp.linear_fc2.weight"]
        for name, pm in perms.items():
            out[name][i] = (w1[pm].contiguous(), b1[pm].contiguous(), w2[:, pm].contiguous())
    return out


def run_p2(device="cuda", dtype=torch.float32, skip_rates=(0.0, 0.3, 0.5),
           do_p1=True, policies=("tau", "tau+drift0.15", "tau+age4"),
           variants=("clustered", "natural", "random")):
    set_deterministic(0)
    enc = VisionEncoder(Weights(dtype, device))
    frames_all = load_all()
    out = {
        "device": str(device), "dtype": str(dtype).replace("torch.", ""),
        "ntok": NTOK, "patch": PATCH, "halo": 0,
        "dense_mac": DENSE_MAC, "dense_mac_total": DENSE_TOTAL,
        "mlp_share_of_block": MLP_SHARE,
        "regimes": {},
    }
    ages = {"tau": None, "tau+age4": 4, "tau+drift0.15": "drift"}
    perm_sets = None
    if do_p1:
        calib = torch.stack([frames_all["V3"][t] for t in range(4)])   # held-out regime
        calib = torch.stack([torch.stack([f, f]) for f in calib])      # [4,2,3,H,W]
        log("P2: computing vision 32-lane permutations on V3 calibration frames ...")
        perm_sets = p1_vision_perms(enc, calib)

    for r in REGIMES:
        frames = frames_all[r]
        dirty0, scores, flush = detect_dirty(frames)
        steps = to_single_steps(frames).to(enc.w.device, dtype=enc.w.dtype)
        pol_masks = {}
        for p in policies:
            if ages[p] is None:
                pol_masks[p] = dirty0
            elif ages[p] == "drift":
                pol_masks[p] = apply_drift_guard(frames, dirty0, 0.15)
            else:
                pol_masks[p] = apply_age_guard(dirty0, ages[p])
        d_ratio = [float(d.float().mean()) for d in dirty0]
        percolation = [float(np.mean([float(dilate(d, k).float().mean()) for d in dirty0]))
                       for k in range(12)]
        rows = {"dirty_ratio_per_frame": d_ratio,
                "halo_percolation_3x3_stage": percolation,
                "mean_dirty_ratio": float(np.mean(d_ratio)),
                "flush_frames": [i + 1 for i, f in enumerate(flush) if f],
                "s_hat": scores,
                "policies": {}}
        with torch.no_grad():
            ref = torch.cat([enc.forward(steps[t][None]) for t in range(len(steps))], 0)
        ref3 = ref.reshape(-1, 49, 1024).float()
        fn = [i for i, f in enumerate(flush) if f]
        for pol, masks in pol_masks.items():
            g_touched = [float(d.reshape(GRID // 2, 2, GRID // 2, 2).any(-1).any(-1).float().mean())
                         for d in masks]
            prow = {"mean_dirty_ratio": float(np.mean([float(d.float().mean()) for d in masks])),
                    "mean_group_touched": float(np.mean(g_touched)),
                    "variants": {}}
            dl = torch.tensor([float(d.float().mean()) for d in masks])
            gl = torch.tensor(g_touched)
            for rate in skip_rates:
                names = ["isomorphic"] if rate == 0 or perm_sets is None else list(variants)
                for name in names:
                    perms = None if (rate == 0 or perm_sets is None) else perm_sets[name]
                    sv = StreamingVision(enc)
                    feats = []
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        for t in range(len(masks)):
                            m, _ = sv.step(steps[t], masks[t], mlp_override=perms, skip_rate=rate)
                            feats.append(m)
                    ms = (time.perf_counter() - t0) / len(masks) * 1e3
                    feats = torch.cat(feats, 0)
                    f3 = feats.reshape(-1, 49, 1024).float()
                    fd = f3[fn] - ref3[fn]
                    rel = (f3 - ref3).norm(dim=-1) / ref3.norm(dim=-1).clamp_min(1e-6)
                    cos = F.cosine_similarity(f3, ref3, dim=-1)
                    factor = 1.0 - MLP_SHARE * (rate / 3.0)
                    mac = (dl * (DENSE_TOTAL - DENSE_MAC["patch_embed"] - DENSE_MAC["merger"]) * factor
                           + dl * DENSE_MAC["patch_embed"] + gl * DENSE_MAC["merger"]).sum().item()
                    key = f"{rate:.1f}|{name}"
                    prow["variants"][key] = {
                        "skip_rate": rate, "perm": name,
                        "ms_per_frame": ms,
                        "flush_frame_max_abs": float(fd.abs().max()),
                        "flush_frame_rel_l2": float((fd.norm(dim=-1) / ref3[fn].norm(dim=-1)
                                                     .clamp_min(1e-6)).mean()) if fn else None,
                        "merged_cos_mean": float(cos.mean()),
                        "pooled_cos": float(F.cosine_similarity(feats.mean(0), ref.mean(0), dim=0)),
                        "merged_rel_l2_mean": float(rel.mean()),
                        "merged_max_abs": float((f3 - ref3).abs().max()),
                        "per_frame_cos": [round(float(x), 4) for x in cos.mean(-1)],
                        "mac_total": mac,
                        "mac_reduction": 1.0 - mac / (DENSE_TOTAL * len(masks)),
                        "ideal_speedup_mac": (DENSE_TOTAL * len(masks)) / mac,
                    }
                    log(f"P2 {r} [{pol}] {key}: dirty={prow['mean_dirty_ratio']:.3f} "
                        f"{ms:.2f} ms/frame cos={float(cos.mean()):.4f} "
                        f"MAC-red={prow['variants'][key]['mac_reduction']*100:.1f}%")
            rows["policies"][pol] = prow
        with torch.no_grad():
            t0 = time.perf_counter()
            for t in range(len(steps)):
                enc.forward(steps[t][None])
            dense_ms = (time.perf_counter() - t0) / len(steps) * 1e3
        rows["dense_ms_per_frame"] = dense_ms
        for prow in rows["policies"].values():
            for v in prow["variants"].values():
                v["wallclock_speedup"] = dense_ms / v["ms_per_frame"]
        out["regimes"][r] = rows
        log(f"P2 {r}: dense {dense_ms:.2f} ms/frame; dirty {rows['mean_dirty_ratio']*100:.1f}%")

    dump_json(out, RESULTS_DIR / f"p2_{device}.json")
    return out


if __name__ == "__main__":
    import sys
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    run_p2(device=dev)
    log("done")
