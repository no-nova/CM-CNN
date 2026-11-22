"""Vision tower stage-by-stage diff: our reference vs official transformers.

    python3 debug_vision.py [--no-rope]

Prints max|Δ| / cosine after patch-embed, pos-embed, every block and the merger,
so a mismatch can be attributed to exactly one stage.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from common import MODEL_DIR, Weights, dump_json, log, RESULTS_DIR
from model_ref import VisionEncoder
from video_data import load_regime, to_pairs

dev = "cuda"
torch.set_grad_enabled(False)


def cmp(a, b, name, rows):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    cos = F.cosine_similarity(a.flatten(), b.flatten(), 0).item()
    rows.append({"stage": name, "max_abs": d.max().item(), "cosine": cos,
                 "a_absmax": a.abs().max().item(), "b_absmax": b.abs().max().item()})
    log(f"  {name:16s} max|Δ|={d.max():.4e} cos={cos:.6f} "
        f"|ref|={a.abs().max():.3f} |off|={b.abs().max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rope", action="store_true")
    args = ap.parse_args()

    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.vision_utils import (get_vision_attention_seqlens,
                                           get_vision_interpolation_indices_and_weights,
                                           get_vision_position_ids)
    off = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
    vis = off.model.visual
    enc = VisionEncoder(Weights(torch.float32, dev))

    frames = to_pairs(load_regime("V2"))[:1].to(dev).float()      # [1,2,3,224,224]
    rows = []

    flat = enc.patchify(frames)
    grid = torch.tensor([[1, 14, 14]], device=dev)
    idx, wts = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=vis.num_grid_per_side, mode=vis.interpolation_mode,
        align_corners=vis.interpolation_align_corners,
        spatial_merge_size=vis.spatial_merge_size)
    pos_ids = get_vision_position_ids(grid, vis.spatial_merge_size)
    cu, msl = get_vision_attention_seqlens(grid, vis.config)

    # --- stage 1: patch tokens
    off_tok = vis.patch_embed(flat)
    ref_tok, shape = enc.embed(frames)
    ref_pe = (vis.pos_embed(idx) * wts[:, :, None]).sum(1)
    ref_patch = F.linear(flat, enc.w["model.visual.patch_embed.proj.weight"].reshape(768, -1),
                         enc.w["model.visual.patch_embed.proj.bias"])
    cmp(ref_patch, off_tok, "patch_embed", rows)
    cmp(enc.pos_embed(14, 14, dev, torch.float32), ref_pe, "pos_embed", rows)
    cmp(ref_tok, off_tok + ref_pe, "tokens+pos", rows)

    # --- rope comparison
    cos_o, sin_o = vis.rotary_pos_emb(ref_tok, pos_ids)
    cos_r, sin_r = enc.vision_rope(14, 14, dev, torch.float32)
    cmp(cos_r, cos_o[:, :32], "rope_cos", rows)
    cmp(sin_r, sin_o[:, :32], "rope_sin", rows)

    if args.no_rope:
        pos_emb = (torch.ones_like(cos_o), torch.zeros_like(sin_o))
    else:
        pos_emb = (cos_o, sin_o)

    # --- block by block
    caps = []
    for i, blk in enumerate(vis.blocks):
        blk.register_forward_hook(lambda m, inp, out, i=i: caps.append(out))

    h_off = vis.patch_embed(flat) + ref_pe
    h_off = h_off + 0 * h_off  # keep dtype
    # run official blocks manually so we can use our own position embeddings
    for blk in vis.blocks:
        h_off = blk(h_off, cu_seqlens=cu, max_seqlen=msl, position_embeddings=pos_emb)

    h_ref, shape = enc.embed(frames)
    for i in range(enc.cfg.depth):
        # our block with the same cos/sin (already equivalent when not --no-rope)
        h_ref = enc.block_dense(h_ref, i, shape)
        cmp(h_ref, caps[i], f"block{i}", rows)
        if False:
            break

    cmp(h_ref, h_off, "final_premerge", rows)
    cmp(enc.merger(h_ref, shape), vis.merger(h_off), "merger", rows)
    dump_json(rows, RESULTS_DIR / ("debug_vision_norope.json" if args.no_rope
                                  else "debug_vision.json"))
    log(f"worst {max(r['max_abs'] for r in rows):.3e}")


if __name__ == "__main__":
    main()
