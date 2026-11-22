"""End-to-end VLM evaluation of P2 (zero-halo temporal token compaction) and
P1+P2 on the six real DERF video regimes V1-V6.

Two model families:
  * Qwen3.5-0.8B  — vision encoder (model_ref.VisionEncoder, streaming variant
                    from p2_stream.StreamingVision) + language backbone fed with
                    the merged image features injected at the image tokens.
  * SmolVLM2-256M — official transformers Idefics3 implementation (reference
                    point for a model that the ecosystem actually ships).

Recorded per regime and per condition:
  vision dense/streaming latency, dirty ratio, MAC reduction,
  VLM next-token Top-1 agreement, KL, feature cosine,
  caption ROUGE-L against the dense-model caption, end-to-end latency.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (MODEL_DIR, ROOT, Weights, build_tokenizer, bench, dump_json,
                    log, set_deterministic, LANE)
from model_ref import (TextBackbone, VisionEncoder, apply_block_mask, block_energy,
                       keep_mask_from_energy, new_state)
from p2_stream import (GRID, NTOK, StreamingVision, apply_age_guard,
                       apply_drift_guard, detect_dirty, p1_vision_perms)
from run_text_eval import first_divergence, rouge_l
from video_data import REGIMES, load_regime

RESULTS_DIR = Path(__file__).resolve().parent / "results"
IMG_TOK = 248056
VIS_START, VIS_END = 248053, 248054
IM_START, IM_END = 248045, 248046          # <|im_start|> / <|im_end|>
SAMPLE_FRAMES = (0, 12, 24, 35)
CONDITIONS = ("P2-tau", "P2-tau+age4", "P2-tau+drift", "P1v+P2-tau")

QUESTIONS = [
    "Describe what happens in this video in one sentence.",
    "Is the camera moving in this video? Answer yes or no.",
]


def template_ids(tok, n_img, question):
    pre = tok("<|im_start|>user\n<|vision_start|>", add_special_tokens=False)["input_ids"]
    # force an empty thinking block so the answer tokens are what we measure
    post = tok(f"<|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n"
               f"<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
    return pre + [IMG_TOK] * n_img + post


@torch.no_grad()
def vlm_logits(tb, tok, feats, question):
    ids = template_ids(tok, feats.shape[0], question)
    emb = F.embedding(torch.tensor(ids, device=feats.device),
                      tb.w[tb.prefix + "embed_tokens.weight"])
    n_pre = len(template_ids(tok, 0, question)) - len(question)
    start = template_ids(tok, 0, question)
    # image block begins right after the vision_start token
    pos = len(tok("<|im_start|>user\n<|vision_start|>", add_special_tokens=False)["input_ids"])
    emb = torch.cat([emb[:pos], feats, emb[pos + feats.shape[0]:]], 0)
    out = tb.forward(embeds=emb)
    return out["logits"][-1].float()


@torch.no_grad()
def vlm_generate(tb, tok, feats, question, n_new=12):
    ids = template_ids(tok, feats.shape[0], question)
    emb = F.embedding(torch.tensor(ids, device=feats.device),
                      tb.w[tb.prefix + "embed_tokens.weight"])
    pos = len(tok("<|im_start|>user\n<|vision_start|>", add_special_tokens=False)["input_ids"])
    emb = torch.cat([emb[:pos], feats, emb[pos + feats.shape[0]:]], 0)
    st = new_state()
    out = tb.forward(embeds=emb, state=st, pos0=0)
    nxt = int(out["logits"][-1].argmax())
    gen = [nxt]
    base = emb.shape[0]
    for t in range(1, n_new):
        o = tb.forward(torch.tensor([nxt], device=feats.device), state=st,
                       pos0=base + t - 1)
        nxt = int(o["logits"][-1].argmax())
        gen.append(nxt)
    return gen


# --------------------------------------------------------------------------- #
def dense_features(enc, frames):
    """Full recompute of the vision encoder, frame by frame (M0)."""
    feats = []
    for t in range(frames.shape[0]):
        pair = torch.stack([frames[t], frames[t]], 0)[None]             # [1,2,3,H,W]
        merged = enc.forward(pair)
        feats.append(merged.reshape(GRID // 2, GRID // 2, -1))
    return torch.stack([f.reshape(-1, f.shape[-1]) for f in feats], 0)


def stream_features(enc, frames, policy, mlp_override=None, skip_rate=0.0):
    sv = StreamingVision(enc)
    dirty, scores, flush = detect_dirty(frames)
    if policy == "tau+age4":
        dirty = apply_age_guard(dirty, 4)
    elif policy == "tau+drift":
        dirty = apply_drift_guard(frames, dirty, 0.15)
    feats, mac_red = [], []
    for t in range(frames.shape[0]):
        frame = torch.stack([frames[t], frames[t]], 0)                 # [2,3,H,W]
        m, _ = sv.step(frame, dirty[t], mlp_override=mlp_override, skip_rate=skip_rate)
        feats.append(m.reshape(-1, m.shape[-1]))
        rat = float(dirty[t].float().mean())
        mac_red.append(1.0 - (0.36 * rat + 0.64 * rat * (1.0 - skip_rate)))
    return torch.stack(feats, 0), float(sum(mac_red) / len(mac_red))


def run(device="cuda", dtype=torch.float32, regimes=("V1", "V2", "V3", "V4", "V5", "V6"),
        policy_list=CONDITIONS, gen_len=12, sample_frames=SAMPLE_FRAMES):
    tb = TextBackbone(Weights(dtype, device))
    enc = VisionEncoder(Weights(dtype, device))
    tok = build_tokenizer(MODEL_DIR)
    out = {"model": "Qwen3.5-0.8B", "device": device, "dtype": str(dtype),
           "sample_frames": list(sample_frames), "gen_len": gen_len,
           "regimes": {}}

    from video_data import to_single_steps
    calib = to_single_steps(load_regime("V3")[:4]).to(device).to(dtype)   # [4,2,3,H,W]
    perms = p1_vision_perms(enc, calib)

    for reg in regimes:
        frames = load_regime(reg).to(device).to(dtype)
        n = frames.shape[0]
        idx = [i for i in sample_frames if i < n]
        row = {"frames": n, "sampled": idx, "conditions": {}}

        if reg == regimes[0]:
            warm = torch.zeros(49, 1024, device=device, dtype=dtype)
            vlm_logits(tb, tok, warm, QUESTIONS[0])
        t0 = time.perf_counter()
        dense = dense_features(enc, frames)
        row["dense_vision_ms"] = (time.perf_counter() - t0) * 1e3

        # reference VLM behaviour (dense vision)
        dense_sel = dense[idx].reshape(-1, dense.shape[-1])
        ref_logits = vlm_logits(tb, tok, dense_sel, QUESTIONS[0])
        ref_caption = vlm_generate(tb, tok, dense_sel, QUESTIONS[0], gen_len)
        row["dense_caption"] = tok.decode(ref_caption)
        t0 = time.perf_counter()
        ref_yn = vlm_logits(tb, tok, dense_sel, QUESTIONS[1])
        row["dense_e2e_ms"] = (time.perf_counter() - t0) * 1e3 + row["dense_vision_ms"]

        for cond in policy_list:
            policy = "tau" if cond in ("P2-tau", "P1v+P2-tau") else cond.replace("P2-", "")
            use_p1 = cond.startswith("P1v")
            skip = 0.3 if use_p1 else 0.0
            t0 = time.perf_counter()
            feats, mac_red = stream_features(
                enc, frames, policy,
                mlp_override=perms["clustered"] if use_p1 else None, skip_rate=skip)
            vis_ms = (time.perf_counter() - t0) * 1e3
            sel = feats[idx].reshape(-1, feats.shape[-1])

            lg = vlm_logits(tb, tok, sel, QUESTIONS[0])
            cap = vlm_generate(tb, tok, sel, QUESTIONS[0], gen_len)
            yn = vlm_logits(tb, tok, sel, QUESTIONS[1])

            kl = float(F.kl_div(F.log_softmax(lg[None].double(), -1),
                                F.log_softmax(ref_logits[None].double(), -1),
                                log_target=True))
            feat_cos = float(F.cosine_similarity(
                dense[idx].mean(0).flatten(), feats[idx].mean(0).flatten(), dim=0))
            row["conditions"][cond] = {
                "mac_reduction": mac_red,
                "dirty_ratio": mac_red,     # reported separately in p2_*.json
                "vision_ms": vis_ms,
                "e2e_ms": vis_ms + row["dense_e2e_ms"] - row["dense_vision_ms"],
                "vis_speedup": row["dense_vision_ms"] / vis_ms,
                "next_token_top1": int(lg.argmax() == ref_logits.argmax()),
                "next_token_kl": kl,
                "next_token_logit_cos": float(F.cosine_similarity(
                    lg[None], ref_logits[None], dim=-1)),
                "yn_top1": int(yn.argmax() == ref_yn.argmax()),
                "feature_cos": feat_cos,
                "caption": tok.decode(cap),
                "caption_rouge_l": rouge_l(tok.decode(cap), tok.decode(ref_caption)),
                "caption_first_div": first_divergence(cap, ref_caption),
            }
            log(f"{reg} {cond:14s} vis {vis_ms:7.1f} ms ({row['dense_vision_ms']/vis_ms:4.2f}x) "
                f"mac_red {mac_red:.3f} top1 {row['conditions'][cond]['next_token_top1']} "
                f"kl {kl:.2e} rougeL {row['conditions'][cond]['caption_rouge_l']:.3f}")
        out["regimes"][reg] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--gen-len", type=int, default=12)
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    set_deterministic(0)
    res = run(args.device, getattr(torch, args.dtype), args.regimes.split(","),
              gen_len=args.gen_len)
    tag = args.tag or args.device
    dump_json(res, RESULTS_DIR / f"vlm_e2e_{tag}.json")
    log(f"wrote results/vlm_e2e_{tag}.json")


if __name__ == "__main__":
    main()
