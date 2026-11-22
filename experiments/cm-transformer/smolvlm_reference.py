"""SmolVLM2-256M-Video-Instruct as an independent, officially-supported
vision-language reference point for the V1-V6 regimes.

This is *not* a CM-accelerated model: it is here to (a) prove that the video
inputs and the captioning protocol are sound with a model the ecosystem ships
natively, and (b) give a qualitative reference caption per regime.

Usage:
    python3 smolvlm_reference.py --device cuda --regimes V1,V4 --gen-len 24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import ROOT, dump_json, log, set_deterministic
from video_data import REGIMES, load_regime

MODEL_DIR = ROOT / "models" / "SmolVLM2-256M-Video-Instruct"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def to_pil(frames_224: torch.Tensor, idx):
    from PIL import Image
    import numpy as np
    out = []
    for i in idx:
        x = frames_224[i]
        a = ((x.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0).clip(0, 255).astype("uint8")
        out.append(Image.fromarray(a))
    return out


@torch.no_grad()
def caption(proc, model, images, prompt, device, gen_len):
    messages = [{"role": "user", "content": [{"type": "image"} for _ in images] +
                 [{"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=text, images=images, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=gen_len, do_sample=False)
    gen = out[0, inputs["input_ids"].shape[1]:]
    return proc.decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--frames", default="0,9,18,27,35")
    ap.add_argument("--gen-len", type=int, default=24)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    set_deterministic(0)
    from transformers import AutoModelForImageTextToText, AutoProcessor
    dt = getattr(torch, args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=dt).to(args.device).eval()
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    idx = [int(x) for x in args.frames.split(",")]
    out = {"model": "SmolVLM2-256M-Video-Instruct", "device": args.device,
           "dtype": args.dtype, "frames": idx, "regimes": {}}
    for reg in args.regimes.split(","):
        frames = load_regime(reg)
        images = to_pil(frames, [i for i in idx if i < frames.shape[0]])
        cap = caption(proc, model, images, "Describe what happens in this video in one sentence.",
                      args.device, args.gen_len)
        out["regimes"][reg] = {"caption": cap.strip(), "n_images": len(images)}
        log(f"{reg}: {cap.strip()!r}")
    dump_json(out, RESULTS_DIR / "smolvlm_reference.json")


if __name__ == "__main__":
    main()
