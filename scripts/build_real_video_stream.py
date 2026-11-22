#!/usr/bin/env python3
"""Build the CMST/CMRGBV real-video benchmark from the extracted DERF frames.

Inputs (already on disk, produced by ffmpeg from the Xiph DERF y4m collection):
    data/real_videos/frames/V1_static_surveillance_01..36.png   (352x288 RGB)
    ... V6_multi_shot_cuts_01..36.png

Outputs:
    experiments/cm-cnn-production/real_stress_stream_{224,320}.bin        6 sequences x 36 frames
    experiments/cm-cnn-production/real_stress_stream_{224,320}_V{i}.bin   one sequence each
    experiments/cm-cnn-production/real_video_torch.json                   torchvision ImageNet-1K
                                                                          top-1/top-5 + margins
    data/real_videos/frames_{224,320}/V*_*.png                            resized frames (cache)

Frame geometry: centre crop to the largest square, then bicubic resize with anti-aliasing
(PIL `Image.BICUBIC` on a pre-reduced image), exactly the pipeline the manuscript documents.
No normalisation is baked into the stream: the engine normalises internally with ImageNet
statistics (mean 0.485/0.456/0.406, std 0.229/0.224/0.225) and the torch reference does the
same, so both sides see identical float32 pixels.
"""
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FR = ROOT / "data" / "real_videos" / "frames"
OUT = ROOT / "experiments" / "cm-cnn-production"
SIZES = (224, 320)
REGIMES = [
    "V1_static_surveillance",
    "V2_local_motion",
    "V3_fast_natural_object",
    "V4_camera_pan",
    "V5_illumination_change",
    "V6_multi_shot_cuts",
]
FRAMES = 36

MODELS = {
    "resnet18": ("resnet18", 224),
    "resnet50": ("resnet50", 224),
    "mobilenet_v2": ("mobilenet_v2", 224),
    "mobilenetv3_small": ("mobilenet_v3_small", 224),
    "shufflenetv2_x0_5": ("shufflenet_v2_x0_5", 224),
    "mbv3_large_320": ("mobilenet_v3_large", 320),
}


def crop_resize(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    im = im.crop((left, top, left + s, top + s))
    return im.resize((size, size), Image.BICUBIC)


def load_regime(tag: str, size: int) -> np.ndarray:
    cache = ROOT / "data" / "real_videos" / f"frames_{size}"
    files = [cache / f"{tag}_{i:02d}.png" for i in range(1, FRAMES + 1)]
    if not all(f.exists() for f in files):
        cache.mkdir(exist_ok=True)
        src = [FR / f"{tag}_{i:02d}.png" for i in range(1, FRAMES + 1)]
        for s, d in zip(src, files):
            crop_resize(Image.open(s).convert("RGB"), size).save(d)
    x = np.stack([np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0 for f in files])
    assert x.shape == (FRAMES, size, size, 3), x.shape
    return x


def write_stream(path: Path, seqs: np.ndarray, labels=None):
    """CMRGBV01 stream: magic(8) seqs,frames,h,w,c (5 x u32) @8, labels @32 padded, f32 NHWC."""
    n, t, h, w, c = seqs.shape
    labels = np.zeros(n, np.uint8) if labels is None else np.asarray(labels, np.uint8)
    with open(path, "wb") as f:
        f.write(b"CMRGBV01")
        f.write(struct.pack("<IIIII", n, t, h, w, c))
        f.write(b"\0" * (32 - 28))
        f.write(labels.tobytes())
        pad = (-(32 + n)) % 4
        f.write(b"\0" * pad)
        f.write(np.ascontiguousarray(seqs, "<f4").tobytes())
    path.chmod(0o444)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def torch_labels(size: int, pix: np.ndarray, tag_by_seq):
    """Official torchvision ImageNet-1K predictions for every frame of every sequence."""
    out = {}
    dev = "cpu"
    torch.set_num_threads(1)
    for name, (ctor, want) in MODELS.items():
        if want != size:
            continue
        mod = getattr(__import__("torchvision.models", fromlist=[ctor]), ctor)(weights="DEFAULT").eval()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = torch.from_numpy(pix.transpose(0, 3, 1, 2)).to(dev)
        x = (x - mean) / std
        with torch.no_grad():
            logits = torch.cat([mod(x[i:i + 8]) for i in range(0, x.shape[0], 8)], 0).numpy()
        top1 = logits.argmax(1)
        srt = np.sort(logits, 1)
        margin = (srt[:, -1] - srt[:, -2]).astype(np.float32)
        top5 = np.argsort(-logits, 1)[:, :5]
        out[name] = {"logits": logits, "top1": top1, "top5": top5, "margin": margin}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torchj = {"generated": "2026-11-20", "source": "Xiph.org DERF y4m collection", "models": {}}
    for size in SIZES:
        seqs = [load_regime(t, size) for t in REGIMES]
        allseq = np.stack(seqs)
        sha_all = write_stream(OUT / f"real_stress_stream_{size}.bin", allseq)
        per = {}
        for tag, s in zip(REGIMES, seqs):
            per[tag] = write_stream(OUT / f"real_stress_stream_{size}_{tag[:2]}.bin", s[None])
        lab = torch_labels(size, allseq.reshape(-1, size, size, 3), [t[:2] for t in REGIMES])
        for name, v in lab.items():
            np.save(OUT / f"real_video_torch_logits_{name}_{size}.npy", v["logits"].astype(np.float32))
            torchj["models"].setdefault(name, {})[str(size)] = {
                "top1": v["top1"].tolist(),
                "top5": v["top5"].tolist(),
                "margin": [round(float(m), 5) for m in v["margin"]],
            }
        torchj[f"stream_{size}"] = {"sha256": sha_all, "per_regime_sha256": per}
        print(f"[{size}] stream sha {sha_all[:16]}  frames {allseq.shape}")
    json.dump(torchj, open(OUT / "real_video_torch.json", "w"), indent=1)
    print("wrote real_video_torch.json")


if __name__ == "__main__":
    main()
