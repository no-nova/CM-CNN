"""P6-T4: build the >=50-sample SmolVLM2 image/video QA set (synthetic, controlled).

Four families, all rendered with PIL from a single deterministic RNG (seed 1234):

  ocr       render a short alphanumeric string, ask for the text
  counting  render N shapes of one colour + distractors, ask for N
  scene     render a fixed, known composition, score ROUGE-L against a reference
  motion    render 3-4 frames of an object moving/growing, ask about the motion

Everything is written under data/p6_vlm/ with a manifest.json that carries, per
sample, the file list, per-file sha256, the prompt and the reference answer, plus
the generator sha256 and the RNG seed.

Dataset is SYNTHETIC (no real photographs): controlled, reproducible, and cheap to
regenerate byte-for-byte.  Documented as such in the manifest and the result JSON.

Usage:
    python make_p6_vlm.py [--seed 1234] [--n-ocr 16] [--n-count 14]
                          [--n-scene 14] [--n-motion 14]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "p6_vlm"
MANIFEST = DATA_DIR / "manifest.json"
SEED = 1234

FONT_CANDIDATES = [
    "/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf",
    "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]

COLORS = {                       # name -> RGB
    "red": (200, 30, 30),
    "green": (30, 150, 60),
    "blue": (35, 70, 200),
    "yellow": (225, 190, 20),
    "purple": (130, 50, 160),
}
SHAPES = ("circle", "square", "triangle")


def _font(size: int, rng: random.Random) -> ImageFont.FreeTypeFont:
    for path in rng.sample(FONT_CANDIDATES, len(FONT_CANDIDATES)):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _draw_shape(d: ImageDraw.ImageDraw, shape: str, box, rgb):
    x0, y0, x1, y1 = box
    if shape == "circle":
        d.ellipse(box, fill=rgb)
    elif shape == "square":
        d.rectangle(box, fill=rgb)
    else:                                   # upward triangle inside the box
        d.polygon([((x0 + x1) / 2, y0), (x0, y1), (x1, y1)], fill=rgb)


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #
OCR_STRINGS = ["K7Q2", "SABLE 42", "MINT-38", "hello 2024", "ZTX 905", "rope7",
               "QUARTZ", "P4L8N3", "north-61", "VIVID 7", "XJ-220", "garden",
               "T9W4M1", "OAK 318", "flux 55", "B6D2K8"]


def gen_ocr(rng: random.Random, n: int):
    out = []
    for i in range(n):
        text = OCR_STRINGS[i % len(OCR_STRINGS)]
        w, h = 512, 224
        bg = rng.randint(225, 252)
        fg = rng.randint(10, 70)
        img = Image.new("RGB", (w, h), (bg, bg, bg))
        d = ImageDraw.Draw(img)
        f = _font(rng.randint(86, 112), rng)
        bb = d.textbbox((0, 0), text, font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        x = (w - tw) // 2 - bb[0] + rng.randint(-6, 6)
        y = (h - th) // 2 - bb[1] + rng.randint(-5, 5)
        d.text((x, y), text, font=f, fill=(fg, fg, fg))
        out.append({
            "id": f"ocr_{i:02d}", "family": "ocr", "images": [img],
            "prompt": "Read the text in this image. Output only the text, no explanation.",
            "reference": text, "scoring": "exact_alnum",
            "meta": {"text": text, "bg": bg, "fg": fg},
        })
    return out


def gen_counting(rng: random.Random, n: int):
    out = []
    for i in range(n):
        colour = rng.choice(list(COLORS))
        shape = rng.choice(SHAPES)
        count = rng.randint(2, 6)
        n_distract = rng.randint(2, 4)
        other_cols = [c for c in COLORS if c != colour]
        cells = [(cx, cy) for cy in range(3) for cx in range(3)]
        rng.shuffle(cells)
        size = 512
        img = Image.new("RGB", (size, size), (250, 250, 250))
        d = ImageDraw.Draw(img)
        cell = size // 3
        used = 0
        for (cx, cy) in cells[:count + n_distract]:
            kind = shape if used < count else rng.choice(SHAPES)
            col = colour if used < count else rng.choice(other_cols)
            used += 1
            rad = rng.randint(38, 52)
            jx = rng.randint(-10, 10)
            jy = rng.randint(-10, 10)
            ccx, ccy = cx * cell + cell // 2 + jx, cy * cell + cell // 2 + jy
            _draw_shape(d, kind, (ccx - rad, ccy - rad, ccx + rad, ccy + rad),
                        COLORS[col])
        out.append({
            "id": f"count_{i:02d}", "family": "counting", "images": [img],
            "prompt": (f"How many {colour} {shape}s are in this image? "
                       "Answer with the number only."),
            "reference": str(count), "scoring": "first_int",
            "meta": {"colour": colour, "shape": shape, "count": count,
                     "distractors": n_distract},
        })
    return out


SCENE_KINDS = ["left_right", "above_below", "big_small", "two_objects", "three_row"]


def gen_scene(rng: random.Random, n: int):
    out = []
    for i in range(n):
        kind = SCENE_KINDS[i % len(SCENE_KINDS)]
        c1, c2 = rng.sample(list(COLORS), 2)
        s1, s2 = rng.sample(SHAPES, 2)
        size = 448
        img = Image.new("RGB", (size, size), (252, 252, 252))
        d = ImageDraw.Draw(img)
        r = 58
        if kind == "left_right":
            _draw_shape(d, s1, (60, size // 2 - r, 60 + 2 * r, size // 2 + r), COLORS[c1])
            _draw_shape(d, s2, (size - 60 - 2 * r, size // 2 - r,
                                size - 60, size // 2 + r), COLORS[c2])
            ref = (f"A {c1} {s1} is on the left and a {c2} {s2} is on the right.")
        elif kind == "above_below":
            _draw_shape(d, s1, (size // 2 - r, 50, size // 2 + r, 50 + 2 * r), COLORS[c1])
            _draw_shape(d, s2, (size // 2 - r, size - 50 - 2 * r, size // 2 + r,
                                size - 50), COLORS[c2])
            ref = (f"A {c1} {s1} is above a {c2} {s2}.")
        elif kind == "big_small":
            _draw_shape(d, s1, (60, 70, 60 + 220, 70 + 220), COLORS[c1])
            _draw_shape(d, s2, (size - 150, size - 150, size - 50, size - 50), COLORS[c2])
            ref = (f"A large {c1} {s1} and a small {c2} {s2}.")
        elif kind == "two_objects":
            _draw_shape(d, s1, (70, size // 2 - r, 70 + 2 * r, size // 2 + r), COLORS[c1])
            _draw_shape(d, s2, (size // 2 + 10, size // 2 - r, size // 2 + 10 + 2 * r,
                                size // 2 + r), COLORS[c2])
            ref = (f"A {c1} {s1} next to a {c2} {s2} on a white background.")
        else:
            for k in range(3):
                x = 40 + k * 130
                _draw_shape(d, s1, (x, size // 2 - r, x + 2 * r, size // 2 + r),
                            COLORS[c1] if k % 2 == 0 else COLORS[c2])
            ref = f"Three {s1}s in a row, alternating {c1} and {c2}."
        out.append({
            "id": f"scene_{i:02d}", "family": "scene", "images": [img],
            "prompt": "Describe this image in one sentence.",
            "reference": ref, "scoring": "rouge_l",
            "meta": {"kind": kind, "colours": [c1, c2], "shapes": [s1, s2]},
        })
    return out


MOTION_KINDS = ["right", "left", "up", "down", "grow", "shrink"]


def gen_motion(rng: random.Random, n: int):
    out = []
    for i in range(n):
        kind = MOTION_KINDS[i % len(MOTION_KINDS)]
        colour = rng.choice(list(COLORS))
        n_frames = 3 + (i % 2)          # alternating 3 / 4 frames
        w, h = 448, 336
        frames = []
        base = 52
        span = 300 if kind in ("right", "left", "up", "down") else 100
        for t in range(n_frames):
            f = t / max(n_frames - 1, 1)
            img = Image.new("RGB", (w, h), (250, 250, 250))
            d = ImageDraw.Draw(img)
            if kind == "right":
                cx, cy, r = 60 + f * span, h // 2, base
            elif kind == "left":
                cx, cy, r = w - 60 - f * span, h // 2, base
            elif kind == "up":
                cx, cy, r = w // 2, h - 60 - f * span, base
            elif kind == "down":
                cx, cy, r = w // 2, 60 + f * span, base
            elif kind == "grow":
                cx, cy, r = w // 2, h // 2, base * (0.55 + 0.75 * f)
            else:
                cx, cy, r = w // 2, h // 2, base * (1.3 - 0.75 * f)
            d.rectangle((cx - r, cy - r, cx + r, cy + r), fill=COLORS[colour])
            frames.append(img)
        if kind in ("right", "left", "up", "down"):
            prompt = ("These frames are in time order. In which direction does the "
                      "square move? Answer with one word: left, right, up or down.")
            ref = kind
        else:
            prompt = ("These frames are in time order. Does the square get bigger or "
                      "smaller? Answer with one word.")
            ref = "bigger" if kind == "grow" else "smaller"
        out.append({
            "id": f"motion_{i:02d}", "family": "motion", "images": frames,
            "prompt": prompt, "reference": ref, "scoring": "word_match",
            "meta": {"kind": kind, "colour": colour, "n_frames": n_frames},
        })
    return out


# --------------------------------------------------------------------------- #
# write
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-ocr", type=int, default=16)
    ap.add_argument("--n-count", type=int, default=14)
    ap.add_argument("--n-scene", type=int, default=14)
    ap.add_argument("--n-motion", type=int, default=14)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    samples = (gen_ocr(rng, args.n_ocr) + gen_counting(rng, args.n_count)
               + gen_scene(rng, args.n_scene) + gen_motion(rng, args.n_motion))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for s in samples:
        sub = DATA_DIR / s["family"]
        sub.mkdir(exist_ok=True)
        paths, shas = [], []
        for k, img in enumerate(s["images"]):
            p = sub / (f"{s['id']}.png" if len(s["images"]) == 1
                       else f"{s['id']}_f{k}.png")
            img.save(p)
            paths.append(str(p.relative_to(ROOT)))
            shas.append(_sha256(p))
        e = {k: v for k, v in s.items() if k != "images"}
        e["files"] = paths
        e["sha256"] = shas
        entries.append(e)

    manifest = {
        "seed": args.seed,
        "generator": "experiments/cm-transformer/make_p6_vlm.py",
        "generator_sha256": _sha256(Path(__file__)),
        "dataset_kind": "synthetic-controlled",
        "note": ("All images are rendered with PIL from random.Random(seed) with a fixed "
                 "draw order: byte-for-byte reproducible from this script, no photographs, "
                 "no network. Every sample carries a reference answer and a scoring rule. "
                 "This is a controlled capability probe, not a natural-image benchmark."),
        "families": {
            "ocr": "rendered alphanumeric string, scored exact-match after lowercasing "
                   "and dropping non-alphanumerics",
            "counting": "N shapes of one colour + differently-coloured distractors on a "
                        "3x3 grid, scored on the first integer in the answer",
            "scene": "fixed known composition, scored ROUGE-L (LCS F1) against the "
                     "reference sentence",
            "motion": "3-4 video frames of one moving/growing square, scored on exact "
                      "word match (direction / bigger / smaller)",
        },
        "counts": {k: sum(1 for e in entries if e["family"] == k)
                   for k in ("ocr", "counting", "scene", "motion")},
        "n_samples": len(entries),
        "samples": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(entries)} samples to {DATA_DIR}")
    print(f"manifest: {MANIFEST}  ({MANIFEST.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
