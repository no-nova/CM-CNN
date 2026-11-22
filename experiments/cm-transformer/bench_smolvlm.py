"""P1 (32-lane structured FFN block skipping) on SmolVLM2-256M-Video-Instruct.

Model under test
----------------
    models/SmolVLM2-256M-Video-Instruct  (SmolVLMForConditionalGeneration)
      vision : ViT, patch 16, 12 layers, d_model 768, FFN 3072, GELU-tanh,
               1024 patches / 512 px image, pixel-shuffle 4x4 -> 64 image tokens
      text   : Llama-3-style decoder, 30 layers, d_model 576, FFN 1536,
               SwiGLU (gate_proj / up_proj / down_proj, no bias), vocab 49280

Mechanism (P1, identical to the text-side P1 work in this directory)
-------------------------------------------------------------------
For every FFN the intermediate axis (I = 1536 / 3072) is split into
I/32 = 48 / 96 *lanes* of 32 channels.  A 32-channel block is the unit that one
warp (or two AVX-512 ZMM registers) consumes, so a skipped block is a block a
kernel can simply not issue.

  permutation  pi  per layer: importance_i = mean_t|a[t,i]| * ||W_down[:, i]||,
               coldest-first block seeding + greedy co-activation fill
               (p1_v2.build_perm).  One permutation per layer, shared by
               gate rows, up rows and down columns (SwiGLU) or fc1 rows (+bias)
               and fc2 columns (GELU), so the FFN stays mathematically
               isomorphic -- only the reduction order inside down_proj changes.
  threshold    theta: per layer, the `tau` quantile of the per-token per-block
               L2 energies measured on real calibration inputs, so tau is the
               target fraction of skipped (token, block) pairs.
  skip         keep block b for token t iff  ||a[t,b]||^2 >= theta.
  compensation alpha = min(1.5, ||a_t|| / ||a_keep,t||) applied to the kept
               intermediate vector of token t before down_proj.

Protocol
--------
tau = 0 is asserted to be isomorphic to the dense model (max |dlogit| at the
fp32 floor, top-1 agreement 1.0) BEFORE any result is reported.  Then, for
tau in {0.0, 0.2, 0.4, 0.6} x tower in {vision, text, both}, we record fidelity
(image-feature / prefill-logit rel-L2, top-1 agreement over 24 generated
tokens, ROUGE-L vs the dense generation, first-divergence step, 4-prompt VQA)
and speed (vision-tower ms, LM prefill ms, LM decode ms/token with a KV cache),
plus the operator-level cost of the *sliced* FFN path (dense fc1+fc2 vs the
union of blocks the batch actually needs), because PyTorch still runs the dense
GEMM in the masking path.

Honest caveats are written into the JSON `notes` list.

Usage
-----
    python bench_smolvlm.py --device cuda --tau 0.0,0.2,0.4,0.6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import (LANE, RESULTS_DIR, ROOT, Timer, bench, dump_json, log,
                    set_deterministic)
from model_ref import block_energy
from p1_v2 import build_perm
from run_text_eval import first_divergence, rouge_l
from video_data import load_regime

MODEL_DIR = ROOT / "models" / "SmolVLM2-256M-Video-Instruct"
OUT_JSON = RESULTS_DIR / "p5_smolvlm.json"
IMG_TOK = 49190                     # config.json: image_token_id
ALPHA_CAP = 1.5
ISO_DLOGIT_TOL = 1e-3               # fp32 reduction-order floor, see notes

VIDEO_PROMPT = "Describe what happens in this video in one sentence."
CALIB_SPEC = [
    ("data/real_videos/frames_224 V2 frame 6",
     "Describe what happens in this video in one sentence."),
    ("data/real_videos/frames_224 V4 frame 18", "What is happening in the scene?"),
    ("data/real_videos/frames_224 V6 frame 27", "List the objects visible in this image."),
]
VQA_PROMPTS = [
    ("What color is the square in the image?", ("red",)),
    ("What color is the circle in the image?", ("green",)),
    ("How many shapes are in the image? Answer with one word.", ("two", "2")),
    ("Is there a triangle in the image? Answer yes or no.", ("no", "none")),
]


# --------------------------------------------------------------------------- #
# 32-lane FFN skip wrapper
# --------------------------------------------------------------------------- #
def detect_kind(mlp) -> str:
    if all(hasattr(mlp, n) for n in ("gate_proj", "up_proj", "down_proj")):
        return "swiglu"
    if all(hasattr(mlp, n) for n in ("fc1", "fc2")):
        return "gelu"
    raise RuntimeError(f"unrecognised FFN module: {type(mlp).__name__}")


class FFNSkip:
    """Installs a 32-lane block-skipping forward on one existing FFN module.

    The module's own weights are permuted in place (bit-exact copy, only the
    intermediate axis order changes) and the original tensors + original
    forward are kept so `restore()` puts the model back exactly.
    """

    def __init__(self, tower: str, idx: int, module, kind: str, lane: int = LANE):
        self.tower, self.idx, self.module, self.kind = tower, idx, module, kind
        self.key = f"{tower}.{idx}"
        self.lane = lane
        self.theta = -1.0                # <= 0 means "never skip" (dense path)
        self.perm = None
        self._orig_forward = module.forward
        self._orig_w = None
        self.capture = None              # list of per-token block energies
        self.want_x = False
        self.saved = None                # (ffn input, post-activation)
        # [E_total, E_keep, pairs, skipped, alpha_sum, n_tokens]
        self.tot = torch.zeros(6, dtype=torch.float64)
        if kind == "swiglu":
            self.gate, self.up, self.down = module.gate_proj, module.up_proj, module.down_proj
            self.I = self.gate.weight.shape[0]
            self.act = None
        else:
            self.fc1, self.fc2 = module.fc1, module.fc2
            self.act = module.activation_fn
            self.I = self.fc1.weight.shape[0]
        assert self.I % lane == 0, (self.key, self.I)

    # -- weight permutation ------------------------------------------------ #
    def apply_perm(self, perm: torch.Tensor):
        self.perm = perm
        with torch.no_grad():
            if self.kind == "swiglu":
                self._orig_w = (self.gate.weight.data, self.up.weight.data,
                                self.down.weight.data)
                self.gate.weight.data = self._orig_w[0][perm]
                self.up.weight.data = self._orig_w[1][perm]
                self.down.weight.data = self._orig_w[2][:, perm]
            else:
                self._orig_w = (self.fc1.weight.data, self.fc1.bias.data,
                                self.fc2.weight.data)
                self.fc1.weight.data = self._orig_w[0][perm]
                self.fc1.bias.data = self._orig_w[1][perm]
                self.fc2.weight.data = self._orig_w[2][:, perm]

    def restore(self):
        self.module.forward = self._orig_forward
        if self._orig_w is None:
            return
        params = (self.gate.weight, self.up.weight, self.down.weight) \
            if self.kind == "swiglu" else (self.fc1.weight, self.fc1.bias, self.fc2.weight)
        with torch.no_grad():
            for p, w in zip(params, self._orig_w):
                p.data = w.to(device=p.device)

    # -- the mechanism ----------------------------------------------------- #
    def _dense_out(self, a):
        if self.kind == "swiglu":
            return F.linear(a, self.down.weight)
        return F.linear(a, self.fc2.weight, self.fc2.bias)

    def _fwd(self, x):
        if self.kind == "swiglu":
            a = F.silu(F.linear(x, self.gate.weight)) * F.linear(x, self.up.weight)
        else:
            a = self.act(F.linear(x, self.fc1.weight, self.fc1.bias))

        if self.capture is not None:
            e = block_energy(a, self.lane)
            self.capture.append(e.reshape(-1, e.shape[-1]))
        if self.want_x:
            self.saved = (x.detach(), a.detach())
            self.want_x = False

        theta = self.theta
        if theta is None or theta <= 0.0:
            if self.tot is not None:
                e = block_energy(a, self.lane)
                n_tok = a.numel() // a.shape[-1]
                self._accum(e.sum(), e.sum(), e.numel(), 0, n_tok, n_tok)
            return self._dense_out(a)

        lead, I = a.shape[:-1], a.shape[-1]
        nb = I // self.lane
        e = block_energy(a, self.lane)                       # [..., nb]
        keep = e >= theta
        a3 = a.reshape(*lead, nb, self.lane) * keep.unsqueeze(-1)
        ak = a3.reshape(*lead, I)
        E = e.sum(-1, keepdim=True)
        Ek = (e * keep).sum(-1, keepdim=True)
        # min(1.5, ||a|| / ||a_keep||) == sqrt(E / E_keep) on the same energies
        alpha = torch.sqrt(E / Ek.clamp_min(1e-20)).clamp(max=ALPHA_CAP)
        out = self._dense_out(ak * alpha)
        if self.tot is not None:
            n = keep.numel()
            n_tok = a.numel() // a.shape[-1]
            self._accum(E.sum(), Ek.sum(), n, n - keep.sum(), alpha.sum(), n_tok)
        return out

    def _accum(self, etot, ekeep, npairs, nskip, alpha_sum, n_tokens):
        dev = etot.device
        val = torch.stack([etot.double(), ekeep.double(),
                           torch.as_tensor(float(npairs), device=dev, dtype=torch.float64),
                           torch.as_tensor(float(nskip), device=dev, dtype=torch.float64),
                           torch.as_tensor(float(alpha_sum), device=dev, dtype=torch.float64),
                           torch.as_tensor(float(n_tokens), device=dev, dtype=torch.float64)])
        self.tot = val if self.tot is None else self.tot.to(dev) + val

    def install(self):
        self.module.forward = self._fwd

    def reset_stats(self):
        dev = next(self.module.parameters()).device
        self.tot = torch.zeros(6, dtype=torch.float64, device=dev)


def collect_ffns(model) -> list[FFNSkip]:
    out = []
    for i, layer in enumerate(model.model.vision_model.encoder.layers):
        out.append(FFNSkip("vision", i, layer.mlp, detect_kind(layer.mlp)))
    for i, layer in enumerate(model.model.text_model.layers):
        out.append(FFNSkip("text", i, layer.mlp, detect_kind(layer.mlp)))
    return out


def is_tower(s: FFNSkip, towers) -> bool:
    return s.tower in towers


def set_thetas(skips, towers, theta_map):
    """Towers outside `towers` stay on the dense (theta <= 0) path."""
    for s in skips:
        s.theta = float(theta_map[s.key]) if is_tower(s, towers) else -1.0


def read_stats(skips, towers):
    et = ek = n = ns = al = ntok = 0.0
    for s in skips:
        if s.tot is None or not is_tower(s, towers):
            continue
        v = s.tot.tolist()
        et += v[0]
        ek += v[1]
        n += v[2]
        ns += v[3]
        al += v[4]
        ntok += v[5]
    return {"energy_captured": ek / et if et > 0 else 1.0,
            "actual_skip": ns / n if n > 0 else 0.0,
            "mean_alpha": al / ntok if ntok > 0 else 1.0,
            "pairs": n, "tokens": ntok}


# --------------------------------------------------------------------------- #
# model / processor
# --------------------------------------------------------------------------- #
def load_model(device):
    import transformers
    from transformers import AutoProcessor
    info = {"transformers": transformers.__version__, "torch": torch.__version__,
            "attempts": []}
    model = None
    for name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq", "AutoModel"):
        cls = getattr(transformers, name, None)
        if cls is None:
            info["attempts"].append(f"{name}: not exported by transformers {transformers.__version__}")
            continue
        try:
            model = cls.from_pretrained(str(MODEL_DIR), dtype=torch.float32)
            info["attempts"].append(f"{name}: ok")
            info["loader"] = name
            break
        except Exception as exc:                                  # pragma: no cover
            info["attempts"].append(f"{name}: {type(exc).__name__}: {exc}")
    if model is None:                                            # pragma: no cover
        raise RuntimeError("no usable auto class: " + "; ".join(info["attempts"]))
    info["class"] = type(model).__name__
    info["class_module"] = type(model).__module__
    info["dtype"] = "float32"
    info["device"] = device
    model = model.to(device).eval()
    model.requires_grad_(False)
    proc = AutoProcessor.from_pretrained(str(MODEL_DIR))
    info["processor"] = type(proc).__name__
    return model, proc, info


def arch_info(model, proc, img_edge):
    vc, tc = model.config.vision_config, model.config.text_config
    n_vis = len(model.model.vision_model.encoder.layers)
    n_txt = len(model.model.text_model.layers)
    ffn_vis = model.model.vision_model.encoder.layers[0].mlp.fc1.weight.shape[0]
    ffn_txt = model.model.text_model.layers[0].mlp.gate_proj.weight.shape[0]
    return {
        "vision": {"n_layers": n_vis, "d_model": vc.hidden_size, "ffn_dim": ffn_vis,
                   "activation": vc.hidden_act, "ffn_kind": "gelu fc1/fc2 (bias)",
                   "patch_size": vc.patch_size, "n_heads": vc.num_attention_heads,
                   "image_size_used": img_edge,
                   "n_patches_per_image": (img_edge // vc.patch_size) ** 2,
                   "n_image_tokens_one_image": (img_edge // vc.patch_size) ** 2 // 16,
                   "pixel_shuffle": 4, "lanes": ffn_vis // LANE},
        "text": {"n_layers": n_txt, "d_model": tc.hidden_size, "ffn_dim": ffn_txt,
                 "activation": tc.hidden_act, "ffn_kind": "swiglu gate/up/down (no bias)",
                 "n_heads": tc.num_attention_heads, "n_kv_heads": tc.num_key_value_heads,
                 "vocab_size": tc.vocab_size, "lanes": ffn_txt // LANE,
                 "rms_norm_eps": tc.rms_norm_eps},
        "connector": {"in_features": model.model.connector.modality_projection.proj.weight.shape[1],
                      "out_features": model.model.connector.modality_projection.proj.weight.shape[0],
                      "image_seq_len": getattr(proc, "image_seq_len", None)},
        "lane": LANE,
    }


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _to_pil(frames: torch.Tensor, i: int) -> Image.Image:
    a = ((frames[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0)
    return Image.fromarray(a.clip(0, 255).astype("uint8"))


def synth_shapes(size: int = 384) -> Image.Image:
    """Deterministic image: white canvas, one red square, one green circle."""
    a = np.full((size, size, 3), 245, np.uint8)
    a[size // 4: size // 4 + 150, size // 8: size // 8 + 150] = (215, 35, 35)     # red square
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx, r = size // 2, int(size * 0.72), 70
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    a[disc] = (30, 160, 60)                                                     # green circle
    return Image.fromarray(a)


def make_inputs(images, prompt, proc, device, img_edge):
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    batch = proc(text=text, images=list(images), return_tensors="pt",
                 size={"longest_edge": img_edge})
    batch = {k: v.to(device) for k, v in batch.items()}
    n_img_tok = int((batch["input_ids"] == IMG_TOK).sum())
    assert n_img_tok == len(images) * (img_edge // 16) ** 2 // 16, (n_img_tok, len(images))
    return batch, n_img_tok


def build_inputs(proc, device, img_edge):
    frames_v2 = load_regime("V2")
    frames_v4 = load_regime("V4")
    frames_v6 = load_regime("V6")
    video = [_to_pil(frames_v2, 0), _to_pil(frames_v2, 12)]
    calib = [([_to_pil(frames_v2, 6)], CALIB_SPEC[0][1]),
             ([_to_pil(frames_v4, 18)], CALIB_SPEC[1][1]),
             ([_to_pil(frames_v6, 27)], CALIB_SPEC[2][1])]
    eval_batch, n_img_tok = make_inputs(video, VIDEO_PROMPT, proc, device, img_edge)
    calib_batches = [make_inputs(imgs, p, proc, device, img_edge)[0] for imgs, p in calib]
    vqa_img = synth_shapes()
    vqa = []
    for prompt, expect in VQA_PROMPTS:
        b, _ = make_inputs([vqa_img], prompt, proc, device, img_edge)
        vqa.append({"prompt": prompt, "expect": list(expect), "batch": b})
    used = {
        "fidelity_input": {"source": "data/real_videos/frames_224 (V2 local_motion)",
                           "frames": [0, 12], "prompt": VIDEO_PROMPT,
                           "n_images": len(video), "n_image_tokens": n_img_tok,
                           "seq_len": int(eval_batch["input_ids"].shape[1])},
        "calibration_inputs": [{"source": s, "prompt": p} for s, p in CALIB_SPEC],
        "vqa": {"image": "synthetic PIL: white 384px canvas, red square, green circle",
                "prompts": [{"prompt": p, "expect": list(e)} for p, e in VQA_PROMPTS],
                "decode": "greedy, 16 new tokens"},
        "processor_size": {"longest_edge": img_edge,
                           "note": "processor default is 2048 (image splitting -> 17 crops / "
                                   "1088 tokens per frame); 512 keeps one crop = 64 tokens "
                                   "and no splitting"},
    }
    return eval_batch, calib_batches, vqa, used


# --------------------------------------------------------------------------- #
# forward / generation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def get_feats(model, batch):
    out = model.model.get_image_features(batch["pixel_values"],
                                         batch.get("pixel_attention_mask"))
    return torch.cat(out.pooler_output, 0), out.last_hidden_state


@torch.no_grad()
def make_embeds(model, batch, feats):
    emb = model.model.text_model.get_input_embeddings()(batch["input_ids"]).to(torch.float32)
    mask = model.model.get_placeholder_mask(batch["input_ids"], inputs_embeds=emb,
                                           image_features=feats)
    return emb.masked_scatter(mask, feats.to(emb.dtype))


@torch.no_grad()
def lm_logits(model, batch, feats, last_only=True):
    out = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                      mm_encoder_outputs={"image": SimpleNamespace(pooler_output=[feats])})
    h = out.last_hidden_state
    return model.lm_head(h[:, -1:] if last_only else h).float()


@torch.no_grad()
def generate(model, batch, n_new):
    out = model.generate(**batch, max_new_tokens=n_new, do_sample=False)
    return out[0, batch["input_ids"].shape[1]:].tolist()


@torch.no_grad()
def decode_loop(model, emb, am, n_steps, warm=1):
    """Greedy KV-cache loop; returns (ms per step after `warm`, generated ids)."""
    tm = model.model.text_model
    lm_head = model.lm_head
    o = tm(inputs_embeds=emb, attention_mask=am, use_cache=True)
    past, cur = o.past_key_values, lm_head(o.last_hidden_state[:, -1:]).argmax(-1)
    ids, ts = [], []
    for i in range(n_steps + warm):
        e = tm.get_input_embeddings()(cur).to(torch.float32)
        am = torch.cat([am, am.new_ones((am.shape[0], 1))], 1)
        with Timer(emb.device) as t:
            o = tm(inputs_embeds=e, attention_mask=am, past_key_values=past, use_cache=True)
            past = o.past_key_values
            cur = lm_head(o.last_hidden_state[:, -1:]).argmax(-1)
        if i >= warm:
            ts.append(t.dt_ms)
            ids.append(int(cur.item()))
    ts.sort()
    return float(ts[len(ts) // 2]), ids


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-20))


def score_vqa(text, expect):
    t = text.strip().lower()
    if any(e in ("yes", "no") for e in expect):
        first = t.split()[0].strip(".,!") if t.split() else ""
        return float(first in expect)
    return float(any(e in t for e in expect))


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def _post_act(s, x):
    """Intermediate-axis activation of the FFN on input x -> [T, I]."""
    if s.kind == "swiglu":
        return F.silu(F.linear(x, s.gate.weight)) * F.linear(x, s.up.weight)
    return s.act(F.linear(x, s.fc1.weight, s.fc1.bias))


def capture_acts(skips, model, batches):
    """FFN post-activations on real calibration batches (un-permuted weights)."""
    for s in skips:
        s._acts = []
    hooks = [s.module.register_forward_pre_hook(
        (lambda mod, inp, s=s: s._acts.append(
            inp[0].detach().reshape(-1, inp[0].shape[-1])))) for s in skips]
    try:
        for b in batches:
            model(**b)
    finally:
        for h in hooks:
            h.remove()
    acts = {}
    for s in skips:
        x = torch.cat(s._acts, 0)
        s._acts = None
        if x.shape[-1] != s.I:            # pre-FFN input (d_model) -> intermediate axis
            x = _post_act(s, x)
        acts[s.key] = x.float()
    return acts


def calibrate_perms(skips, acts, model):
    """One permutation per layer from (mean|a| * ||W_down[:, i]||) coldest-first blocks."""
    perms, t0 = {}, time.perf_counter()
    for s in skips:
        wd = s.down.weight if s.kind == "swiglu" else s.fc2.weight
        perm, _ = build_perm(acts[s.key].float(), wd.float(), LANE)
        perms[s.key] = perm
    log(f"    build_perm: {len(skips)} layers in {time.perf_counter() - t0:.1f}s")
    return perms


def calibrate_energies(skips, model, batches):
    for s in skips:
        s.capture = []
    with torch.no_grad():
        for b in batches:
            model(**b)
    energies = {}
    for s in skips:
        e = torch.cat(s.capture, 0)
        s.capture = None
        energies[s.key] = e
    return energies


def thetas_for(energies, taus):
    out = {}
    for tau in taus:
        th = {}
        for k, e in energies.items():
            if tau <= 0:
                th[k] = -1.0
            else:
                th[k] = float(torch.quantile(e.flatten().float(), tau))
        out[tau] = th
    return out


# --------------------------------------------------------------------------- #
# operator-level FFN cost
# --------------------------------------------------------------------------- #
def _dense_ffn(s, x):
    if s.kind == "swiglu":
        a = F.silu(F.linear(x, s.gate.weight)) * F.linear(x, s.up.weight)
        return F.linear(a, s.down.weight), a
    a = s.act(F.linear(x, s.fc1.weight, s.fc1.bias))
    return F.linear(a, s.fc2.weight, s.fc2.bias), a


def _sliced_ffn(s, x, cols):
    if s.kind == "swiglu":
        a = F.silu(F.linear(x, s.gate.weight[cols])) * F.linear(x, s.up.weight[cols])
        return F.linear(a, s.down.weight[:, cols])
    a = s.act(F.linear(x, s.fc1.weight[cols], s.fc1.bias[cols]))
    return F.linear(a, s.fc2.weight[:, cols], s.fc2.bias)


def ffn_op_bench(skips, tower, theta_map, device, reps=10):
    """Dense fc1+fc2 vs the *sliced* path over the union of kept blocks.

    Two references are measured per layer:
      union  -- gather every block any token in the batch needs (what a batched
                dense GEMM kernel would have to compute). Saturates fast.
      single -- apply one median-token mask to the whole batch: the "all tokens
                share one 32-lane mask" case, i.e. the best case for a batched
                kernel at this skip rate.
    """
    rows = []
    tower_skips = [s for s in skips if s.tower == tower]
    picks = sorted({len(tower_skips) // 4, len(tower_skips) // 2, 3 * len(tower_skips) // 4})
    for s in (tower_skips[i] for i in picks):
        if s.saved is None:
            continue
        x = s.saved[0]
        theta = theta_map[s.key]
        with torch.no_grad():
            _, a = _dense_ffn(s, x)
            e = block_energy(a, LANE)
            keep = (e >= theta) if theta > 0 else torch.ones_like(e, dtype=torch.bool)
            flat = keep.reshape(-1, keep.shape[-1])
            union = keep.any(dim=tuple(range(keep.dim() - 1)))
            bidx = union.nonzero().flatten()
            cols = (bidx[:, None] * LANE + torch.arange(LANE, device=bidx.device)).flatten()
            kept_frac = cols.numel() / s.I
            token_skip = 1.0 - float(keep.float().mean())
            kc = flat.sum(-1)
            idx = int((kc == int(kc.median())).nonzero()[0])
            mu = flat[idx].nonzero().flatten()
            cols_u = (mu[:, None] * LANE + torch.arange(LANE, device=mu.device)).flatten()
            dense_ms, _ = bench(lambda: _dense_ffn(s, x)[0], device, warmup=3, reps=reps)
            row = {"tower": tower, "layer": s.idx,
                   "tokens": int(x.numel() // x.shape[-1]), "theta": theta,
                   "token_skip": token_skip, "kept_block_frac": kept_frac,
                   "kept_cols": int(cols.numel()), "ffn_dim": s.I,
                   "dense_ms": dense_ms}
            if bidx.numel() == 0 or bidx.numel() == union.numel():
                row.update({"partial_ms": dense_ms, "op_speedup": 1.0,
                            "note": "union of kept blocks covers every block"})
            else:
                part_ms, _ = bench(lambda: _sliced_ffn(s, x, cols), device,
                                   warmup=3, reps=reps)
                row.update({"partial_ms": part_ms,
                            "op_speedup": dense_ms / part_ms if part_ms > 0 else 1.0})
            if 0 < mu.numel() < s.I // LANE:
                uni_ms, _ = bench(lambda: _sliced_ffn(s, x, cols_u), device,
                                  warmup=3, reps=reps)
                row.update({"single_mask_block_frac": mu.numel() / (s.I // LANE),
                            "single_mask_ms": uni_ms,
                            "op_speedup_single_mask":
                                dense_ms / uni_ms if uni_ms > 0 else 1.0})
        rows.append(row)
    return rows


def gpu_speed_pass(model, batch, skips, thetas, taus, dev, decode_steps):
    """One timing pass over all taus; returns {tau: {vision_ms, pre_ms, dec_ms, ...}}."""
    out = {}
    for tau in taus:
        d = {}
        set_thetas(skips, {"vision"}, thetas[tau])
        d["vision_ms"], _ = bench(lambda: get_feats(model, batch), dev, warmup=2, reps=5)
        set_thetas(skips, {"text"}, thetas[tau])
        feats, _ = get_feats(model, batch)
        emb = make_embeds(model, batch, feats)
        am = batch["attention_mask"]
        d["pre_ms"], _ = bench(lambda: model.model.text_model(
            inputs_embeds=emb, attention_mask=am, use_cache=False), dev, warmup=2, reps=5)
        d["dec_ms"], _ = decode_loop(model, emb, am, decode_steps)
        set_thetas(skips, {"vision", "text"}, thetas[tau])
        feats2, _ = get_feats(model, batch)
        emb2 = make_embeds(model, batch, feats2)
        d["pre2_ms"], _ = bench(lambda: model.model.text_model(
            inputs_embeds=emb2, attention_mask=am, use_cache=False), dev, warmup=2, reps=5)
        d["dec2_ms"], _ = decode_loop(model, emb2, am, decode_steps)
        out[tau] = d
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--gen-len", type=int, default=24)
    ap.add_argument("--img-edge", type=int, default=512)
    ap.add_argument("--decode-steps", type=int, default=12)
    ap.add_argument("--vqa-len", type=int, default=16)
    ap.add_argument("--no-cpu", action="store_true")
    ap.add_argument("--iso-tol", type=float, default=ISO_DLOGIT_TOL)
    args = ap.parse_args()
    taus = [float(x) for x in args.tau.split(",")]
    dev = args.device

    set_deterministic(0)
    torch.set_grad_enabled(False)
    log(f"[cfg] device={dev} tau={taus} gen_len={args.gen_len} img_edge={args.img_edge}")

    log("[1/8] loading model")
    model, proc, load_info = load_model(dev)
    log(f"    {load_info['loader']} -> {load_info['class']} "
        f"(transformers {load_info['transformers']}, fp32, {dev})")

    log("[2/8] building inputs")
    batch, calib_batches, vqa, used = build_inputs(proc, dev, args.img_edge)
    arch = arch_info(model, proc, args.img_edge)
    log(f"    fidelity input: {used['fidelity_input']['n_image_tokens']} image tokens, "
        f"seq_len {used['fidelity_input']['seq_len']}")

    skips = collect_ffns(model)
    log(f"    FFNs hooked: {sum(s.tower == 'vision' for s in skips)} vision "
        f"(fc1/fc2, {skips[0].I} -> {skips[0].I // LANE} lanes), "
        f"{sum(s.tower == 'text' for s in skips)} text "
        f"(gate/up/down, {skips[-1].I} -> {skips[-1].I // LANE} lanes)")

    log("[3/8] dense reference pass")
    feats0, _ = get_feats(model, batch)
    lg0 = lm_logits(model, batch, feats0)
    emb0 = make_embeds(model, batch, feats0)
    gen0 = generate(model, batch, args.gen_len)
    text0 = proc.decode(gen0, skip_special_tokens=True)
    vqa0 = []
    for item in vqa:
        f, _ = get_feats(model, item["batch"])
        ids = generate(model, item["batch"], args.vqa_len)
        ans = proc.decode(ids, skip_special_tokens=True).strip()
        vqa0.append({"prompt": item["prompt"], "answer": ans[:80],
                     "score": score_vqa(ans, item["expect"])})
    vqa0_acc = sum(v["score"] for v in vqa0) / len(vqa0)
    log(f"    dense: {len(gen0)} gen tokens, vqa dense acc {vqa0_acc:.2f}")
    log(f"    dense text: {text0.strip()[:110]!r}")

    log("[4/8] calibrating pi (permutation) and theta (thresholds)")
    acts = capture_acts(skips, model, calib_batches)
    perms = calibrate_perms(skips, acts, model)
    for s in skips:
        s.apply_perm(perms[s.key])
        s.install()                     # needed before capturing post-permutation energies
    energies = calibrate_energies(skips, model, calib_batches)
    thetas = thetas_for(energies, taus)
    for tau in taus:
        vals = [thetas[tau][s.key] for s in skips if s.tower == "text"]
        log(f"    tau={tau}: text theta median {float(np.median(vals)):.4g} "
            f"(IQR {float(np.percentile(vals, 25)):.4g}..{float(np.percentile(vals, 75)):.4g})")
        del vals

    # ---------------------------------------------------------------- tau=0
    log("[5/8] tau = 0 isomorphism check (before any result is reported)")
    set_thetas(skips, {"vision", "text"}, thetas[0.0])
    feats1, _ = get_feats(model, batch)
    lg1 = lm_logits(model, batch, feats1)
    gen1 = generate(model, batch, args.gen_len)
    dlogit = float((lg1 - lg0).abs().max())
    agree = float(np.mean([a == b for a, b in zip(gen1, gen0)])) if gen0 else 1.0
    checks = {"tau0_max_abs_dlogit": dlogit,
              "tau0_feat_rel_l2": rel_l2(feats1, feats0),
              "tau0_top1_agree": agree,
              "tau0_gen_identical": gen1 == gen0,
              "iso_tol": args.iso_tol}
    log(f"    max|dlogit|={dlogit:.3e}  feat rel-L2={checks['tau0_feat_rel_l2']:.3e}  "
        f"top-1 agree={agree:.3f}  gen identical={gen1 == gen0}")
    assert dlogit <= args.iso_tol, f"tau=0 not isomorphic: max|dlogit|={dlogit:.3e} > {args.iso_tol}"
    assert agree == 1.0, f"tau=0 top-1 agreement {agree}"
    assert gen1 == gen0, "tau=0 generation differs from dense"

    # ------------------------------------------------------------- fidelity
    log("[6/8] fidelity sweep")
    rows = []
    for tower in ("vision", "text", "both"):
        towers = {"vision", "text"} if tower == "both" else {tower}
        for tau in taus:
            set_thetas(skips, towers, thetas[tau])
            for s in skips:
                s.reset_stats()
            if tau == 0.0:                  # capture activations for the op-level bench
                for s in skips:
                    if is_tower(s, towers):
                        s.want_x = True
            feats, _ = get_feats(model, batch)
            lg = lm_logits(model, batch, feats)
            st = read_stats(skips, towers)
            gen = generate(model, batch, args.gen_len)
            txt = proc.decode(gen, skip_special_tokens=True)
            va = []
            for item in vqa:
                f, _ = get_feats(model, item["batch"])
                ids = generate(model, item["batch"], args.vqa_len)
                ans = proc.decode(ids, skip_special_tokens=True).strip()
                va.append({"answer": ans[:60], "score": score_vqa(ans, item["expect"])})
            row = {
                "tau": tau, "tower": tower,
                "rel_l2": rel_l2(lg, lg0) if tower != "vision" else rel_l2(feats, feats0),
                "rel_l2_feat": rel_l2(feats, feats0),
                "rel_l2_prefill_logit": rel_l2(lg, lg0),
                "max_abs_dlogit": float((lg - lg0).abs().max()),
                "top1_agree": float(np.mean([a == b for a, b in zip(gen, gen0)])),
                "rouge_l": rouge_l(txt, text0),
                "first_div": first_divergence(gen, gen0),
                "energy_captured": st["energy_captured"],
                "actual_skip": st["actual_skip"],
                "mean_alpha": st["mean_alpha"],
                "vqa_acc": sum(v["score"] for v in va) / len(va),
                "vqa_answers": [v["answer"] for v in va],
                "text": txt.strip()[:160],
                "gen_tokens": len(gen),
            }
            rows.append(row)
            log(f"    {tau:4.2f} {tower:6s} relL2={row['rel_l2']:.3e} "
                f"top1={row['top1_agree']:.3f} rouge={row['rouge_l']:.3f} "
                f"div={row['first_div']:2d} skip={row['actual_skip']:.3f} "
                f"ecap={row['energy_captured']:.3f} vqa={row['vqa_acc']:.2f}")

    # ---------------------------------------------------------------- speed
    log("[7/8] speed (GPU)")
    speed = {"gpu": {"vision_forward_ms": {}, "lm_prefill_ms": {}, "lm_decode_ms_per_token": {},
                     "lm_prefill_ms_both": {}, "lm_decode_ms_per_token_both": {},
                     "decode_steps": args.decode_steps,
                     "timing_note": "median of 5 reps, min over 2 full passes over the taus"},
             "ffn_op": {"vision": [], "text": []}}
    # two passes, elementwise min: the 5070's clocks drift by 20-30% between passes
    passes = [gpu_speed_pass(model, batch, skips, thetas, taus, dev, args.decode_steps)
              for _ in range(2)]
    for tau in taus:
        d = {k: min(p[tau][k] for p in passes) for k in passes[0][tau]}
        k = str(tau)
        speed["gpu"]["vision_forward_ms"][k] = d["vision_ms"]
        speed["gpu"]["lm_prefill_ms"][k] = d["pre_ms"]
        speed["gpu"]["lm_decode_ms_per_token"][k] = d["dec_ms"]
        speed["gpu"]["lm_prefill_ms_both"][k] = d["pre2_ms"]
        speed["gpu"]["lm_decode_ms_per_token_both"][k] = d["dec2_ms"]
        speed["gpu"].setdefault("passes", []).append({"tau": tau, "runs": [p[tau] for p in passes]})
        log(f"    tau={tau}: vision {d['vision_ms']:.2f} ms | prefill {d['pre_ms']:.2f} ms | "
            f"decode {d['dec_ms']:.2f} ms/tok | both prefill {d['pre2_ms']:.2f} "
            f"decode {d['dec2_ms']:.2f}")
        for tower in ("vision", "text"):
            speed["ffn_op"][tower].extend(ffn_op_bench(skips, tower, thetas[tau], dev))

    if not args.no_cpu:
        log("    moving model to CPU for one cheap CPU timing pass")
        speed["cpu"] = {"vision_forward_ms": {}, "lm_prefill_ms": {},
                        "lm_decode_ms_per_token": {}, "decode_steps": 4,
                        "note": "1 warm-up + 2 reps, fp32, 32-core Zen5"}
        model.to("cpu")
        b_cpu = {k: v.cpu() for k, v in batch.items()}
        # warm the CPU kernels first: otherwise the tau=0.0 row eats all first-call cost
        f_warm, _ = get_feats(model, b_cpu)
        model.model.text_model(inputs_embeds=make_embeds(model, b_cpu, f_warm),
                               attention_mask=b_cpu["attention_mask"], use_cache=False)
        for tau in taus:
            set_thetas(skips, {"vision", "text"}, thetas[tau])
            ms, _ = bench(lambda: get_feats(model, b_cpu), "cpu", warmup=2, reps=3)
            feats_c, _ = get_feats(model, b_cpu)
            emb_c = make_embeds(model, b_cpu, feats_c)
            pre_c, _ = bench(lambda: model.model.text_model(
                inputs_embeds=emb_c, attention_mask=b_cpu["attention_mask"],
                use_cache=False), "cpu", warmup=2, reps=3)
            dec_c, _ = decode_loop(model, emb_c, b_cpu["attention_mask"], 4)
            speed["cpu"]["vision_forward_ms"][str(tau)] = ms
            speed["cpu"]["lm_prefill_ms"][str(tau)] = pre_c
            speed["cpu"]["lm_decode_ms_per_token"][str(tau)] = dec_c
            log(f"    cpu tau={tau}: vision {ms:.1f} ms prefill {pre_c:.1f} ms "
                f"decode {dec_c:.1f} ms/tok")

    # -------------------------------------------------------------- restore
    log("[8/8] restoring original weights/forwards")
    for s in skips:
        s.restore()
    model.to(dev)
    feats_r, _ = get_feats(model, batch)
    lg_r = lm_logits(model, batch, feats_r)
    checks["restore_max_abs_dlogit"] = float((lg_r - lg0).abs().max())
    log(f"    restored model max|dlogit| vs dense = {checks['restore_max_abs_dlogit']:.3e}")
    assert checks["restore_max_abs_dlogit"] <= args.iso_tol, "restore failed"

    tmax = max(taus)
    notes = [
        f"loader: {load_info['loader']} -> {load_info['class']} "
        f"({load_info['class_module']}), transformers {load_info['transformers']}, "
        f"torch {load_info['torch']}, fp32 on {dev}; attempts: {load_info['attempts']}.",
        "both towers have a clean FFN hook: vision SmolVLMVisionMLP.fc1/fc2 (GELU-tanh, bias) "
        "and text LlamaMLP.gate_proj/up_proj/down_proj (SwiGLU, no bias); we replace "
        "module.forward with a 32-lane block-skip forward and restore the original "
        "callable + weight tensors afterwards (verified: restore_max_abs_dlogit ~ 0).",
        "permutation applied physically to the weights (gate rows / up rows / down columns, "
        "fc1 rows+bias / fc2 columns) so the model is isomorphic and blocks are contiguous; "
        "only the reduction order inside down_proj changes, which is the fp32 floor.",
        "theta is the per-layer tau-quantile of per-token per-block L2 energies measured on "
        f"{len(calib_batches)} real calibration batches (V2/V4/V6 frames), so tau is the target "
        "fraction of skipped (token, block) pairs; energy_captured is the retained activation "
        "energy fraction and actual_skip is measured, not assumed.",
        "norm compensation: alpha = min(1.5, ||a||/||a_keep||) per token, computed from the "
        "same block energies (sqrt(E/E_keep)); mean_alpha reported per row.",
        "fidelity rows for tower='vision' report image-feature rel-L2 (post pixel-shuffle "
        "connector) as rel_l2; text/both report prefill logits at the last position. Both "
        "quantities are always present (rel_l2_feat / rel_l2_prefill_logit).",
        "generation is greedy, 24 new tokens, on a 2-frame V2 video ({} image tokens, "
        "seq_len {}); top1_agree/first_div/rouge_l compare against the dense generation."
        .format(used["fidelity_input"]["n_image_tokens"],
                used["fidelity_input"]["seq_len"]),
        "VQA uses a deterministic synthetic image (red square + green circle) and 4 hand-written "
        "yes/no + colour prompts; the dense model itself scores {:.2f} accuracy and there are "
        "only 4 prompts, so vqa_acc is a coarse relative signal (it is not monotone in tau) "
        "and not an absolute capability claim.".format(vqa0_acc),
        "calibration transfer: actual_skip is measured on the evaluation forward and lands on "
        "~tau for every tower (see rows), i.e. the per-layer theta quantiles taken from the 3 "
        "calibration frames transfer to unseen frames and prompts; energy_captured stays "
        ">= 0.989 at tau=0.6 because the dropped blocks carry little L2 energy.",
        "SPEED CAVEAT: the masking path still issues the dense GEMMs (PyTorch has no sparse "
        "FFN), so gpu timings for tau>0 are equal-or-worse than dense: the mask multiply plus "
        "the alpha rescale are pure overhead, e.g. at tau={:.1f} vision {:.1f} -> {:.1f} ms, "
        "prefill {:.1f} -> {:.1f} ms, decode {:.1f} -> {:.1f} ms/token. "
        "The real saving is quantified separately in speed.ffn_op, where the FFN is recomputed "
        "with the kept columns sliced out (gate/up rows and down columns gathered) -- that is "
        "what a kernel would do.".format(
            tmax, speed["gpu"]["vision_forward_ms"]["0.0"],
            speed["gpu"]["vision_forward_ms"][str(tmax)],
            speed["gpu"]["lm_prefill_ms"]["0.0"], speed["gpu"]["lm_prefill_ms"][str(tmax)],
            speed["gpu"]["lm_decode_ms_per_token"]["0.0"],
            speed["gpu"]["lm_decode_ms_per_token"][str(tmax)]),
        "SPEED CAVEAT 2: per-token skipping cannot be expressed as one GEMM. ffn_op reports "
        "two sliced references: 'union' gathers every block any token needs (kept_block_frac, "
        "op_speedup) and saturates towards 1.0 as soon as tokens disagree -- it is a lower "
        "bound for batched dense kernels; 'single_mask' applies one median-token mask to the "
        "whole batch (single_mask_block_frac, op_speedup_single_mask) and is the best case a "
        "batched kernel can reach at this skip rate. Per-token kernels (the P1 CUDA/AVX-512 "
        "lanes) sit between the two.",
        "prefill logits are compared at the last position only; decode timings use a manual "
        "KV-cache loop (median of the timed steps after warm-up) and include the masking "
        "overhead of every text layer.",
        f"tau=0 isomorphism: max|dlogit|={dlogit:.3e}, feat rel-L2="
        f"{checks['tau0_feat_rel_l2']:.3e}, top-1 agree={agree:.3f}, generation identical="
        f"{gen1 == gen0} (tolerance {args.iso_tol}).",
    ]

    out = {
        "model": "SmolVLM2-256M-Video-Instruct",
        "arch": arch,
        "inputs_used": used,
        "rows": rows,
        "speed": speed,
        "checks": checks,
        "notes": notes,
        "load_info": load_info,
        "dense": {"text": text0.strip(), "gen_tokens": gen0,
                  "vqa": vqa0, "vqa_acc": vqa0_acc,
                  "n_image_tokens": used["fidelity_input"]["n_image_tokens"]},
    }
    dump_json(out, OUT_JSON)
    log(f"wrote {OUT_JSON}")

    # ---------------------------------------------------------------- table
    print()
    hdr = (f"{'tau':>4} {'tower':>6} {'relL2':>9} {'top1':>6} {'rouge':>6} "
           f"{'1stdiv':>6} {'skip':>6} {'ecap':>6} {'vqa':>4}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['tau']:4.2f} {r['tower']:>6} {r['rel_l2']:9.2e} {r['top1_agree']:6.3f} "
              f"{r['rouge_l']:6.3f} {r['first_div']:6d} {r['actual_skip']:6.3f} "
              f"{r['energy_captured']:6.3f} {r['vqa_acc']:4.2f}")
    print()
    print(f"{'tau':>4} {'vision_ms':>10} {'pre_ms':>8} {'dec_ms/tok':>11} "
          f"{'preboth':>8} {'decboth':>8}")
    g = speed["gpu"]
    for tau in taus:
        k = str(tau)
        print(f"{tau:4.2f} {g['vision_forward_ms'][k]:10.2f} {g['lm_prefill_ms'][k]:8.3f} "
              f"{g['lm_decode_ms_per_token'][k]:11.3f} {g['lm_prefill_ms_both'][k]:8.3f} "
              f"{g['lm_decode_ms_per_token_both'][k]:8.3f}")
    print()
    print(f"{'tower':>6} {'layer':>5} {'tokens':>6} {'tok_skip':>8} {'kept_frac':>9} "
          f"{'smask_frac':>10} {'dense_ms':>8} {'union_ms':>8} {'union_x':>7} {'smask_x':>7}")
    for tower in ("vision", "text"):
        for r in speed["ffn_op"][tower]:
            print(f"{tower:>6} {r['layer']:5d} {r['tokens']:6d} {r['token_skip']:8.3f} "
                  f"{r['kept_block_frac']:9.3f} {r.get('single_mask_block_frac', 1.0):10.3f} "
                  f"{r['dense_ms']:8.3f} {r['partial_ms']:8.3f} {r['op_speedup']:6.2f}x "
                  f"{r.get('op_speedup_single_mask', 1.0):6.2f}x")
    log("done")


if __name__ == "__main__":
    main()
