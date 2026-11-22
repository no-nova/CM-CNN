"""P6-T4: 58-sample SmolVLM2 evaluation (dense vs 32-lane skipping, real kernels).

What this measures
------------------
1. A controlled, >=50-sample image/video QA set built by make_p6_vlm.py
   (data/p6_vlm/manifest.json, seed 1234, per-file sha256), four families:
   OCR (exact match), counting (first integer), scene description (ROUGE-L),
   multi-frame motion (exact word).
2. Accuracy / ROUGE-L / first-divergence / feature and logit rel-L2 for
   tau in {0.10, 0.20, 0.30, 0.40} x tower in {vision, text, both} on all
   samples, with a 1000-resample bootstrap 95% CI on accuracy.
3. End-to-end CPU wall clock with the real AVX-512 block-skipping kernel
   (lanemlp_avx512.cpp: down_skip_avx512_auto2) mounted as the FFN down
   projection: ViT prefill ms/image and LM decode ms/token, dense vs each tau.
4. Tau-0 isomorphism check with the permuted weights (must be at the fp32 floor).

Runs on CPU by default: the box's GPU is shared and NVML is currently broken, so
CUDA timing is attempted only briefly and skipped on any error (recorded).

Usage (from experiments/cm-transformer):
    OMP_NUM_THREADS=16 OMP_WAIT_POLICY=passive PYTHONPATH=. \
        <venv>/bin/python bench_p6_smolvlm.py --taus 0.10,0.20,0.30,0.40
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import RESULTS_DIR, ROOT, bench, dump_json, log, set_deterministic
from model_ref import block_energy
from run_text_eval import first_divergence, rouge_l
from bench_smolvlm import (ISO_DLOGIT_TOL, LANE, FFNSkip, arch_info, capture_acts,
                           calibrate_energies, calibrate_perms, detect_kind, get_feats,
                           load_model, make_embeds, make_inputs, read_stats, rel_l2,
                           set_thetas, thetas_for)

DATA_DIR = ROOT / "data" / "p6_vlm"
MANIFEST = DATA_DIR / "manifest.json"
OUT_JSON = RESULTS_DIR / "p6_smolvlm_50.json"
CPU_LIB = Path(__file__).resolve().parent / "lanemlp_avx512.so"


# --------------------------------------------------------------------------- #
# CPU block-skipping down projection (the real kernel)
# --------------------------------------------------------------------------- #
class KernelDown:
    """out = A @ W.T over the kept 32-lane blocks only, via AVX-512 + OpenMP."""

    def __init__(self, lib_path=CPU_LIB):
        self.lib = ctypes.CDLL(str(lib_path))
        self.fn = self.lib.down_skip_avx512_auto2
        self.fn.restype = None
        self.fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_float),
                            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.calls = 0
        self.active = 0
        self.total = 0

    def __call__(self, a: torch.Tensor, w: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        lead = a.shape[:-1]
        I = a.shape[-1]
        O = w.shape[0]
        if a.is_cuda:
            # GPU quality passes: same arithmetic, dense masked path (the AVX-512
            # operator is a CPU kernel; accuracy does not depend on who computes it)
            self.calls += 1
            self.active += int(keep.sum())
            self.total += keep.numel()
            mask = keep.to(a.dtype).repeat_interleave(LANE, dim=-1)
            return torch.nn.functional.linear(a * mask, w)
        a2 = np.ascontiguousarray(a.detach().reshape(-1, I).float().numpy())
        w2 = np.ascontiguousarray(w.detach().float().numpy())
        k2 = np.ascontiguousarray(
            keep.detach().reshape(-1, I // LANE).to(torch.uint8).numpy())
        out = np.zeros((a2.shape[0], O), dtype=np.float32)
        self.calls += 1
        self.active += int(k2.sum())
        self.total += k2.size
        self.fn(a2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                w2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                k2.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                a2.shape[0], I, O)
        return torch.from_numpy(out).reshape(*lead, O)

    def skip_rate(self):
        return self.active / max(self.total, 1)


class FFNSkipKernel(FFNSkip):
    """FFNSkip whose down projection is the real AVX-512 skipping kernel.

    Same 32-lane mechanism (threshold on per-token block energies, keep mask,
    min(1.5, ||a||/||a_keep||) compensation), but the down GEMM is issued as
    `down_skip_avx512_auto2(A, W, mask, out)` so skipped blocks are never touched.
    theta <= 0 keeps the plain dense path (used for the tau=0 isomorphism check).
    """

    kernel: KernelDown | None = None

    def _fwd(self, x):
        if self.kind == "swiglu":
            a = F.silu(F.linear(x, self.gate.weight)) * F.linear(x, self.up.weight)
            bias = None
        else:
            a = self.act(F.linear(x, self.fc1.weight, self.fc1.bias))
            bias = self.fc2.bias

        if self.capture is not None:
            e = self.capture_energy(a)
            self.capture.append(e.reshape(-1, e.shape[-1]))
        if self.want_x:
            self.saved = (x.detach(), a.detach())
            self.want_x = False

        theta = self.theta
        if theta is None or theta <= 0.0 or self.kernel is None:
            return super()._fwd(x)

        I = a.shape[-1]
        nb = I // self.lane
        e = self.capture_energy(a)                          # [..., nb]
        keep = e >= theta
        E = e.sum(-1, keepdim=True)
        Ek = (e * keep).sum(-1, keepdim=True)
        alpha = torch.sqrt(E / Ek.clamp_min(1e-20)).clamp(max=1.5)
        out = self.kernel(a * alpha, self.down_w, keep)
        if self.tot is not None:
            self._accum(E.sum(), Ek.sum(), keep.numel(), keep.numel() - keep.sum(),
                        alpha.sum(), a.numel() // I)
        if bias is not None:
            out = out + bias
        return out

    # helpers kept tiny so the math is auditable against FFNSkip._fwd
    def capture_energy(self, a):
        return block_energy(a, self.lane)

    @property
    def down_w(self):
        return self.down.weight if self.kind == "swiglu" else self.fc2.weight


def collect_kernel_ffns(model, kernel) -> list[FFNSkipKernel]:
    out = []
    for i, layer in enumerate(model.model.vision_model.encoder.layers):
        out.append(FFNSkipKernel("vision", i, layer.mlp, detect_kind(layer.mlp)))
    for i, layer in enumerate(model.model.text_model.layers):
        out.append(FFNSkipKernel("text", i, layer.mlp, detect_kind(layer.mlp)))
    for s in out:
        s.kernel = kernel
    return out


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def load_manifest():
    m = json.loads(MANIFEST.read_text())
    bad = []
    for s in m["samples"]:
        for rel, want in zip(s["files"], s["sha256"]):
            p = ROOT / rel
            if not p.exists() or sha256_file(p) != want:
                bad.append(rel)
    if bad:
        raise RuntimeError(f"manifest sha256 mismatch / missing files: {bad[:5]}")
    log(f"    {m['n_samples']} samples, all sha256 verified ({m['counts']})")
    return m


def build_all(samples, proc, device, img_edge):
    batches = []
    for s in samples:
        imgs = [Image.open(ROOT / p).convert("RGB") for p in s["files"]]
        b, n_tok = make_inputs(imgs, s["prompt"], proc, device, img_edge)
        batches.append({"batch": b, "n_img_tok": n_tok, "n_images": len(imgs),
                        "seq_len": int(b["input_ids"].shape[1])})
    return batches


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def norm_alnum(t: str) -> str:
    return "".join(ch for ch in t.lower() if ch.isalnum())


def ocr_text(answer: str) -> str:
    """First non-empty line with quotes/backticks/edges stripped -- the OCR field."""
    for line in answer.splitlines():
        line = line.strip().strip("\"'`*.:;,")
        if norm_alnum(line):
            return line
    return ""


def first_int(t: str):
    cur = ""
    for ch in t:
        if ch.isdigit():
            cur += ch
        elif cur:
            return int(cur)
    return int(cur) if cur else None


def score(sample, answer: str) -> float:
    kind = sample["scoring"]
    ref = sample["reference"]
    if kind == "exact_alnum":
        return float(norm_alnum(ocr_text(answer)) == norm_alnum(ref))
    if kind == "first_int":
        return float(first_int(answer) == int(ref))
    if kind == "rouge_l":
        return rouge_l(answer.strip().lower(), ref.lower())
    if kind == "word_match":
        words = [w.strip(".,!?\"'()*") for w in answer.strip().lower().split()]
        words = [w for w in words if w]
        if not words:
            return 0.0
        head = words[0]
        ref_ = ref.lower()
        return float(head == ref_ or (ref_ in words[:3])
                     or (ref_ == "bigger" and head in ("bigger", "larger", "grows"))
                     or (ref_ == "smaller" and head in ("smaller", "shrinks")))
    raise ValueError(kind)


def bootstrap_ci(values, n_boot=1000, seed=1234, lo=2.5, hi=97.5):
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_decode(model, emb, am, n_new, eos_id=None):
    """Greedy KV-cache decode from precomputed input embeddings.

    Returns (ids, first-step last-position logits, per-step ms list).
    """
    tm, head = model.model.text_model, model.lm_head
    o = tm(inputs_embeds=emb, attention_mask=am, use_cache=True)
    lg0 = head(o.last_hidden_state[:, -1:]).float()
    cur = lg0.argmax(-1)
    ids, ts = [], []
    for _ in range(n_new):
        i = int(cur.item())
        ids.append(i)
        if eos_id is not None and i == eos_id:
            break
        e = tm.get_input_embeddings()(cur).to(torch.float32)
        am = torch.cat([am, am.new_ones((am.shape[0], 1))], 1)
        t0 = time.perf_counter()
        o = tm(inputs_embeds=e, attention_mask=am, past_key_values=o.past_key_values,
               use_cache=True)
        cur = head(o.last_hidden_state[:, -1:]).argmax(-1)
        ts.append((time.perf_counter() - t0) * 1e3)
    return ids, lg0, ts


def eos_of(model, proc):
    """The model's real stop token: <end_of_utterance> (49279).

    config.json has eos_token_id=None and the tokenizer reports <|im_end|> (2,
    the pad token), which never fires -- generation_config.json holds the truth.
    """
    gc = getattr(model, "generation_config", None)
    for src in (gc, model.config, getattr(model.config, "text_config", None)):
        v = getattr(src, "eos_token_id", None) if src is not None else None
        if isinstance(v, int) and v != 2:
            return v
    tok = getattr(proc, "tokenizer", None)
    v = getattr(tok, "eos_token_id", None)
    return v if v != 2 else None


def run_pass(model, proc, samples, built, skips, towers, theta_map, gen_len, eos_id,
             want_stats=True, kernel=None):
    """One full pass over all samples under the given tower/tau configuration."""
    set_thetas(skips, towers, theta_map)
    if want_stats:
        for s in skips:
            s.reset_stats()
        if kernel is not None:
            kernel.active = kernel.total = kernel.calls = 0
    rows = []
    for sample, item in zip(samples, built):
        b = item["batch"]
        feats, _ = get_feats(model, b)
        emb = make_embeds(model, b, feats)
        ids, lg0, ts = greedy_decode(model, emb, b["attention_mask"], gen_len, eos_id)
        txt = proc.decode(ids, skip_special_tokens=True).strip()
        rows.append({"id": sample["id"], "family": sample["family"],
                     "feats": feats.detach().clone(),
                     "logit": lg0.detach().clone(), "ids": ids,
                     "answer": txt, "score": score(sample, txt),
                     "decode_ms": (float(np.median(ts)) if ts else 0.0)})
    return rows


def aggregate(rows, dense_rows, tower, tau):
    by_id = {r["id"]: r for r in dense_rows}
    accs = [r["score"] for r in rows]
    rouge = [rouge_l(r["answer"].lower(), d["answer"].lower())
             for r, d in zip(rows, dense_rows)]
    divs = [first_divergence(r["ids"], d["ids"]) for r, d in zip(rows, dense_rows)]
    feat_l2 = [rel_l2(r["feats"], d["feats"]) for r, d in zip(rows, dense_rows)]
    logit_l2 = [rel_l2(r["logit"], d["logit"]) for r, d in zip(rows, dense_rows)]
    lo, hi = bootstrap_ci(accs)
    fam = {}
    for f in ("ocr", "counting", "scene", "motion"):
        v = [r["score"] for r in rows if r["family"] == f]
        fam[f] = float(np.mean(v)) if v else float("nan")
    return {
        "tower": tower, "tau": tau,
        "acc": float(np.mean(accs)), "acc_ci_lo": lo, "acc_ci_hi": hi,
        "rouge_l": float(np.mean(rouge)),
        "first_div": float(np.mean(divs)),
        "feat_rel_l2": float(np.mean(feat_l2)),
        "logit_rel_l2": float(np.mean(logit_l2)),
        "acc_by_family": fam,
        "n_samples": len(rows),
        "exact_match_ocr": fam["ocr"],
        "exact_match_motion": fam["motion"],
        "mean_decode_ms_per_token": float(np.mean([r["decode_ms"] for r in rows])),
        "answers_sample": [{"id": r["id"], "answer": r["answer"][:70],
                            "score": r["score"]} for r in rows[:4]],
        "dense_acc_check": float(np.mean([by_id[r["id"]]["score"] for r in rows])),
    }


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def timed_config(model, skips, towers, theta_map, sample_batch, n_images, kernel,
                 reps=3, decode_steps=8):
    """ViT prefill ms/image, LM prefill ms, LM decode ms/token for one config."""
    set_thetas(skips, towers, theta_map)
    if kernel is not None:
        kernel.active = kernel.total = 0
    b = sample_batch
    warm, _ = get_feats(model, b)                      # warm-up once, always
    emb = make_embeds(model, b, warm)
    model.model.text_model(inputs_embeds=emb, attention_mask=b["attention_mask"],
                           use_cache=False)
    vis_ms, _ = bench(lambda: get_feats(model, b), "cpu", warmup=1, reps=reps)
    pre_ms, _ = bench(lambda: model.model.text_model(
        inputs_embeds=emb, attention_mask=b["attention_mask"], use_cache=False),
        "cpu", warmup=1, reps=reps)
    ts = []
    for _ in range(2):                                 # 2 runs, keep the faster median
        ids, _, t = greedy_decode(model, emb, b["attention_mask"], decode_steps)
        ts.append(float(np.median(t)))
    out = {"vit_prefill_ms": vis_ms, "vit_prefill_ms_per_image": vis_ms / n_images,
           "lm_prefill_ms": pre_ms, "lm_decode_ms_per_token": min(ts),
           "n_images": n_images, "decode_steps": decode_steps}
    if kernel is not None:
        out["kernel_keep_rate"] = kernel.skip_rate()
        out["kernel_block_skip"] = 1.0 - kernel.skip_rate()
        out["kernel_calls"] = kernel.calls
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--taus", default="0.10,0.20,0.30,0.40")
    ap.add_argument("--gen-len", type=int, default=16)
    ap.add_argument("--img-edge", type=int, default=512)
    ap.add_argument("--n-calib", type=int, default=8)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--cuda-probe", action="store_true", default=True)
    ap.add_argument("--no-cuda-probe", dest="cuda_probe", action="store_false")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: keep at most N samples, stratified by family")
    args = ap.parse_args()

    taus = [float(x) for x in args.taus.split(",")]
    dev = args.device
    os.environ.setdefault("OMP_WAIT_POLICY", "passive")
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)
    set_deterministic(1234)
    torch.set_grad_enabled(False)
    log(f"[cfg] device={dev} taus={taus} gen_len={args.gen_len} img_edge={args.img_edge} "
        f"threads={args.threads}")

    log("[1/8] manifest")
    man = load_manifest()
    samples = man["samples"]
    if args.limit and args.limit < len(samples):
        per = max(args.limit // 4, 1)
        keep = []
        for fam in ("ocr", "counting", "scene", "motion"):
            keep += [s for s in samples if s["family"] == fam][:per]
        samples = keep[:args.limit]
        log(f"    --limit: {len(samples)} samples, "
            f"{[s['id'] for s in samples]}")

    log("[2/8] loading model")
    model, proc, load_info = load_model(dev)
    arch = arch_info(model, proc, args.img_edge)
    eos_id = eos_of(model, proc)
    log(f"    {load_info['class']} (transformers {load_info['transformers']}, "
        f"torch {load_info['torch']}, fp32, {dev}); eos={eos_id}")

    log("[3/8] building inputs")
    t0 = time.time()
    built = build_all(samples, proc, dev, args.img_edge)
    log(f"    {len(built)} batches in {time.time() - t0:.1f}s; image tokens "
        f"{sum(b['n_img_tok'] for b in built)} over "
        f"{sum(b['n_images'] for b in built)} images")

    log("[4/8] calibration (permutation + thresholds) on held-in probe batch")
    kernel = KernelDown()
    skips = collect_kernel_ffns(model, kernel)
    calib_idx = [i for i in range(0, len(samples), max(len(samples) // args.n_calib, 1))]
    calib_batches = [built[i]["batch"] for i in calib_idx[:args.n_calib]]
    calib_desc = [{"id": samples[i]["id"], "family": samples[i]["family"]}
                  for i in calib_idx[:args.n_calib]]
    acts = capture_acts(skips, model, calib_batches)
    perms = calibrate_perms(skips, acts, model)
    for s in skips:
        s.apply_perm(perms[s.key])
        s.install()
    energies = calibrate_energies(skips, model, calib_batches)
    thetas = thetas_for(energies, taus)
    log(f"    calibrated on {calib_desc}")
    for tau in taus:
        v = [thetas[tau][s.key] for s in skips if s.tower == "text"]
        log(f"    tau={tau}: text theta median {float(np.median(v)):.4g}")

    log("[5/8] dense reference pass")
    t0 = time.time()
    dense_rows = run_pass(model, proc, samples, built, skips, set(), thetas[taus[0]],
                          args.gen_len, eos_id, want_stats=False)
    dense_acc = float(np.mean([r["score"] for r in dense_rows]))
    fam_acc = {f: float(np.mean([r["score"] for r in dense_rows
                                 if r["family"] == f]))
               for f in ("ocr", "counting", "scene", "motion")}
    log(f"    dense acc={dense_acc:.3f} by family {fam_acc} "
        f"({time.time() - t0:.0f}s for {len(samples)} samples)")

    log("[6/8] tau=0 isomorphism check (permuted weights, no skipping)")
    zero_rows = run_pass(model, proc, samples, built, skips, {"vision", "text"},
                         {k: -1.0 for k in thetas[taus[0]]}, args.gen_len, eos_id,
                         want_stats=False)
    dlogit = max(float((a["logit"] - b["logit"]).abs().max())
                 for a, b in zip(zero_rows, dense_rows))
    feat_l2_0 = max(rel_l2(a["feats"], b["feats"]) for a, b in zip(zero_rows, dense_rows))
    agree = float(np.mean([a == b for a, b in
                           ((r["ids"], d["ids"]) for r, d in zip(zero_rows, dense_rows))]))
    ident = all(r["ids"] == d["ids"] for r, d in zip(zero_rows, dense_rows))
    checks = {"tau0_max_abs_dlogit": dlogit, "tau0_feat_rel_l2": feat_l2_0,
              "tau0_top1_agree": agree, "tau0_gen_identical": ident,
              "tau0_iso_tol": ISO_DLOGIT_TOL,
              "manifest_files_sha256_verified": True,
              "n_samples": len(samples)}
    log(f"    max|dlogit|={dlogit:.3e} feat rel-L2={feat_l2_0:.3e} "
        f"ids identical={ident}")
    assert dlogit <= ISO_DLOGIT_TOL, f"tau=0 not isomorphic: {dlogit:.3e}"
    assert ident, "tau=0 generation differs from dense"

    log("[7/8] sweep: dense + 4 taus x 3 tower configs")
    d_lo, d_hi = bootstrap_ci([r["score"] for r in dense_rows])
    rows = [{"tower": "dense", "tau": 0.0, "acc": dense_acc,
             "acc_ci_lo": d_lo, "acc_ci_hi": d_hi,
             "rouge_l": float(np.mean([rouge_l(r["answer"].lower(), d["answer"].lower())
                                       for r, d in zip(dense_rows, dense_rows)])),
             "first_div": float(args.gen_len), "feat_rel_l2": 0.0, "logit_rel_l2": 0.0,
             "acc_by_family": fam_acc, "n_samples": len(samples),
             "exact_match_ocr": fam_acc["ocr"], "exact_match_motion": fam_acc["motion"],
             "mean_decode_ms_per_token": float(np.mean([r["decode_ms"]
                                                        for r in dense_rows]))}]
    for tower in ("text", "vision", "both"):
        towers = {"vision", "text"} if tower == "both" else {tower}
        for tau in taus:
            t0 = time.time()
            r = run_pass(model, proc, samples, built, skips, towers, thetas[tau],
                         args.gen_len, eos_id, kernel=kernel)
            st = read_stats(skips, towers)
            row = aggregate(r, dense_rows, tower, tau)
            row.update({"actual_skip": st["actual_skip"],
                        "energy_captured": st["energy_captured"],
                        "mean_alpha": st["mean_alpha"],
                        "kernel_block_skip": 1.0 - kernel.skip_rate(),
                        "wall_s": time.time() - t0})
            rows.append(row)
            log(f"    {tower:6s} tau={tau:.2f} acc={row['acc']:.3f} "
                f"[{row['acc_ci_lo']:.3f},{row['acc_ci_hi']:.3f}] "
                f"rouge={row['rouge_l']:.3f} div={row['first_div']:.2f} "
                f"featL2={row['feat_rel_l2']:.3e} logitL2={row['logit_rel_l2']:.3e} "
                f"skip={st['actual_skip']:.3f} kskip={kernel.skip_rate():.3f} "
                f"({time.time() - t0:.0f}s)")

    log("[8/8] timing (CPU, kernels mounted)")
    # One 4-image motion sample as the timing workload (worst case: most image tokens)
    ti = next(i for i, b in enumerate(built) if b["n_images"] == max(x["n_images"] for x in built))
    tb, n_img = built[ti]["batch"], built[ti]["n_images"]
    log(f"    workload: {samples[ti]['id']} ({n_img} images, "
        f"{built[ti]['n_img_tok']} image tokens, seq {built[ti]['seq_len']}), "
        f"{args.threads} threads, OMP_WAIT_POLICY=passive")

    timing = {"cpu": {"device": "cpu", "arch": arch["vision"]["activation"],
                      "threads": args.threads, "nproc": os.cpu_count(),
                      "kernel": "down_skip_avx512_auto2 (AVX-512, OpenMP)",
                      "workload": {"sample": samples[ti]["id"], "n_images": n_img,
                                   "n_image_tokens": built[ti]["n_img_tok"],
                                   "seq_len": built[ti]["seq_len"]},
                      "warmup": "1 full forward + 1 prefill per config before timing; "
                                "median of 3 timed reps (decode: faster of 2 runs, "
                                "median of 8 steps)",
                      "rows": []},
               "gpu": {}}
    # Two interleaved passes, elementwise min: CPU clocks drift with the co-tenant
    # load, so each config is timed twice and the faster run is kept.
    passes = []
    for _ in range(2):
        cfg = [dict(timed_config(model, skips, set(), thetas[taus[0]], tb, n_img, None),
                    config="dense", tau=0.0, tower="dense")]
        for tau in taus:
            cfg.append(dict(timed_config(model, skips, {"vision", "text"}, thetas[tau],
                                         tb, n_img, kernel),
                            config=f"kernel both tau={tau}", tau=tau, tower="both"))
        passes.append(cfg)
    keys = ("vit_prefill_ms", "vit_prefill_ms_per_image", "lm_prefill_ms",
            "lm_decode_ms_per_token")
    for i in range(len(passes[0])):
        r = {k: passes[0][i][k] for k in ("config", "tau", "tower")}
        for k in keys:
            vals = [p[i][k] for p in passes]
            r[k] = min(vals)
        r["n_images"] = passes[0][i]["n_images"]
        r["decode_steps"] = passes[0][i]["decode_steps"]
        if "kernel_block_skip" in passes[0][i]:
            r["kernel_block_skip"] = passes[0][i]["kernel_block_skip"]
            r["kernel_keep_rate"] = passes[0][i]["kernel_keep_rate"]
        r["runs"] = [{k: p[i][k] for k in keys} for p in passes]
        timing["cpu"]["rows"].append(r)
        log(f"    {r['config']:22s} vit {r['vit_prefill_ms_per_image']:6.1f} ms/img | "
            f"prefill {r['lm_prefill_ms']:6.1f} ms | "
            f"decode {r['lm_decode_ms_per_token']:6.1f} ms/tok"
            + (f" | block skip {r['kernel_block_skip']:.3f}"
               if "kernel_block_skip" in r else ""))

    # Kernel-only micro benchmark on REAL post-activation rows of one text layer:
    # dense F.linear vs the AVX-512 skipping kernel, same shapes, same threads.
    ts_layer = skips[len(skips) // 2]
    set_thetas(skips, {"vision", "text"}, thetas[taus[0]])
    ts_layer.want_x = True
    model.model.text_model(inputs_embeds=make_embeds(model, tb, get_feats(model, tb)[0]),
                           attention_mask=tb["attention_mask"], use_cache=False)
    a_real = ts_layer.saved[1].detach().reshape(-1, ts_layer.I).clone()
    micro = []
    for tau in taus:
        keep = block_energy(a_real, LANE) >= float(thetas[tau][ts_layer.key])
        dms, _ = bench(lambda: F.linear(a_real, ts_layer.down_w), "cpu", warmup=2, reps=5)
        kms, _ = bench(lambda: kernel(a_real, ts_layer.down_w, keep), "cpu", warmup=2, reps=5)
        micro.append({"tau": tau, "layer": ts_layer.key, "T": int(a_real.shape[0]),
                      "I": int(a_real.shape[1]), "O": int(ts_layer.down_w.shape[0]),
                      "blk_skip": float(1 - keep.float().mean()),
                      "dense_f_linear_ms": dms, "kernel_ms": kms,
                      "speedup": dms / kms if kms > 0 else 0.0})
        log(f"    kernel micro tau={tau:.2f}: F.linear {dms:.2f} ms vs kernel "
            f"{kms:.2f} ms ({dms / kms:.2f}x, block skip "
            f"{1 - float(keep.float().mean()):.3f})")
    timing["cpu"]["kernel_micro"] = micro

    # brief CUDA probe: the box's GPU is shared / NVML is broken -> expect a skip
    if args.cuda_probe:
        timing["gpu"] = {"attempted": True}
        try:
            ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
            if not ok:
                timing["gpu"]["status"] = "unavailable"
            else:
                z = torch.zeros(8, device="cuda") + 1.0
                torch.cuda.synchronize()
                timing["gpu"].update({"status": "usable",
                                      "device": torch.cuda.get_device_name(0),
                                      "free_mem_mb": torch.cuda.mem_get_info()[0] / 2**20})
                log(f"    cuda: {timing['gpu']['device']}")
        except Exception as exc:                       # pragma: no cover
            timing["gpu"].update({"status": "unusable",
                                  "error": f"{type(exc).__name__}: {exc}"})
            log(f"    cuda probe failed: {type(exc).__name__}: {exc}")
    else:
        timing["gpu"] = {"attempted": False}

    # --------------------------------------------------------------- restore
    for s in skips:
        s.restore()
    feats_r, _ = get_feats(model, built[0]["batch"])
    checks["restore_feat_rel_l2"] = rel_l2(feats_r, dense_rows[0]["feats"])

    notes = [
        "dataset is SYNTHETIC and controlled (PIL, seed 1234, no photographs): OCR strings, "
        "coloured-shape counts on a 3x3 grid, fixed compositions for ROUGE-L, and 3-4 frame "
        "motion sequences. Every image is byte-reproducible from make_p6_vlm.py; per-file "
        "sha256, prompt and reference answer are in data/p6_vlm/manifest.json. It is a "
        "controlled capability probe, not a natural-image benchmark.",
        "dense baseline generation is the same model with permuted-but-unskipped weights "
        "(tau=0 path); the tau=0 check verifies this is bit-comparable to the unpermuted "
        f"model (max|dlogit|={dlogit:.2e}, identical token ids on all "
        f"{len(samples)} samples).",
        "scoring: OCR = exact match on the first non-empty answer line after stripping "
        "quotes/backticks and non-alphanumerics (the model often appends commentary); counting = "
        "first integer in the answer equals the rendered count; scene = ROUGE-L F1 against a "
        "reference sentence; motion = exact match of the first answer word "
        "(direction / bigger / smaller). Aggregate acc is the unweighted mean per-sample "
        "score across all families, so scene/motion contribute fractional ROUGE-style credit; "
        "acc_by_family gives the per-family view.",
        "tau rows use the per-layer theta quantiles of per-token block energies calibrated on "
        f"{len(calib_batches)} images from this set ({', '.join(c['family'] for c in calib_desc)}); "
        "actual_skip is measured on the eval pass, energy_captured is the retained activation "
        "energy fraction.",
        "feature rel-L2 is on the connector image features, logit rel-L2 on the prefill logits "
        "at the last prompt position, averaged over samples. first_div is the mean index of "
        "the first token where the greedy answer differs from dense (gen_len is the cap).",
        "TIMING: CPU fp32, "
        f"{args.threads} threads (OMP_WAIT_POLICY=passive), the real AVX-512 kernel "
        "down_skip_avx512_auto2 replaces BOTH down projections (vision fc2 and text down_proj); "
        "a warm-up forward+prefill runs before every timed config, then medians are taken. "
        "The dense row uses plain F.linear (torch/MKL). Kernel and F.linear are both given the "
        "same thread count, so the comparison is honest but kernel-vs-MKL: at low tau the mask "
        "bookkeeping (energy + threshold + numpy/ctypes hand-off) can cost more than the "
        "skipped FMAs, see kernel_micro for the pure GEMM comparison at the same shapes.",
        "GPU timing is deliberately not attempted beyond a liveness probe: the 8 GB GPU is "
        f"shared with another job ({timing['gpu'].get('status')}). The P5 numbers in "
        "results/p5_smolvlm.json remain the GPU reference for the masking path.",
        f"p5_smolvlm.json is untouched; this file is the 58-sample replacement for its "
        f"4-prompt VQA weakness.",
    ]

    out = {
        "provenance": {
            "task": "P6-T4 50+ sample SmolVLM2 evaluation with end-to-end timing",
            "model": "SmolVLM2-256M-Video-Instruct",
            "model_dir": "models/SmolVLM2-256M-Video-Instruct",
            "script": "experiments/cm-transformer/bench_p6_smolvlm.py",
            "script_sha256": sha256_file(Path(__file__)),
            "data_generator": "experiments/cm-transformer/make_p6_vlm.py",
            "dataset": "data/p6_vlm (synthetic-controlled)",
            "device": dev, "dtype": "float32",
            "transformers": load_info["transformers"], "torch": torch.__version__,
            "python": sys.version.split()[0],
            "threads": args.threads, "nproc": os.cpu_count(),
            "kernel": "experiments/cm-transformer/lanemlp_avx512.cpp -> "
                      "down_skip_avx512_auto2 (AVX-512, OpenMP)",
            "build": "g++ -O3 -fopenmp -mavx512f -mavx512bw -mfma -march=native -shared -fPIC",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": os.popen("git -C %s rev-parse --short HEAD" % ROOT).read().strip(),
        },
        "seed": man["seed"],
        "n_samples": len(samples),
        "manifest_sha256": sha256_file(MANIFEST),
        "manifest_counts": man["counts"],
        "arch": arch,
        "rows": rows,
        "timing": timing,
        "checks": checks,
        "calibration": calib_desc,
        "gen_len": args.gen_len,
        "img_edge": args.img_edge,
        "notes": notes,
    }
    dump_json(out, OUT_JSON)
    log(f"wrote {OUT_JSON}")

    print()
    hdr = (f"{'tower':>6} {'tau':>4} {'acc':>6} {'95% CI':>15} {'rouge':>6} "
           f"{'1stdiv':>6} {'featL2':>9} {'logitL2':>9} {'skip':>6} "
           f"{'ocr':>5} {'cnt':>5} {'scn':>5} {'mot':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        f = r["acc_by_family"]
        print(f"{r['tower']:>6} {r['tau']:4.2f} {r['acc']:6.3f} "
              f"[{r['acc_ci_lo']:5.3f},{r['acc_ci_hi']:5.3f}] {r['rouge_l']:6.3f} "
              f"{r['first_div']:6.2f} {r['feat_rel_l2']:9.2e} {r['logit_rel_l2']:9.2e} "
              f"{r.get('actual_skip', 0.0):6.3f} {f['ocr']:5.2f} {f['counting']:5.2f} "
              f"{f['scene']:5.2f} {f['motion']:5.2f}")
    print()
    print(f"{'config':>22} {'vit ms/img':>11} {'prefill ms':>11} {'decode ms/tok':>14} "
          f"{'kernel skip':>12}")
    for t in timing["cpu"]["rows"]:
        print(f"{t['config']:>22} {t['vit_prefill_ms_per_image']:11.1f} "
              f"{t['lm_prefill_ms']:11.1f} {t['lm_decode_ms_per_token']:14.1f} "
              f"{t.get('kernel_block_skip', 0.0):12.3f}")
    log("done")


if __name__ == "__main__":
    main()
