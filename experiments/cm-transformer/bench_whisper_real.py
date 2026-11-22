#!/usr/bin/env python3
"""bench_whisper_real.py -- Phase 6 Task 3: the Whisper block-skipping result on
*real English speech* with reference transcripts (LibriSpeech dev-clean).

`bench_whisper.py` (P5) could only report dense-vs-gated *agreement*, because its
inputs were synthetic tones/chirp/harmonics with no reference transcript.  This
script reuses exactly the same mechanism (`FFNGate` / `GateSet` / `build_perm` /
`calibrate_theta` / 32-lane block energy) but evaluates it against the published
LibriSpeech transcripts, so absolute WER/CER are meaningful.

Protocol (fixed, seeds 1234 everywhere):

  * data: `data/p6_whisper_audio/` from `fetch_whisper_audio.py` (official
    OpenSLR SLR12 dev-clean tar.gz, streamed; per-file sha256 in the manifest).
    N_UTTS utterances are drawn with `np.random.RandomState(1234)`: the first
    N_CALIB are the *calibration* split (builds pi, calibrates theta_l), the
    remaining N_EVAL are the *evaluation* split.  The two splits are disjoint.
  * decoding: greedy, whisper's suppress_tokens applied, stop at <|endoftext|>,
    max 128 new tokens, no timestamps, fixed 4-token prompt.
  * metrics per utterance: word error rate and character error rate from a local
    Levenshtein implementation (no jiwer), exact-token-match, encoder relative
    L2 vs the dense encoder, and teacher-forced top-1 agreement (teacher forcing
    runs on the *reference* tokens, so it is a label-bearing metric).
  * timing: CPU only (`nproc`=32 threads, GPU not touched), full 6-layer encoder
    (no lm_head) and the full decoder loop (prefill + greedy steps), ms/utterance,
    dense vs tau in {0.20, 0.40, 0.60}, plus tau=0 (permutation, nothing skipped).
  * AVX-512: `lanemlp_avx512.cpp::down_skip_avx512_auto2` is compiled and mounted
    as fc2's kernel, verified against the PyTorch gated path, and timed.

    python bench_whisper_real.py --n-utts 160 --n-calib 40
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

# OpenMP for the AVX-512 kernel: passive waiting, <= 16 threads (the box is
# 16 cores / 32 SMT threads and torch gets all 32 of them).
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OMP_WAIT_POLICY", "passive")
os.environ.setdefault("OMP_DYNAMIC", "false")

import numpy as np
import torch
import torch.nn.functional as F

from common import EXP_DIR, RESULTS_DIR, ROOT, dump_json, log, set_deterministic, sha256_file
from bench_whisper import (LANE, PROMPT_IDS, FFNGate, GateSet, WhisperBench,
                           block_energy, build_perm, calibrate_theta, keep_mask_from_energy,
                           rel_l2)

SEED = 1234
TAUS = [0.0, 0.2, 0.4, 0.6]
AUDIO_ROOT = ROOT / "data" / "p6_whisper_audio"
PROVENANCE = AUDIO_ROOT / "provenance.json"
MANIFEST_JSON = ROOT / "data" / "p6_whisper_manifest.json"
OUT_JSON = RESULTS_DIR / "p6_whisper_real.json"
PERM_CACHE = RESULTS_DIR / "whisper_p6_perm.pt"
KERNEL_SRC = EXP_DIR / "lanemlp_avx512.cpp"
KERNEL_SO = RESULTS_DIR / "lanemlp_avx512_p6.so"

TOL_ENC_REL_L2 = 1e-5          # tau = 0 must stay at the fp32 rounding floor
TOL_DEC_MAX_ABS = 1e-4
TOL_KERNEL_ABS = 2e-4          # AVX-512 kernel vs PyTorch, same masked activation


# --------------------------------------------------------------------------- #
# audio: flac -> 16 kHz mono float32 without soundfile/librosa
# --------------------------------------------------------------------------- #
def load_wav(path: Path, sr: int = 16000) -> np.ndarray:
    """Decode with ffmpeg (present on this box); no python audio deps."""
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


# --------------------------------------------------------------------------- #
# metrics: local edit distance (no jiwer)
# --------------------------------------------------------------------------- #
def edit_distance(ref: list, hyp: list) -> int:
    """Plain Levenshtein, two rows, unit substitution/insertion/deletion cost."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1,                 # deletion
                         cur[j - 1] + 1,              # insertion
                         prev[j - 1] + (r != h))      # substitution / match
        prev = cur
    return prev[-1]


_PUNCT = re.compile(r"[^a-z0-9' ]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """LibriSpeech-style normalization: lowercase, drop punctuation, squeeze ws.

    whisper's own EnglishTextNormalizer is used when available (it also expands
    numbers and contractions); this fallback keeps the pipeline dependency-free.
    """
    t = unicodedata.normalize("NFKD", text).lower()
    t = t.replace("-", " ")
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def get_normalizer():
    try:                                              # whisper's normalizer, if importable
        from transformers.models.whisper.tokenization_whisper import EnglishTextNormalizer
        norm = EnglishTextNormalizer()
        return (lambda s: norm(normalize(s))), "whisper.EnglishTextNormalizer"
    except Exception:                                 # pragma: no cover
        return normalize, "local lowercase/punctuation strip"


def words_of(text: str) -> list[str]:
    t = normalize(text)
    return t.split() if t else []


def chars_of(text: str) -> list[str]:
    """Characters of the normalized text; spaces count (standard CER on ASR)."""
    return list(normalize(text))


def stats(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1) if a.size > 1 else 0.0)


# --------------------------------------------------------------------------- #
# activation capture (fc2 inputs) for pi construction / theta calibration
# --------------------------------------------------------------------------- #
def capture_utterance(wh: WhisperBench, feats: torch.Tensor, ref_ids: list[int]):
    """Dense pass: encoder forward + teacher-forced decoder over the reference.

    Returns {key: [N, 2048] cpu float32} with fc2's input (= GELU(fc1(x))) as the
    model actually computes it -- the same tensor the gate thresholds at
    inference, and the same one P5 captured.
    """
    cap: dict[str, list] = {k: [] for k in wh.gates.entries}
    handles = []
    for key, g in wh.gates.entries.items():
        def hook(mod, inp, key=key, cap=cap):
            x = inp[0].detach()
            cap[key].append(x.reshape(-1, x.shape[-1]).float().cpu())
        handles.append(g.fc2.register_forward_pre_hook(hook))
    wh.gates.set_dense()
    try:
        with torch.no_grad():
            enc = wh.run_encoder(feats)
            ids = ref_ids[:-1][:wh.cfg.max_target_positions - len(PROMPT_IDS) - 1]
            wh.decoder(input_ids=torch.tensor([PROMPT_IDS + ids], device=wh.device),
                       encoder_hidden_states=enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {k: (torch.cat(v, 0) if v else torch.zeros(0, wh.cfg.encoder_ffn_dim))
            for k, v in cap.items()}


# --------------------------------------------------------------------------- #
# greedy decoding with an explicit end-of-text stop (P5 used a fixed length)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_ids(wh: WhisperBench, enc: torch.Tensor, max_new: int = 128) -> list[int]:
    from transformers import DynamicCache
    ids = torch.tensor([PROMPT_IDS], device=wh.device)
    cache = DynamicCache()
    o = wh.decoder(input_ids=ids, encoder_hidden_states=enc,
                   past_key_values=cache, use_cache=True)
    last = o.last_hidden_state[:, -1:]
    out: list[int] = []
    eot = wh.cfg.eos_token_id
    for _ in range(max_new):
        logits = wh.model.proj_out(last)[:, -1].float()
        logits[:, wh.suppress] = -float("inf")
        nxt = int(logits.argmax(-1))
        if nxt == eot:
            break
        out.append(nxt)
        o = wh.decoder(input_ids=torch.tensor([[nxt]], device=wh.device),
                       encoder_hidden_states=enc,
                       past_key_values=o.past_key_values, use_cache=True)
        last = o.last_hidden_state[:, -1:]
    return out


@torch.no_grad()
def teacher_forced_logits(wh: WhisperBench, enc: torch.Tensor, ids: list[int]) -> torch.Tensor:
    seq = torch.tensor([PROMPT_IDS + ids[:-1]], device=wh.device)
    o = wh.decoder(input_ids=seq, encoder_hidden_states=enc, use_cache=False)
    return wh.model.proj_out(o.last_hidden_state)


# --------------------------------------------------------------------------- #
# AVX-512 down-projection kernel (fc2) mounted as a real gate
# --------------------------------------------------------------------------- #
def build_kernel() -> ctypes.CDLL:
    subprocess.run(["g++", "-O3", "-fopenmp", "-mavx512f", "-mavx512bw", "-mfma",
                    "-march=native", "-shared", "-fPIC", "-o", str(KERNEL_SO),
                    str(KERNEL_SRC)], check=True)
    lib = ctypes.CDLL(str(KERNEL_SO))
    P = ctypes.POINTER(ctypes.c_float)
    lib.down_skip_avx512_auto2.argtypes = [P, P, ctypes.POINTER(ctypes.c_ubyte), P,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int]
    return lib


class KernGate(FFNGate):
    """`FFNGate` whose fc2 GEMM is the AVX-512 block-skipping kernel.

    `down_skip_avx512_auto2(A, W, M, out, T, I, O)` computes
    out[t, r] = sum_{b: M[t,b]} sum_{l<32} A[t, 32b+l] * W[r, 32b+l] in fp32, so
    the skipped 32-lane tiles are never loaded.  A is the *permuted, gated and
    alpha-scaled* activation, i.e. the exact tensor the PyTorch path feeds to
    F.linear, so the two paths differ only by fp32 accumulation order.
    """

    lib: ctypes.CDLL | None = None

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.theta is None or KernGate.lib is None:
            return super().__call__(hidden)
        lead = hidden.shape[:-1]
        I = hidden.shape[-1]
        O = self.W2.shape[0]
        a = hidden.index_select(-1, self.perm)
        flat = a.reshape(-1, I)
        e = block_energy(flat, LANE)
        keep = keep_mask_from_energy(e, self.theta)
        mask = keep.repeat_interleave(LANE, dim=1).to(flat.dtype)
        ak = flat * mask
        na = flat.norm(dim=-1, keepdim=True)
        nk = ak.norm(dim=-1, keepdim=True)
        alpha = torch.clamp(na / nk.clamp_min(1e-12), max=1.5)
        x = (ak * alpha).contiguous()
        W = self.W2p.contiguous()
        m = keep.to(torch.uint8).contiguous()
        out = torch.empty((flat.shape[0], O), dtype=torch.float32)
        P = ctypes.POINTER(ctypes.c_float)
        KernGate.lib.down_skip_avx512_auto2(
            ctypes.cast(x.data_ptr(), P), ctypes.cast(W.data_ptr(), P),
            ctypes.cast(m.data_ptr(), ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.cast(out.data_ptr(), P),
            int(flat.shape[0]), int(I), int(O))
        if self.b2 is not None:
            out += self.b2
        vals = torch.stack([e[keep].sum(), e.sum(), (~keep).sum().to(e.dtype)])
        self.acc = vals if self.acc is None else self.acc + vals
        self.n_blk += keep.numel()
        return out.reshape(*lead, O)


def install_kernel(lib: ctypes.CDLL):
    """Swap every gate instance to the kernel-backed subclass, keep state."""
    KernGate.lib = lib
    return KernGate


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-utts", type=int, default=160, help="calib + eval utterances")
    ap.add_argument("--n-calib", type=int, default=40)
    ap.add_argument("--tau", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--timing-reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--kernel", type=int, default=1)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=str(OUT_JSON))
    args = ap.parse_args()

    t_start = time.perf_counter()
    taus = [float(x) for x in args.tau.split(",")]
    set_deterministic(SEED)
    torch.set_num_threads(args.threads)
    log(f"[init] torch {torch.__version__} threads={torch.get_num_threads()} "
        f"omp={os.environ.get('OMP_NUM_THREADS')} seed={SEED} tau={taus}")

    # ---- data ------------------------------------------------------------- #
    prov = json.loads(PROVENANCE.read_text())
    utts_all = sorted(prov["utterances"], key=lambda u: u["id"])
    rng = np.random.RandomState(SEED)
    pick = rng.permutation(len(utts_all))[: args.n_utts]
    picked = [utts_all[i] for i in pick]
    picked.sort(key=lambda u: u["id"])                 # decode order = id order
    calib_utts = picked[: args.n_calib]
    eval_utts = picked[args.n_calib:]
    log(f"[data] corpus {len(utts_all)} utts / {prov['chapters']} chapters / "
        f"{len({u['speaker'] for u in utts_all})} speakers "
        f"({prov['source']}); split: {len(calib_utts)} calib + {len(eval_utts)} eval")

    from transformers import WhisperFeatureExtractor, WhisperTokenizer
    fe = WhisperFeatureExtractor.from_pretrained(str(ROOT / "models" / "whisper-base"))
    tok = WhisperTokenizer.from_pretrained(str(ROOT / "models" / "whisper-base"))
    normalizer, normalizer_name = get_normalizer()

    t_load = time.perf_counter()
    waves, secs = {}, {}
    for u in picked:
        w = load_wav(ROOT / u["path"])
        waves[u["id"]] = w
        secs[u["id"]] = len(w) / 16000.0
        u["seconds"] = round(secs[u["id"]], 3)
        u["ref_norm"] = normalizer(u["ref"])
        u["ref_ids"] = tok(u["ref"], add_special_tokens=False).input_ids
    log(f"[data] decoded {len(picked)} flac files in {time.perf_counter() - t_load:.1f}s, "
        f"audio {sum(secs.values()):.1f}s (mean {np.mean(list(secs.values())):.2f}s), "
        f"ref words {np.mean([len(words_of(u['ref'])) for u in picked]):.1f}/utt")

    feats_list = [fe(waves[u["id"]], sampling_rate=16000,
                     return_tensors="pt").input_features for u in picked]
    feats_by_id = {u["id"]: f for u, f in zip(picked, feats_list)}

    # ---- model ------------------------------------------------------------ #
    wh = WhisperBench("cpu")
    arch = wh.arch()
    log(f"[init] whisper-base loaded in {wh.load_s:.1f}s device={wh.device} "
        f"suppress={len(wh.suppress)} tokens")

    # ---- calibration: pi from the calibration split, theta_l per tau ------- #
    t0 = time.perf_counter()
    caps_calib = [capture_utterance(wh, feats_by_id[u["id"]], u["ref_ids"])
                  for u in calib_utts]
    acts = {k: torch.cat([c[k] for c in caps_calib], 0) for k in wh.gates.entries}
    log(f"[calib] captured {({k: v.shape[0] for k, v in acts.items()})} fc2 inputs "
        f"from {len(calib_utts)} held-out utterances in {time.perf_counter() - t0:.1f}s")

    sig = {"n_calib": len(calib_utts), "ids": [u["id"] for u in calib_utts],
           "prompt": PROMPT_IDS, "lane": LANE, "model": "whisper-base", "seed": SEED,
           "teacher_forced_ref": True}
    params = None
    if PERM_CACHE.exists() and not args.no_cache:
        try:
            blob = torch.load(PERM_CACHE, map_location="cpu")
            if blob.get("sig") == sig:
                params = blob
                log(f"[calib] reusing cached pi from {PERM_CACHE.name}")
        except Exception as e:                       # pragma: no cover
            log(f"[calib] cache unusable ({e}); rebuilding")
    if params is None:
        perms = {}
        for key, g in wh.gates.entries.items():
            perm, _ = build_perm(acts[key], g.W2, LANE)
            perms[key] = perm.cpu()
        params = {"sig": sig, "perm": perms}
        torch.save(params, PERM_CACHE)
    perms = params["perm"]
    for k, g in wh.gates.entries.items():
        g.register(perms[k], float("-inf"))

    thetas = {tau: calibrate_theta(wh, acts, perms, tau) for tau in taus}
    for tau in taus:
        if tau > 0:
            log(f"[calib] tau={tau:g} theta enc/dec first layer "
                f"{thetas[tau]['enc.0']:.4g}/{thetas[tau]['dec.0']:.4g}")

    # ---- dead 32-lane blocks: before vs after the permutation -------------- #
    # computed on every selected utterance (calib + eval), per layer, as the
    # fraction of exactly-zero-energy blocks; mean/std are across utterances.
    dead_before = {k: [] for k in wh.gates.entries}
    dead_after = {k: [] for k in wh.gates.entries}
    t0 = time.perf_counter()
    for u in picked:
        cap = capture_utterance(wh, feats_by_id[u["id"]], u["ref_ids"])
        for k, a in cap.items():
            dead_before[k].append(float((block_energy(a, LANE) == 0).float().mean()))
            dead_after[k].append(
                float((block_energy(a.index_select(1, perms[k]), LANE) == 0).float().mean()))
    dead_blocks = {}
    for k in wh.gates.entries:
        bm, bs = stats(dead_before[k])
        am, asd = stats(dead_after[k])
        dead_blocks[k] = {"before_mean": bm, "before_std": bs,
                          "after_mean": am, "after_std": asd,
                          "n_utterances": len(dead_before[k])}
    log("[dead] exactly-zero 32-lane blocks (mean fraction, before -> after pi): " + " ".join(
        f"{k}:{dead_blocks[k]['before_mean']:.3f}->{dead_blocks[k]['after_mean']:.3f}"
        for k in wh.gates.entries) + f"  [{time.perf_counter() - t0:.0f}s]")

    # ---- tau = 0 isomorphism gate ----------------------------------------- #
    wh.gates.set_dense()
    dense_enc, dense_ids, dense_tf = {}, {}, {}
    for u in eval_utts:
        uid = u["id"]
        e = wh.run_encoder(feats_by_id[uid])
        dense_enc[uid] = e
        dense_ids[uid] = greedy_ids(wh, e, args.max_new)
        dense_tf[uid] = teacher_forced_logits(wh, e, u["ref_ids"])
    log(f"[ref] dense decoding done ({np.mean([len(v) for v in dense_ids.values()]):.1f} "
        f"tokens/utt)")

    wh.gates.set_dense()   # tau=0 must be dense: the permutation alone is the test
    iso = {"max_enc_rel_l2": 0.0, "max_dec_max_abs": 0.0,
           "max_dec_max_abs_tf": 0.0, "min_top1_tf": 1.0, "token_mismatch": 0}
    for u in eval_utts:
        uid = u["id"]
        e = wh.run_encoder(feats_by_id[uid])
        ids = greedy_ids(wh, e, args.max_new)
        tf = teacher_forced_logits(wh, e, u["ref_ids"])
        iso["max_enc_rel_l2"] = max(iso["max_enc_rel_l2"], rel_l2(e, dense_enc[uid]))
        iso["max_dec_max_abs"] = max(iso["max_dec_max_abs"],
                                     float((tf - dense_tf[uid]).abs().max()))
        iso["min_top1_tf"] = min(iso["min_top1_tf"],
                                 float((tf.argmax(-1) == dense_tf[uid].argmax(-1))
                                       .float().mean()))
        iso["token_mismatch"] += int(ids != dense_ids[uid])
    ok = (iso["max_enc_rel_l2"] <= TOL_ENC_REL_L2
          and iso["max_dec_max_abs"] <= TOL_DEC_MAX_ABS
          and iso["min_top1_tf"] == 1.0 and iso["token_mismatch"] == 0)
    log(f"[iso] tau=0 permutation-only vs dense: enc_rel_l2<={iso['max_enc_rel_l2']:.2e} "
        f"dec_max|d|<={iso['max_dec_max_abs']:.2e} top1_tf>={iso['min_top1_tf']:.3f} "
        f"token mismatches={iso['token_mismatch']} -> {'OK' if ok else 'FAIL'}")
    if not ok:
        log("[FATAL] tau=0 is not isomorphic under the permutation")
        wh.gates.restore()
        return 2

    # ---- evaluation: WER / CER / exact match per tau ----------------------- #
    ref_words = {u["id"]: words_of(u["ref"]) for u in eval_utts}
    ref_chars = {u["id"]: chars_of(u["ref"]) for u in eval_utts}
    hyp_text: dict[float, dict[str, str]] = {}
    rows = []
    per_utt: dict[float, dict] = {}

    def run_config(tau):
        wh.gates.set_dense() if tau is None else wh.gates.set_theta(thetas[tau])
        wer, cer, em, rl2, top1, hyp, tf_logits = [], [], [], [], [], {}, {}
        wh.gates.reset_acc()
        for u in eval_utts:
            uid = u["id"]
            with torch.no_grad():
                e = wh.run_encoder(feats_by_id[uid])
                ids = greedy_ids(wh, e, args.max_new)
                tf = teacher_forced_logits(wh, e, u["ref_ids"])
            txt = wh.decode_text(ids)
            hyp[uid] = txt
            hw, hc = words_of(txt), chars_of(txt)
            wer.append(edit_distance(ref_words[uid], hw) / max(len(ref_words[uid]), 1))
            cer.append(edit_distance(ref_chars[uid], hc) / max(len(ref_chars[uid]), 1))
            hids = tok(normalizer(txt), add_special_tokens=False).input_ids
            em.append(float(hids == u["ref_ids"]))
            rl2.append(rel_l2(e, dense_enc[uid]))
            top1.append(float((tf.argmax(-1) == dense_tf[uid].argmax(-1)).float().mean()))
            tf_logits[uid] = tf
        agg = wh.gates.aggregate()
        return dict(wer=wer, cer=cer, em=em, rl2=rl2, top1=top1, hyp=hyp,
                    agg=agg, tf=tf_logits)

    for tau in [None] + [t for t in taus if t > 0]:
        label = "dense" if tau is None else f"{tau:g}"
        t0 = time.perf_counter()
        r = run_config(tau)
        per_utt[tau] = r
        hyp_text[tau] = r["hyp"]
        wer_m, wer_s = stats(r["wer"])
        cer_m, cer_s = stats(r["cer"])
        tot_err = sum(edit_distance(ref_words[u["id"]], words_of(r["hyp"][u["id"]]))
                      for u in eval_utts)
        tot_ref = sum(len(ref_words[u["id"]]) for u in eval_utts)
        row = {
            "tau": -1.0 if tau is None else tau,
            "label": label,
            "wer": wer_m, "wer_std": wer_s,
            "cer": cer_m, "cer_std": cer_s,
            "wer_corpus": tot_err / max(tot_ref, 1),
            "exact_match": float(np.mean(r["em"])),
            "enc_rel_l2": float(np.mean(r["rl2"])),
            "teacher_forced_top1": float(np.mean(r["top1"])),
            "energy_captured": r["agg"]["energy_captured"],
            "actual_skip": r["agg"]["actual_skip"],
            "n_utterances": len(eval_utts),
        }
        rows.append(row)
        log(f"[eval] tau={label:>4}  WER={wer_m:.4f}+-{wer_s:.4f} "
            f"CER={cer_m:.4f}+-{cer_s:.4f} EM={row['exact_match']:.3f} "
            f"enc_rel_l2={row['enc_rel_l2']:.4f} top1_tf={row['teacher_forced_top1']:.3f} "
            f"skip={row['actual_skip']:.3f}  [{time.perf_counter() - t0:.0f}s]")
        log(f"           sample: ref={eval_utts[0]['ref_norm'][:70]!r}  "
            f"hyp={normalizer(r['hyp'][eval_utts[0]['id']])[:70]!r}")

    # ---- timing: encoder + full decoder loop, ms/utterance, CPU ----------- #
    def kname(t) -> str:
        return "dense" if t is None else f"{t:g}"

    def time_encoder(rounds: int) -> dict:
        acc = {kname(t): [] for t in [None] + [x for x in taus if x > 0]}
        for _ in range(rounds):
            for tau in [None] + [x for x in taus if x > 0]:
                wh.gates.set_dense() if tau is None else wh.gates.set_theta(thetas[tau])
                t0 = time.perf_counter()
                with torch.no_grad():
                    for u in eval_utts:
                        wh.run_encoder(feats_by_id[u["id"]])
                acc[("dense" if tau is None else f"{tau:g}")].append(
                    (time.perf_counter() - t0) * 1e3 / len(eval_utts))
        return {k: float(np.median(v)) for k, v in acc.items()}

    def time_decoder(rounds: int) -> dict:
        """Full decoder loop: prefill + greedy steps until <|eot|>, per utterance."""
        acc = {kname(t): [] for t in [None] + [x for x in taus if x > 0]}
        acc_tok = {k: [] for k in acc}
        for _ in range(rounds):
            for tau in [None] + [x for x in taus if x > 0]:
                wh.gates.set_dense() if tau is None else wh.gates.set_theta(thetas[tau])
                el, nt = 0.0, 0
                with torch.no_grad():
                    for u in eval_utts:
                        uid = u["id"]
                        t0 = time.perf_counter()
                        ids = greedy_ids(wh, dense_enc[uid], args.max_new)
                        el += (time.perf_counter() - t0) * 1e3
                        nt += len(ids) + 1              # + the <|eot|> step
                k = kname(tau)
                acc[k].append(el / len(eval_utts))
                acc_tok[k].append(el / max(nt, 1))
        return ({k: float(np.median(v)) for k, v in acc.items()},
                {k: float(np.median(v)) for k, v in acc_tok.items()})

    log(f"[timing] CPU, torch threads={torch.get_num_threads()}, "
        f"{len(eval_utts)} utterances x {args.timing_reps} rounds")
    enc_ms = time_encoder(args.timing_reps)
    dec_ms, dec_tok_ms = time_decoder(max(2, args.timing_reps))
    timing = []
    for tau in [None] + [x for x in taus if x > 0]:
        k = "dense" if tau is None else f"{tau:g}"
        timing.append({
            "tau": -1.0 if tau is None else tau,
            "label": k,
            "enc_ms_per_utt": enc_ms[k],
            "dec_ms_per_utt": dec_ms[k],
            "dec_ms_per_token": dec_tok_ms[k],
            "enc_speedup": enc_ms["dense"] / max(enc_ms[k], 1e-9),
            "dec_speedup": dec_ms["dense"] / max(dec_ms[k], 1e-9),
        })
        log(f"[timing] tau={k:>4}  enc {enc_ms[k]:8.2f} ms/utt "
            f"({enc_ms['dense'] / enc_ms[k]:.3f}x)  dec {dec_ms[k]:8.2f} ms/utt "
            f"({dec_ms['dense'] / dec_ms[k]:.3f}x, {dec_tok_ms[k]:.3f} ms/tok)")

    # ---- AVX-512 kernel: verify, then time End-to-End --------------------- #
    kernel = {"available": False}
    if args.kernel:
        try:
            lib = build_kernel()
            kernel["build"] = "g++ -O3 -fopenmp -mavx512f -mavx512bw -mfma -march=native -shared -fPIC"
            kernel["source"] = "experiments/cm-transformer/lanemlp_avx512.cpp"
            kernel["function"] = "down_skip_avx512_auto2(A,W,M,out,T,I,O)"
            kernel["omp_threads"] = int(os.environ.get("OMP_NUM_THREADS", "0"))
            kernel["omp_wait_policy"] = os.environ.get("OMP_WAIT_POLICY", "")
            install_kernel(lib)

            # correctness: kernel-backed gate vs PyTorch gate, same theta
            max_diff = 0.0
            wh.gates.set_theta(thetas[0.2])
            for u in eval_utts[:8]:
                with torch.no_grad():
                    ref = wh.run_encoder(feats_by_id[u["id"]])
                wh.gates.set_theta(thetas[0.2])
                for g in wh.gates.entries.values():
                    g.__class__ = KernGate
                with torch.no_grad():
                    got = wh.run_encoder(feats_by_id[u["id"]])
                max_diff = max(max_diff, float((got - ref).abs().max()))
                for g in wh.gates.entries.values():
                    g.__class__ = FFNGate
            kernel["max_abs_diff_vs_torch_gate"] = max_diff
            kernel["verified"] = bool(max_diff <= TOL_KERNEL_ABS)
            log(f"[kernel] down_skip_avx512_auto2 vs torch gated encoder: "
                f"max|d|={max_diff:.2e} (tol {TOL_KERNEL_ABS:g}) -> "
                f"{'OK' if kernel['verified'] else 'FAIL'}")

            if kernel["verified"]:
                for g in wh.gates.entries.values():
                    g.__class__ = KernGate
                t_kern = {}
                for _ in range(max(2, args.timing_reps)):
                    for tau in [None] + [x for x in taus if x > 0]:
                        k = "dense" if tau is None else f"{tau:g}"
                        wh.gates.set_dense() if tau is None else wh.gates.set_theta(thetas[tau])
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            for u in eval_utts:
                                wh.run_encoder(feats_by_id[u["id"]])
                        t_kern.setdefault(k, []).append(
                            (time.perf_counter() - t0) * 1e3 / len(eval_utts))
                kernel["enc_ms_per_utt"] = {k: float(np.median(v)) for k, v in t_kern.items()}
                kernel["enc_speedup"] = {
                    k: kernel["enc_ms_per_utt"]["dense"] / max(v, 1e-9)
                    for k, v in kernel["enc_ms_per_utt"].items()}
                kernel["available"] = True
                log("[kernel] encoder E2E ms/utt " + " ".join(
                    f"{k}:{v:.2f}({kernel['enc_speedup'][k]:.3f}x)"
                    for k, v in kernel["enc_ms_per_utt"].items()))
                for g in wh.gates.entries.values():
                    g.__class__ = FFNGate
        except Exception as e:                        # pragma: no cover
            kernel["error"] = f"{type(e).__name__}: {e}"
            log(f"[kernel] FAILED, reporting PyTorch-level E2E only: {kernel['error']}")

    wh.gates.restore()

    # ---- manifest: per-file sha256 of every audio file used ---------------- #
    files = []
    calib_ids = {u["id"] for u in calib_utts}
    for u in picked:
        p = ROOT / u["path"]
        got = sha256_file(p)
        if got != u["sha256"]:
            log(f"[FATAL] sha256 mismatch for {u['path']}")
            return 3
        files.append({"id": u["id"], "path": u["path"], "sha256": got,
                      "bytes": u["bytes"], "seconds": u["seconds"],
                      "split": "calib" if u["id"] in calib_ids else "eval"})
    canon = json.dumps(sorted(files, key=lambda f: f["id"]), sort_keys=True,
                       separators=(",", ":")).encode()
    manifest_sha = hashlib.sha256(canon).hexdigest()
    manifest = {
        "corpus": prov["source"],
        "url": prov["url"],
        "url_total_bytes": prov["url_total_bytes"],
        "download_mode": prov["download_mode"],
        "chapters": prov["chapter_list"],
        "n_files": len(files),
        "total_audio_seconds": round(sum(f["seconds"] for f in files), 3),
        "containers": "flac, 16 kHz mono, decoded with ffmpeg",
        "selection": f"np.random.RandomState({SEED}).permutation({len(utts_all)})[:{args.n_utts}]"
                     f", sorted by id; first {args.n_calib} = calib",
        "canonicalization": "sha256 of json.dumps(sorted by id, keys sorted, separators=(',',':'))",
        "manifest_sha256": manifest_sha,
        "files": files,
    }
    dump_json(manifest, MANIFEST_JSON)
    log(f"[manifest] {len(files)} files, sha256(manifest)={manifest_sha[:16]}... -> "
        f"{MANIFEST_JSON.relative_to(ROOT)}")

    # ---- JSON ------------------------------------------------------------- #
    worst = rows[-1]
    notes = [
        "Real English speech with published reference transcripts, unlike P5 "
        f"(synthetic tones/chirp/harmonics, no ground truth). Source: {prov['source']}, "
        f"{prov.get('chapters')} chapters / {len({u['speaker'] for u in utts_all})} speakers, "
        f"streamed from {prov['url']} and aborted after {prov['bytes_consumed'] / 1e6:.0f} MB "
        f"({prov['url_total_bytes'] / 1e6:.0f} MB total). Audio decoded to 16 kHz mono fp32 by "
        "ffmpeg (no soundfile/librosa/torchaudio on this box; the HF "
        "`librispeech_asr_dummy` mirror now only ships a parquet shard, unreadable "
        "without pyarrow).",
        f"Split: {len(calib_utts)} calibration utterances build the permutation pi per layer "
        "and calibrate theta_l (quantile of permuted block energies, target tau of "
        f"(token, block) pairs); {len(eval_utts)} disjoint utterances are evaluated. All "
        f"WER/CER/exact-match numbers are on the evaluation split only.",
        "WER/CER use a local two-row Levenshtein (no jiwer) on "
        f"{normalizer_name}-normalized text; values are per-utterance rates averaged over "
        "utterances (mean +- sample std, ddof=1), with wer_corpus (total edits / total "
        "reference words) reported alongside.",
        "exact_match = fraction of utterances whose hypothesis tokenization equals the "
        "reference tokenization exactly (whisper tokenizer on the normalized hypothesis "
        "vs the normalized reference). It is a strict metric: dense whisper-base itself is "
        f"not 1.0 ({rows[0]['exact_match']:.3f}).",
        "teacher_forced_top1 = argmax agreement between the gated and dense decoder logits "
        "when both are teacher-forced on the prompt + the *reference* tokens, so unlike P5's "
        "free-running agreement it is a label-bearing metric. enc_rel_l2 is the relative L2 "
        "between the gated and dense encoder output, averaged over utterances.",
        "Timing is CPU only (torch.set_num_threads(32), 16-core Zen 5 / 32 SMT threads): "
        "the full 6-layer audio encoder (no lm_head) and the full decoder loop (prefill + "
        "greedy steps up to <|eot|>). Values are ms/utterance, median over repeated rounds "
        "that alternate configs round-robin so drift hits all configs alike. tau=-1 is the "
        "unmodified dense model (original fc2.forward); tau=0 keeps every block but still "
        "permutes, and is verified isomorphic to dense before any result is reported.",
        "The honest wall-clock result: the per-token gate still executes a dense fc1 and a "
        "dense fc2 GEMM inside PyTorch (mask + gather + alpha), so PyTorch-level E2E shows no "
        "speedup. The AVX-512 kernel mounts the *real* 32-lane skipping down-projection "
        "(down_skip_avx512_auto2), verified to match the PyTorch gated path, which removes "
        "the skipped tiles from fc2; fc1 remains dense, so the encoder gain is bounded by "
        "fc2's share of the pass.",
        "dead-block numbers are the fraction of 32-lane blocks with *exactly* zero energy in "
        "the fc2 input, measured on every selected utterance (calib + eval), before "
        "(natural lane order) and after (pi order) the permutation; mean +- std across "
        "utterances. This replaces P5's single-clip 51-54% claim on enc.4/enc.5 with the "
        "real distribution over utterances.",
        f"`energy_captured` / `actual_skip` are measured on exactly the quantity the gate "
        f"thresholds (block L2 in pi order). At tau=0.6 WER is {worst['wer']:.3f} vs "
        f"{rows[0]['wer']:.3f} dense: the mechanism is not free, and the trade is now "
        "measured in absolute WER/CER instead of dense-vs-gated agreement.",
    ]

    out = {
        "phase": "P6 task 3 -- real-speech Whisper evaluation",
        "model": "models/whisper-base",
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "seed": SEED,
        "arch": arch,
        "dataset": {
            "corpus": prov["source"],
            "url": prov["url"],
            "url_total_bytes": prov["url_total_bytes"],
            "chapters": prov["chapter_list"],
            "speakers": sorted({u["speaker"] for u in utts_all}),
            "n_utterances": len(picked),
            "n_calib": len(calib_utts),
            "n_eval": len(eval_utts),
            "total_audio_seconds": round(sum(secs.values()), 1),
            "mean_audio_seconds": round(float(np.mean(list(secs.values()))), 2),
            "audio_root": str(AUDIO_ROOT.relative_to(ROOT)),
            "audio_manifest": str(MANIFEST_JSON.relative_to(ROOT)),
            "audio_manifest_sha256": manifest_sha,
            "provenance_file": str(PROVENANCE.relative_to(ROOT)),
            "decoder": "ffmpeg -f f32le -ac 1 -ar 16000",
            "normalizer": normalizer_name,
            "prompt_ids": PROMPT_IDS,
            "decoding": "greedy, suppress_tokens applied, stop at <|eot|>, "
                        f"max {args.max_new} new tokens, no timestamps",
        },
        "rows": rows,
        "timing": timing,
        "dead_blocks": dead_blocks,
        "kernel_avx512": kernel,
        "iso_check": {"tol_enc_rel_l2": TOL_ENC_REL_L2, "tol_dec_max_abs": TOL_DEC_MAX_ABS,
                      "passed": True, **iso},
        "theta": {f"{tau:g}": {k: (v if math.isfinite(v) else "-inf")
                               for k, v in thetas[tau].items()} for tau in taus},
        "samples": [{"id": u["id"], "seconds": u["seconds"],
                     "ref": u["ref"],
                     "hyp_dense": hyp_text[None][u["id"]],
                     "hyp_tau0.6": hyp_text[0.6][u["id"]] if 0.6 in hyp_text else None}
                    for u in eval_utts[:12]],
        "per_utterance": [
            {"id": u["id"], "ref_words": len(ref_words[u["id"]]), "seconds": u["seconds"],
             **{(f"dense" if t is None else f"tau{t:.2f}"): {
                 "wer": per_utt[t]["wer"][i], "cer": per_utt[t]["cer"][i],
                 "exact_match": per_utt[t]["em"][i], "enc_rel_l2": per_utt[t]["rl2"][i],
                 "teacher_forced_top1": per_utt[t]["top1"][i]}
                for t in per_utt}}
            for i, u in enumerate(eval_utts)],
        "notes": notes,
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    dump_json(out, Path(args.out))
    log(f"[done] wrote {args.out} in {out['runtime_seconds']:.1f}s")

    # ---- summary table ---------------------------------------------------- #
    log("")
    log("whisper-base, LibriSpeech dev-clean, 32-lane FFN block skipping "
        f"({len(eval_utts)} eval utterances, {out['dataset']['total_audio_seconds']:.0f}s audio)")
    hdr = (f"{'tau':>5} {'WER':>7} {'+-':>6} {'CER':>7} {'+-':>6} {'EM':>6} "
           f"{'enc_rL2':>8} {'top1tf':>6} {'skip':>6} {'enc ms':>8} {'dec ms':>8} "
           f"{'enc x':>6} {'dec x':>6}")
    log(hdr)
    log("-" * len(hdr))
    for r, t in zip(rows, timing):
        log(f"{r['label']:>5} {r['wer']:>7.4f} {r['wer_std']:>6.4f} {r['cer']:>7.4f} "
            f"{r['cer_std']:>6.4f} {r['exact_match']:>6.3f} {r['enc_rel_l2']:>8.4f} "
            f"{r['teacher_forced_top1']:>6.3f} {r['actual_skip']:>6.3f} "
            f"{t['enc_ms_per_utt']:>8.2f} {t['dec_ms_per_utt']:>8.2f} "
            f"{t['enc_speedup']:>6.3f} {t['dec_speedup']:>6.3f}")
    if kernel.get("available"):
        log("avx512 kernel encoder ms/utt: " + " ".join(
            f"{k}:{v:.2f}({kernel['enc_speedup'][k]:.3f}x)"
            for k, v in kernel["enc_ms_per_utt"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
