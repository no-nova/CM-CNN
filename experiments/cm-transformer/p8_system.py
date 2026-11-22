"""Phase-8 system-level measurements: batch/context sweep, speedup ladder, counters.

GPU timings use the reference incremental decoder (one token per step with a
warm cache), so they include attention, the gated-DeltaNet recurrence and the
sampling head, not only the FFN.  CPU counters come from `perf stat` when the
kernel allows it; otherwise the byte-count model is reported instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import LANE, log, set_deterministic
import p6_sota as P6
from model_ref import new_state
from run_text_eval import load_jsonl

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DATA = HERE.parents[1] / "data" / "benchmarks"
SEED = 1234


def kw(cfg):
    """Map build_cfg keys onto TextBackbone.forward keywords."""
    cfg = cfg or {}
    out = {}
    if cfg.get("override") is not None:
        out["mlp_override"] = cfg["override"]
    if cfg.get("scale") is not None:
        out["skip_scale"] = cfg["scale"]
    if cfg.get("theta") is not None:
        out["skip_theta"] = cfg["theta"]
    if cfg.get("norm_gamma"):
        out["norm_gamma"] = cfg["norm_gamma"]
    out["skip_rate"] = float(cfg.get("rate", 0.0))
    return out


def pct(x, q):
    return float(np.percentile(np.asarray(x), q))


@torch.no_grad()
def decode_steps(tb, ids, n_steps=16, cfg=None):
    k = kw(cfg)
    st = new_state()
    tb.forward(ids, state=st, pos0=0, **k)
    pos = len(ids)
    cur = ids[-1:]
    lat = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        out = tb.forward(cur, state=st, pos0=pos, **k)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e3)
        pos += 1
        cur = out["logits"].argmax(-1).reshape(1).to(ids.device)
    return lat


@torch.no_grad()
def prefill_time(tb, ids, cfg=None, reps=3):
    k = kw(cfg)
    best = 1e9
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tb.forward(ids, **k)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def corpus_ids(tb_tok, n):
    rows = load_jsonl(DATA / "wikitext2_test.jsonl")
    text = " ".join(r["text"] for r in rows[300:340])
    return P6.P.encode(tb_tok, [text], n)[0]


def task_system(device="cuda", batch=(1, 2, 4, 8), ctx=(128, 512, 2048)):
    from p1_v2 import build_perm
    from model_ref import block_energy
    import p8_reviewer_defense as P8

    out = {"batch_sweep": {}, "ctx_sweep": {}, "ladder": {}, "counters": {}}
    for model in ("qwen3.5-0.8b", "qwen3-0.6b"):
        runner = P6.make_runner(model, device, torch.float32)
        tb = runner.tb if isinstance(runner, P6.RefP1) else None
        if tb is None:
            log(f"[system] {model}: HF path skipped for the incremental ladder")
            continue
        log(f"[system] {model}")
        ids = corpus_ids(runner.tok, max(ctx) + 32).to(device)
        # ---- batch: B sequences decoded in parallel, T=1 steps
        b_rows = {}
        for B in batch:
            try:
                if B == 1:
                    lat = decode_steps(tb, ids[:512], n_steps=12)
                    b_rows[B] = {"mean": float(np.mean(lat)), "p50": pct(lat, 50),
                                 "p95": pct(lat, 95), "ms_per_token": float(np.mean(lat) / B)}
                    continue
                st = [new_state() for _ in range(B)]
                for s in st:
                    tb.forward(ids[:512], state=s, pos0=0)
                lat = []
                for _ in range(12):
                    t0 = time.perf_counter()
                    for s in st:
                        tb.forward(ids[511:512], state=s, pos0=512)
                    torch.cuda.synchronize()
                    lat.append((time.perf_counter() - t0) * 1e3)
                b_rows[B] = {"mean": float(np.mean(lat)), "p50": pct(lat, 50),
                             "p95": pct(lat, 95), "ms_per_token": float(np.mean(lat) / B)}
            except torch.OutOfMemoryError:
                b_rows[B] = {"oom": True}
                torch.cuda.empty_cache()
            log(f"  batch B={B}: {b_rows[B]}")
        out["batch_sweep"][model] = b_rows

        # ---- context length: prefill time for L tokens
        c_rows = {}
        for L in ctx:
            try:
                ms = prefill_time(tb, ids[:L])
                c_rows[L] = {"prefill_ms": ms, "tokens_per_s": L / ms * 1e3}
            except torch.OutOfMemoryError:
                c_rows[L] = {"oom": True}
                torch.cuda.empty_cache()
            log(f"  ctx L={L}: {c_rows[L]}")
        out["ctx_sweep"][model] = c_rows

        # ---- ladder at tau = 0.20 / 0.40
        texts = P6.CALIB_PROMPTS + [r["text"] for r in load_jsonl(DATA / "wikitext2_test.jsonl")[200:260]]
        acts = runner.capture(P6.P.encode(runner.tok, texts, 256))
        perm = [P8.strategy_perm(acts[i], P8.mlp_weights(runner, i)[2], "ours")
                for i in range(runner.L)]
        lad = {"dense": {"end_to_end_ms": float(np.median(decode_steps(tb, ids[:512])))}}
        for rate in (0.2, 0.4):
            cfg = P8.build_cfg(runner, acts, perm, rate, "abs")
            lad[f"tau{rate:.2f}"] = {
                "end_to_end_ms": float(np.median(decode_steps(tb, ids[:512], cfg=cfg)))}
            del cfg
            torch.cuda.empty_cache()
        dense_ms = lad["dense"]["end_to_end_ms"]
        for k, v in lad.items():
            v["speedup"] = dense_ms / v["end_to_end_ms"]
        lad["break_even_tau"] = next(
            (float(k[3:]) for k in lad if k.startswith("tau")
             and lad[k]["speedup"] >= 1.0), None)
        out["ladder"][model] = lad
        log(f"  ladder: {json.dumps(lad)}")
        del runner
        torch.cuda.empty_cache()
    return out


def task_counters():
    """perf stat on a single-layer block-major vs row-major GEMV (CPU)."""
    cmd = ("perf stat -e L1-dcache-load-misses,LLC-load-misses,cycles "
           f"{sys.executable} bench_p4_iq4.py --only-dense 2>/dev/null")
    out = {"perf_available": False}
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=240)
        out["perf_available"] = r.returncode == 0 and "LLC-load-misses" in r.stderr
        out["raw"] = r.stderr[-2000:]
    except Exception as e:                                    # pragma: no cover
        out["error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    set_deterministic(SEED)
    t0 = time.time()
    out = task_system(a.device)
    out["counters"] = task_counters()
    out.update({"seed": SEED, "device": a.device, "elapsed_s": round(time.time() - t0, 1)})
    p = RES / f"p8_system{a.tag}.json"
    p.write_text(json.dumps(out, indent=1))
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
