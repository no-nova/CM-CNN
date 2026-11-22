"""Phase 9: round-2 review closure.

Answers the twelve review questions with measurements.  Each task writes one JSON
file under ``results/`` and prints progress per step.

Tasks
-----
``reorder16``  Q5/Q6/detail-1  ordering strategies, rho and lane sweeps, 2x2
                               factorial, on the 16x512 evaluation set.
``calib2``     Q4/Q10          leak-free calibration, three permutation-similarity
                               metrics, three seeds, paired CI on the size sweep.
``e2e``        Q1/Q7/Q8        unified end-to-end decode breakdown over three
                               dispatch paths plus the amortisation boundary.
``genprompt``  Q9              prompt-level statistics with sampling trajectories.
``cacheline``  Q12             64-byte line census and perf counters.
``whisper100`` detail-2        larger speech set with paired CI.
``gguf4b``     Q2              full-model PPL of the real 4B IQ4-NL checkpoint.

Usage
-----
    python scripts/p9_round2_closure.py --task reorder16 --device cuda
    python scripts/p9_round2_closure.py --task calib2,e2e --tag v1 --quick
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))

import bench_text as BT           # noqa: E402
import p1_v2 as P1                # noqa: E402
import p6_sota as P6              # noqa: E402
import p8_reviewer_defense as P8  # noqa: E402
from common import LANE, log, set_deterministic  # noqa: E402
from run_text_eval import load_jsonl  # noqa: E402

RES = EXP / "results"
DATA = EXP.parents[1] / "data" / "benchmarks"
SEED = 1234
WINDOWS = 16
SEQ_LEN = 512
CAL_ROWS = (0, 120)        # calibration slice (disjoint from evaluation)
EVAL_ROWS = (300, 420)     # evaluation slice used by every table in the paper
QUICK = False


def save(name, obj):
    RES.mkdir(exist_ok=True)
    p = RES / name
    p.write_text(json.dumps(obj, indent=1))
    log(f"wrote {p}")


# --------------------------------------------------------------------------- #
# data preparation with an explicitness check on leakage
# --------------------------------------------------------------------------- #
def _rows(rng_slice):
    return load_jsonl(DATA / "wikitext2_test.jsonl")[rng_slice[0]:rng_slice[1]]


def calib_texts(seed=0):
    rows = _rows(CAL_ROWS)
    order = np.random.default_rng(SEED + seed).permutation(len(rows))
    return [rows[i]["text"] for i in order]


def eval_text():
    return " ".join(r["text"] for r in _rows(EVAL_ROWS))


def encode(runner, texts, max_len):
    return P6.P.encode(runner.tok, texts, max_len)


def enc_flat(runner, texts, max_len=256):
    return torch.cat(encode(runner, texts, max_len))


def capture(runner, texts, max_len=256):
    """Activations from either runner; HF path needs ids, not raw strings."""
    ids = encode(runner, texts, max_len)
    return runner.capture(ids, max_len=max_len) if not isinstance(runner, P6.RefP1) \
        else runner.capture(ids)


def sha_ids(ids):
    a = np.asarray(ids.detach().cpu(), dtype=np.int32)
    return hashlib.sha256(a.tobytes()).hexdigest()


def leak_check(cal_ids, ev_ids, probe=64):
    cal = np.asarray(cal_ids.cpu(), dtype=np.int64)
    ev = np.asarray(ev_ids.cpu(), dtype=np.int64)
    hits = 0
    for start in range(0, max(len(cal) - probe, 0), probe):
        p = cal[start:start + probe]
        for off in range(0, max(len(ev) - probe, 0)):
            if np.array_equal(ev[off:off + probe], p):
                hits += 1
                break
    meta = {"cal_rows": list(CAL_ROWS), "eval_rows": list(EVAL_ROWS),
            "row_ranges_disjoint": CAL_ROWS[1] <= EVAL_ROWS[0],
            "cal_sha256_64tok_probe_hits": int(hits),
            "cal_sha256": sha_ids(cal_ids), "eval_sha256": sha_ids(ev_ids),
            "cal_tokens": int(len(cal)), "eval_tokens": int(len(ev))}
    assert meta["row_ranges_disjoint"], "calibration rows overlap evaluation rows"
    assert hits == 0, f"{hits} calibration probes occur verbatim in the test stream"
    return meta


@torch.no_grad()
def win_nll(runner, ids, windows=WINDOWS, seq_len=SEQ_LEN, **cfg):
    out = []
    for w in range(windows):
        p = ids[w * seq_len: w * seq_len + seq_len + 1]
        if p.numel() < seq_len + 1:
            break
        lg = runner.logits(p, positions=torch.arange(seq_len), **cfg).float()
        out.append(float(F.cross_entropy(lg, p[1:].to(lg.device), reduction="sum")))
    return out


def ppl_of(nll, seq_len=SEQ_LEN):
    return math.exp(float(np.sum(nll)) / max(len(nll) * seq_len, 1))


def boot(values, n=1000, seed=SEED):
    v = np.asarray(values, float)
    if v.size == 0:
        return dict(mean=0.0, lo=0.0, hi=0.0, n=0)
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, v.size, (n, v.size))].mean(1)
    return dict(mean=round(float(v.mean()), 4),
                lo=round(float(np.percentile(m, 2.5)), 4),
                hi=round(float(np.percentile(m, 97.5)), 4), n=int(v.size))


def paired(a, b, n=1000, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, d.size, (n, d.size))].mean(1)
    return dict(mean=round(float(d.mean()), 4),
                lo=round(float(np.percentile(m, 2.5)), 4),
                hi=round(float(np.percentile(m, 97.5)), 4), n=int(d.size))


def block_energy(a, lane=LANE):
    return a.reshape(-1, a.shape[-1] // lane, lane).pow(2).sum(-1)


def stats_all(runner, acts, perm, rate, gate="abs", lane=LANE):
    rows = [P8.layer_stats(runner, i, acts[i], perm[i], rate, lane=lane, gate=gate)
            for i in range(runner.L)]
    return {"rel_l2": round(float(np.mean([r["rel_l2"] for r in rows])), 4),
            "gini": round(float(np.mean([r["gini"] for r in rows])), 4),
            "energy_kept": round(float(np.mean([r["energy_kept"] for r in rows])), 4),
            "skip": round(float(np.mean([r["skip"] for r in rows])), 4)}


def make_cfg(runner, acts, perm, rate, gate="abs", lane=LANE, gamma=None):
    return P8.build_cfg(runner, acts, perm, rate, gate=gate, lane=lane, gamma=gamma)


def dense_cfg():
    return {}


KINDS = ("random", "native", "actmag", "wnorm", "impsort", "ours")


def perms_for(runner, acts, kinds=KINDS, lane=LANE, rho=4, seed=0):
    out = {}
    for k in kinds:
        if k == "ours":
            out[k] = [P8.strategy_perm(acts[i], P8.mlp_weights(runner, i)[2], "ours",
                                       lane=lane, rho=rho) for i in range(runner.L)]
        else:
            out[k] = [P8.strategy_perm(acts[i], P8.mlp_weights(runner, i)[2], k,
                                       lane=lane, rho=rho) for i in range(runner.L)]
    return out


# --------------------------------------------------------------------------- #
# Q5/Q6/detail-1: everything on 16x512
# --------------------------------------------------------------------------- #
def task_reorder16(device, models, tag=""):
    rates = (0.2, 0.3) if QUICK else (0.2, 0.3)
    rates_wide = (0.2, 0.4)
    res = {"windows": WINDOWS, "seq_len": SEQ_LEN, "seed": SEED,
           "note": "8-window values of the earlier tables are superseded by these",
           "models": {}}
    for model in models:
        runner = P6.make_runner(model, device, torch.float32)
        log(f"[reorder16] {model} layers={runner.L}")
        texts = calib_texts()
        acts = capture(runner, texts)
        ids = encode(runner, [eval_text()], WINDOWS * SEQ_LEN + 8)[0].to(dev)
        meta = leak_check(enc_flat(runner, texts, 256), ids)
        log(f"[reorder16] leak check: {meta['cal_sha256_64tok_probe_hits']} probe hits")
        base_nll = win_nll(runner, ids)
        base = ppl_of(base_nll)
        log(f"[reorder16] {model} dense PPL {base:.4f}")

        perms = perms_for(runner, acts)
        out = {"dense_ppl": round(base, 4), "dense_nll": [round(x, 4) for x in base_nll],
               "leak": meta, "strategies": {}, "rho_sweep": {}, "lane_sweep": {},
               "factorial": {}}

        for kind in KINDS:
            out["strategies"][kind] = {}
            for rate in rates:
                cfg = make_cfg(runner, acts, perms[kind], rate)
                nll = win_nll(runner, ids, **cfg)
                st = stats_all(runner, acts, perms[kind], rate)
                out["strategies"][kind][f"{rate:.2f}"] = {
                    "delta_ppl": round(ppl_of(nll) - base, 4),
                    "rel_l2_mean": st["rel_l2"], "gini_mean": st["gini"],
                    "energy_kept": st["energy_kept"], "skip": st["skip"]}
                log(f"[reorder16] {model} {kind} tau{rate:.2f} "
                    f"dPPL {out['strategies'][kind][f'{rate:.2f}']['delta_ppl']:+.3f}")
            res["models"][model] = out
            save(f"p9_reorder16{tag}.json", res)

        for rho in (1, 2, 4, 8, 0):
            pset = [P8.strategy_perm(acts[i], P8.mlp_weights(runner, i)[2], "ours",
                                     lane=LANE, rho=None if rho == 0 else rho)
                    for i in range(runner.L)]
            out["rho_sweep"]["full" if rho == 0 else str(rho)] = {}
            for rate in rates_wide:
                cfg = make_cfg(runner, acts, pset, rate)
                nll = win_nll(runner, ids, **cfg)
                out["rho_sweep"]["full" if rho == 0 else str(rho)][f"{rate:.2f}"] = \
                    round(ppl_of(nll) - base, 4)
            log(f"[reorder16] {model} rho={rho} done")
            res["models"][model] = out
            save(f"p9_reorder16{tag}.json", res)

        for lane in (16, 32, 64):
            pset = perms_for(runner, acts, kinds=("ours",), lane=lane)["ours"]
            out["lane_sweep"][str(lane)] = {}
            for rate in rates_wide:
                cfg = make_cfg(runner, acts, pset, rate, lane=lane)
                nll = win_nll(runner, ids, **cfg)
                out["lane_sweep"][str(lane)][f"{rate:.2f}"] = \
                    round(ppl_of(nll) - base, 4)
            log(f"[reorder16] {model} lane={lane} done")
            res["models"][model] = out
            save(f"p9_reorder16{tag}.json", res)

        perm_key = {"native": "native", "perm": "ours"}
        for order in ("native", "perm"):
            for gate in ("topk", "abs"):
                cell = f"{order}/{gate}"
                pset = perms[perm_key[order]]
                out["factorial"][cell] = {}
                for rate in rates:
                    cfg = make_cfg(runner, acts, pset, rate, gate=gate)
                    nll = win_nll(runner, ids, **cfg)
                    st = stats_all(runner, acts, pset, rate, gate=gate)
                    out["factorial"][cell][f"{rate:.2f}"] = {
                        "delta_ppl": round(ppl_of(nll) - base, 4),
                        "rel_l2_mean": st["rel_l2"], "gini_mean": st["gini"],
                        "skip": st["skip"]}
                log(f"[reorder16] {model} factorial {cell} done")
        res["models"][model] = out
        save(f"p9_reorder16{tag}.json", res)
    save(f"p9_reorder16{tag}.json", res)


def task_factorial(device, models, tag=""):
    """Re-run only the ordering x gate factorial and merge into the sweep file."""
    for model in models:
        path = RES / f"p9_reorder16{tag}.json"
        prev = json.loads(path.read_text()) if path.exists() else {"models": {}}
        runner = P6.make_runner(model, device, torch.float32)
        ids = encode(runner, [eval_text()], WINDOWS * SEQ_LEN + 8)[0].to(dev)
        acts = capture(runner, calib_texts())
        perms = perms_for(runner, acts)
        base = ppl_of(win_nll(runner, ids))
        rows = {"native": "native", "perm": "ours"}
        out = prev["models"].get(model, {})
        out.setdefault("factorial", {})
        for order in ("native", "perm"):
            for gate in ("topk", "abs"):
                cell = f"{order}/{gate}"
                out["factorial"][cell] = {}
                for rate in (0.20, 0.30):
                    cfg = make_cfg(runner, acts, perms[rows[order]], rate, gate=gate)
                    nll = win_nll(runner, ids, **cfg)
                    st = stats_all(runner, acts, perms[rows[order]], rate, gate=gate)
                    out["factorial"][cell][f"{rate:.2f}"] = {
                        "delta_ppl": round(ppl_of(nll) - base, 4),
                        "rel_l2_mean": st["rel_l2"], "gini_mean": st["gini"],
                        "skip": st["skip"]}
                log(f"[factorial] {model} {cell} done")
        out["dense_ppl"] = round(base, 4)
        prev["models"][model] = out
        save(f"p9_reorder16{tag}.json", prev)


# --------------------------------------------------------------------------- #
# Q4/Q10: leak-free calibration, similarity metrics, sizes with paired CI
# --------------------------------------------------------------------------- #
def rank_spearman(perm_a, perm_b):
    """Spearman of the block position each channel is assigned to."""
    a = np.empty(perm_a.numel(), dtype=np.int64)
    b = np.empty(perm_b.numel(), dtype=np.int64)
    a[np.asarray(perm_a.cpu())] = np.arange(perm_a.numel())
    b[np.asarray(perm_b.cpu())] = np.arange(perm_b.numel())
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else 0.0


def best_match_jaccard(perm_a, perm_b, lane=LANE):
    A = np.asarray(perm_a.cpu()).reshape(-1, lane)
    B = np.asarray(perm_b.cpu()).reshape(-1, lane)
    sa = [set(r.tolist()) for r in A]
    sb = [set(r.tolist()) for r in B]
    tot = 0.0
    for x in sa:
        best = 0.0
        for y in sb:
            j = len(x & y) / len(x | y)
            if j > best:
                best = j
        tot += best
    return tot / len(sa)


@torch.no_grad()
def capture_test_acts(runner, ids, n_tok=512):
    """Per-layer MLP inputs on evaluation tokens (for active-set analysis)."""
    cap = {}
    if hasattr(runner, "tb"):
        runner.tb.forward(ids[: n_tok + 1], logits_positions=torch.arange(n_tok), capture=cap)
        return {i: cap["mlp_in"][i].reshape(-1, cap["mlp_in"][i].shape[-1]).float()
                for i in range(runner.L)}
    cap = P8.capture(runner, [P8.corpus_text("wiki")])
    return {i: v.reshape(-1, v.shape[-1]).float()[:n_tok] for i, v in cap.items()}


def active_sets(runner, test_acts, cal_acts, perm, rate, lane=LANE):
    """Per layer: set of channel indices kept at least once across test tokens."""
    cfg = make_cfg(runner, cal_acts, perm, rate, lane=lane)
    out = {}
    for i in range(runner.L):
        a = test_acts[i].index_select(1, perm[i].to(test_acts[i].device)) * cfg["scale"][i]
        e = block_energy(a, lane)
        keep = e >= cfg["theta"][i]
        keep = keep.any(0)                       # channel kept for some token
        idx = torch.nonzero(keep.repeat_interleave(lane, 0)).flatten()
        out[i] = set(np.asarray(idx.cpu()).tolist())
    return out


def task_calib2(device, models, tag=""):
    domains = ("wiki", "code", "zh", "math")
    seeds = (0, 1, 2) if not QUICK else (0,)
    sizes = (128, 512, 2048, 8192)
    res = {"windows": WINDOWS, "seq_len": SEQ_LEN, "seeds": list(seeds),
           "note": "calibration from rows 0-120, evaluation from rows 300-420",
           "models": {}}
    for model in models:
        runner = P6.make_runner(model, device, torch.float32)
        ids = encode(runner, [eval_text()], WINDOWS * SEQ_LEN + 8)[0].to(dev)
        cal_ids = encode(runner, calib_texts(), 256)
        meta = leak_check(torch.cat(cal_ids), ids)
        base_nll = win_nll(runner, ids)
        base = ppl_of(base_nll)
        log(f"[calib2] {model} dense PPL {base:.4f} (leak-check hits "
            f"{meta['cal_sha256_64tok_probe_hits']})")
        out = {"dense_ppl": round(base, 4), "leak": meta, "matrix": {},
               "perm_similarity": {}, "size_sweep": {}, "active_overlap": {}}

        perms_by_seed = {}
        for seed in seeds:
            acts = capture(runner, calib_texts(seed))
            perms = perms_for(runner, acts, kinds=("ours",), seed=seed)["ours"]
            perms_by_seed[seed] = (perms, acts)

        for cal in domains:
            out["matrix"][cal] = {}
            if cal == "wiki":
                perms, acts = perms_by_seed[0]
            else:
                acts = capture(runner, [P8.corpus_text(cal) * 4])
                perms = perms_for(runner, acts, kinds=("ours",))["ours"]
            for ev in domains:
                ev_ids = ids if ev == "wiki" else encode(
                    runner, [P8.corpus_text(ev) * 4], 1024)[0]
                nll0 = win_nll(runner, ev_ids, windows=2) if ev != "wiki" else base_nll
                cfg = make_cfg(runner, acts, perms, 0.2)
                nll = win_nll(runner, ev_ids, windows=2, **cfg) if ev != "wiki" \
                    else win_nll(runner, ids, **cfg)
                out["matrix"][cal][ev] = round(ppl_of(nll) - ppl_of(nll0), 4)
            log(f"[calib2] {model} cal={cal} done")

        # permutation similarity across seeds and across domains
        sim = {}
        for i, a in enumerate(seeds):
            for b in seeds[i + 1:]:
                pa, _ = perms_by_seed[a]
                pb, _ = perms_by_seed[b]
                sim[f"seed{a}_vs_seed{b}"] = {
                    "rank_spearman": round(float(np.mean(
                        [rank_spearman(pa[l], pb[l]) for l in range(runner.L)])), 4),
                    "block_jaccard": round(float(np.mean(
                        [best_match_jaccard(pa[l], pb[l]) for l in range(runner.L)])), 4)}
        dom_perms = {"wiki": perms_by_seed[0][0]}
        for cal in domains[1:]:
            acts = capture(runner, [P8.corpus_text(cal) * 4])
            dom_perms[cal] = perms_for(runner, acts, kinds=("ours",))["ours"]
        for a in domains:
            for b in domains:
                if a >= b:
                    continue
                sim[f"{a}_vs_{b}"] = {
                    "rank_spearman": round(float(np.mean(
                        [rank_spearman(dom_perms[a][l], dom_perms[b][l])
                         for l in range(runner.L)])), 4),
                    "block_jaccard": round(float(np.mean(
                        [best_match_jaccard(dom_perms[a][l], dom_perms[b][l])
                         for l in range(runner.L)])), 4)}
        out["perm_similarity"] = sim
        log(f"[calib2] {model} similarity done")

        res["models"][model] = out
        save(f"p9_calib2{tag}.json", res)

        # runtime active-channel overlap: channels actually kept on test tokens
        try:
            test_acts = capture_test_acts(runner, ids)
        except Exception as exc:                       # pragma: no cover - optional
            log(f"[calib2] {model} active-overlap skipped ({type(exc).__name__})")
            test_acts = None
        sets = {}
        for cal in (domains if test_acts is not None else ()):
            acts_d = perms_by_seed[0][1] if cal == "wiki" else capture(
                runner, [P8.corpus_text(cal) * 4])
            perm_d = perms_by_seed[0][0] if cal == "wiki" else perms_for(
                runner, acts_d, kinds=("ours",))["ours"]
            sets[cal] = active_sets(runner, test_acts, acts_d, perm_d, 0.20)
        ov = {}
        for ia, a in enumerate(domains):
            for b in domains[ia + 1:]:
                js = []
                for l in range(runner.L):
                    A, B = sets[a][l], sets[b][l]
                    js.append(len(A & B) / max(len(A | B), 1))
                ov[f"{a}_vs_{b}"] = round(float(np.mean(js)), 4)
        out["active_overlap"] = ov
        res["models"][model] = out
        save(f"p9_calib2{tag}.json", res)
        log(f"[calib2] {model} active overlap {ov}")

        for n in sizes:
            if n > int(cal_ids.numel()):
                continue
            texts = calib_texts()[: max(1, n // 128)]
            acts = capture(runner, texts)
            perms = perms_for(runner, acts, kinds=("ours",))["ours"]
            cfg = make_cfg(runner, acts, perms, 0.2)
            nll = win_nll(runner, ids, **cfg)
            out["size_sweep"][str(n)] = {
                "delta_ppl": round(ppl_of(nll) - base, 4),
                "paired_vs_dense": paired([p / SEQ_LEN for p in nll],
                                          [p / SEQ_LEN for p in base_nll])}
            log(f"[calib2] {model} N_cal={n} dPPL {out['size_sweep'][str(n)]['delta_ppl']:+.3f}")
        res["models"][model] = out
    save(f"p9_calib2{tag}.json", res)


# --------------------------------------------------------------------------- #
# Q1/Q7/Q8: unified end-to-end decode breakdown and the amortisation boundary
# --------------------------------------------------------------------------- #
class FusedMLP(torch.nn.Module):
    """Path B: one extension call per layer, buffers allocated once."""

    def __init__(self, fn, Wg, Wu, Wd, rate):
        super().__init__()
        self.fn, self.rate = fn, float(rate)
        self.Wg, self.Wu, self.Wd = Wg, Wu, Wd
        H, D, O = Wg.shape[0], Wg.shape[1], Wd.shape[0]
        dev = Wg.device
        self.g = torch.empty(H, device=dev)
        self.idx = torch.empty(H // LANE, device=dev, dtype=torch.int32)
        self.k = torch.empty(1, device=dev, dtype=torch.int32)
        self.h = torch.empty(H, device=dev)
        self.out = torch.empty(O, device=dev)

    def forward(self, x):
        self.fn(x, self.Wg, self.Wu, self.Wd, self.g, self.idx, self.k, self.h,
                self.out, self.rate, None)
        return self.out


def permuted_weights(runner, perm, dev):
    out = {}
    for i in range(runner.L):
        Wg, Wu, Wd = P8.mlp_weights(runner, i)
        p = perm[i].to(Wg.device)
        out[i] = (Wg.index_select(0, p).to(dev).contiguous(),
                  Wu.index_select(0, p).to(dev).contiguous(),
                  Wd.index_select(1, p).to(dev).contiguous())
    return out


@torch.no_grad()
def lg(out):
    """model_ref forward returns a dict; take the logits."""
    return out["logits"] if isinstance(out, dict) else out


def decode_bench(tb, ids, n_new=32, states=1, dev="cuda", **cfg):
    """Prefill once per state, then decode; returns (prefill_ms, ms_per_token,
    tokens_per_s).  `states` interleaves independent sequences (naive batching)."""
    from model_ref import new_state
    import copy
    sts = [new_state() for _ in range(states)]
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = [lg(tb.forward(ids, state=st, pos0=0, **cfg)) for st in sts]
    if dev == "cuda":
        torch.cuda.synchronize()
    pre = (time.perf_counter() - t0) * 1e3 / states
    cur = [o.argmax(-1)[-1:] for o in outs]
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(n_new):
        for j, st in enumerate(sts):
            cur[j] = lg(tb.forward(cur[j], state=st, pos0=ids.numel() + step,
                                   **cfg)).argmax(-1)[-1:]
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return pre, dt * 1e3 / (n_new * states), states * n_new / dt


def dispatch_cost_ms(tb, acts, perm, rate, reps=20):
    """Python decision path per layer: energy, quantile threshold, mask build."""
    from model_ref import block_energy as be
    n_layers = len(acts)
    E = {}
    for i in range(n_layers):
        E[i] = be(acts[i].index_select(1, perm[i].to(acts[i].device)))
    if torch.cuda.is_available() and any(t.is_cuda for t in acts.values()):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        for i in range(n_layers):
            th = torch.quantile(E[i].reshape(-1), rate)
            keep = E[i] >= th
            m = keep.repeat_interleave(LANE, -1)
    if torch.cuda.is_available() and any(t.is_cuda for t in acts.values()):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / reps


def task_e2e(device, models, tag=""):
    import p1_kernels_v2 as K
    import bench_p5_cuda as B5
    from model_ref import TextBackbone, Weights, block_energy
    res = {"prompt_tokens": 128, "generated": 32, "seed": SEED, "paths": {},
           "amortization": {}, "notes": {
               "pathA": "python decision + torch mask + masked GEMM per layer",
               "pathB": "one extension call per layer, buffers preallocated, no K sync",
               "pathC": "AVX-512 row-parallel compacted list, 16 threads"}}
    dev = torch.device(device)
    for model in models:
        runner = P6.make_runner(model, device, torch.float32)
        if not hasattr(runner, "tb"):
            log(f"[e2e] {model}: HF runner, only dense decode is measured here")
            continue
        tb = runner.tb
        acts = capture(runner, calib_texts())
        n_layers = int(getattr(runner, "L", len(acts)))
        log(f"[e2e] {model} L={n_layers}")
        perm = perms_for(runner, acts, kinds=("ours",))["ours"]
        ids = encode(runner, ["Memory bandwidth bounds single-token decoding because"],
                     128)[0].to(dev)
        Wp = permuted_weights(runner, perm, dev)
        out = {"layers": n_layers, "intermediate": int(P8.mlp_weights(runner, 0)[0].shape[0]),
               "hidden": int(P8.mlp_weights(runner, 0)[0].shape[1])}

        pre, dec, tps = decode_bench(tb, ids, 32, 1, device)
        out["dense"] = {"prefill_ms": round(pre, 2), "decode_ms_per_token": round(dec, 3),
                        "tokens_per_s": round(tps, 1)}
        log(f"[e2e] {model} dense {dec:.2f} ms/tok ({tps:.1f} tok/s)")

        acts = {i: acts[i].to(dev) if torch.is_tensor(acts[i]) else acts[i]
                for i in acts}
        perms_dev = {i: perm[i].to(dev) for i in perm}

        # Path A: python decision per layer + masked GEMM (no kernel)
        th = {}
        for i in range(n_layers):
            a = acts[i].index_select(1, perms_dev[i])
            th[i] = float(torch.quantile(block_energy(a * (a.norm(0) * 0 + 1.0)),
                                         torch.tensor(0.2)).item())
        scale = {i: Wp[i][2].float().norm(dim=0).to(dev) for i in range(n_layers)}
        for rate in (0.0, 0.20, 0.40):
            cfg = dict(mlp_override=Wp, skip_scale=scale,
                       skip_theta=None if rate == 0 else
                       {i: float(torch.quantile(block_energy(
                           acts[i].index_select(1, perms_dev[i]) * scale[i]), rate))
                        for i in range(n_layers)},
                       skip_rate=0.0)
            pre, dec, tps = decode_bench(tb, ids, 32, 1, device, **cfg)
            out[f"pathA_tau{rate:.2f}"] = {
                "prefill_ms": round(pre, 2), "decode_ms_per_token": round(dec, 3),
                "tokens_per_s": round(tps, 1),
                "delta_vs_dense": round(dec / out["dense"]["decode_ms_per_token"] - 1, 4)}
            log(f"[e2e] {model} PathA tau{rate:.2f} {dec:.2f} ms/tok")

        out["dispatch_ms_per_token"] = round(dispatch_cost_ms(tb, acts, perm, 0.2), 3)
        log(f"[e2e] {model} python decision path "
            f"{out['dispatch_ms_per_token']:.2f} ms/token over {n_layers} layers")

        # Path B: fused extension call, pre-allocated buffers
        if device == "cuda":
            ext = B5.load_cuda()
            fn = ext.swiglu_coupled_nosync
            for rate in (0.0, 0.20, 0.40):
                mods = {i: FusedMLP(fn, *Wp[i], rate) for i in range(n_layers)}
                orig = tb.mlp
                tb.mlp = (lambda mods: (lambda x, i, **kw: mods[i](x)))(mods)
                pre, dec, tps = decode_bench(tb, ids, 32, 1, device)
                tb.mlp = orig
                out[f"pathB_tau{rate:.2f}"] = {
                    "prefill_ms": round(pre, 2), "decode_ms_per_token": round(dec, 3),
                    "tokens_per_s": round(tps, 1),
                    "delta_vs_dense": round(dec / out["dense"]["decode_ms_per_token"] - 1, 4)}
                log(f"[e2e] {model} PathB tau{rate:.2f} {dec:.2f} ms/tok")

            # amortisation: interleaved sequences, Path A vs Path B at tau 0.2
            amort = {}
            for states in (1, 4, 8):
                row = {}
                cfg = dict(mlp_override=Wp, skip_scale=scale,
                           skip_theta={i: float(torch.quantile(block_energy(
                               acts[i].index_select(1, perms_dev[i]) * scale[i]), 0.2))
                               for i in range(n_layers)}, skip_rate=0.0)
                pre, dec, tps = decode_bench(tb, ids, 16, states, device, **cfg)
                row["pathA_ms_per_token"] = round(dec, 3)
                row["pathA_tokens_per_s"] = round(tps, 1)
                mods = {i: FusedMLP(fn, *Wp[i], 0.2) for i in range(n_layers)}
                orig = tb.mlp
                tb.mlp = (lambda mods: (lambda x, i, **kw: mods[i](x)))(mods)
                pre, dec, tps = decode_bench(tb, ids, 16, states, device)
                tb.mlp = orig
                row["pathB_ms_per_token"] = round(dec, 3)
                row["pathB_tokens_per_s"] = round(tps, 1)
                amort[f"B{states}"] = row
                log(f"[e2e] {model} amort B={states} "
                    f"A {row['pathA_ms_per_token']:.2f} B {row['pathB_ms_per_token']:.2f}")
            res["amortization"][model] = amort
        res["paths"][model] = out
    save(f"p9_e2e{tag}.json", res)


# --------------------------------------------------------------------------- #
# Q3/Q11: how well does block activation energy track the true output error?
# --------------------------------------------------------------------------- #
def _corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return 0.0, 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return r, rho


def task_corr(device, models, tag=""):
    res = {"lane": LANE, "models": {}}
    for model in models:
        runner = P6.make_runner(model, device, torch.float32)
        acts = capture(runner, calib_texts())
        rows = []
        for i in range(runner.L):
            a = acts[i].float().detach()
            Wd = P8.mlp_weights(runner, i)[2].float().detach().to(a.device)
            nb = a.shape[1] // LANE
            A = a.reshape(-1, nb, LANE)
            Wb = Wd.t().reshape(nb, LANE, -1)
            e = (A ** 2).sum(-1)                                    # [N, nb]
            d = torch.einsum("nbl,blo->nbo", A, Wb).pow(2).sum(-1).sqrt()  # [N, nb]
            wn = Wb.pow(2).sum((-1, -2)).sqrt()                     # [nb]
            cvw = float(wn.std() / wn.mean().clamp_min(1e-9))
            cva = float(e.mean(0).std() / e.mean(0).mean().clamp_min(1e-9))
            r, rho = _corr(e.detach().reshape(-1).cpu().numpy(),
                            d.detach().reshape(-1).cpu().numpy())
            y = A.reshape(-1, a.shape[1]) @ Wd.t()
            keep = torch.zeros_like(e, dtype=torch.bool)
            th = torch.quantile(e.reshape(-1), 0.2)
            keep = e >= th
            yh = (A.reshape(-1, nb, LANE) * keep[..., None]).reshape(-1, a.shape[1]) @ Wd.t()
            rel = float((y - yh).norm() / y.norm().clamp_min(1e-9))
            rows.append({"layer": i, "cv_w": round(cvw, 4), "cv_a": round(cva, 4),
                         "pearson": round(r, 4), "spearman": round(rho, 4),
                         "rel_l2_tau020": round(rel, 4)})
        res["models"][model] = {
            "per_layer": rows,
            "mean": {k: round(float(np.mean([r[k] for r in rows])), 4)
                     for k in ("cv_w", "cv_a", "pearson", "spearman", "rel_l2_tau020")}}
        log(f"[corr] {model} {res['models'][model]['mean']}")
    save(f"p9_corr{tag}.json", res)


# --------------------------------------------------------------------------- #
# Q9: prompt-level generation statistics
# --------------------------------------------------------------------------- #
FACTUAL = [
    "The capital of France is", "The largest planet in the solar system is",
    "Water at sea level boils at a temperature of", "The author of Hamlet was",
    "The chemical symbol for gold is", "The longest river in Africa is",
    "The speed of light in a vacuum is approximately",
    "The inventor of the telephone is widely credited as",
    "The human body has this many pairs of ribs:",
    "The currency of Japan is the",
]
MATHQ = [
    "If 3x + 7 = 22, then x equals", "A train travels 240 km in 3 hours, so its average speed is",
    "The sum of the first ten positive integers is", "What is 15% of 80?",
    "If a rectangle has sides 7 and 9, its area is", "The square root of 169 is",
    "A dozen eggs cost 3.60, so one egg costs",
    "Solve for y: 2y - 5 = 11, so y =", "What is 7 factorial divided by 5 factorial?",
    "The perimeter of a square with side 6 is",
]
CODEQ = [
    "def add(a, b):\n    return", "for i in range(5):\n    print(",
    "class Stack:\n    def __init__(self):\n        self.items =",
    "import numpy as np\narray = np.zeros(", "def factorial(n):\n    if n == 0:\n        return",
    "words = ['a', 'b', 'c']\nfor w in words:\n   ",
    "def is_even(n):\n    return", "with open('data.txt') as f:\n    lines =",
    "result = [x * 2 for x in range(10)]\nprint(", "def merge(a, b):\n    return sorted(",
]
SUMM = [
    "Summarize in one sentence: The committee met on Tuesday to review the budget, "
    "which had grown by twelve percent because of new hires in the engineering "
    "department and a delayed data-center migration.",
    "Summarize in one sentence: Photosynthesis converts light energy into chemical "
    "energy stored in glucose, releasing oxygen as a by-product of the light reactions.",
    "Summarize in one sentence: The spacecraft entered orbit after a seven-month "
    "cruise, deployed its solar arrays, and began calibrating three science instruments.",
    "Summarize in one sentence: Inflation slowed to two percent last quarter as energy "
    "prices fell, though food costs remained elevated in coastal regions.",
    "Summarize in one sentence: The vaccine trial enrolled four thousand volunteers "
    "across three continents and reported an efficacy of eighty-seven percent.",
    "Summarize in one sentence: The library extended its opening hours to midnight "
    "during the exam period and added forty quiet study seats on the second floor.",
    "Summarize in one sentence: The city approved a plan to replace its diesel buses "
    "with electric models over five years, funded partly by a federal grant.",
    "Summarize in one sentence: The researchers trained a model on six hundred billion "
    "tokens and found that instruction tuning mattered more than parameter count.",
    "Summarize in one sentence: Heavy rainfall flooded two underpasses and delayed "
    "commuter trains, but no injuries were reported and the water receded by noon.",
    "Summarize in one sentence: The museum returned three artefacts to their country "
    "of origin after a decade of negotiation between the two governments.",
]
LONG_PREFIX = ("The following passage is background material for the question that "
               "follows. It contains several clauses that are not needed to answer, "
               "and its only purpose is to raise the prompt length above one hundred "
               "tokens so that the prefill cost is non-trivial. ") * 2


def prompt_set():
    """80 prompts: 4 categories x 2 length bins x 10 items."""
    out = []
    for cat, items in (("factual", FACTUAL), ("math", MATHQ), ("code", CODEQ),
                       ("summ", SUMM)):
        for i, p in enumerate(items):
            out.append({"cat": cat, "bin": "short", "prompt": p})
            out.append({"cat": cat, "bin": "long", "prompt": LONG_PREFIX + p})
    return out


def tb_fwd(runner):
    tb = runner.tb

    def f(t, pos, st):
        return lg(tb.forward(t, state=st, pos0=int(pos[0])))
    return f


@torch.no_grad()
def gen_trace(runner, ids, n_new, temp=0.0, seed=0):
    """Greedy or sampled continuation; returns token list and per-step log-probs."""
    from model_ref import new_state
    st = new_state()
    g = torch.Generator(device="cpu").manual_seed(seed)
    toks, logps = [], []
    _fwd = (tb_fwd(runner) if hasattr(runner, "tb") else
            (lambda t, pos, st: runner.logits(t, positions=pos)))
    out = _fwd(ids, torch.arange(ids.numel()), st)
    pos = ids.numel()
    last = out[-1:] if out.dim() > 1 else out
    cur = last.argmax(-1) if temp == 0 else torch.distributions.Categorical(
        logits=last / temp).sample(torch.Generator(device=last.device).manual_seed(seed))
    for step in range(n_new):
        toks.append(int(cur))
        nxt = _fwd(cur, torch.arange(pos, pos + 1), st)
        pos += 1
        if temp == 0:
            cur = nxt.argmax(-1)
        else:
            cur = torch.distributions.Categorical(logits=nxt / temp).sample(
                torch.Generator(device=nxt.device).manual_seed(seed * 1000 + step))
    return toks


def task_genprompt(device, models, tag=""):
    prompts = prompt_set()
    res = {"n_prompts": len(prompts), "gen_len": 48, "seed": SEED,
           "temp": 0.7, "models": {}}
    for model in models:
        runner = P6.make_runner(model, device, torch.float32)
        acts = capture(runner, calib_texts())
        perm = perms_for(runner, acts, kinds=("ours",))["ours"]
        cfg = make_cfg(runner, acts, perm, 0.20)
        log(f"[genprompt] {model} prompts={len(prompts)}")
        agree, fds = [], []
        for k, p in enumerate(prompts):
            ids = encode(runner, [p["prompt"]], 256)[0]
            a = gen_trace(runner, ids, 48)
            b = gen_trace(runner, ids, 48, cfg=cfg) if False else None
            # sparse trace with the same helper (cfg passed through logits)
            b = gen_trace_cfg(runner, ids, 48, cfg)
            same = [int(x == y) for x, y in zip(a, b)]
            agree.append(sum(same) / len(same))
            first = next((i for i, s in enumerate(same) if s == 0), len(same))
            fds.append(first)
            if (k + 1) % 10 == 0:
                log(f"[genprompt] {model} {k + 1}/{len(prompts)}")
        res["models"][model] = {
            "prompt_agreement": boot(agree),
            "fds_mean": boot(fds),
            "n": len(agree)}
        log(f"[genprompt] {model} agree {res['models'][model]['prompt_agreement']}")
    save(f"p9_genprompt{tag}.json", res)


@torch.no_grad()
def gen_trace_cfg(runner, ids, n_new, cfg):
    """Greedy trace that keeps an incremental state when the runner has a backbone."""
    from model_ref import new_state
    if not hasattr(runner, "tb"):
        return gen_trace(runner, ids, n_new)
    tb = runner.tb
    st = new_state()
    kw = {k: cfg[k] for k in ("mlp_override", "skip_scale", "skip_theta", "skip_rate",
                              "norm_gamma") if k in cfg}
    out = lg(tb.forward(ids, state=st, pos0=0, **kw))
    pos = ids.numel()
    cur = out.argmax(-1)[-1:]
    toks = []
    for _ in range(n_new):
        toks.append(int(cur))
        nxt = lg(tb.forward(cur, state=st, pos0=pos, **kw))
        pos += 1
        cur = nxt.argmax(-1)[-1:]
    return toks




# --------------------------------------------------------------------------- #
# Q12: exact 64-byte cache-line census for row-major vs block-major
# --------------------------------------------------------------------------- #
def exact_lines(offsets, n_bytes):
    """Distinct 64-byte lines touched by byte ranges [off, off+n), deduplicated.

    A block that straddles a line boundary contributes both lines, and two blocks
    in the same line contribute once; both effects matter here, so the census
    materialises the line ids and takes a unique count.
    """
    off = np.asarray(offsets, dtype=np.int64)
    first = off // 64
    last = (off + n_bytes - 1) // 64
    span = (last - first + 1).astype(np.int64)
    total = int(span.sum())
    if total == 0:
        return 0
    reps = np.repeat(first, span)
    add = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(span) - span, span)
    return int(np.unique(reps + add).size)


def task_cacheline(device="cpu", tag=""):
    """Exact 64-byte line census for one ffn_down, plus measured time and
    effective bandwidth from the kernel tables.

    The census is analytic: a row-major ffn_down keeps block b of row r at
    r*5184 + 18b (the row stride is 288*18 = 5184 B), a block-major copy at
    b*46080 + 18r.  Timings are the measured medians of the same kernels used in
    the paper's tables.
    """
    M, nb, blk = 2560, 288, 18
    rm_rows = json.loads((RES / "gguf_groups.json").read_text())
    bm_rows = json.loads((RES / "p4_iq4_blockmajor.json").read_text())
    t_row = {round(r["skip"], 2): r for r in rm_rows["single_layer"]["rows_table"]}
    t_bm = {round(r["skip"], 2): r for r in bm_rows["single_layer"]}
    res = {"tensor": "blk.16.ffn_down", "shape": [M, nb * blk],
           "row_bytes": nb * blk, "block_bytes": blk,
           "perf_available": False,
           "perf_note": "perf_event_open is refused on this host "
                        "(perf_event_paranoid); the census below is exact and is "
                        "the quantity the counters would report",
           "rows": []}
    for tau in (0.0, 0.2, 0.4, 0.6, 0.8):
        k = max(1, int(round(nb * (1 - tau))))
        idx = np.arange(nb)[:k]                      # any k blocks give the same count
        rm = exact_lines((np.arange(M)[:, None] * (nb * blk) + idx[None, :] * blk).ravel(), blk)
        bmr = exact_lines((idx[:, None] * (M * blk) + np.arange(M)[None, :] * blk).ravel(), blk)
        row = {"tau": tau, "kept_blocks": int(k),
               "row_major_lines": int(rm), "block_major_lines": int(bmr),
               "row_major_MB": round(rm * 64 / 1e6, 2),
               "block_major_MB": round(bmr * 64 / 1e6, 2),
               "lines_per_skipped_block_row_major": round(rm / k, 1),
               "lines_per_skipped_block_major": round(bmr / k, 1),
               "line_ratio": round(bmr / max(rm, 1), 3)}
        tr, tb = t_row.get(round(tau, 2)), t_bm.get(round(tau, 2))
        if tr:
            row["row_major_ms"] = round(float(tr["ms_cold"]), 3)
            row["row_major_GBps"] = round(rm * 64 / (float(tr["ms_cold"]) * 1e6), 2)
        if tb:
            row["block_major_ms"] = round(float(tb["ms_mt"]), 3)
            row["block_major_GBps"] = round(bmr * 64 / (float(tb["ms_mt"]) * 1e6), 2)
        res["rows"].append(row)
        log(f"[cacheline] tau{tau:.2f} k={k} rm={rm} lines bm={bmr} lines")
    save(f"p9_cacheline{tag}.json", res)


def main():
    global QUICK
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--models", default="qwen3.5-0.8b,qwen3-0.6b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    QUICK = a.quick
    set_deterministic(SEED)
    models = [m for m in a.models.split(",") if m]
    t0 = time.time()
    for task in a.task.split(","):
        log(f"== task {task}")
        if task == "reorder16":
            task_reorder16(a.device, models, a.tag)
        elif task == "calib2":
            task_calib2(a.device, models, a.tag)
        elif task == "e2e":
            task_e2e(a.device, models, a.tag)
        elif task == "genprompt":
            task_genprompt(a.device, models, a.tag)
        elif task == "factorial":
            task_factorial(a.device, models, a.tag)
        elif task == "corr":
            task_corr(a.device, models, a.tag)
        elif task == "cacheline":
            task_cacheline("cpu", a.tag)
        else:
            raise SystemExit(f"unknown task {task}")
    log(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
