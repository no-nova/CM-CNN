"""Phase 6 / Task 1 — named literature baselines, ablation ladder, bootstrap CIs.

Five named methods, all at the same average 32-lane block-skip budget tau:

  cats_block      native order + per-layer *absolute* calibrated threshold
                  (CATS, Lee et al., ICML 2024 — channel-level -> 32-lane blocks)
  teal_block      native order + per-token Top-K uniform across layers
                  (TEAL, 2024 — layer-uniform magnitude allocation)
  static_wanda    native order + static block mask from activation-weighted
                  norms (Wanda-style: one mask for all tokens, zero per-token work)
  cm_uncoupled    pi + adaptive threshold + alpha norm rescaling (our P1, down only)
  cm_full         pi + adaptive threshold + w_b importance weighting + alpha
                  + coupled up/down criterion (our full stack)

Ablation ladder: native_uniform -> +pi -> +adaptive -> +alpha -> +w_b -> +coupled

Every accuracy carries a 1000-sample bootstrap 95% CI and a *paired* bootstrap CI
against the dense model (the number a reviewer asks for).

Usage:
  python p6_sota.py --stage main --model both
  python p6_sota.py --stage gsm  --model qwen3.5-0.8b --gsm-limit 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import MODEL_DIR
from model_ref import LANE, block_energy, new_state
import p1_v2 as P
from p1_v2 import (CALIB_PROMPTS, FILES, HFP1, PREP, RefP1, build_params,
                   load_jsonl, log)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA = HERE.parents[1] / "data" / "benchmarks"
SEED = 1234
TAUS = (0.10, 0.20, 0.30, 0.40)
DATASETS = ("arc", "piqa", "boolq")

METHODS = {
    # name          perm    gate      alpha  wb    coupled
    "cats_block":   (None,  "abs",    False, False, False),
    "teal_block":   (None,  "topk",   False, False, False),
    "static_wanda": (None,  "static", False, True,  False),
    "cm_uncoupled": ("perm", "abs",   True,  False, False),
    "cm_full":      ("perm", "abs",   True,  True,  True),
}
LADDER = [
    ("S0 native_uniform", None,  "topk",   False, False, False),
    ("S1 +pi",            "perm", "topk",  False, False, False),
    ("S2 +adaptive",      "perm", "abs",   False, False, False),
    ("S3 +alpha",         "perm", "abs",   True,  False, False),
    ("S4 +wb",            "perm", "abs",   True,  True,  False),
    ("S5 +coupled",       "perm", "abs",   True,  True,  True),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boot_ci(flags, n=1000, seed=SEED):
    a = np.asarray(flags, dtype=np.float64)
    if a.size == 0:
        return dict(mean=0.0, lo=0.0, hi=0.0, n=0)
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, a.size, (n, a.size))].mean(1)
    return dict(mean=float(a.mean()), lo=float(np.quantile(m, .025)),
                hi=float(np.quantile(m, .975)), n=int(a.size))


def paired_ci(base, ours, n=1000, seed=SEED):
    d = np.asarray(ours, float) - np.asarray(base, float)
    if d.size == 0:
        return dict(mean=0.0, lo=0.0, hi=0.0)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, d.size, (n, d.size))].mean(1)
    return dict(mean=float(d.mean()), lo=float(np.quantile(m, .025)),
                hi=float(np.quantile(m, .975)))


# --------------------------------------------------------------------------- #
def build_bank(runner, params, seed=SEED):
    """Permuted weights, w_b and block energies for NATIVE and pi order.

    Activations are captured once (native order) and permuted afterwards, which
    is exact: permuting neuron order of (a, W_up rows, W_down cols) commutes with
    the permutation of the block partition.
    """
    prompts = P.encode(runner.tok, CALIB_PROMPTS +
                       [r["text"] for r in load_jsonl("wikitext2_test.jsonl")[200:260]], 256)
    log("capturing calibration activations ...")
    acts = runner.capture(prompts)                      # {layer: [N, I]} post-activation
    gate_acts = capture_gates(runner, prompts)
    bank = {}
    for key in (None, "perm"):
        W, scale, wb, E, Ew, Eg, Egw = {}, {}, {}, {}, {}, {}, {}
        for i in range(runner.L):
            wd = runner.down_w(i).detach()
            perm = None if key is None else params[key][i].to(runner.device)
            Wd = wd if perm is None else wd.index_select(1, perm)
            Wg = (runner.tb.w[f"{runner.tb.prefix}layers.{i}.mlp.gate_proj.weight"]
                  if isinstance(runner, RefP1) else runner.layers[i].mlp.gate_proj.weight)
            Wu = (runner.tb.w[f"{runner.tb.prefix}layers.{i}.mlp.up_proj.weight"]
                  if isinstance(runner, RefP1) else runner.layers[i].mlp.up_proj.weight)
            if perm is not None:
                Wg, Wu = Wg.index_select(0, perm), Wu.index_select(0, perm)
            a = acts[i].to(runner.device)
            if perm is not None:
                a = a.index_select(1, perm)
            # per-neuron output norm
            s = Wd.float().norm(dim=0)
            # Wanda-style block importance: RMS_t ||a_b|| * ||W_down[:, b]||_F
            # (a is the post-activation act(gate x) * (up x); Wanda's saliency is
            # |activation| * ||weight column||, taken here per 32-lane block).
            rms = a.float().pow(2).mean(0).reshape(-1, LANE).sum(-1).sqrt()
            w = rms * Wd.float().reshape(Wd.shape[0], -1, LANE).pow(2).sum((0, 2)).sqrt()
            w = (w / w.mean().clamp_min(1e-8)).clamp(0.01, 8.0)
            W[i], scale[i], wb[i] = (Wg.detach(), Wu.detach(), Wd.detach()), s, w
            E[i] = block_energy(a * s).float().cpu()
            Ew[i] = block_energy(a * s * w.repeat_interleave(LANE).sqrt()).float().cpu()
            # coupled criterion scores the GATE, not the product: S_b = w_b||g_b||
            g = gate_acts[i].to(runner.device)
            if perm is not None:
                g = g.index_select(1, perm)
            Eg[i] = block_energy(g * s).float().cpu()
            Egw[i] = block_energy(g * s * w.repeat_interleave(LANE).sqrt()).float().cpu()
        bank[key] = {"W": W, "scale": scale, "wb": wb, "E": E, "Ew": Ew,
                     "Eg": Eg, "Egw": Egw}
    return bank


def capture_gates(runner, prompts):
    """Gate pre-activation g = SiLU(W_gate x) per layer, native order."""
    if isinstance(runner, RefP1):
        out = {i: [] for i in range(runner.L)}
        for ids in prompts:
            cap = {}
            with torch.no_grad():
                runner.tb.forward(ids.to(runner.device), capture=cap,
                                  logits_positions=[0])
            for i in range(runner.L):
                x = cap["mlp_in"][i]
                wg = runner.tb.w[f"{runner.tb.prefix}layers.{i}.mlp.gate_proj.weight"]
                out[i].append(F.silu(F.linear(x, wg)).float().cpu())
        return {i: torch.cat(v, 0) for i, v in out.items()}
    cap = {i: [] for i in range(runner.L)}
    for l in runner.layers:
        l.mlp._cm_capg = cap
    with torch.no_grad():
        for ids in prompts:
            runner.model(input_ids=ids.unsqueeze(0).to(runner.device), use_cache=False)
    for l in runner.layers:
        l.mlp._cm_capg = None
    return {i: torch.cat(v, 0) for i, v in cap.items()}


def cfg_for(bank, gate, tau, alpha, wb_weight, coupled):
    """Thresholds / static masks for one (gate, tau) under a bank entry.

    `bank["E"]` are energies in unweighted 32-lane order, `bank["Ew"]` the same
    energies with the w_b importance factor folded in (which is exactly what the
    kernel computes as S_b = w_b * ||g_b||), so the calibrated quantile and the
    runtime score always live in the same space.
    """
    scale, theta, static = {}, None, None
    for i in range(len(bank["W"])):
        s = bank["scale"][i]
        if wb_weight:
            s = s * bank["wb"][i].to(s.device).repeat_interleave(LANE).sqrt()
        scale[i] = s
    if coupled:
        src = bank["Egw"] if wb_weight else bank["Eg"]
    else:
        src = bank["Ew"] if wb_weight else bank["E"]
    if gate == "abs":
        theta = {i: float(torch.quantile(e.reshape(-1), tau)) for i, e in src.items()}
    elif gate == "static":
        static = {}
        for i, w in bank["wb"].items():
            static[i] = (w >= torch.quantile(w, tau)).to(w.device)
    return {"override": bank["W"], "scale": scale, "theta": theta,
            "rate": tau if gate == "topk" else 0.0,
            "norm_gamma": 1.5 if alpha else None,
            "coupled": coupled,
            "static": static}


# --------------------------------------------------------------------------- #
@torch.no_grad()
def mc_items(runner, dataset, limit, cfg):
    """Per-item correctness (for bootstrap CIs), lm-eval length-normalised."""
    items = PREP[dataset](load_jsonl(FILES[dataset]))[:limit]
    flags, rows = [], []
    for it in items:
        ctx = runner.tok(it["ctx"], add_special_tokens=False)["input_ids"][-1024:]
        cand = []
        for c in it["choices"]:
            cid = runner.tok(" " + c.strip(), add_special_tokens=False)["input_ids"]
            ids = ctx + cid
            pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(cid))
            lg = runner.logits(ids, positions=pos, **cfg)
            lp = F.log_softmax(lg.float(), -1)
            ll = sum(float(lp[j, t]) for j, t in enumerate(cid))
            cand.append({"choice": c, "loglik": ll, "n_tok": len(cid),
                         "n_bytes": len((" " + c.strip()).encode())})
        pred = max(range(len(cand)), key=lambda j: cand[j]["loglik"] / cand[j]["n_tok"])
        ok = int(pred == int(it["answer"]))
        flags.append(ok)
        rows.append(ok)
    return flags, rows


@torch.no_grad()
def eval_ppl(runner, ids, windows, cfg, seq_len=512):
    nll, ntok = 0.0, 0
    for w in range(windows):
        p = ids[w * seq_len: w * seq_len + seq_len + 1]
        if p.numel() < seq_len + 1:
            break
        lg = runner.logits(p, positions=torch.arange(seq_len), **cfg).float()
        nll += float(F.cross_entropy(lg, p[1:].to(lg.device), reduction="sum"))
        ntok += seq_len
    return float(np.exp(nll / max(ntok, 1)))


@torch.no_grad()
def gsm_items(runner, limit, max_new, cfg):
    from bench_text import GSM_FEWSHOT, extract_answer
    rows = load_jsonl(DATA / "gsm8k_test200.jsonl")[:limit]
    shots = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in GSM_FEWSHOT)
    flags = []
    t0 = time.perf_counter()
    for k, r in enumerate(rows):
        prompt = f"{shots}\n\nQuestion: {r['question']}\nAnswer:"
        ids = runner.tok(prompt, add_special_tokens=False)["input_ids"]
        gen = runner.generate(ids, max_new, stop_on_answer=False, **cfg)
        txt = runner.tok.decode(gen)
        pred = extract_answer(txt)
        gold = r["gold"]
        flags.append(int(pred is not None and str(pred).replace(",", "") == gold))
        if k % 25 == 0:
            log(f"  gsm {k}/{len(rows)} acc {np.mean(flags):.3f}  "
                f"({(time.perf_counter()-t0)/(k+1):.1f} s/item)")
    return flags


# --------------------------------------------------------------------------- #
def make_runner(name, device, dtype):
    if name == "qwen3.5-0.8b":
        return RefP1(device=device, dtype=dtype)
    return HFP1(str(MODEL_DIR.parent / "Qwen3-0.6B"), device=device, dtype=dtype)


def run_model(name, device, dtype, mc_limit, windows, gsm_limit, gsm_new,
              stage, tag, taus=TAUS, ladder=True, only=None, skip_sweep=False):
    runner = make_runner(name, device, dtype)
    log(f"== {name} on {device} ({dtype})")
    cache = RESULTS / f"p1v2_{'q35' if name.startswith('qwen3.5') else 'q3'}_params.pt"
    params = build_params(runner, cache=cache)
    bank = build_bank(runner, params)

    ids = P.encode_tokens(runner.tok, load_jsonl("wikitext2_test.jsonl"),
                          windows * 512 + 1) if hasattr(P, "encode_tokens") else None
    if ids is None:
        text = "".join(r["text"] for r in load_jsonl("wikitext2_test.jsonl"))
        ids = torch.tensor(runner.tok(text, add_special_tokens=False)["input_ids"][:windows * 512 + 1])

    out = {"model": name, "device": device, "dtype": str(dtype), "seed": SEED,
           "taus": list(taus), "mc_limit": mc_limit, "ppl_windows": windows,
           "data_sha256": {f.name: sha256(f) for f in sorted(DATA.glob("*.jsonl"))},
           "methods": {}, "ladder": [], "dense": {}}

    # ---- dense reference -------------------------------------------------- #
    dense_cfg = {"override": None, "scale": None, "theta": None, "rate": 0.0,
                 "norm_gamma": None, "coupled": False, "static": None}
    t0 = time.perf_counter()
    out["dense"]["ppl"] = eval_ppl(runner, ids, windows, dense_cfg)
    out["dense"]["acc"], out["dense"]["items"] = {}, {}
    for d in (() if skip_sweep else DATASETS):
        flags, _ = mc_items(runner, d, mc_limit, dense_cfg)
        out["dense"]["items"][d] = flags
        out["dense"]["acc"][d] = boot_ci(flags)
    log(f"  dense ppl {out['dense']['ppl']:.4f}  "
        + " ".join(f"{d} {out['dense']['acc'][d]['mean']:.3f}"
                   for d in (() if skip_sweep else DATASETS))
        + f"  ({time.perf_counter()-t0:.0f}s)")

    # ---- five named methods ---------------------------------------------- #
    for mname, (perm, gate, alpha, wbw, coupled) in METHODS.items():
        if skip_sweep:
            break
        if only and mname not in only:
            continue
        rec = {"ppl": {}, "acc": {}, "items": {}, "paired": {}}
        for tau in taus:
            cfg = cfg_for(bank[perm], gate, tau, alpha, wbw, coupled)
            rec["ppl"][tau] = eval_ppl(runner, ids, windows, cfg)
            for d in DATASETS:
                flags, _ = mc_items(runner, d, mc_limit, cfg)
                rec["items"].setdefault(d, {})[tau] = flags
                rec["acc"].setdefault(d, {})[tau] = boot_ci(flags)
                rec["paired"].setdefault(d, {})[tau] = paired_ci(
                    out["dense"]["items"][d], flags)
            log(f"  {mname:13s} tau {tau:.2f}  ppl {rec['ppl'][tau]:7.3f}"
                f"  dPPL {rec['ppl'][tau]-out['dense']['ppl']:+6.3f}  "
                + " ".join(f"{d} {rec['acc'][d][tau]['mean']:.3f}" for d in DATASETS))
        rec["delta_ppl"] = {t: rec["ppl"][t] - out["dense"]["ppl"] for t in taus}
        out["methods"][mname] = rec

    # ---- ablation ladder (0.8B only: identical harness, cheaper) ---------- #
    if ladder and not skip_sweep and name.startswith("qwen3.5"):
        for step, (sname, perm, gate, alpha, wbw, coupled) in enumerate(LADDER):
            rec = {"step": step, "name": sname, "ppl": {}, "acc": {}, "delta": {}}
            for tau in taus:
                cfg = cfg_for(bank[perm], gate, tau, alpha, wbw, coupled)
                rec["ppl"][tau] = eval_ppl(runner, ids, windows, cfg)
                rec["delta"][tau] = rec["ppl"][tau] - out["dense"]["ppl"]
                for d in DATASETS:
                    flags, _ = mc_items(runner, d, mc_limit, cfg)
                    rec.setdefault("items", {}).setdefault(d, {})[tau] = flags
                    rec["acc"].setdefault(d, {})[tau] = boot_ci(flags)
                log(f"  {sname:18s} tau {tau:.2f} ppl {rec['ppl'][tau]:7.3f} "
                    f"dPPL {rec['delta'][tau]:+6.3f}")
            out["ladder"].append(rec)

    # ---- GSM8K at the operating point ------------------------------------ #
    if stage == "gsm" and gsm_limit:
        out["gsm"] = {"limit": gsm_limit, "max_new": gsm_new, "rows": []}
        taus_g = (0.20,)
        out["gsm"]["dense"] = gsm_items(runner, gsm_limit, gsm_new, dense_cfg)
        out["gsm"]["dense_ci"] = boot_ci(out["gsm"]["dense"])
        log(f"  gsm dense acc {np.mean(out['gsm']['dense']):.3f}")
        for mname, (perm, gate, alpha, wbw, coupled) in METHODS.items():
            cfg = cfg_for(bank[perm], gate, taus_g[0], alpha, wbw, coupled)
            fl = gsm_items(runner, gsm_limit, gsm_new, cfg)
            out["gsm"]["rows"].append({"method": mname, "tau": taus_g[0],
                                       "acc": boot_ci(fl),
                                       "paired": paired_ci(out["gsm"]["dense"], fl),
                                       "items": fl})
            log(f"  gsm {mname:13s} tau {taus_g[0]:.2f} acc {np.mean(fl):.3f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"p6_sota_{tag}.json").write_text(json.dumps(out, indent=1))
    log(f"wrote results/p6_sota_{tag}.json")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both",
                    choices=("qwen3.5-0.8b", "qwen3-0.6b", "both"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--stage", default="main", choices=("main", "gsm"))
    ap.add_argument("--mc-limit", type=int, default=200)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--gsm-limit", type=int, default=200)
    ap.add_argument("--gsm-new", type=int, default=128)
    ap.add_argument("--tag", default="")
    ap.add_argument("--taus", default=",".join(str(t) for t in TAUS))
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--methods", default="")
    ap.add_argument("--skip-sweep", action="store_true")
    a = ap.parse_args()
    dt = getattr(torch, a.dtype)
    models = (["qwen3.5-0.8b", "qwen3-0.6b"] if a.model == "both" else [a.model])
    for m in models:
        tag = a.tag or (("gsm_" if a.stage == "gsm" else "") + m.replace(".", "_"))
        run_model(m, a.device, dt, a.mc_limit, a.windows, a.gsm_limit,
                  a.gsm_new, a.stage, tag,
                  taus=tuple(float(x) for x in a.taus.split(",")),
                  ladder=not a.no_ladder,
                  only=set(x for x in a.methods.split(",") if x),
                  skip_sweep=a.skip_sweep)


if __name__ == "__main__":
    main()
