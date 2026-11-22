"""Phase-8 reviewer defence experiments.

Tasks
  perm       5 channel-ordering strategies, rho / lane-width sweeps
  factorial  2x2 (native|permuted) x (uniform Top-K|calibrated threshold)
  calib      cross-domain calibration (wiki / code / chinese / math), block overlap,
             calibration-set size sweep
  quant      IQ4_NL simulated on every FFN, full model forward (attention included)

Every task writes results/p8_<task>.json and prints a compact table.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import LANE, MODEL_DIR, log, set_deterministic
from run_text_eval import load_jsonl
import p6_sota as P6
from p1_v2 import build_perm, importance, output_gram
from model_ref import block_energy

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DATA = HERE.parents[1] / "data" / "benchmarks"
SEED = 1234


# --------------------------------------------------------------------------- #
def mlp_weights(runner, i):
    if isinstance(runner, P6.RefP1):
        p = runner.tb.prefix
        w = runner.tb.w
        return (w[f"{p}layers.{i}.mlp.gate_proj.weight"],
                w[f"{p}layers.{i}.mlp.up_proj.weight"],
                w[f"{p}layers.{i}.mlp.down_proj.weight"])
    m = runner.layers[i].mlp
    return m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight


@torch.no_grad()
def capture(runner, texts, max_len=256):
    """Reference path takes token ids, the HF path takes raw strings."""
    if isinstance(runner, P6.RefP1):
        return runner.capture(P6.P.encode(runner.tok, texts, max_len))
    return runner.capture(texts, max_len=max_len)


def gini(x):
    v = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = v.size
    if n == 0 or v.sum() <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


# --------------------------------------------------------------------------- #
def strategy_perm(acts, Wd, kind, lane=LANE, rho=4, pool=None, seed=SEED):
    """Return a permutation of the intermediate dimension for one layer."""
    acts = acts.detach().float().cpu()
    Wd = Wd.detach().float().cpu()
    I = acts.shape[1]
    if kind == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.randperm(I, generator=g)
    if kind == "native":
        return torch.arange(I)
    if kind == "actmag":
        key = acts.float().abs().mean(0)
        return torch.argsort(key, descending=True)
    if kind == "impsort":
        return torch.argsort(importance(acts, Wd), descending=True)
    if kind == "wnorm":
        return torch.argsort(Wd.float().norm(dim=0), descending=True)
    if kind == "ours":
        return build_perm(acts, Wd, lane=lane, pool=pool or rho or I)[0]
    raise ValueError(kind)


@torch.no_grad()
def layer_stats(runner, i, acts_i, perm, rate, lane=LANE, gate="abs"):
    """Per-layer output reconstruction error and block-energy Gini for one perm."""
    Wg, Wu, Wd = mlp_weights(runner, i)
    dev = Wd.device
    perm = perm.to(dev)
    a = acts_i.to(dev).index_select(1, perm)
    s = Wd.float().index_select(1, perm).norm(dim=0)
    e = block_energy(a.float() * s).float()                     # [N, n_blocks]
    if gate == "abs":
        theta = torch.quantile(e.reshape(-1), rate)
        keep_b = (e >= theta)
    else:                                                        # uniform top-k
        nb = e.shape[-1]
        k = max(1, round(nb * (1 - rate)))
        idx = torch.topk(e, k, dim=-1).indices
        keep_b = torch.zeros_like(e, dtype=torch.bool).scatter_(-1, idx, True)
    keep = keep_b.repeat_interleave(lane, dim=-1)
    Wd_p = Wd.float().index_select(1, perm)
    y = a.float() @ Wd_p.t()
    y_hat = (a.float() * keep) @ Wd_p.t()
    rel = float((y - y_hat).norm() / y.norm().clamp_min(1e-9))
    return {"rel_l2": rel, "gini": gini(e.reshape(-1).cpu().numpy()),
            "energy_kept": float((e * keep_b).sum() / e.sum().clamp_min(1e-9)),
            "skip": float(1.0 - keep_b.float().mean())}


def build_cfg(runner, acts, perm, rate, gate="abs", lane=LANE, gamma=None, coupled=False):
    """Thresholds + permuted weights for a whole model, from one permutation set."""
    override, scale, theta, static = {}, {}, {}, None
    for i in range(runner.L):
        Wg, Wu, Wd = mlp_weights(runner, i)
        p = perm[i].to(Wd.device)
        override[i] = (Wg.index_select(0, p).detach(), Wu.index_select(0, p).detach(),
                       Wd.index_select(1, p).detach())
        s = override[i][2].float().norm(dim=0)
        scale[i] = s
        a = acts[i].to(Wd.device).index_select(1, p).float() * s
        e = block_energy(a)
        if gate == "abs":
            theta[i] = float(torch.quantile(e.reshape(-1), rate))
        elif gate == "static":
            static = static or {}
            static[i] = torch.ones(e.shape[-1], dtype=torch.bool, device=e.device)
    return {"override": override, "scale": scale,
            "theta": theta if gate == "abs" else None,
            "rate": rate if gate == "topk" else 0.0,
            "norm_gamma": gamma, "coupled": coupled, "static": static}


@torch.no_grad()
def ppl(runner, ids, windows=8, seq_len=512, **cfg):
    nll, n = 0.0, 0
    for w in range(windows):
        p = ids[w * seq_len: w * seq_len + seq_len + 1]
        if p.numel() < seq_len + 1:
            break
        lg = runner.logits(p, positions=torch.arange(seq_len), **cfg).float()
        nll += float(F.cross_entropy(lg, p[1:].to(lg.device), reduction="sum"))
        n += seq_len
    return math.exp(nll / max(n, 1))


def corpus_text(kind):
    """Raw calibration / evaluation text of one domain."""
    if kind == "wiki":
        rows = load_jsonl(DATA / "wikitext2_test.jsonl")
        text = " ".join(r["text"] for r in rows[300:420])
    elif kind == "code":
        parts = []
        for pat in ("*.py", "*.cpp"):
            for f in sorted(HERE.glob(pat))[:6]:
                parts.append(f.read_text(errors="ignore"))
        text = "\n".join(parts)
    elif kind == "zh":
        text = ("量子力学是研究微观粒子运动规律的物理学分支。它揭示了微观世界中粒子的波粒二象性，"
                "并给出了描述粒子状态演化的薛定谔方程。在深度学习领域，注意力机制通过查询、键和值"
                "三个矩阵计算得到输出，其计算复杂度随序列长度的平方增长。为了降低推理成本，研究者"
                "提出了稀疏化、量化和结构化剪枝等多种方法。") * 12
    elif kind == "math":
        rows = load_jsonl(DATA / "gsm8k_test200.jsonl")
        text = "\n".join(f"Question: {r['question']}\nAnswer: {r['answer']}" for r in rows[:40])
    else:
        raise ValueError(kind)
    return text


def corpus(runner, kind, max_len=2560):
    """Token ids for one domain (first 2,560 tokens)."""
    return P6.P.encode(runner.tok, [corpus_text(kind)], max_len)[0]


# --------------------------------------------------------------------------- #
def task_perm(device, model, rates=(0.2, 0.3)):
    runner = P6.make_runner(model, device, torch.float32)
    log(f"[perm] {model} L={runner.L}")
    texts = P6.CALIB_PROMPTS + [r["text"] for r in load_jsonl(DATA / "wikitext2_test.jsonl")[200:260]]
    acts = capture(runner, texts)
    ids = corpus(runner, "wiki")
    base = ppl(runner, ids)
    log(f"[perm] dense PPL {base:.4f}")
    out = {"model": model, "dense_ppl": base, "strategies": {}, "rho_sweep": {},
           "lane_sweep": {}}

    for kind in ("random", "native", "actmag", "wnorm", "impsort", "ours"):
        perm = [strategy_perm(acts[i], mlp_weights(runner, i)[2], kind) for i in range(runner.L)]
        row = {"rates": {}}
        for rate in rates:
            cfg = build_cfg(runner, acts, perm, rate, "abs")
            p = ppl(runner, ids, **cfg)
            stats = [layer_stats(runner, i, acts[i], perm[i], rate) for i in range(runner.L)]
            row["rates"][f"{rate:.2f}"] = {
                "ppl": p, "delta_ppl": p - base,
                "rel_l2_mean": float(np.mean([s["rel_l2"] for s in stats])),
                "rel_l2_worst": float(np.max([s["rel_l2"] for s in stats])),
                "rel_l2_best": float(np.min([s["rel_l2"] for s in stats])),
                "gini_mean": float(np.mean([s["gini"] for s in stats])),
                "energy_kept": float(np.mean([s["energy_kept"] for s in stats])),
                "skip": float(np.mean([s["skip"] for s in stats])),
            }
            log(f"  {kind:8s} tau={rate:.2f} dPPL={p - base:+.3f} "
                f"relL2 mean {row['rates'][f'{rate:.2f}']['rel_l2_mean']:.3f} "
                f"gini {row['rates'][f'{rate:.2f}']['gini_mean']:.3f}")
        out["strategies"][kind] = row
        torch.cuda.empty_cache()

    for rho in (1, 2, 4, 8, "full"):
        pool = 1000 if rho == "full" else int(rho)
        perm = [strategy_perm(acts[i], mlp_weights(runner, i)[2], "ours", pool=pool) for i in range(runner.L)]
        cfg = build_cfg(runner, acts, perm, 0.2, "abs")
        p = ppl(runner, ids, **cfg)
        out["rho_sweep"][str(rho)] = {"ppl": p, "delta_ppl": p - base}
        log(f"  rho={rho} tau=0.20 dPPL={p - base:+.3f}")
        torch.cuda.empty_cache()

    for lane in (16, 64):
        perm = [strategy_perm(acts[i], mlp_weights(runner, i)[2], "ours", lane=lane) for i in range(runner.L)]
        cfg = build_cfg(runner, acts, perm, 0.2, "abs", lane=lane)
        p = ppl(runner, ids, **cfg)
        out["lane_sweep"][str(lane)] = {"ppl": p, "delta_ppl": p - base}
        log(f"  lane={lane} tau=0.20 dPPL={p - base:+.3f}")
        torch.cuda.empty_cache()

    return out


# --------------------------------------------------------------------------- #
def task_factorial(device, models, rate=0.2):
    res = {"rate": rate, "models": {}}
    for model in models:
        runner = P6.make_runner(model, device, torch.float32)
        log(f"[factorial] {model}")
        texts = P6.CALIB_PROMPTS + [r["text"] for r in load_jsonl(DATA / "wikitext2_test.jsonl")[200:260]]
        acts = capture(runner, texts)
        ids = corpus(runner, "wiki")
        base = ppl(runner, ids)
        perms = {"native": [torch.arange(acts[0].shape[1]) for _ in range(runner.L)],
                 "perm": [strategy_perm(acts[i], mlp_weights(runner, i)[2], "ours")
                          for i in range(runner.L)]}
        cells = {}
        for pname, perm in perms.items():
            for gate in ("topk", "abs"):
                cfg = build_cfg(runner, acts, perm, rate, gate)
                p = ppl(runner, ids, **cfg)
                stats = [layer_stats(runner, i, acts[i], perm[i], rate, gate=gate)
                         for i in range(runner.L)]
                cells[f"{pname}_{gate}"] = {
                    "ppl": p, "delta_ppl": p - base,
                    "rel_l2_mean": float(np.mean([s["rel_l2"] for s in stats])),
                    "rel_l2_worst": float(np.max([s["rel_l2"] for s in stats])),
                    "gini_mean": float(np.mean([s["gini"] for s in stats])),
                    "skip": float(np.mean([s["skip"] for s in stats])),
                }
                log(f"  {pname:6s} {gate:4s} dPPL={p - base:+.3f} "
                    f"skip={cells[f'{pname}_{gate}']['skip']:.3f}")
                torch.cuda.empty_cache()
        res["models"][model] = {"dense_ppl": base, "cells": cells}
    return res


# --------------------------------------------------------------------------- #
def task_calib(device, models, rate=0.2, n_cal=(128, 512, 2048, 8192)):
    res = {"rate": rate, "models": {}}
    domains = ("wiki", "code", "zh", "math")
    for model in models:
        runner = P6.make_runner(model, device, torch.float32)
        log(f"[calib] {model}")
        base = {}
        for d in domains:
            ids = corpus(runner, d)
            base[d] = ppl(runner, ids)
            log(f"  dense {d}: {base[d]:.3f}")
        perms, act_cache = {}, {}
        for cal in domains:
            texts = P6.P.encode(runner.tok,
                                P6.CALIB_PROMPTS + [corpus_text(cal)[:6000]], 256)
            acts = runner.capture([corpus(runner, cal, 256)])
            perms[cal] = [strategy_perm(acts[i], mlp_weights(runner, i)[2], "ours")
                          for i in range(runner.L)]
            act_cache[cal] = {i: acts[i] for i in range(runner.L)}
            torch.cuda.empty_cache()
        matrix = {}
        for cal in domains:
            row = {}
            for d in domains:
                ids = corpus(runner, d)
                cfg = build_cfg(runner, act_cache[cal], perms[cal], rate, "abs")
                p = ppl(runner, ids, **cfg)
                del cfg
                torch.cuda.empty_cache()
                row[d] = {"ppl": p, "delta_ppl": p - base[d]}
            matrix[cal] = row
            log("  cal=" + cal + " " + " ".join(f"{d}:{row[d]['delta_ppl']:+.3f}" for d in domains))

        # block overlap between the four permutations
        overlap = {}
        for a in domains:
            for b in domains:
                if a >= b:
                    continue
                same, tot = 0, 0
                for i in range(runner.L):
                    ba = (perms[a][i] // LANE).reshape(-1, LANE)
                    seta = {frozenset(x.tolist()) for x in ba}
                    bb = (perms[b][i] // LANE).reshape(-1, LANE)
                    setb = {frozenset(x.tolist()) for x in bb}
                    same += len(seta & setb)
                    tot += len(seta)
                overlap[f"{a}|{b}"] = same / max(tot, 1)

        # calibration-size sweep on wiki
        size_sweep = {}
        wid = corpus(runner, "wiki")
        for n in n_cal:
            texts = P6.P.encode(runner.tok, [runner.tok.decode(wid[:n].tolist())], 300)
            acts = runner.capture([wid[:n]])
            perm = [strategy_perm(acts[i], mlp_weights(runner, i)[2], "ours") for i in range(runner.L)]
            cfg = build_cfg(runner, acts, perm, rate, "abs")
            del acts
            p = ppl(runner, wid, **cfg)
            del cfg
            size_sweep[str(n)] = {"ppl": p, "delta_ppl": p - base["wiki"]}
            log(f"  n_cal={n}: dPPL={p - base['wiki']:+.3f}")
            torch.cuda.empty_cache()
        res["models"][model] = {"dense_ppl": base, "matrix": matrix,
                                "block_overlap": overlap, "size_sweep": size_sweep}
    return res


# --------------------------------------------------------------------------- #
KVALS = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10,
                      1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.float32)


@torch.no_grad()
def iq4nl_roundtrip(w):
    """Simulate the IQ4_NL block codec: 32 weights -> one fp16 scale + 16 indices."""
    O, I = w.shape
    assert I % 32 == 0
    x = w.float().reshape(O, I // 32, 32)
    amax = x.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    d = amax / 127.0
    q = (x / d)                                    # in kvalue units
    idx = (q.unsqueeze(-1) - KVALS.to(w.device)).abs().argmin(-1)
    deq = KVALS.to(w.device)[idx] * d
    return deq.reshape(O, I).to(w.dtype), d.squeeze(-1), idx


def set_mlp_weight(runner, i, Wg, Wu, Wd):
    """Write FFN weights back in place, so no duplicate copy is held on the GPU."""
    if isinstance(runner, P6.RefP1):
        pfx = runner.tb.prefix
        w = runner.tb.w
        w._cache[f"{pfx}layers.{i}.mlp.gate_proj.weight"] = Wg
        w._cache[f"{pfx}layers.{i}.mlp.up_proj.weight"] = Wu
        w._cache[f"{pfx}layers.{i}.mlp.down_proj.weight"] = Wd
    else:
        m = runner.layers[i].mlp
        with torch.no_grad():
            m.gate_proj.weight.copy_(Wg)
            m.up_proj.weight.copy_(Wu)
            m.down_proj.weight.copy_(Wd)


def task_quant(device, models, rates=(0.0, 0.1, 0.2, 0.3, 0.4)):
    res = {"models": {}}
    for model in models:
        runner = P6.make_runner(model, device, torch.float32)
        log(f"[quant] {model}")
        ids = corpus(runner, "wiki")
        base = ppl(runner, ids)
        p0 = ids[:257]
        ref = runner.logits(p0, positions=torch.arange(256)).float()

        # IQ4_NL round-trip on every FFN weight, written back in place
        for i in range(runner.L):
            set_mlp_weight(runner, i,
                           *[iq4nl_roundtrip(w)[0].detach() for w in mlp_weights(runner, i)])
        torch.cuda.empty_cache()
        q_ppl = ppl(runner, ids)
        qlg = runner.logits(p0, positions=torch.arange(256)).float()
        agree = float((ref.argmax(-1) == qlg.argmax(-1)).float().mean())
        rows = {"0.00": {"ppl": q_ppl, "delta_ppl": q_ppl - base,
                         "delta_ppl_vs_quant_dense": 0.0, "top1_vs_fp32": agree}}
        log(f"  IQ4_NL dense PPL {q_ppl:.3f} (fp32 {base:.3f}), top1 {agree:.3f}")

        texts = P6.CALIB_PROMPTS + [r["text"] for r in load_jsonl(DATA / "wikitext2_test.jsonl")[200:260]]
        acts = capture(runner, texts)
        perm = [build_perm(acts[i].detach().float().cpu(),
                           mlp_weights(runner, i)[2].detach().float().cpu(), LANE, 4)[0]
                for i in range(runner.L)]
        for rate in rates:
            if rate == 0.0:
                continue
            cfg = build_cfg(runner, acts, perm, rate, "abs")
            p = ppl(runner, ids, **cfg)
            lg = runner.logits(p0, positions=torch.arange(256), **cfg).float()
            a2 = float((ref.argmax(-1) == lg.argmax(-1)).float().mean())
            rows[f"{rate:.2f}"] = {"ppl": p, "delta_ppl": p - base,
                                   "delta_ppl_vs_quant_dense": p - q_ppl,
                                   "top1_vs_fp32": a2}
            log(f"  tau={rate:.2f} PPL {p:.3f} (+{p - q_ppl:.3f} vs quant-dense) top1 {a2:.3f}")
            del cfg
            torch.cuda.empty_cache()
        res["models"][model] = {"fp32_ppl": base, "rows": rows}
        del runner
        torch.cuda.empty_cache()
    return res


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=("perm", "factorial", "calib", "quant"))
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--models", default="qwen3.5-0.8b,qwen3-0.6b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    set_deterministic(SEED)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    t0 = time.time()
    if a.task == "perm":
        out = task_perm(a.device, a.model)
    elif a.task == "factorial":
        out = task_factorial(a.device, models)
    elif a.task == "calib":
        out = task_calib(a.device, models)
    else:
        out = task_quant(a.device, models)
    out.update({"task": a.task, "seed": SEED, "device": a.device,
                "elapsed_s": round(time.time() - t0, 1)})
    name = RES / f"p8_{a.task}{a.tag}.json"
    name.write_text(json.dumps(out, indent=1))
    log(f"wrote {name} in {out['elapsed_s']}s")


if __name__ == "__main__":
    main()
