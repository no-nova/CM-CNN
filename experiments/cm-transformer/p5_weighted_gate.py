"""Phase 5, Task 1b: offline importance-weighted gate.

    S_b = w_b * ||g_b||_2 ,   w_b = RMS(u_b) * ||W_down[:, b]||_F

`w_b` is one float per 32-lane block, computed once during calibration, so the
runtime gate costs nothing extra.  Compares WikiText-2 perplexity of the plain
gate with the weighted gate at the same skip rate.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import LANE, dump_json  # noqa: E402
from run_text_eval import load_jsonl  # noqa: E402
import p1_v2 as P  # noqa: E402
from p1_v2 import RefP1, HFP1, build_params, eval_one, encode, log  # noqa: E402

MODELS = {"qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3-0.6b": "Qwen3-0.6B"}
ROOT = HERE.parents[1]


@torch.no_grad()
def block_weight(p1, prompts, max_len=192):
    """w_b = RMS(u_b) * ||W_down[:, b]||_F, one float per block, mean-normalised.

    `u_b` is the raw up_proj output of the block (before the silu(gate) product),
    captured on the calibration set; the norm term is the block's own slice of
    the down projection, so w_b is the block's expected output contribution."""
    sums = {i: None for i in range(p1.L)}
    cnt = {i: 0 for i in range(p1.L)}
    for ids in prompts:
        cap = {}
        p1.tb.forward(ids[:max_len].to(p1.device), capture=cap, logits_positions=[0])
        for i in range(p1.L):
            Wu = (p1.tb.w[f"{p1.tb.prefix}layers.{i}.mlp.up_proj.weight"]
                  if isinstance(p1, RefP1) else p1.layers[i].mlp.up_proj.weight)
            u = torch.nn.functional.linear(cap["mlp_in"][i].float(), Wu.detach().float())
            nb = u.shape[1] // LANE
            e = u.reshape(-1, nb, LANE).pow(2).sum(-1).sum(0)     # sum_t ||u_b(t)||^2
            sums[i] = e if sums[i] is None else sums[i] + e
            cnt[i] += u.shape[0]
    out = {}
    for i in range(p1.L):
        Wd = p1.down_w(i).detach().float()
        nb = sums[i].shape[0]
        rms = (sums[i] / max(cnt[i], 1)).sqrt()          # RMS(u_b)
        wdb = Wd.reshape(Wd.shape[0], nb, LANE)
        nrm = wdb.pow(2).sum((0, 2)).sqrt()
        w = rms * nrm
        out[i] = (w / w.mean()).clamp(1e-2, 8.0)
    return out


def scales(p1, wb, weighted):
    """Per-neuron gate scale; energy ||a * s||^2 then equals w_b * ||g_b||^2."""
    s = {}
    for i in range(p1.L):
        n = p1.down_w(i).detach().float().norm(dim=0)
        if weighted:
            n = n * wb[i].repeat_interleave(LANE).to(n.device).sqrt()
        s[i] = n
    return s


@torch.no_grad()
def calibrate_theta(p1, sc, rate, texts, max_len=192):
    prompts = encode(p1.tok, texts, max_len)
    en = {i: [] for i in range(p1.L)}
    for p in prompts:
        eo = {}
        p1.tb.forward(p.to(p1.device), energy_out=eo, skip_rate=0.99, skip_scale=sc,
                      logits_positions=[0])
        for i, e in eo.items():
            en[i].append(e.reshape(-1).cpu())
    th = {}
    for i, v in en.items():
        e = torch.cat(v, 0).flatten()
        th[i] = float(torch.quantile(e, rate))
    return th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rates", default="0.2,0.4,0.6")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rates = [float(x) for x in a.rates.split(",")]
    dev = a.device
    if a.model.startswith("qwen3.5"):
        p1 = RefP1(device=dev, dtype=torch.float32, name=a.model)
    else:
        p1 = HFP1(str(ROOT / "models" / MODELS[a.model]), device=dev, dtype=torch.float32)
    params = build_params(p1, cache=HERE / "results" / f"p1v2_{a.model}_params.pt")
    calib_texts = P.CALIB_PROMPTS + [r["text"] for r in load_jsonl("wikitext2_test.jsonl")[100:160]]
    wb = block_weight(p1, encode(p1.tok, calib_texts, 192))
    log("w_b  min %.3f  max %.3f" % (min(float(v.min()) for v in wb.values()),
                                     max(float(v.max()) for v in wb.values())))
    rows = []
    for tag, weighted in (("plain", False), ("weighted", True)):
        sc = scales(p1, wb, weighted)
        th0 = {i: float("-inf") for i in range(p1.L)}
        base = eval_one(p1, params, "natural", 0.0, theta=th0, mc_limit=0,
                        datasets="", windows=a.windows, scale=sc)
        rows.append({"gate": tag, "rate": 0.0, "ppl": base["ppl"], "delta_ppl": 0.0})
        log("%-9s tau 0.00  ppl %.4f" % (tag, base["ppl"]))
        for r in rates:
            th = calibrate_theta(p1, sc, r, calib_texts)
            res = eval_one(p1, params, "natural", r, theta=th, mc_limit=0,
                           datasets="", windows=a.windows, scale=sc)
            rows.append({"gate": tag, "rate": r, "ppl": res["ppl"],
                         "delta_ppl": res["ppl"] - base["ppl"]})
            log("%-9s tau %.2f  ppl %.4f  d %+.4f"
                % (tag, r, res["ppl"], res["ppl"] - base["ppl"]))
    dump_json({"model": a.model, "windows": a.windows, "rows": rows},
              HERE / "results" / ("p5_weighted_gate%s.json" % (("_" + a.tag) if a.tag else "")))
    log("done")


if __name__ == "__main__":
    main()
