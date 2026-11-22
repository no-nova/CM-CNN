"""Why the coupled criterion loses: how well does ||g_b|| rank blocks vs ||a_b||?

The coupled kernel must decide which 32-lane blocks to keep *before* the up
projection is computed, so it can only score the gate: S_b = w_b ||g_b||_2.
The uncoupled P1 kernel scores the product a = silu(gate) * up, which is what
actually enters down_proj.  This script quantifies the gap between the two
criteria on the calibration distribution and on real decode activations:

  * Spearman rank correlation per layer between ||g_b|| and ||a_b||
  * top-k overlap (Jaccard) of the two block rankings at each tau
  * energy captured by the gate-selected blocks, measured in the *product* metric

Writes results/p6_criterion.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import MODEL_DIR
from model_ref import LANE, block_energy
import p6_sota as S
import p1_v2 as P

OUT = Path(__file__).resolve().parent / "results" / "p6_criterion.json"


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    torch.manual_seed(S.SEED)
    runner = S.make_runner("qwen3.5-0.8b", "cuda", torch.float32)
    params = P.build_params(runner, cache=S.RESULTS / "p1v2_q35_params.pt")
    prompts = P.encode(runner.tok, P.CALIB_PROMPTS +
                       [r["text"] for r in P.load_jsonl("wikitext2_test.jsonl")[200:260]], 256)

    decode_prompts = []

    rows = {}
    for tag, pr in (("calibration", prompts),):
        caps, capg = {i: [] for i in range(runner.L)}, {i: [] for i in range(runner.L)}
        for ids in pr:
            cap = {}
            with torch.no_grad():
                runner.tb.forward(ids[:256].to(runner.device), capture=cap, logits_positions=[0])
            for i in range(runner.L):
                x = cap["mlp_in"][i]
                pre = f"{runner.tb.prefix}layers.{i}.mlp."
                g = F.silu(F.linear(x, runner.tb.w[pre + "gate_proj.weight"]))
                a = g * F.linear(x, runner.tb.w[pre + "up_proj.weight"])
                perm = params["perm"][i].to(x.device)
                caps[i].append(block_energy(a.index_select(1, perm)).float().cpu())
                capg[i].append(block_energy(g.index_select(1, perm)).float().cpu())
        out = {"spearman": {}, "overlap": {}}
        for i in range(runner.L):
            Ea = torch.cat(caps[i], 0).reshape(-1).numpy()
            Eg = torch.cat(capg[i], 0).reshape(-1).numpy()
            out["spearman"][i] = spearman(Eg, Ea)
        for tau in S.TAUS:
            o = {}
            for i in range(runner.L):
                Ea = torch.cat(caps[i], 0).reshape(-1).numpy()
                Eg = torch.cat(capg[i], 0).reshape(-1).numpy()
                k = max(1, int(round(len(Ea) * (1 - tau))))
                ig = set(np.argsort(-Eg)[:k].tolist())
                ia = set(np.argsort(-Ea)[:k].tolist())
                o[i] = len(ig & ia) / k
            out["overlap"][tau] = o
        out["spearman_mean"] = float(np.mean(list(out["spearman"].values())))
        out["overlap_mean"] = {t: float(np.mean(list(v.values())))
                               for t, v in out["overlap"].items()}
        rows[tag] = out
        print(f"{tag:12s} spearman {out['spearman_mean']:.4f}  "
              f"overlap " + " ".join(f"{t}:{v:.3f}" for t, v in out["overlap_mean"].items()))

    OUT.write_text(json.dumps({"seed": S.SEED, "tau": list(S.TAUS), "runs": rows},
                              indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
