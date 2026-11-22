"""Phase 10, Q2: bridge the FFN-only residual error to the full-model ΔPPL.

Two things are measured for each model at every block budget:

  (i)  rel-L2 of the FFN-only residual stream, i.e. a network made only of the
       embedding, the per-layer RMSNorm and the (optionally masked) MLP.  This is
       exactly the quantity a weight-streaming evaluator can see without running
       attention.
  (ii) the full model's ΔPPL and the KL divergence between its next-token
       distribution and the dense one.

If ΔPPL grows like the square of (i), a streaming evaluator that only replaces the
FFN is a valid proxy for model quality, which is the argument the manuscript makes
for the 4B checkpoint.

The "4B full-model PPL" experiment (evidence A) is not implemented: it needs a
GGUF GatedDeltaNet/attention runtime, and building one is a separate project. This
script therefore reports the bridge on the two models where a full forward exists,
in both fp32 and with the FFN weights quantised to IQ4_NL.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import p1_v2 as P1  # noqa: E402
import p6_sota as P6  # noqa: E402
import p8_reviewer_defense as P8  # noqa: E402
import p9_round2_closure as P9  # noqa: E402

RES = EXP / "results"
RATES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
WINDOWS, SEQ = 8, 512


def log(*a):
    print(*a, flush=True)


def ffn_stream(tb, ids, Wp, scale, theta, n_layers):
    """Embedding -> [RMSNorm -> masked MLP] x L. Residual stream without attention."""
    from model_ref import rms_norm
    h = torch.nn.functional.embedding(ids, tb.w[tb.prefix + "embed_tokens.weight"])
    for i in range(n_layers):
        hn = rms_norm(h, tb.w[f"{tb.prefix}layers.{i}.post_attention_layernorm.weight"],
                      tb.cfg.eps)
        kw = {} if theta is None else {"skip_scale": {i: scale[i]},
                                       "skip_theta": {i: theta[i]}, "skip_rate": 0.0}
        h = h + tb.mlp(hn, i, W=Wp[i], **kw)
    return h


def ffn_stream_hf(runner, ids, Wp, scale, theta, n_layers):
    """Same FFN-only residual stream for the transformers model."""
    import torch.nn.functional as F
    m = runner.model
    h = m.model.embed_tokens(ids)
    for i in range(n_layers):
        hn = m.model.layers[i].post_attention_layernorm(h)
        Wg, Wu, Wd = Wp[i]
        g = F.linear(hn.float(), Wg)
        u = F.linear(hn.float(), Wu)
        a = F.silu(g) * u
        if theta is not None:
            s = scale[i].to(a.device)
            e = a.reshape(*a.shape[:-1], -1, 32).pow(2).sum(-1) * (s.reshape(-1, 32)
                                                                  .pow(2).sum(-1))
            keep = e >= theta[i]
            a = a * keep.repeat_interleave(32, -1)
        h = h + F.linear(a, Wd)
    return h


def logits_of(runner, ids, cfg=None):
    cfg = cfg or {}
    n = min(ids.numel(), 64)
    if hasattr(runner, "tb"):
        kw = {"mlp_override": cfg.get("override"), "skip_scale": cfg.get("scale"),
              "skip_theta": cfg.get("theta"), "skip_rate": cfg.get("rate", 0.0),
              "norm_gamma": cfg.get("norm_gamma")}
        return runner.tb.forward(ids[:n], logits_positions=torch.arange(n), **kw)["logits"]
    return runner.logits(ids[:n], **cfg)


def ppl_of_ids(runner, ids, cfg):
    """Summed NLL over the evaluation windows with explicit runner kwargs."""
    import torch.nn.functional as F
    if hasattr(runner, "tb"):
        kw = {"mlp_override": cfg["override"], "skip_scale": cfg["scale"],
              "skip_theta": cfg["theta"], "skip_rate": cfg["rate"],
              "norm_gamma": cfg["norm_gamma"]}
        total, n = 0.0, 0
        for w in range(WINDOWS):
            p = ids[w * SEQ: w * SEQ + SEQ + 1]
            if p.numel() < SEQ + 1:
                break
            lg = runner.tb.forward(p, logits_positions=torch.arange(SEQ), **kw)["logits"]
            total += float(F.cross_entropy(lg.float(), p[1:], reduction="sum"))
            n += SEQ
        return math.exp(total / max(n, 1))
    return math.exp(sum(P9.win_nll(runner, ids, windows=WINDOWS, seq_len=SEQ, **cfg))
                   / (WINDOWS * SEQ))


def full_stats(runner, ids, cfg, dense_logits):
    """ΔPPL and KL of the sparse model against the dense one on the same stream."""
    ppl = ppl_of_ids(runner, ids, cfg)
    out = logits_of(runner, ids, cfg)
    p = torch.softmax(out.float(), -1)
    q = torch.softmax(dense_logits.float(), -1)
    kl = float(torch.sum(p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log()), -1).mean())
    top1 = float((out.argmax(-1) == dense_logits.argmax(-1)).float().mean())
    return ppl, kl, top1


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"rates": list(RATES), "windows": WINDOWS, "seq_len": SEQ,
           "evidenceA": {"status": "not implemented",
                         "reason": "requires a GGUF GatedDeltaNet+attention runtime; "
                                   "the bridge below measures how well FFN-only residual "
                                   "propagation tracks full-model ΔPPL instead"},
           "models": {}}
    for model in ("qwen3.5-0.8b", "qwen3-0.6b"):
        runner = P6.make_runner(model, dev, torch.float32)
        ids = P9.encode(runner, [P8.corpus_text("wiki") * 6], WINDOWS * SEQ + 8)[0].to(dev)
        acts = {i: v.detach().cpu() for i, v in
                P9.capture(runner, P9.calib_texts()[:1]).items()}
        n_layers = int(getattr(runner, "L", len(acts)))
        perm = P9.perms_for(runner, acts, kinds=("ours",))["ours"]
        Wp = P9.permuted_weights(runner, perm, dev)
        scale = {i: Wp[i][2].float().norm(dim=0) for i in range(n_layers)}
        from model_ref import block_energy
        # thresholds are computed on the CPU so the 8 GB card only holds the model
        E = {i: block_energy(acts[i].cpu().index_select(1, perm[i].cpu())
                             * scale[i].cpu()) for i in range(n_layers)}

        dense_ppl = math.exp(sum(P9.win_nll(runner, ids, windows=WINDOWS, seq_len=SEQ))
                             / (WINDOWS * SEQ))
        dense_logits = logits_of(runner, ids)
        if hasattr(runner, "tb"):
            hd = ffn_stream(runner.tb, ids[:SEQ], Wp, scale, None, n_layers)
        else:
            hd = ffn_stream_hf(runner, ids[:SEQ], Wp, scale, None, n_layers)
        rows = []
        for rate in RATES:
            theta = {i: float(torch.quantile(E[i].reshape(-1), rate)) for i in range(n_layers)}
            cfg = {"override": Wp, "scale": scale, "theta": theta, "rate": 0.0,
                   "norm_gamma": None}
            ppl, kl, top1 = full_stats(runner, ids, cfg, dense_logits)
            if hasattr(runner, "tb"):
                h = ffn_stream(runner.tb, ids[:SEQ], Wp, scale, theta, n_layers)
            else:
                h = ffn_stream_hf(runner, ids[:SEQ], Wp, scale, theta, n_layers)
            rel = float((h - hd).norm() / hd.norm().clamp_min(1e-9))
            rows.append({"rate": rate, "rel_l2": round(rel, 5),
                         "delta_ppl": round(ppl - dense_ppl, 4),
                         "kl": round(kl, 6), "top1": round(top1, 4)})
            log(f"[bridge] {model} tau={rate:.2f} rel {rel:.5f} dPPL {ppl - dense_ppl:+.4f} "
                f"KL {kl:.2e} top1 {top1:.3f}")
        x = np.array([r["rel_l2"] ** 2 for r in rows]); y = np.array([r["delta_ppl"] for r in rows])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        r2 = 1 - float(((y - pred) ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1e-12)
        pear = float(np.corrcoef(x, y)[0, 1])
        spear = float(np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1])
        out["models"][model] = {"dense_ppl": round(dense_ppl, 4), "rows": rows, "layers": n_layers,
                                "fit": {"form": "dPPL = a*relL2^2 + b", "a": float(coef[0]),
                                        "b": float(coef[1]), "r2": round(r2, 5),
                                        "pearson": round(pear, 4), "spearman": round(spear, 4)}}
        log(f"[bridge] {model} fit R2={r2:.4f} pearson={pear:.4f} spearman={spear:.4f}")
        (RES / "p10_bridge.json").write_text(json.dumps(out, indent=1))
    log("wrote results/p10_bridge.json")


if __name__ == "__main__":
    main()
