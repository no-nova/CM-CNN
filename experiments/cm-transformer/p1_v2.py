"""P1 v2 — output-error-aware 32-lane permutation and adaptive energy gating.

Why v2: v1 clustered neurons by scale-invariant cosine co-activation, which
(i) ignored the output weight norm ||W_down[:, i]||_2 and (ii) actually *groups
positively correlated neurons together*, so their truncation errors
sum_i h_i W_down[:, i] add constructively inside a dropped block.  At matched
MAC cut v1-clustered therefore retained the same energy as natural and lost on
dPPL.

v2 optimises the quantity that matters: the squared norm of the *output*
truncation error of a dropped block,

    E ||e_B||^2 = sum_{i,j in B} C[i,j],   C[i,j] = E[h_i h_j] * <Wd[:,i], Wd[:,j]>

so the permutation is chosen greedily to minimise it (errors cancel inside a
block), while neurons are assigned coldest-first so that the 32-lane blocks are
importance-homogeneous in s_i = E|h_i| * ||Wd[:,i]|| — the blocks that survive
top-k then hold the hot neurons.

Gating: instead of a rigid per-layer top-k count, a threshold theta_l per layer
is calibrated on held-out text so that the *average* skip fraction equals the
target tau-bar; at inference each token skips its own low-energy blocks.

    python3 p1_v2.py --model qwen3.5-0.8b --stage build     # build + save params
    python3 p1_v2.py --model qwen3.5-0.8b --stage eval --rates 0.1,0.2 --mc-limit 500
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (LANE, MODEL_DIR, Weights, build_tokenizer, log,
                    set_deterministic)
from model_ref import TextBackbone
from bench_text import (FILES, PREP, extract_answer, gold_answer, score_items,
                        summarize, run_gsm8k, NUM)
from run_text_eval import BENCH, load_jsonl

RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
LANES = {"qwen3.5-0.8b": 32, "qwen3-0.6b": 32}
CALIB_PROMPTS = [
    "The history of the Roman Empire spans more than a thousand years, from the",
    "In computer science, an algorithm is a finite sequence of well-defined",
    "The cells of the human body contain many organelles, each of which",
    "Economic growth in the twentieth century was driven by",
    "Photosynthesis is the process by which green plants",
    "A proof in mathematics must be logically valid, meaning that",
    "The Pacific Ocean is the largest and deepest of",
    "Neural networks consist of layers of artificial neurons that",
    "During the Industrial Revolution, cities grew rapidly because",
    "The theory of general relativity describes gravity as",
    "In a democracy, the legitimacy of government derives from",
    "Proteins are large molecules composed of chains of",
    "The stock market is influenced by interest rates because",
    "Shakespeare's plays are traditionally divided into",
    "The immune system recognises pathogens through",
    "Quantum mechanics predicts that a particle's position",
    "The Ottoman Empire controlled trade routes between",
    "Machine translation systems map a source sentence",
    "The water cycle describes the continuous movement of",
    "In statistics, a confidence interval quantifies",
]


# --------------------------------------------------------------------------- #
# permutation construction
# --------------------------------------------------------------------------- #
def output_gram(H: torch.Tensor, Wd: torch.Tensor, max_tokens: int = 512):
    """C[i,j] = E[h_i h_j] * <Wd[:,i], Wd[:,j]>  (block truncation-error Gram)."""
    if H.shape[0] > max_tokens:
        idx = torch.linspace(0, H.shape[0] - 1, max_tokens).long()
        H = H[idx]
    H = H.float()
    H = H - H.mean(0, keepdim=True)
    G = (H.t() @ H) / max(H.shape[0] - 1, 1)
    A = Wd.float().t() @ Wd.float()                     # [I,I]
    return G * A


def importance(H: torch.Tensor, Wd: torch.Tensor):
    return H.float().abs().mean(0) * Wd.float().norm(dim=0)


def build_perm(acts: torch.Tensor, Wd: torch.Tensor, lane: int = LANE,
               pool: int = 4):
    """Greedy coldest-first block construction minimising dropped-block error."""
    H = acts.float()
    if H.shape[0] > 512:
        idx = torch.linspace(0, H.shape[0] - 1, 512).long()
        H = H[idx]
    imp = importance(H, Wd)
    C = output_gram(H, Wd)                              # [I,I]
    I = imp.numel()
    nb = I // lane
    order = torch.argsort(imp)                          # coldest first
    rank = torch.empty_like(order)
    rank[order] = torch.arange(I, device=imp.device)
    taken = torch.zeros(I, dtype=torch.bool, device=imp.device)
    blocks, costs = [], []
    for b in range(nb):
        # seed = coldest neuron still free
        free = (~taken).nonzero().flatten()
        seed = free[torch.argmin(rank[free])]
        members = [int(seed)]
        taken[seed] = True
        for _ in range(lane - 1):
            free = (~taken).nonzero().flatten()
            cand = free[torch.argsort(rank[free])[: lane * pool]]
            # marginal error energy if this neuron joins the block
            delta = C[cand][:, members].sum(1)
            j = int(cand[torch.argmin(delta)])
            members.append(j)
            taken[j] = True
        blocks.append(members)
        m = torch.tensor(members, device=imp.device)
        costs.append(float(C[m][:, m].sum()))
    perm = torch.tensor([i for blk in blocks for i in blk], device=imp.device)
    return perm, {"importance": imp.cpu(), "block_error": costs,
                  "est_total_dropped_error": float(np.sum(costs)),
                  "est_retained_energy": float(
                      imp.pow(2).sum() - imp.pow(2).sum())}    # filled by caller


def block_stats(perm, C):
    """Sum of within-block output-error energy (lower = better cancellation)."""
    blocks = perm.reshape(-1, LANE)
    return float(sum(C[b][:, b].sum() for b in blocks))


# --------------------------------------------------------------------------- #
# calibration of adaptive thresholds
# --------------------------------------------------------------------------- #
def block_scale(Wd):
    """Per-neuron ||Wd[:, i]||_2 so that ||s_B||^2 = energy(a * scale)."""
    return Wd.float().norm(dim=0)


def collect_energies(tb, prompts, scale, max_len=256):
    """Per-layer block energies [T,B] on calibration text."""
    out = {}
    for i in range(tb.cfg.layers):
        out[i] = []
    with torch.no_grad():
        for p in prompts:
            eo = {}
            tb.forward(p, energy_out=eo, skip_rate=1e-9,
                       skip_scale=scale, logits_positions=[0])
            for i, e in eo.items():
                out[i].append(e.reshape(-1, e.shape[-1]).cpu())
    return {i: torch.cat(v, 0) for i, v in out.items()}


def calibrate_theta(energies, rate):
    """Per-layer threshold hitting exactly `rate` skipped blocks on average."""
    th = {}
    for i, e in energies.items():
        q = torch.quantile(e.flatten(), rate) if rate > 0 else torch.tensor(-1.0)
        th[i] = q
    return {i: float(v) for i, v in th.items()}


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ppl(tb, ids, seq_len=512, windows=16, override=None, rate=0.0,
        scale=None, theta=None, norm_gamma=None):
    nll, ntok = 0.0, 0
    for w in range(windows):
        p = ids[w * seq_len: w * seq_len + seq_len + 1]
        if p.numel() < seq_len + 1:
            break
        o = tb.forward(p, mlp_override=override, skip_rate=rate,
                       skip_scale=scale, skip_theta=theta,
                       norm_gamma=norm_gamma,
                       logits_positions=torch.arange(seq_len))
        lg = o["logits"].float()
        nll += float(F.cross_entropy(lg, p[1:], reduction="sum"))
        ntok += seq_len
    return float(np.exp(nll / max(ntok, 1)))


@torch.no_grad()
def mc(runner, dataset, limit, **kw):
    items = PREP[dataset](load_jsonl(FILES[dataset]))[:limit]
    rows = []
    tok = runner.tok
    for it in items:
        ctx = tok(it["ctx"], add_special_tokens=False)["input_ids"][-1024:]
        cand = []
        for c in it["choices"]:
            cid = tok(" " + c.strip(), add_special_tokens=False)["input_ids"]
            ids = ctx + cid
            pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(cid))
            lg = runner.logits(ids, positions=pos, **kw)
            lp = F.log_softmax(lg.float(), -1)
            ll = sum(float(lp[j, t]) for j, t in enumerate(cid))
            cand.append({"choice": c, "loglik": ll, "n_tok": len(cid),
                         "n_bytes": len((" " + c.strip()).encode())})
        rows.append({"id": it["id"], "answer": it["answer"], "cands": cand})
    return summarize(rows)


@torch.no_grad()
def gsm8k(p1, limit=50, max_new=192, **kw):
    tok = p1.tok
    from bench_text import GSM_FEWSHOT
    items = load_jsonl("gsm8k_test.jsonl")[:limit]
    shots = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in GSM_FEWSHOT)
    ok = 0
    rows = []
    for it in items:
        prompt = f"{shots}\n\nQuestion: {it['question']}\nAnswer:"
        ids = torch.tensor(tok(prompt, add_special_tokens=False)["input_ids"],
                           device=p1.device)
        # no early stop: a partial digit would truncate "15" to "1"
        gen = p1.generate(ids, max_new, stop_on_answer=False, **kw)
        text = tok.decode(gen)
        pred, gold = extract_answer(text), gold_answer(it)
        good = pred is not None and pred == gold
        ok += int(good)
        rows.append({"id": it["id"], "pred": pred, "gold": gold, "ok": good,
                     "text": text[:200]})
    return {"acc": ok / max(len(items), 1), "exact": f"{ok}/{len(items)}", "rows": rows}


@torch.no_grad()
def generate(tb, ids, n_new, stop_on_answer=True, tok=None, **kw):
    from model_ref import new_state
    if "override" in kw:                      # RefP1 keyword is mlp_override
        kw["mlp_override"] = kw.pop("override")
    return _generate(tb, ids, n_new, stop_on_answer, tok, **kw)


def _generate(tb, ids, n_new, stop_on_answer, tok, **kw):
    from model_ref import new_state
    st = new_state()
    o = tb.forward(ids, state=st, pos0=0, **kw)
    cur = int(o["logits"][-1].argmax())
    gen = [cur]
    for k in range(1, n_new):
        o = tb.forward(torch.tensor([cur], device=ids.device), state=st,
                       pos0=ids.numel() + k - 1, **kw)
        cur = int(o["logits"][-1].argmax())
        gen.append(cur)
        if stop_on_answer and len(gen) > 12 and tok is not None:
            # stop once the numeric answer after "####" is complete
            if re.search(r"####\s*-?[0-9][0-9,]*", tok.decode(gen)):
                break
    return gen


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #
class RefP1:
    """Qwen3.5-0.8B text backbone via the verified reference implementation."""

    def __init__(self, device="cuda", dtype=torch.float32, name="qwen3.5-0.8b"):
        self.name = name
        self.device = device
        self.tb = TextBackbone(Weights(dtype, device))
        self.tok = build_tokenizer(MODEL_DIR)
        self.L = self.tb.cfg.layers
        self.I = self.tb.cfg.inter
        self.dtype = dtype

    def down_w(self, i):
        return self.tb.w[f"{self.tb.prefix}layers.{i}.mlp.down_proj.weight"]

    def capture(self, prompts, max_len=256):
        acts = {i: [] for i in range(self.L)}
        for ids in prompts:
            cap = {}
            with torch.no_grad():
                self.tb.forward(ids[:max_len].to(self.device), capture=cap,
                                logits_positions=[0])
            for i in range(self.L):
                x = cap["mlp_in"][i]
                p = f"{self.tb.prefix}layers.{i}.mlp."
                a = F.silu(F.linear(x, self.tb.w[p + "gate_proj.weight"])) * \
                    F.linear(x, self.tb.w[p + "up_proj.weight"])
                acts[i].append(a.float().cpu())
        return {i: torch.cat(v, 0) for i, v in acts.items()}

    def logits(self, ids, positions=None, override=None, rate=0.0, scale=None,
               theta=None, norm_gamma=None, coupled=False, static=None):
        pos = positions if positions is not None else torch.arange(len(ids))
        self.tb.static_keep = static
        with torch.no_grad():
            o = self.tb.forward(torch.as_tensor(ids, device=self.device),
                                mlp_override=override, skip_rate=rate,
                                skip_scale=scale, skip_theta=theta,
                                norm_gamma=norm_gamma, coupled=coupled,
                                logits_positions=torch.as_tensor(pos, device="cpu"))
        return o["logits"].float()

    def generate(self, ids, n_new, override=None, rate=0.0, scale=None, theta=None,
                 stop_on_answer=True, norm_gamma=None, coupled=False, static=None):
        self.tb.static_keep = static
        return generate(self.tb, torch.as_tensor(ids, device=self.device), n_new,
                        tok=self.tok, mlp_override=override, skip_rate=rate,
                        skip_scale=scale, skip_theta=theta, norm_gamma=norm_gamma,
                        coupled=coupled, stop_on_answer=stop_on_answer)


class HFP1:
    """Qwen3-0.6B (dense attention) through the official transformers model.

    P1 is injected by swapping each decoder layer's MLP forward with a version
    that applies the permutation scale and the block mask.
    """

    def __init__(self, model_dir, device="cuda", dtype=torch.float32,
                 name="qwen3-0.6b"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = name
        self.device, self.dtype = device, dtype
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.layers = self.model.model.layers
        self.L = len(self.layers)
        self.I = self.layers[0].mlp.down_proj.in_features
        self.cfg = None            # {"override", "scale", "theta", "rate"}
        self._orig = [type(l.mlp).forward for l in self.layers]
        self._install()

    def down_w(self, i):
        return self.layers[i].mlp.down_proj.weight

    def _install(self):
        for i, l in enumerate(self.layers):
            l.mlp.forward = self._fwd(i).__get__(l.mlp)

    def _fwd(self, i):
        def fwd(self_mlp, x):
            c = self.cfg
            ov = None if c is None or c["override"] is None else c["override"].get(i)
            if ov is not None:
                wg, wu, wd = ov
            else:
                wg, wu, wd = (self_mlp.gate_proj.weight, self_mlp.up_proj.weight,
                              self_mlp.down_proj.weight)
            coupled = bool(c is not None and c.get("coupled"))
            cap0 = getattr(self_mlp, "_cm_capg", None)
            if coupled or cap0 is not None:
                g = self_mlp.act_fn(F.linear(x, wg))
                if cap0 is not None:
                    cap0.setdefault(i, []).append(
                        g.detach().float().reshape(-1, g.shape[-1]).cpu())
                a = g * F.linear(x, wu)
            else:
                a = self_mlp.act_fn(F.linear(x, wg)) * F.linear(x, wu)
            cap = getattr(self_mlp, "_cm_cap", None)
            if cap is not None:
                cap.setdefault(i, []).append(
                    a.detach().float().reshape(-1, a.shape[-1]).cpu())
            gamma = c.get("gamma") if c is not None else None
            keep = None
            if c is not None:
                sc = c["scale"]
                src = g if coupled else a
                e = block_energy(src if sc is None else src * sc[i])
                if c.get("wb") is not None:
                    e = e * c["wb"][i].to(e.device)
                n_full = a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                if c.get("static") is not None:
                    keep = c["static"][i].to(a.device).expand(a.shape[0], -1)
                elif c["theta"] is not None:
                    th = c["theta"][i] if isinstance(c["theta"], dict) else c["theta"]
                    keep = e >= th
                elif c["rate"]:
                    keep = keep_mask_from_energy(e, c["rate"])
                if keep is not None:
                    a = apply_block_mask(a, keep)
                    if gamma:
                        n_keep = a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        a = a * (n_full / n_keep).clamp(max=float(gamma))
            if (coupled and keep is not None and keep.dim() == 2
                    and keep.shape[0] == 1 and not bool(keep.all())):
                # single-token decode: emulate the fused kernel, only the kept
                # rows of up_proj are ever touched.  (For T>1 tokens each token
                # needs its own row set, so the masked path below is used; it is
                # numerically identical.)
                k = keep[0]
                blk = torch.nonzero(k, as_tuple=False).flatten()
                idx = (blk[:, None] * 32 +
                       torch.arange(32, device=blk.device)).flatten()
                g2 = self_mlp.act_fn(F.linear(x, wg))
                a = (g2.index_select(-1, idx) *
                     F.linear(x, wu.index_select(0, idx)))
                if gamma:
                    n_keep = a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    n_full = (g2 * F.linear(x, wu)).norm(dim=-1, keepdim=True)
                    a = a * (n_full / n_keep).clamp(max=float(gamma))
                return F.linear(a, wd.index_select(1, idx))
            kern = getattr(self, "_cm_kernel", None)
            if kern is not None and keep is not None and sc is not None:
                return kern(a, wd, keep)
            return F.linear(a, wd)
        return fwd

    def set_cfg(self, override=None, rate=0.0, scale=None, theta=None,
                gamma=None, coupled=False, wb=None, static=None):
        self.cfg = {"override": override, "rate": rate, "scale": scale,
                    "theta": theta, "gamma": gamma, "coupled": coupled,
                    "wb": wb, "static": static}

    def capture(self, prompts, max_len=256):
        """Post-activation a = act(gate x) * (up x), captured inside the patched
        MLP forward (the module's own forward is never called, so a module hook
        would never fire)."""
        cap = {}
        for l in self.layers:
            l.mlp._cm_cap = cap
        with torch.no_grad():
            for ids in prompts:
                self.model(input_ids=ids[:max_len].unsqueeze(0).to(self.device),
                           use_cache=False)
        for l in self.layers:
            l.mlp._cm_cap = None
        return {i: torch.cat(v, 0) for i, v in cap.items()}

    def logits(self, ids, positions=None, override=None, rate=0.0, scale=None,
               theta=None, norm_gamma=None, coupled=False, static=None):
        self.set_cfg(override, rate, scale, theta, norm_gamma,
                     coupled=coupled, wb=None, static=static)
        with torch.no_grad():
            o = self.model(input_ids=torch.as_tensor(ids, device=self.device).unsqueeze(0),
                           use_cache=False).logits[0].float()
        return o if positions is None else o[torch.as_tensor(positions, device="cpu")]

    def generate(self, ids, n_new, override=None, rate=0.0, scale=None, theta=None,
                 stop_on_answer=True, norm_gamma=None, coupled=False, static=None):
        self.set_cfg(override, rate, scale, theta, norm_gamma,
                     coupled=coupled, wb=None, static=static)
        with torch.no_grad():
            o = self.model.generate(
                input_ids=torch.as_tensor(ids, device=self.device).unsqueeze(0),
                max_new_tokens=n_new, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        return o[0, len(ids):].tolist()


def block_energy(a, lane=LANE):
    from model_ref import block_energy as be
    return be(a, lane)


def apply_block_mask(a, keep, lane=LANE):
    from model_ref import apply_block_mask as abm
    return abm(a, keep, lane)


def keep_mask_from_energy(e, rate):
    from model_ref import keep_mask_from_energy as km
    return km(e, rate)


# --------------------------------------------------------------------------- #
# build / evaluate
# --------------------------------------------------------------------------- #
def encode(tok, texts, max_len=256, device="cpu"):
    return [torch.tensor(tok(t, add_special_tokens=False)["input_ids"][:max_len],
                         device=device) for t in texts]


def build_params(p1, cache=None, seed=1234):
    """Compute the importance permutation for every layer + the natural/random refs."""
    if cache is not None and cache.exists():
        return torch.load(cache, map_location="cpu")
    texts = CALIB_PROMPTS + [r["text"] for r in load_jsonl("wikitext2_test.jsonl")[100:160]]
    prompts = encode(p1.tok, texts, 256)
    log("capturing calibration activations ...")
    acts = p1.capture(prompts)
    out = {"perm": {}, "block_error": {}, "importance": {}, "natural": {},
           "random": {}}
    g = torch.Generator().manual_seed(seed)
    for i in range(p1.L):
        Wd = p1.down_w(i).detach().float().cpu()
        H = acts[i]
        C = output_gram(H, Wd)
        perm, meta = build_perm(H, Wd)
        # natural blocks = identity; random = random partition
        nat = torch.arange(H.shape[1])
        rnd = torch.randperm(H.shape[1], generator=g)
        out["perm"][i] = perm.cpu()
        out["natural"][i] = nat
        out["random"][i] = rnd
        out["importance"][i] = meta["importance"]
        out["block_error"][i] = {"ours": block_stats(perm.cpu(), C.cpu()),
                                 "natural": block_stats(nat, C.cpu()),
                                 "random": block_stats(rnd, C.cpu())}
        log(f"layer {i:2d} block error  ours {out['block_error'][i]['ours']:.4g}  "
            f"natural {out['block_error'][i]['natural']:.4g}  "
            f"random {out['block_error'][i]['random']:.4g}")
    dump = {"perm": out["perm"], "natural": out["natural"], "random": out["random"],
            "importance": out["importance"], "block_error": out["block_error"],
            "calibration_tokens": int(H.shape[0])}
    if cache is not None:
        torch.save(dump, cache)
    return dump


def make_cfg(p1, params, variant, rate=None, theta=None):
    perms = params["perm" if variant == "ours" else variant]
    override = {}
    scale = {}
    for i in range(p1.L):
        perm = perms[i].to(p1.device)
        W = (p1.tb.w[f"{p1.tb.prefix}layers.{i}.mlp.gate_proj.weight"],
             p1.tb.w[f"{p1.tb.prefix}layers.{i}.mlp.up_proj.weight"],
             p1.down_w(i)) if isinstance(p1, RefP1) else (
            p1.layers[i].mlp.gate_proj.weight, p1.layers[i].mlp.up_proj.weight,
            p1.down_w(i))
        from model_ref import permute_mlp
        override[i] = permute_mlp(W, perm)
        n = p1.down_w(i).detach().float().norm(dim=0)      # ||Wd[:, i]||
        scale[i] = n[perm]                                 # permuted order
    return {"override": override, "scale": scale, "rate": rate or 0.0,
            "theta": theta}


@torch.no_grad()
def calibrate(p1, params, rate, texts, max_len=192, override=None, variant="natural"):
    """Per-layer thresholds giving exactly `rate` skipped blocks on average.

    Energies are measured in the SAME weight ordering as evaluation: when a
    permutation is supplied the permuted weights are used, so the quantile
    refers to the permuted 32-lane blocks and not to the native ones."""
    prompts = encode(p1.tok, texts, max_len)
    energies = {i: [] for i in range(p1.L)}
    perm = params["perm" if variant == "ours" else variant]
    for p in prompts:
        if isinstance(p1, RefP1):
            eo = {}
            sc = {i: p1.down_w(i).detach().float().norm(dim=0) for i in range(p1.L)}
            if override is not None:
                sc = {i: sc[i][perm[i].to(sc[i].device)] for i in sc}
            p1.tb.forward(p.to(p1.device), energy_out=eo, skip_rate=0.5,
                          skip_scale=sc, mlp_override=override,
                          logits_positions=[0])
            for i, e in eo.items():
                energies[i].append(e.reshape(-1, e.shape[-1]).cpu())
        else:
            acts = p1.capture([p], max_len=None)
            for i in range(p1.L):
                a = acts[i]
                sc = p1.down_w(i).detach().float().norm(dim=0).cpu()
                if variant != "natural":
                    idx = perm[i].cpu()
                    a, sc = a[:, idx], sc[idx]
                energies[i].append(block_energy(a * sc).reshape(-1, p1.I // LANE))
    th = {}
    for i, v in energies.items():
        e = torch.cat(v, 0).flatten()
        # keep == (e >= theta); rate 0 must keep everything, so theta = -inf
        th[i] = float(torch.quantile(e, rate)) if rate > 0 else float("-inf")
    return th


def eval_one(p1, params, variant, rate, theta=None, mc_limit=500,
             datasets="arc,piqa,boolq", gsm=0, windows=16, override=None,
             scale=None, norm_gamma=None):
    if override is not None:      # reuse a config built once per variant
        cfg = {"override": override, "scale": scale,
               "rate": rate if theta is None else 0.0, "theta": theta,
               "norm_gamma": norm_gamma}
    else:
        cfg = make_cfg(p1, params, variant,
                       rate=rate if theta is None else 0.0, theta=theta)
        cfg["norm_gamma"] = norm_gamma
    kw = dict(override=cfg["override"], rate=cfg["rate"], scale=cfg["scale"],
              theta=cfg["theta"], norm_gamma=cfg["norm_gamma"])
    wt = load_jsonl("wikitext2_test.jsonl")
    text = "\n\n".join(r["text"] for r in wt)
    ids = torch.tensor(p1.tok(text, add_special_tokens=False)["input_ids"],
                       device=p1.device)
    t0 = time.perf_counter()
    p = ppl(p1.tb, ids, windows=windows, **kw) if isinstance(p1, RefP1) else _hf_ppl(p1, ids, windows, **kw)
    out = {"ppl": p, "seconds": time.perf_counter() - t0}
    res = {}
    for d in (datasets or "").split(","):
        if not d:
            continue
        res[d] = mc(p1, d, mc_limit, **kw)
        log(f"  {d:6s} acc {res[d]['acc']:.3f} (n={res[d]['n']})")
    out["mc"] = res
    if gsm:
        g = gsm8k(p1, limit=gsm, **kw)
        out["gsm8k"] = {"acc": g["acc"], "exact": g["exact"]}
        log(f"  gsm8k  acc {g['acc']:.3f} ({g['exact']})")
    return out


@torch.no_grad()
def _hf_ppl(p1, ids, windows=16, seq_len=512, **kw):
    nll, ntok = 0.0, 0
    p1.set_cfg(kw.get("override"), kw.get("rate", 0.0), kw.get("scale"),
               kw.get("theta"), kw.get("norm_gamma"))
    for w in range(windows):
        seg = ids[w * seq_len: w * seq_len + seq_len + 1]
        if seg.numel() < seq_len + 1:
            break
        lg = p1.model(input_ids=seg.unsqueeze(0), use_cache=False).logits[0].float()
        nll += float(F.cross_entropy(lg[:-1], seg[1:], reduction="sum"))
        ntok += seq_len
    return float(np.exp(nll / max(ntok, 1)))


@torch.no_grad()
def retained_energy(p1, params, variant, rate=None, theta=None, prompts=None,
                    override=None):
    """Fraction of output-weighted block energy kept, averaged over tokens."""
    if not isinstance(p1, RefP1):
        # HF backbone: energies come from captured activations, permuted the same
        # way the weights are, so the numbers are directly comparable to the 0.8B.
        acts = p1.capture([torch.tensor(p1.tok(t, add_special_tokens=False)["input_ids"],
                                        device=p1.device)
                           for t in (prompts or CALIB_PROMPTS[:4])])
        perm = params["perm" if variant == "ours" else variant]
        tot, kept = 0.0, 0.0
        for i in range(p1.L):
            sc = p1.down_w(i).detach().float().norm(dim=0).cpu()
            a = acts[i]
            if variant != "natural":
                idx = perm[i].cpu()
                a, sc = a[:, idx], sc[idx]
            e = block_energy(a * sc).reshape(-1, p1.I // LANE)
            th = (torch.quantile(e.flatten(), rate) if theta is None
                  else torch.tensor(theta[i], device=e.device))
            tot += float(e.sum())
            kept += float(e[e >= th].sum())
        return kept / max(tot, 1e-9)
    texts = prompts or CALIB_PROMPTS[:8]
    pids = encode(p1.tok, texts, 192)
    perm = params["perm" if variant == "ours" else variant]
    tot, kept = 0.0, 0.0
    sc = {i: p1.down_w(i).detach().float().norm(dim=0) for i in range(p1.L)}
    if override is not None:                 # energies already in block order
        sc = {i: sc[i][perm[i].to(sc[i].device)] for i in sc}
    for p in pids:
        eo = {}
        p1.tb.forward(p.to(p1.device), energy_out=eo, skip_rate=0.5,
                      skip_scale=sc, mlp_override=override, logits_positions=[0])
        for i, e in eo.items():
            e = e.reshape(-1, e.shape[-1])
            if override is None:             # native order: map to block order
                e = block_energy(e * sc[i], LANE)
            th = (torch.quantile(e.flatten(), rate) if theta is None
                  else torch.tensor(theta[i], device=e.device))
            tot += float(e.sum())
            kept += float(e[e >= th].sum())
    return kept / max(tot, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b",
                    choices=["qwen3.5-0.8b", "qwen3-0.6b"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--stage", default="all", choices=["build", "eval", "all"])
    ap.add_argument("--rates", default="0.1,0.2,0.3")
    ap.add_argument("--mc-limit", type=int, default=500)
    ap.add_argument("--gsm", type=int, default=0)
    ap.add_argument("--datasets", default="arc,piqa,boolq")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--norm-gamma", type=float, default=None,
                    help="Defect-4 norm-preserving rescale cap (e.g. 1.5)")
    ap.add_argument("--configs", default="ours:adaptive,ours:uniform,natural:uniform,random:uniform")
    ap.add_argument("--gsm-first-only", action="store_true",
                    help="run GSM8K only for the first rate of the first config")
    args = ap.parse_args()
    set_deterministic(0)
    dt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[args.dtype]
    if args.model == "qwen3.5-0.8b":
        p1 = RefP1(args.device, dt)
    else:
        p1 = HFP1(str(Path(__file__).resolve().parents[2] / "models/Qwen3-0.6B"),
                  args.device, dt)
    if args.device == "cuda":       # the NVML init inside torch is flaky on this
        for _ in range(20):         # box; force it once, early, and wait it out
            try:
                torch.cuda.mem_get_info()
                break
            except Exception:       # noqa: BLE001
                time.sleep(1)
    cache = RES / f"p1v2_{args.model}_params.pt"
    params = build_params(p1, cache)
    if args.stage == "build":
        return
    rates = [float(r) for r in args.rates.split(",")]
    calib_texts = [r["text"] for r in load_jsonl("wikitext2_test.jsonl")[500:560]]
    tag = args.tag or f"{args.model}_{args.dtype}"
    out_path = RES / f"p1v2_{tag}.json"
    out = {"model": args.model, "device": args.device, "dtype": args.dtype,
           "lane": LANE, "inter": p1.I, "layers": p1.L,
           "block_error": {str(k): v for k, v in params["block_error"].items()},
           "configs": {}}
    if out_path.exists():                      # resume after an NVML/OOM crash
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("model") == args.model:
                out["configs"] = prev.get("configs", {})
                if "m0" in prev:
                    out["m0"] = prev["m0"]
                log(f"resuming {out_path.name}: {sum(len(v) for v in out['configs'].values())} rows")
        except Exception as e:                 # noqa: BLE001
            log(f"could not resume: {e}")
    be_sum = {k: sum(params["block_error"][i][k] for i in range(p1.L))
              for k in ("ours", "natural", "random")}
    out["block_error_total"] = be_sum
    log("total within-block output-error energy (ours/natural/random): "
        f"{be_sum['ours']:.4g} / {be_sum['natural']:.4g} / {be_sum['random']:.4g}")

    if "m0" in out:
        m0 = out["m0"]
        log(f"M0 (cached) PPL {m0['ppl']:.3f}")
    else:
        log("M0 baseline ...")
        m0 = eval_one(p1, params, "natural", 0.0, mc_limit=args.mc_limit,
                      datasets=args.datasets, gsm=args.gsm, windows=args.windows)
        out["m0"] = m0
        out_path.write_text(json.dumps(out, indent=1))
        log(f"M0 PPL {m0['ppl']:.3f}")

    wanted = [c.strip().replace(":", "_") for c in args.configs.split(",") if c.strip()]
    for variant, gate in (("ours", "adaptive"), ("ours", "uniform"),
                          ("natural", "uniform"), ("random", "uniform")):
        name = f"{variant}_{gate}"
        if name not in wanted:
            continue
        todo = [r for r in rates
                if not any(abs(r0["rate"] - r) < 1e-9
                           for r0 in out["configs"].get(name, []))]
        if not todo:
            log(f"{name}: all rates done")
            continue
        # one permuted-weight set per variant, reused for every rate
        cfg0 = make_cfg(p1, params, variant,
                        rate=todo[0] if gate == "uniform" else 0.0, theta=None)
        # thresholds are calibrated in the permuted block space of THIS variant
        thetas = {rate: calibrate(p1, params, rate, calib_texts,
                                  override=cfg0["override"], variant=variant)
                  for rate in todo}
        for rate in todo:
            th = thetas[rate]
            th_v = th if gate == "adaptive" else None
            gsm = args.gsm
            if args.gsm_first_only and (name != wanted[0] or rate != rates[0]):
                gsm = 0
            r = eval_one(p1, params, variant, rate,
                         theta=th_v, mc_limit=args.mc_limit,
                         datasets=args.datasets, gsm=gsm,
                         windows=args.windows,
                         override=cfg0["override"],
                         scale=cfg0["scale"] if gate == "uniform" or th_v else None,
                         norm_gamma=args.norm_gamma)
            r["rate"] = rate
            r["variant"] = variant
            r["gate"] = gate
            r["norm_gamma"] = args.norm_gamma
            r["retained_energy"] = retained_energy(
                p1, params, variant,
                rate=rate if gate == "uniform" else None, theta=th_v,
                override=cfg0["override"])
            out["configs"].setdefault(name, []).append(r)
            out_path.write_text(json.dumps(out, indent=1))
            if args.device == "cuda":
                torch.cuda.empty_cache()
            log(f"{name:18s} tau={rate:.2f} PPL {r['ppl']:.3f} "
                f"(d{r['ppl'] - m0['ppl']:+.3f}) Ekeep={r['retained_energy']}")
    out_path.write_text(json.dumps(out, indent=1))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
