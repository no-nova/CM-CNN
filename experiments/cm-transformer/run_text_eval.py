"""Text-side capability + speed evaluation for P1 (32-lane FFN block skipping).

Uniform runner interface so the same harness drives:
  * RefRunner  — Qwen3.5-0.8B through the verified reference implementation
                 (experiments/cm-transformer/model_ref.py)
  * HFRunner   — Qwen3-0.6B (and SmolVLM2's LM) through official transformers

Metrics
  * WikiText-2 perplexity (teacher forcing, non-overlapping windows)
  * ARC-Easy / BoolQ / PIQA(-or-HellaSwag) multiple-choice accuracy
  * GSM8K greedy 32-token generation: ROUGE-L + mean first-divergence step vs M0

Gating is per-token top-k block energy over the 32-lane intermediate dimension,
applied inside down_proj (see model_ref.block_energy / keep_mask_from_energy).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (ROOT, MODEL_DIR, Weights, build_tokenizer, bench, dump_json,
                    log, set_deterministic, silu, BAND)
from model_ref import (TextBackbone, block_energy, keep_mask_from_energy,
                       apply_block_mask, permute_mlp, identity_perm, new_state)

BENCH = ROOT / "data" / "benchmarks"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RATES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
VARIANTS = ("clustered", "natural", "random")


# --------------------------------------------------------------------------- #
# benchmark loading
# --------------------------------------------------------------------------- #
def load_jsonl(name: str):
    p = BENCH / name
    if not p.exists():
        log(f"[skip] missing {p}")
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# runners
# --------------------------------------------------------------------------- #
class RefRunner:
    """Qwen3.5-0.8B reference implementation."""

    name = "Qwen3.5-0.8B"

    def __init__(self, device="cuda", dtype=torch.float32):
        self.device, self.dtype = device, dtype
        self.tb = TextBackbone(Weights(dtype, device))
        self.tok = build_tokenizer(MODEL_DIR)
        self.vocab = self.tb.cfg.vocab
        self.override = None
        self.rate = 0.0
        self.mode = "natural"

    def set_p1(self, override, rate, mode):
        self.override, self.rate, self.mode = override, rate, mode

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        out = self.tb.forward(ids, mlp_override=self.override, skip_rate=self.rate)
        return out["logits"].float()

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n_new: int) -> list[int]:
        st = new_state()
        out = self.tb.forward(ids, mlp_override=self.override, skip_rate=self.rate,
                              state=st, pos0=0)
        nxt = int(out["logits"][-1].argmax())
        gen = [nxt]
        for t in range(1, n_new):
            o = self.tb.forward(torch.tensor([nxt], device=ids.device),
                                mlp_override=self.override, skip_rate=self.rate,
                                state=st, pos0=ids.numel() + t - 1)
            nxt = int(o["logits"][-1].argmax())
            gen.append(nxt)
        return gen


class HFRunner:
    """Official transformers model (Qwen3 dense). P1 is injected by replacing
    each decoder layer's MLP forward with a permuted + energy-gated version."""

    def __init__(self, model_dir, device="cuda", dtype=torch.float32, name=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.name = name or Path(model_dir).name
        self.device, self.dtype = device, dtype
        self.vocab = self.model.config.vocab_size
        self.layers = self.model.model.layers
        self.inter = self.layers[0].mlp.down_proj.in_features
        self._orig_forward = [type(l.mlp).forward for l in self.layers]
        self.override, self.rate = None, 0.0

    # --- P1 injection ---------------------------------------------------- #
    @staticmethod
    def _patched_forward(layer_idx, inter, get_cfg):
        def fwd(self_mlp, x):
            cfg = get_cfg()
            ov = None if cfg["override"] is None else cfg["override"].get(layer_idx)
            if ov is not None:
                wg, wu, wd = ov
            else:
                wg, wu, wd = (self_mlp.gate_proj.weight, self_mlp.up_proj.weight,
                              self_mlp.down_proj.weight)
            a = self_mlp.act_fn(F.linear(x, wg)) * F.linear(x, wu)
            rate = cfg["rate"]
            if rate:
                a = apply_block_mask(a, keep_mask_from_energy(block_energy(a, BAND), rate))
            return F.linear(a, wd)
        return fwd

    def set_p1(self, override, rate, mode):
        if override is None:
            for l, of in zip(self.layers, self._orig_forward):
                l.mlp.forward = of.__get__(l.mlp)
            self.override, self.rate = None, 0.0
            return
        cfg = {"override": override, "rate": rate}
        for i, l in enumerate(self.layers):
            l.mlp.forward = self._patched_forward(i, self.inter, lambda c=cfg: c).__get__(l.mlp)
        self.override, self.rate = override, rate

    @torch.no_grad()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids.unsqueeze(0), use_cache=False).logits[0].float()

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, n_new: int) -> list[int]:
        out = self.model.generate(ids.unsqueeze(0), max_new_tokens=n_new, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id or 0)
        return out[0, ids.numel():].tolist()


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _nll_sum(lg: torch.Tensor, ids: torch.Tensor, chunk: int = 64) -> float:
    """Position-chunked cross entropy (avoids a float64 [T,V] temporary)."""
    tot = 0.0
    for s in range(0, lg.shape[0], chunk):
        lo = lg[s:s + chunk].float()
        tgt = ids[s + 1:s + 1 + lo.shape[0]]
        tot += float(F.cross_entropy(lo, tgt, reduction="sum"))
    return tot


@torch.no_grad()
def perplexity(runner, token_ids, seq_len, stride=None, max_windows=None):
    """Teacher-forced perplexity over non-overlapping windows."""
    stride = stride or seq_len
    nll_sum, n_tok = 0.0, 0
    wins = 0
    for start in range(0, len(token_ids) - seq_len, stride):
        chunk = token_ids[start:start + seq_len]
        ids = torch.tensor(chunk, device=runner.device)
        lg = runner.logits(ids)
        nll_sum += _nll_sum(lg[:-1], ids)
        del lg
        n_tok += ids.numel() - 1
        wins += 1
        if max_windows and wins >= max_windows:
            break
    return math.exp(nll_sum / max(n_tok, 1)), n_tok


@torch.no_grad()
def choice_logprob(runner, prompt_ids, choice_text) -> float:
    """Mean log-prob of `choice_text` continuation."""
    c_ids = runner.tok(choice_text, add_special_tokens=False)["input_ids"]
    if not c_ids:
        return -1e9
    ids = torch.tensor(prompt_ids + c_ids, device=runner.device)
    lg = runner.logits(ids)
    lp = F.log_softmax(lg[len(prompt_ids) - 1:-1].double(), dim=-1)
    tgt = torch.tensor(c_ids, device=runner.device)
    return float(lp.gather(-1, tgt[:, None]).mean())


@torch.no_grad()
def mc_accuracy(runner, items, limit=None):
    """Single-forward multiple choice: compare log P(first token of each choice)
    at the final prompt position (lm-eval style, cheap and standard)."""
    correct, n = 0, 0
    for it in items[:limit or len(items)]:
        ids = torch.tensor(it["_prompt_ids"], device=runner.device)
        lg = runner.logits(ids)[-1]
        lp = F.log_softmax(lg.double(), dim=-1)
        scores = []
        for c in it["choices"]:
            cid = it["_choice_ids"][c][0]
            scores.append(float(lp[cid]))
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct += int(pred == it["answer"])
        n += 1
    return correct / max(n, 1), n


def rouge_l(hyp: list[str], ref: list[str]) -> float:
    """LCS-based ROUGE-L F1 on tokens."""
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return float(int(h == r))
    prev = [0] * (len(r) + 1)
    for i in range(1, len(h) + 1):
        cur = [0] * (len(r) + 1)
        for j in range(1, len(r) + 1):
            cur[j] = prev[j - 1] + 1 if h[i - 1] == r[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, rc = lcs / len(h), lcs / len(r)
    return 2 * p * rc / (p + rc)


def first_divergence(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


# --------------------------------------------------------------------------- #
# benchmark-specific prompt builders
# --------------------------------------------------------------------------- #
def prep_arc(items):
    out = []
    for it in items:
        body = it["question"] + "\n" + "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(it["choices"])) + "\nAnswer:"
        it = dict(it, _text=body, _choices=[chr(65 + i) for i in range(len(it["choices"]))])
        out.append(it)
    return out


def prep_boolq(items):
    for it in items:
        it["choices"] = ["yes", "no"]
        it["answer"] = 0 if it["answer"] else 1
        it["_text"] = f"{it['passage']}\nQuestion: {it['question']}\nAnswer:"
    return items


def prep_piqa(items):
    for it in items:
        it["_text"] = it["goal"] + "\nAnswer:"
        it["choices"] = it["choices"]
    return items


def prep_hellaswag(items):
    for it in items:
        it["choices"] = it["choices"]
        it["_text"] = it["ctx"] + "\nAnswer:"
    return items


def prep_pope(items):
    for it in items:
        it["_text"] = it["question"] + "\nAnswer:"
    return items


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def calibration_profiles(runner, prompts, max_tokens=384):
    """Per-layer |SiLU(gate)*up| profiles from calibration prompts (P1 input)."""
    tb = runner.tb
    cap = {}
    for p in prompts:
        ids = torch.tensor(runner.tok(p, add_special_tokens=False)["input_ids"][:128],
                           device=runner.device)
        tb.forward(ids, capture=cap)
        acts = cap["mlp_in"]
        if not hasattr(runner, "_prof"):
            runner._prof = {i: [] for i in acts}
        for i, x in acts.items():
            p_ = f"{tb.prefix}layers.{i}.mlp."
            a = silu(F.linear(x, tb.w[p_ + "gate_proj.weight"])) * \
                F.linear(x, tb.w[p_ + "up_proj.weight"])
            runner._prof[i].append(a)
        cap.clear()
    return {i: torch.cat(v, 0)[:max_tokens] for i, v in runner._prof.items()}


def run_one(runner, profile, perm_map, rate, variant, cfg, eval_items):
    """Evaluate a single (variant, rate) configuration."""
    override = None if rate == 0.0 else perm_map
    runner.set_p1(override, rate, variant)
    row = {"variant": variant, "rate": rate}

    if cfg["ppl_ids"] is not None:
        row["ppl"], row["ppl_tokens"] = perplexity(
            runner, cfg["ppl_ids"], cfg["seq_len"], max_windows=cfg["ppl_windows"])
        row["delta_ppl"] = row["ppl"] - cfg["m0_ppl"]

    for bench in ("arc", "boolq", "piqa", "hellaswag", "pope"):
        key = f"{bench}_acc"
        if bench in eval_items:
            acc, n = mc_accuracy(runner, eval_items[bench], limit=cfg["mc_limit"])
            row[key] = acc
            row[f"{bench}_n"] = n
            row[f"delta_{bench}"] = acc - cfg["m0_acc"].get(bench, acc)

    if cfg["gen_items"] and cfg["m0_gen"] is not None:
        rl, fd, outs = [], [], []
        for it, ref_gen in zip(cfg["gen_items"], cfg["m0_gen"]):
            out = runner.generate(torch.tensor(it["_ids"], device=runner.device), cfg["gen_len"])
            rl.append(rouge_l(runner.tok.decode(out), it["answer"].split("####")[-1]))
            fd.append(first_divergence(out, ref_gen))
            outs.append(out)
        row["gen_rouge_l"] = sum(rl) / len(rl)
        row["gen_first_div"] = sum(fd) / len(fd)
        row["gen_top1_agree"] = sum(1 for x, y in zip(outs, cfg["m0_gen"]) if x[:1] == y[:1]) / len(outs)
    runner.set_p1(None, 0.0, variant)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b",
                    choices=["qwen3.5-0.8b", "qwen3-0.6b"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--ppl-windows", type=int, default=24)
    ap.add_argument("--mc-limit", type=int, default=100)
    ap.add_argument("--gen-n", type=int, default=20)
    ap.add_argument("--gen-len", type=int, default=24)
    ap.add_argument("--rates", default=",".join(str(r) for r in RATES))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--latency-reps", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    set_deterministic(0)
    dt = getattr(torch, args.dtype)
    rates = [float(r) for r in args.rates.split(",")]
    variants = args.variants.split(",")

    if args.model == "qwen3.5-0.8b":
        runner = RefRunner(args.device, dt)
        from video_data import CALIB_PROMPTS
        calib_prompts = CALIB_PROMPTS
    else:
        runner = HFRunner(MODEL_DIR.parent / "Qwen3-0.6B", args.device, dt)
        calib_prompts = ["The capital of France is", "def fib(n):", "In 1969 humans",
                         "The quick brown fox", "量子力学是", "Machine learning models",
                         "select * from users where", "The recipe calls for"]

    log(f"model={runner.name} device={args.device} dtype={args.dtype}")

    # ---- data ----
    wt = load_jsonl("wikitext2_test.jsonl")
    ppl_ids = None
    if wt:
        text = "\n\n".join(r["text"] for r in wt)
        ppl_ids = runner.tok(text, add_special_tokens=False)["input_ids"]
        log(f"wikitext-2 test: {len(wt)} rows, {len(ppl_ids)} tokens")

    eval_items = {}
    files = {"arc": "arc_easy_test.jsonl", "boolq": "boolq_val.jsonl",
             "piqa": "piqa_val.jsonl", "hellaswag": "hellaswag_val.jsonl",
             "pope": "pope_test.jsonl"}
    for name, prep in (("arc", prep_arc), ("boolq", prep_boolq), ("piqa", prep_piqa),
                       ("hellaswag", prep_hellaswag), ("pope", prep_pope)):
        items = load_jsonl(files[name])
        if items:
            items = prep(items)
            for it in items:
                if "_prompt_ids" not in it:
                    it["_prompt_ids"] = runner.tok(it["_text"], add_special_tokens=False)["input_ids"]
                it["_choice_ids"] = {c: runner.tok(" " + c.strip(), add_special_tokens=False)["input_ids"]
                                     or runner.tok(c.strip(), add_special_tokens=False)["input_ids"]
                                     for c in it["choices"]}
            eval_items[name] = items
    log(f"benchmarks: { {k: len(v) for k, v in eval_items.items()} }")

    gen_items = load_jsonl("gsm8k_test.jsonl")[:args.gen_n]
    for it in gen_items:
        it["_ids"] = runner.tok(it["question"] + "\n", add_special_tokens=False)["input_ids"]

    # ---- M0 ----
    runner.set_p1(None, 0.0, "natural")
    m0 = {"variant": "M0", "rate": 0.0}
    if ppl_ids:
        m0["ppl"], m0["ppl_tokens"] = perplexity(runner, ppl_ids, args.seq_len,
                                                 max_windows=args.ppl_windows)
    m0_acc = {}
    for name, items in eval_items.items():
        acc, n = mc_accuracy(runner, items, limit=args.mc_limit)
        m0[f"{name}_acc"], m0[f"{name}_n"] = acc, n
        m0_acc[name] = acc
    m0_gen = None
    if gen_items:
        m0_gen = [runner.generate(torch.tensor(it["_ids"], device=runner.device), args.gen_len)
                  for it in gen_items]
        m0["gen_rouge_l"] = sum(rouge_l(runner.tok.decode(g), it["answer"].split("####")[-1])
                               for g, it in zip(m0_gen, gen_items)) / len(gen_items)
        m0["gen_first_div"] = 0.0
    log(f"M0: {json.dumps({k: v for k, v in m0.items() if isinstance(v, float)}, indent=None)}")

    # ---- P1 permutations ----
    perms = {}
    if hasattr(runner, "tb"):
        from p1_ffn import cluster_blocks
        import random as _r
        profile = calibration_profiles(runner, calib_prompts)
        inter = profile[0].shape[1]
        nlayers = runner.tb.cfg.layers
        g = torch.Generator().manual_seed(1234)
        perms = {
            "natural": {i: torch.arange(inter) for i in range(nlayers)},
            "clustered": {i: cluster_blocks(profile[i], BAND, seed=1234)[0]
                          for i in range(nlayers)},
            "random": {i: torch.randperm(inter, generator=g) for i in range(nlayers)},
        }
        ov = {}
        for v in variants:
            ov[v] = {}
            for i in range(nlayers):
                pre = f"{runner.tb.prefix}layers.{i}.mlp."
                ov[v][i] = permute_mlp((runner.tb.w[pre + "gate_proj.weight"],
                                        runner.tb.w[pre + "up_proj.weight"],
                                        runner.tb.w[pre + "down_proj.weight"]), perms[v][i])
    else:
        # HF model: permute the live weight tensors in place, keep references
        from p1_ffn import cluster_blocks
        prof = {}
        caps = []
        for p_ in calib_prompts:
            ids = torch.tensor(runner.tok(p_, add_special_tokens=False)["input_ids"][:96],
                               device=runner.device)
            with torch.no_grad():
                runner.model(input_ids=ids.unsqueeze(0))
            caps.append(None)
        # profiles via hooks
        A = {}

        def mk(i):
            def hook(m, inp, out):
                x = inp[0]
                a = m.act_fn(m.gate_proj(x)) * m.up_proj(x)
                A.setdefault(i, []).append(a.detach())
            return hook
        hs = [l.mlp.register_forward_hook(mk(i)) for i, l in enumerate(runner.layers)]
        for p_ in calib_prompts:
            ids = torch.tensor(runner.tok(p_, add_special_tokens=False)["input_ids"][:96],
                               device=runner.device)
            with torch.no_grad():
                runner.model(input_ids=ids.unsqueeze(0))
        for h in hs:
            h.remove()
        profile = {i: torch.cat([t.reshape(-1, t.shape[-1]) for t in v], 0)[:384]
                   for i, v in A.items()}
        inter = runner.inter
        gtorch = torch.Generator().manual_seed(1234)
        perms = {
            "natural": {i: torch.arange(inter) for i in range(len(runner.layers))},
            "clustered": {i: cluster_blocks(profile[i], BAND, seed=1234)[0]
                          for i in range(len(runner.layers))},
            "random": {i: torch.randperm(inter, generator=gtorch)
                       for i in range(len(runner.layers))},
        }
        ov = {}
        for v in variants:
            ov[v] = {}
            for i, l in enumerate(runner.layers):
                p = perms[v][i].to(runner.device)
                g, u, d = l.mlp.gate_proj.weight, l.mlp.up_proj.weight, l.mlp.down_proj.weight
                ov[v][i] = (g[p].contiguous(), u[p].contiguous(), d[:, p].contiguous())

    # isomorphism check at tau=0 for the clustered permutation
    iso = {}
    if True:
        ids = eval_items.get("arc", [{"_prompt_ids": [1] * 12}])[0]["_prompt_ids"]
        ids = torch.tensor(ids[:32], device=runner.device)
        runner.set_p1(None, 0.0, "natural")
        base = runner.logits(ids)
        runner.set_p1(ov["clustered"], 0.0, "clustered")
        perm_lg = runner.logits(ids)
        d = (perm_lg - base).abs()
        iso = {"max_abs_dlogit": float(d.max()), "rel_l2": float(d.norm() / base.norm()),
               "top1_agreement": float((perm_lg.argmax(-1) == base.argmax(-1)).float().mean())}
        runner.set_p1(None, 0.0, "natural")
        log(f"P1 isomorphism tau=0 (clustered): {iso}")

    # ---- latency (per generated token, GPU/CPU) ----
    lat = {}
    probe = gen_items[0]["_ids"] if gen_items else eval_items.get("arc", [{"_prompt_ids": [1] * 16}])[0]["_prompt_ids"]
    probe = torch.tensor(probe[:24], device=runner.device)
    for r in (0.0, 0.2, 0.4, 0.6, 0.8):
        runner.set_p1(None if r == 0 else ov.get("clustered"), r, "clustered")
        ms, _ = bench(lambda: runner.logits(probe), args.device, warmup=2, reps=args.latency_reps)
        lat[f"{r:.1f}"] = ms
    runner.set_p1(None, 0.0, "natural")
    dense_ms = lat["0.0"]
    for k in lat:
        lat[k] = {"ms": lat[k], "speedup": dense_ms / lat[k] if lat[k] else 0.0}

    # ---- sweep ----
    cfg = {"ppl_ids": ppl_ids, "seq_len": args.seq_len, "ppl_windows": args.ppl_windows,
           "mc_limit": args.mc_limit, "gen_items": gen_items, "gen_len": args.gen_len,
           "m0_ppl": m0.get("ppl", 0.0), "m0_acc": m0_acc, "m0_gen": m0_gen}
    rows = [m0]
    for v in variants:
        for r in rates:
            if r == 0.0 and v != "natural":
                continue
            row = run_one(runner, profile, ov.get(v), r, v, cfg, eval_items)
            rows.append(row)
            log(f"{v:9s} tau={r:.2f} " + " ".join(
                f"{k}={row[k]:.4f}" for k in
                ("ppl", "delta_ppl", "arc_acc", "boolq_acc", "piqa_acc", "gen_rouge_l",
                 "gen_first_div") if k in row))

    # safe threshold: first rate where dPPL > 0.5 or dAcc > 1pp for the clustered permutation
    tau_star = None
    for row in rows:
        if row.get("variant") != "clustered":
            continue
        bad = (row.get("delta_ppl", 0) > 0.5 or
               any(row.get(f"delta_{b}", 0) < -0.01 for b in m0_acc))
        if bad:
            tau_star = row["rate"]
            break

    out = {"model": runner.name, "device": args.device, "dtype": args.dtype,
           "seq_len": args.seq_len, "ppl_windows": args.ppl_windows,
           "mc_limit": args.mc_limit, "gen_n": len(gen_items), "gen_len": args.gen_len,
           "n_wikitext_tokens": None if ppl_ids is None else len(ppl_ids),
           "rows": rows, "latency": lat, "isomorphism": iso,
           "tau_star_first_violation": tau_star,
           "benchmarks": {k: len(v) for k, v in eval_items.items()},
           "permutation_coherence": None}
    tag = args.tag or args.model.replace(".", "_")
    dump_json(out, RESULTS_DIR / f"capability_{tag}.json")
    log(f"wrote results/capability_{tag}.json")


if __name__ == "__main__":
    main()
