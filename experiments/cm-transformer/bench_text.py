"""Text-capability benchmarks with lm-evaluation-harness scoring.

Why this file exists: the first pass scored multiple-choice items with a single
uncalibrated first token of each option, which is why PIQA sat at 0.510 and
ARC-Easy wobbled around chance.  Here every option is scored the way the harness
does it:

    loglik(c) = sum_j log P(tok_j | prompt + tok_<j)      (only choice tokens)
    acc       = argmax loglik / n_choice_tokens           (length-normalised)
    acc_norm  = argmax loglik / len(choice_bytes)         (byte-normalised)

Both are reported; `acc` is the length-normalised variant requested for this
study and `acc_norm` reproduces the public leaderboard convention.

Datasets: ARC-Easy, PIQA, BoolQ (multiple choice) and GSM8K (greedy generation,
answer extracted after `####`).  Also 64-token greedy-generation stability
(ROUGE-L and first-divergence step) on a fixed prompt set.

    python3 bench_text.py --model qwen3.5-0.8b --device cuda --dtype float32 \
        --datasets arc,piqa,boolq --limit 500 --tag base
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from run_text_eval import BENCH, HFRunner, RefRunner, load_jsonl, log, rouge_l, first_divergence

RES = Path(__file__).resolve().parent / "results"
RES.mkdir(exist_ok=True)
MODELS = {"qwen3.5-0.8b": str(Path(__file__).resolve().parents[2] / "models/Qwen3.5-0.8B"),
          "qwen3-0.6b": str(Path(__file__).resolve().parents[2] / "models/Qwen3-0.6B")}

# --------------------------------------------------------------------------- #
# prompt construction (lm-eval style)
# --------------------------------------------------------------------------- #
def prep_arc(items):
    out = []
    for it in items:
        body = it["question"] + "\n" + "\n".join(
            f"{chr(65 + i)}. {c}" for i, c in enumerate(it["choices"])) + "\nAnswer:"
        out.append({"id": it["id"], "ctx": body,
                    "choices": [chr(65 + i) for i in range(len(it["choices"]))],
                    "answer": it["answer"]})
    return out


def prep_piqa(items):
    return [{"id": it["id"], "ctx": it["goal"] + "\nAnswer:",
             "choices": list(it["choices"]), "answer": it["answer"]} for it in items]


def prep_boolq(items):
    return [{"id": it["id"],
             "ctx": it["passage"] + "\nQuestion: " + it["question"] + "\nAnswer:",
             "choices": ["yes", "no"],
             "answer": 0 if it["answer"] else 1} for it in items]


PREP = {"arc": prep_arc, "piqa": prep_piqa, "boolq": prep_boolq}
FILES = {"arc": "arc_easy_test.jsonl", "piqa": "piqa_val.jsonl", "boolq": "boolq_val.jsonl"}


# --------------------------------------------------------------------------- #
# multiple choice
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_items(runner, items, max_tokens=1024):
    """Return rows of per-choice (loglik, n_tokens, n_bytes) plus metadata."""
    rows = []
    tok = runner.tok
    for it in items:
        ctx = tok(it["ctx"], add_special_tokens=False)["input_ids"][-max_tokens:]
        cand = []
        for c in it["choices"]:
            cid = tok(" " + c.strip(), add_special_tokens=False)["input_ids"]
            if not cid:                                  # empty continuation
                continue
            ids = ctx + cid
            # lm-eval: predict the choice tokens only, conditioned on the prompt
            pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(cid))
            lg = runner.logits_at(torch.tensor(ids, device=runner.device), pos)
            lp = F.log_softmax(lg.float(), dim=-1)
            ll = 0.0
            for j, t in enumerate(cid):
                ll += float(lp[j, t])
            cand.append({"choice": c, "loglik": ll, "n_tok": len(cid),
                         "n_bytes": len((" " + c.strip()).encode("utf-8")),
                         "ids": cid})
        if not cand:
            continue
        rows.append({"id": it["id"], "answer": it["answer"], "cands": cand})
    return rows


def summarize(rows):
    acc = acc_norm = 0
    for r in rows:
        a = r["answer"]
        p = max(range(len(r["cands"])), key=lambda i: r["cands"][i]["loglik"] / r["cands"][i]["n_tok"])
        q = max(range(len(r["cands"])), key=lambda i: r["cands"][i]["loglik"] / r["cands"][i]["n_bytes"])
        acc += int(p == a)
        acc_norm += int(q == a)
    n = max(len(rows), 1)
    return {"acc": acc / n, "acc_norm": acc_norm / n, "n": len(rows)}


def run_dataset(runner, name, limit, max_tokens=1024):
    items = PREP[name](load_jsonl(FILES[name]))[:limit]
    t0 = time.perf_counter()
    rows = score_items(runner, items, max_tokens=max_tokens)
    out = summarize(rows)
    out["seconds"] = time.perf_counter() - t0
    out["items_per_s"] = out["n"] / max(out["seconds"], 1e-6)
    return out, rows


# --------------------------------------------------------------------------- #
# GSM8K
# --------------------------------------------------------------------------- #
GSM_FEWSHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
     "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. There will be 21 trees after planting. So the workers planted "
     "21 - 15 = 6 trees.\n#### 6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "3 + 2 = 5 cars in the parking lot.\n#### 5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally they had 32 + 42 = 74 chocolates. After eating 35, they have 74 - 35 = 39.\n#### 39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
     "How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops, and has 12 left, so he gave 20 - 12 = 8 lollipops.\n#### 8"),
]

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str):
    """GSM8K convention: the number right after the FIRST '####'.  (Taking the
    last one picks up the next few-shot example when the model keeps generating.)"""
    if "####" in text:
        m = NUM.search(text.split("####", 1)[1])
        if m:
            return m.group(0).replace(",", "")
    nums = NUM.findall(text)
    return nums[-1].replace(",", "") if nums else None


def gold_answer(item):
    m = NUM.search(str(item["answer"]).split("####")[-1])
    return m.group(0).replace(",", "") if m else str(item["answer"]).strip()


@torch.no_grad()
def run_gsm8k(runner, limit=50, max_new=192):
    items = load_jsonl("gsm8k_test.jsonl")[:limit]
    shots = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in GSM_FEWSHOT)
    correct, rows, n_tok = 0, [], 0
    t0 = time.perf_counter()
    for it in items:
        prompt = f"{shots}\n\nQuestion: {it['question']}\nAnswer:"
        ids = torch.tensor(runner.tok(prompt, add_special_tokens=False)["input_ids"],
                           device=runner.device)
        gen = runner.generate(ids, max_new)
        n_tok += len(gen)
        text = runner.tok.decode(gen)
        pred, gold = extract_answer(text), gold_answer(it)
        ok = pred is not None and pred == gold
        correct += int(ok)
        rows.append({"id": it["id"], "pred": pred, "gold": gold, "ok": ok,
                     "text": text[:220]})
    dt = time.perf_counter() - t0
    return {"acc": correct / max(len(items), 1), "n": len(items),
            "exact": f"{correct}/{len(items)}", "seconds": dt,
            "tokens_per_s": n_tok / max(dt, 1e-6), "rows": rows}


# --------------------------------------------------------------------------- #
# generation stability
# --------------------------------------------------------------------------- #
GEN_PROMPTS = [
    "The industrial revolution began in Britain because",
    "A healthy diet consists of",
    "The function of the mitochondria is to",
    "In 1969, humans first",
    "The three laws of thermodynamics are",
    "To solve a quadratic equation, one can",
    "The main causes of the First World War were",
    "Machine learning models are trained by",
    "The capital of France is",
    "Water boils at a temperature of",
    "A compiler translates source code into",
    "The most important property of a hash table is",
    "Photosynthesis converts light energy into",
    "The principle of supply and demand states that",
    "Neural networks approximate functions using",
    "A binary search tree maintains the invariant that",
    "The industrial policy of Japan after 1945 focused on",
    "In a distributed system, consensus is required when",
    "The difference between TCP and UDP is that",
    "An eigenvalue of a matrix satisfies",
    "Write a Python function that reverses a string:",
    "The role of the liver in the human body is to",
    "Shakespeare wrote Hamlet in",
    "A prime number is defined as",
    "The greenhouse effect is caused by",
    "Gradient descent updates parameters by",
    "The main advantage of a relational database is",
    "Electric current is defined as",
    "The Pythagorean theorem states that",
    "Reinforcement learning differs from supervised learning because",
    "The boiling point of water decreases with",
    "A proof by contradiction assumes",
    "The Roman Empire fell because",
    "The purpose of a cache is to",
    "In statistics, the central limit theorem says",
    "A transformer model attends to",
    "The first law of motion states that",
    "Vaccines work by",
    "The value of pi is approximately",
    "A deadlock occurs when",
    "The chemical formula for water is",
    "An operating system scheduler decides",
    "The difference between a stack and a queue is",
    "Interest rates affect inflation because",
    "The largest planet in the solar system is",
    "A convolutional layer detects",
    "The goal of regularization is to",
    "Bitcoin achieves consensus through",
    "The normal distribution is characterised by",
    "A foreign key enforces",
]


def run_generation(runner, n_prompts=32, n_new=64):
    prompts = GEN_PROMPTS[:n_prompts]
    outs, t0 = [], time.perf_counter()
    for p in prompts:
        ids = torch.tensor(runner.tok(p, add_special_tokens=False)["input_ids"],
                           device=runner.device)
        outs.append(runner.generate(ids, n_new))
    dt = time.perf_counter() - t0
    return {"prompts": [{"prompt": p, "ids": o} for p, o in zip(prompts, outs)],
            "seconds": dt, "tokens_per_s": n_prompts * n_new / max(dt, 1e-6)}


def compare_generation(base, other):
    sims, divs = [], []
    for a, b in zip(base["prompts"], other["prompts"]):
        ta, tb = _decode_cached(a, b)
        sims.append(rouge_l(tb, ta))
        divs.append(first_divergence(a["ids"], b["ids"]))
    n = max(len(sims), 1)
    return {"rouge_l": sum(sims) / n, "first_div": sum(divs) / n,
            "top1_agreement": sum(1 for a, b in zip(base["prompts"], other["prompts"])
                                  if a["ids"][:1] == b["ids"][:1]) / n,
            "n": len(sims)}


_DECODER = {}


def _decode_cached(a, b):
    tok = _DECODER.get("tok")
    if tok is None:
        from common import build_tokenizer, MODEL_DIR
        tok = _DECODER["tok"] = build_tokenizer(MODEL_DIR)
    return tok.decode(a["ids"]), tok.decode(b["ids"])


# --------------------------------------------------------------------------- #
def make_runner(model: str, device: str, dtype: str):
    dt = {"float32": torch.float32, "float16": torch.float16,
          "bfloat16": torch.bfloat16}[dtype]
    if model == "qwen3.5-0.8b":
        r = RefRunner(device=device, dtype=dt)
        r.logits_at = lambda ids, pos: r.tb.forward(ids, logits_positions=pos)["logits"].float()
        return r
    r = HFRunner(MODELS[model], device=device, dtype=dt, name=model)
    r.logits_at = lambda ids, pos: r.model(input_ids=ids.unsqueeze(0),
                                           use_cache=False).logits[0][pos].float()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b", choices=list(MODELS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--datasets", default="arc,piqa,boolq")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--gsm8k", type=int, default=50)
    ap.add_argument("--gen-prompts", type=int, default=32)
    ap.add_argument("--gen-len", type=int, default=64)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--skip-mc", action="store_true")
    args = ap.parse_args()

    runner = make_runner(args.model, args.device, args.dtype)
    out = {"model": args.model, "device": args.device, "dtype": args.dtype,
           "datasets": {}, "tag": args.tag}
    for name in ([] if args.skip_mc else args.datasets.split(",")):
        res, rows = run_dataset(runner, name, args.limit)
        out["datasets"][name] = res
        log(f"{name:6s} acc {res['acc']:.3f}  acc_norm {res['acc_norm']:.3f}  "
            f"n={res['n']}  {res['seconds']:.1f}s ({res['items_per_s']:.1f} it/s)")
        out.setdefault("rows", {})[name] = rows
    if args.gsm8k:
        g = run_gsm8k(runner, limit=args.gsm8k)
        out["gsm8k"] = g
        log(f"gsm8k  acc {g['acc']:.3f} ({g['exact']})  {g['tokens_per_s']:.1f} tok/s")
    if not args.no_gen:
        base = run_generation(runner, args.gen_prompts, args.gen_len)
        out["generation"] = base
        log(f"generation {base['tokens_per_s']:.1f} tok/s "
            f"({args.gen_prompts}x{args.gen_len})")
    p = RES / f"textcap_{args.tag}.json"
    p.write_text(json.dumps(out, indent=1))
    log(f"wrote {p}")


if __name__ == "__main__":
    main()
