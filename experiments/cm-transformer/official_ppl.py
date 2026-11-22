"""Independent perplexity cross-check with the official transformers model.

Used to prove that the WikiText-2 perplexity reported by run_text_eval.py is a
property of the checkpoint, not of the in-repo reference implementation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from common import MODEL_DIR, ROOT, build_tokenizer, dump_json, log, set_deterministic

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_windows(tok, seq_len, n_windows):
    rows = [json.loads(l) for l in (ROOT / "data/benchmarks/wikitext2_test.jsonl").open()]
    ids = tok("\n\n".join(r["text"] for r in rows), add_special_tokens=False)["input_ids"]
    return [ids[i * seq_len:(i + 1) * seq_len] for i in range(n_windows)]


@torch.no_grad()
def ppl_official(windows, model_dir, device="cuda", dtype=torch.float32):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
    nll, ntok = 0.0, 0
    for w in windows:
        ids = torch.tensor(w, device=device)
        lg = m(input_ids=ids.unsqueeze(0), use_cache=False).logits[0]
        for s in range(0, lg.shape[0] - 1, 64):
            n = min(64, lg.shape[0] - 1 - s)
            lo = lg[s:s + n].float()
            tgt = ids[s + 1:s + 1 + n]
            nll += float(F.cross_entropy(lo, tgt, reduction="sum"))
            ntok += lo.shape[0]
    return math.exp(nll / ntok), ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--windows", type=int, default=16)
    args = ap.parse_args()
    set_deterministic(0)
    tok = build_tokenizer(Path(args.model_dir))
    wins = load_windows(tok, args.seq_len, args.windows)
    ppl, n = ppl_official(wins, Path(args.model_dir), args.device, getattr(torch, args.dtype))
    log(f"official PPL {ppl:.4f} over {n} tokens ({len(wins)} windows)")
    out = RESULTS_DIR / "official_ppl.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev[Path(args.model_dir).name + f"_{args.device}_{args.dtype}"] = {
        "ppl": ppl, "tokens": n, "seq_len": args.seq_len, "windows": len(wins),
        "implementation": "transformers AutoModelForCausalLM", "dtype": args.dtype}
    dump_json(prev, out)


if __name__ == "__main__":
    main()
