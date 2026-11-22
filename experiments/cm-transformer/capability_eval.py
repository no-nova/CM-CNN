"""Capability preservation sweep for P1 (32-lane FFN block skipping).

FIXED in this revision: the previous version scored multiple-choice items with a
single uncalibrated token (the first token of each option), which is why PIQA
stayed at 0.510 and ARC-Easy wobbled around chance.  Scoring now follows
lm-evaluation-harness exactly:

    loglik(c) = sum_j log P(tok_j | prompt + tok_<j)     (choice tokens only)
    acc       = argmax loglik / n_choice_tokens          (length-normalised)
    acc_norm  = argmax loglik / len(choice_bytes)        (byte-normalised)

All the scoring lives in `bench_text.py` (one implementation, no duplicates);
all the P1 permutation / gating logic lives in `p1_v2.py`.  This module is the
driver that ties them together and writes the capability tables:

    python3 capability_eval.py --model qwen3.5-0.8b --rates 0.1,0.2,0.3 \
        --mc-limit 500 --datasets arc,piqa,boolq --gsm 50

Outputs `results/p1v2_<tag>.json` with, for every (variant, gate, tau):
WikiText-2 PPL, ARC-Easy/PIQA/BoolQ acc + acc_norm, GSM8K exact-match,
retained output-weighted block energy, and the per-layer within-block output
error energies used to justify the permutation.
"""
from __future__ import annotations

import sys

import torch

from bench_text import (PREP, FILES, run_dataset, run_gsm8k, score_items,
                        summarize)
from run_text_eval import load_jsonl


def mc_accuracy(runner, dataset, limit=500, **kw):
    """Length-normalised choice loglik accuracy (lm-eval `acc`)."""
    items = PREP[dataset](load_jsonl(FILES[dataset]))[:limit]
    return summarize(score_items(runner, items, **kw))


def main(argv=None):
    from p1_v2 import main as p1_main
    sys.argv = [sys.argv[0]] + (list(argv) if argv is not None else sys.argv[1:])
    p1_main()


if __name__ == "__main__":
    main()
