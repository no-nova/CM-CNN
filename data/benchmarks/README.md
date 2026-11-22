# Evaluation sets used by the CM-Transform experiments

These files are fixed-format JSONL snapshots of public evaluation splits. They are
committed so that every number in `paper2_bundle/CM_Transform_Paper2_Final.pdf` can be
reproduced without network access, and so that the digests in
`experiments/cm-transformer/results/*.json` refer to immutable inputs.

All files are read-only (`chmod 444`) and are listed with row counts, byte counts and
SHA-256 digests in `manifest.json`. Regenerate the corpus with

```bash
python3 scripts/download_benchmarks.py --out data/benchmarks
```

## Files, sources and licences

| File | Rows | Source split | Licence |
|---|---|---|---|
| `wikitext2_test.jsonl` | 2,891 | WikiText-2, `test` (`merity/wikitext-2-raw-v1`) | CC BY-SA 3.0 |
| `arc_easy_test.jsonl` | 500 | AI2 Reasoning Challenge, `ARC-Easy/test` | CC BY-SA 4.0 |
| `boolq_val.jsonl` | 500 | BoolQ, `validation` | CC BY-SA 3.0 |
| `piqa_val.jsonl` | 500 | PIQA, `validation` (via the `nthngdy/piqa` mirror) | AFL-3.0 |
| `gsm8k_test.jsonl` | 100 | GSM8K, `test` | MIT |
| `gsm8k_test200.jsonl` | 300 | GSM8K, `test`, items 0–299 | MIT |
| `pope_test.jsonl` | 300 | POPE, `test` | BSD-3-Clause |

Each JSONL row carries the columns used by the loader in
`experiments/cm-transformer/bench_text.py`: `question`/`choices`/`answer` for
ARC-Easy, `passage`/`question`/`answer` for BoolQ, `goal`/`sol1`/`sol2`/`label` for
PIQA, `question`/`answer` (with the `####` marker) for GSM8K, and the raw text field
for WikiText-2.

Only the evaluation splits are redistributed; no training data, no model weights and
no derived annotations are included. If you reuse these files, cite the original
dataset papers listed in the manuscript's References.
