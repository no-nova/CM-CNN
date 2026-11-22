# CM-CNN

Code and manuscripts for structured-sparsity work on convolutional and transformer
inference.

## Paper 2 — CM-Transform: co-activation permuted block sparsity with cache-line aligned execution

`paper2_bundle/CM_Transform_Paper2_Final.pdf` — final manuscript (20 pages, 45 tables,
6 figures). Source: `paper2_bundle/CM_Transform_Paper2.typ` (Typst 0.13).

What the paper measures, on one consumer GPU (RTX 5070 Laptop, 8 GB) and one Zen 5
CPU (AVX-512):

* a functionally isomorphic 32-lane permutation of the FFN intermediate dimension
  (`τ = 0` reproduces the dense forward pass to `1e-7`);
* a per-layer adaptive energy threshold with norm-preserving rescaling, compared
  against CATS-Block, TEAL-Block and Wanda-style static pruning at matched skip
  budgets, with bootstrap intervals;
* device-side compaction and row-parallel CPU execution for the skip kernel;
* a block-major repack of IQ4_NL weights so that skipping 18-byte blocks reduces
  64-byte cache-line traffic (`1.74×` at 0% skip, up to `5.67×` at 80%);
* the same mechanism on speech (`whisper-base`) and vision-language
  (`SmolVLM2-256M`) towers, including where it does not pay.

### Layout

| Path | Contents |
|---|---|
| `paper2_bundle/` | manuscript source, compiled PDF, vector figures, fragments, digests |
| `experiments/cm-transformer/` | reference model, kernels, drivers, tests, result JSONs, `REPORT_TEXT.md` |
| `data/benchmarks/` | fixed evaluation splits (WikiText-2, ARC-Easy, BoolQ, PIQA, GSM8K, POPE) |
| `data/p6_vlm/` | synthetic multimodal probe images, regenerable from `make_p6_vlm.py` |
| `scripts/download_benchmarks.py` | regenerates `data/benchmarks/` |

### Reproduce

```bash
# manuscript
cd paper2_bundle && typst compile CM_Transform_Paper2.typ CM_Transform_Paper2_Final.pdf

# numerical invariants and paper-source checks
cd experiments/cm-transformer && python3 -m pytest tests/ -q

# evaluation data
python3 scripts/download_benchmarks.py --out data/benchmarks
```

Experiments were run with Python 3.12 + PyTorch 2.15.0 (CUDA 13.5, sm_120) and
`transformers` 5.23.0. Kernels build with `nvcc` and `g++ -O3 -mavx512f -mavx512bw
-mfma -fopenmp -march=native`; CPU numbers assume `OMP_WAIT_POLICY=passive`.

### Not in this repository

Model weights (Qwen3-0.6B, Qwen3.5-0.8B, Qwen3.5-4B-IQ4_NL, whisper-base,
SmolVLM2-256M) and LibriSpeech audio are third-party assets under their own licences
and are not redistributed. `paper2_bundle/hashes.json` records the digests of the
checkpoints used, and `experiments/cm-transformer/REPORT_TEXT.md` states which script
regenerates each result file.

## Licence

Code and manuscript: see `LICENSE`. Evaluation splits in `data/benchmarks/` keep the
licences of their source datasets, listed in `data/benchmarks/README.md`.
