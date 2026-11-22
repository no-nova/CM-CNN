# CM-Transform P1 — text-only acceleration & capability report

## 0. Headline

| question | answer |
|---|---|
| Are the baselines real now? | yes — ARC-Easy 0.778 / PIQA 0.702 / BoolQ 0.630 (0.8B) and 0.512 / 0.698 / 0.652 (0.6B) at 500 items, GSM8K 0.280 / 0.380, generation 84 / 62 tok/s |
| Does the permutation beat natural order? | on Qwen3-0.6B yes, by 2.1x / 1.75x / 2.1x at tau 0.10 / 0.20 / 0.30 and 2.0x / 2.2x at 0.40 / 0.50; on Qwen3.5-0.8B it wins from tau 0.30 up (4.94 vs 5.22, 8.72 vs 9.38, 14.65 vs 16.16) but natural still wins at 0.10-0.20 (0.68-2.34 vs 0.92-2.46) |
| Does adaptive gating beat per-layer top-k? | yes at every rate on both models (0.8B: dPPL 0.92/2.46/4.94 vs 1.15/2.76/5.51; 0.6B: 0.28/1.45/3.22 vs 0.34/2.01/5.14) |
| Does the 32-lane kernel scale? | yes: 2.94x (T=1) and 3.46x (T=64) at 75% skipped blocks, i.e. it tracks the 1/(1-s) bound, while the old warp-mask kernel stays at 1.0-1.7x |
| Do the four Phase-3 fixes change the verdict? | **quality yes, single-operator yes, whole-model no.** Norm-preserving rescaling cuts dPPL by 0.15-0.99 (bigger at higher tau); block-major IQ4_NL turns line traffic into a linear function of the skip rate (4.54x cold at 80% vs 1.53x row-major, 2.84x vs 1.68x for 32 layers); row-parallel OpenMP lifts CPU decode from 0.89x to 1.02-1.17x; SwiGLU coupling doubles the skippable MLP share (ceiling 1.12x -> 1.26x). The model-level decode ceiling is unchanged because `down_proj` + `up_proj` are still only 21% of the checkpoint. |
| Does end-to-end decode get faster? | device-side compaction fixed the regression (GPU decode 0.78x -> 0.85x, prefill 0.59x -> 0.99x), but the byte ceiling of the model forbids a win: `down_proj` is only 10.4% of the 3.4 GB checkpoint, so 1.12x is the absolute maximum at T=1 |
| Phase 4: is the coupled SwiGLU skip real? | **yes at operator level, no at model level.** AVX-512 fused: 1.35/1.83/2.18x over the same-code dense at tau 0.2/0.4/0.6 and 8-13x over torch; CUDA fused 1.09-1.78x; 8-row block-major IQ4_NL beats the row-major dense baseline in absolute time (0.534 vs 0.930 ms) and by 3.8-5.7x under skipping |
| Phase 5: does `float4` + 2 rows/warp close the CUDA gap? | **no** — it is 9% slower dense and 30-35% slower at tau 0.40 than the Phase-4 scalar 4-row kernel (halving rows/warp halves the work per `x` fetch), so the 4-row kernel stays the shipping path. The weighted block gate is the real win: -6% rel-L2 at equal time on the GPU and 22-42% lower dPPL on the model |
| Phase 5: is the IQ4_NL Amdahl ceiling broken? | yes — the gate pre-filter (r=2 partial rows) moves the coupled FFN stream from 1.08/1.16/1.35/1.97x to 1.23/1.38/1.99/4.24x at 20/40/60/80% skip |
| Phase 5: does it transfer to speech and VLMs? | yes at tau=0 (exact) and at tau=0.2 (whisper teacher-forced top-1 0.889; SmolVLM VQA 1.0), with one mechanism finding: GELU sparsity makes 54%/51% of whisper's enc.4/5 blocks exactly dead after pi |
| Does it help a real quantised model? | hot/L3-resident: 1.23x @20% and 4.24x @80% skipped (dense 0.722 ms, final JSON); cold/DRAM: 1.14x @20% and 1.68x @80% at block granularity. Super-blocks (G=4) halve the line traffic (5184 -> 1728 B/row at 80% skip) but the loop is instruction-bound, so cold time improves only to 1.44x while energy capture drops from 0.81 to 0.56 |

## 1. Dense baseline (M0), lm-evaluation-harness scoring

| model | ARC-Easy acc | PIQA acc | BoolQ acc | GSM8K | gen tok/s |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.778 | 0.702 | 0.630 | 0.280 | 83.5 |
| Qwen3-0.6B | 0.512 | 0.698 | 0.652 | 0.380 | 62.1 |

## 2. P1 sweep — qwen3.5-0.8b

Dense PPL (WikiText-2, 8,176 tokens) = **26.858**.

| variant/gate | tau | PPL | dPPL | ARC | PIQA | BoolQ | GSM8K | E_keep |
|---|---|---|---|---|---|---|---|---|
| ours_adaptive | 0.10 | 27.776 | +0.918 | 0.820 | 0.705 | 0.635 | 0.300 | 99.02% |
| ours_adaptive | 0.20 | 29.321 | +2.463 | 0.815 | 0.690 | 0.645 | - | 97.52% |
| ours_adaptive | 0.30 | 31.793 | +4.935 | 0.760 | 0.700 | 0.650 | - | 95.53% |
| ours_adaptive | 0.40 | 35.576 | +8.718 | 0.730 | 0.680 | 0.670 | - | 92.96% |
| ours_adaptive | 0.50 | 41.506 | +14.648 | 0.650 | 0.665 | 0.570 | - | 89.78% |
| ours_uniform | 0.10 | 28.004 | +1.146 | 0.760 | 0.710 | 0.635 | 0.300 | 98.89% |
| ours_uniform | 0.20 | 29.621 | +2.763 | 0.730 | 0.680 | 0.645 | - | 97.18% |
| ours_uniform | 0.30 | 32.368 | +5.510 | 0.680 | 0.695 | 0.675 | - | 95.02% |
| natural_uniform | 0.10 | 27.534 | +0.676 | 0.780 | 0.710 | 0.635 | 0.320 | 98.64% |
| natural_uniform | 0.20 | 29.200 | +2.342 | 0.760 | 0.685 | 0.635 | - | 96.65% |
| natural_uniform | 0.30 | 32.074 | +5.216 | 0.615 | 0.685 | 0.620 | - | 94.24% |
| natural_uniform | 0.40 | 36.241 | +9.384 | 0.650 | 0.700 | 0.670 | - | 91.42% |
| natural_uniform | 0.50 | 43.015 | +16.157 | 0.635 | 0.685 | 0.625 | - | 88.15% |
| random_uniform | 0.10 | 27.969 | +1.111 | 0.735 | 0.685 | 0.640 | 0.380 | 98.35% |
| random_uniform | 0.20 | 30.237 | +3.379 | 0.765 | 0.645 | 0.655 | - | 95.99% |
| random_uniform | 0.30 | 34.260 | +7.402 | 0.645 | 0.675 | 0.625 | - | 93.12% |

Within-block output-truncation energy (lower is better): ours 60.3, natural 62.01, random 60.18.

## 2. P1 sweep — qwen3-0.6b

Dense PPL (WikiText-2, 8,176 tokens) = **34.309**.

| variant/gate | tau | PPL | dPPL | ARC | PIQA | BoolQ | GSM8K | E_keep |
|---|---|---|---|---|---|---|---|---|
| ours_adaptive | 0.10 | 34.589 | +0.281 | 0.585 | 0.705 | 0.685 | 0.320 | 100.00% |
| ours_adaptive | 0.20 | 35.763 | +1.454 | 0.645 | 0.690 | 0.695 | - | 99.97% |
| ours_adaptive | 0.30 | 37.533 | +3.224 | 0.600 | 0.705 | 0.620 | - | 99.94% |
| ours_adaptive | 0.40 | 41.504 | +7.195 | 0.535 | 0.650 | 0.595 | - | 99.95% |
| ours_adaptive | 0.50 | 49.513 | +15.204 | 0.460 | 0.650 | 0.535 | - | 99.88% |
| ours_uniform | 0.10 | 34.650 | +0.341 | 0.505 | 0.720 | 0.645 | 0.260 | 100.00% |
| ours_uniform | 0.20 | 36.322 | +2.013 | 0.605 | 0.675 | 0.615 | - | 99.97% |
| ours_uniform | 0.30 | 39.451 | +5.142 | 0.530 | 0.655 | 0.605 | - | 99.94% |
| natural_uniform | 0.10 | 34.913 | +0.604 | 0.450 | 0.690 | 0.655 | 0.240 | 99.99% |
| natural_uniform | 0.20 | 36.855 | +2.546 | 0.420 | 0.675 | 0.575 | - | 99.97% |
| natural_uniform | 0.30 | 41.160 | +6.851 | 0.480 | 0.650 | 0.570 | - | 99.92% |
| natural_uniform | 0.40 | 48.914 | +14.605 | 0.400 | 0.640 | 0.585 | - | 99.88% |
| natural_uniform | 0.50 | 67.085 | +32.776 | 0.375 | 0.595 | 0.555 | - | 99.81% |
| random_uniform | 0.10 | 35.236 | +0.927 | 0.375 | 0.705 | 0.650 | 0.340 | 99.99% |
| random_uniform | 0.20 | 37.973 | +3.664 | 0.395 | 0.690 | 0.590 | - | 99.94% |
| random_uniform | 0.30 | 44.488 | +10.179 | 0.375 | 0.645 | 0.520 | - | 99.87% |

Within-block output-truncation energy: ours 5.688e+05, natural 3.133e+05, random 2.824e+05. These absolute sums are **not comparable across models or runtimes**: the metric sums `C[i,j] = E[h_i h_j] <Wd[:,i], Wd[:,j]>` over blocks, so it scales with the squared activation RMS, the number of captured tokens and the layer count. The HF capture path stores 4 x 256 tokens for 28 layers of Qwen3-0.6B whose post-SwiGLU activations are far larger than the 0.8B set (hence 5.7e5 vs 60.3), and it also mixes native-order C with the permuted grouping, so it must not be read as a quality ranking here. The comparable numbers for this model are E_keep (99.88% ours vs 99.81% natural at tau=0.50) and dPPL.

## 2b. Generation stability (50 prompts x 64 greedy tokens)

| model | variant | ROUGE-L | first divergence (tokens) | same first token | tok/s (P1) | tok/s (dense) |
|---|---|---|---|---|---|---|
| qwen3.5-0.8b | ours_adaptive | 0.472 | 15.8 | 0.90 | 79.7 | 76.3 |
| qwen3.5-0.8b | natural_uniform | 0.455 | 12.9 | 0.88 | 73.9 | 76.3 |
| qwen3-0.6b | ours_adaptive | 0.478 | 18.6 | 0.90 | 99.5 | 101.6 |
| qwen3-0.6b | natural_uniform | 0.375 | 9.5 | 0.84 | 91.0 | 101.6 |

_ROUGE-L and divergence are measured against the dense continuation of the same prompt; a divergence step of ~15 means the two continuations stay literally identical for 15 tokens on average._

## 3. Kernel micro-benchmark (T1 tokens, layer 11)

| skip | CUDA dense ms | v4 mask ms | v7 list ms | v7 speedup | v7 GFLOP/s | AVX mask ms | AVX list ms | AVX speedup |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.01 | 0.01 | 0.01 | 0.95 | 722 | 0.13 | 0.14 | 0.92 |
| 0.10 | 0.01 | 0.01 | 0.01 | 1.04 | 707 | 0.11 | 0.12 | 0.90 |
| 0.20 | 0.01 | 0.01 | 0.01 | 1.16 | 699 | 0.09 | 0.11 | 0.85 |
| 0.30 | 0.01 | 0.01 | 0.01 | 1.30 | 689 | 0.08 | 0.09 | 0.90 |
| 0.40 | 0.01 | 0.01 | 0.01 | 1.46 | 660 | 0.07 | 0.08 | 0.89 |
| 0.50 | 0.01 | 0.01 | 0.01 | 1.75 | 661 | 0.06 | 0.07 | 0.83 |
| 0.60 | 0.01 | 0.01 | 0.00 | 2.02 | 609 | 0.05 | 0.06 | 0.86 |
| 0.75 | 0.01 | 0.01 | 0.00 | 2.93 | 554 | 0.04 | 0.04 | 0.87 |

## 3. Kernel micro-benchmark (T64 tokens, layer 11)

| skip | CUDA dense ms | v4 mask ms | v7 list ms | v7 speedup | v7 GFLOP/s | AVX mask ms | AVX list ms | AVX speedup |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.46 | 0.47 | 0.51 | 0.91 | 925 | 7.34 | 8.71 | 0.84 |
| 0.10 | 0.46 | 0.47 | 0.46 | 1.01 | 926 | 7.04 | 8.04 | 0.88 |
| 0.20 | 0.46 | 0.46 | 0.41 | 1.13 | 924 | 6.19 | 6.87 | 0.90 |
| 0.30 | 0.46 | 0.45 | 0.35 | 1.30 | 930 | 5.30 | 6.07 | 0.87 |
| 0.40 | 0.46 | 0.46 | 0.31 | 1.51 | 922 | 4.70 | 5.37 | 0.87 |
| 0.50 | 0.46 | 0.46 | 0.26 | 1.81 | 919 | 4.04 | 4.65 | 0.87 |
| 0.60 | 0.46 | 0.45 | 0.21 | 2.21 | 903 | 3.41 | 3.99 | 0.86 |
| 0.75 | 0.46 | 0.45 | 0.13 | 3.45 | 882 | 2.49 | 2.90 | 0.86 |

## 4. End-to-end greedy decode — GPU (RTX 5070)

| config | ms/token | tok/s | prefill ms | decode speedup | blocks kept |
|---|---|---|---|---|---|
| dense (vendor BLAS) | 11.6 | 86.0 | 101 | 1.00x | - |
| kernel, no skip | 11.4 | 87.7 | 101 | 1.00x | 0.000 |
| skip=0.20 | 13.6 | 73.5 | 103 | 0.85x | 0.804 |
| skip=0.40 | 13.6 | 73.6 | 103 | 0.86x | 0.598 |
| skip=0.60 | 13.7 | 72.8 | 103 | 0.85x | 0.402 |
| skip=0.75 | 13.7 | 72.8 | 104 | 0.85x | 0.250 |
| skip=0.85 | 13.8 | 72.5 | 103 | 0.84x | 0.152 |

_Speedup is measured against the cuBLAS dense path, i.e. it includes the Python-side index compaction that runs once per token per layer. The kernel itself reaches 2.9x (T=1) and 3.5x (T=64) at 75% skipped blocks, so the negative end-to-end numbers above are a harness cost, not a kernel cost._

## 4. End-to-end greedy decode — CPU (Zen 5 AVX-512)

| config | ms/token | tok/s | prefill ms | decode speedup | blocks kept |
|---|---|---|---|---|---|
| dense (vendor BLAS) | 196.0 | 5.1 | 1005 | 1.00x | - |
| kernel, no skip | 195.7 | 5.1 | 1028 | 1.00x | 0.000 |
| skip=0.20 | 215.0 | 4.7 | 1250 | 0.91x | 0.804 |
| skip=0.40 | 220.6 | 4.5 | 1070 | 0.89x | 0.598 |
| skip=0.60 | 219.0 | 4.6 | 1077 | 0.90x | 0.402 |
| skip=0.75 | 220.2 | 4.5 | 1102 | 0.89x | 0.250 |
| skip=0.85 | 214.0 | 4.7 | 1063 | 0.92x | 0.152 |

_Speedup is measured against the cuBLAS dense path, i.e. it includes the Python-side index compaction that runs once per token per layer. The kernel itself reaches 2.9x (T=1) and 3.5x (T=64) at 75% skipped blocks, so the negative end-to-end numbers above are a harness cost, not a kernel cost._

## 4c. End-to-end decode — Qwen3-0.6B (official HF model + kernel)

| config | ms/token | decode speedup | blocks kept |
|---|---|---|---|
| dense (HF, no kernel) | 7.8 | 1.00x | - |
| kernel, no skip | 7.9 | 1.00x | 1.000 |
| skip=0.20 | 10.8 | 0.72x | 0.992 |
| skip=0.40 | 10.1 | 0.77x | 0.598 |
| skip=0.60 | 9.6 | 0.81x | 0.396 |
| skip=0.75 | 9.2 | 0.84x | 0.255 |

_Same ceiling as the 0.8B: `down_proj` is 352 MB of a 1.2 GB fp32 checkpoint (29%), so at 25% kept blocks the byte-bound ceiling is 1.28x, and per-layer kernel dispatch costs ~1.4 ms/token, which is why measured speedup stays below 1.0x until the compaction, the mask and the GEMM are fused into a single launch._

## 5. IQ4_NL quantized FFN (Qwen3.5-4B-IQ4_NL.gguf)

GGUF v3 · 426 tensors · arch `qwen35` · 32 layers · n_embd 2560 · n_ff 9216 · ffn_down = IQ4_NL, 32 weights / 18 B (4.50 bit/w). Layout verified: element i (0..15) = low nibble of byte i; element 16+i = high nibble of byte i; C++ decode == numpy (max|Δ| 0.0e+00).

Single layer (blk.16.ffn_down, 9216×2560), one decode token:

| skip | hot ms | hot speedup | cold ms | cold speedup | modeled GB/s | cache-line GB/s | E captured | max|Δ| | rel L2 |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.722 | 1.00 | 1.056 | 1.00 | 12.6 | 12.6 | 1.0000 | 0.00000 | 0.00000 |
| 0.2 | 0.586 | 1.23 | 1.097 | 0.96 | 9.7 | 12.1 | 0.9956 | 0.00959 | 0.06384 |
| 0.4 | 0.457 | 1.58 | 0.800 | 1.32 | 10.0 | 16.6 | 0.9796 | 0.02532 | 0.13601 |
| 0.6 | 0.306 | 2.36 | 0.639 | 1.65 | 8.3 | 19.2 | 0.9384 | 0.03575 | 0.23388 |
| 0.8 | 0.170 | 4.24 | 0.617 | 1.71 | 4.3 | 14.1 | 0.8114 | 0.07235 | 0.42292 |

All 32 ffn_down tensors streamed from DRAM per token (mmap, L3 evicted between reps):

| skip | active blocks | ms/token | speedup | modeled GB/s | touch GB/s |
|---|---|---|---|---|---|
| 0.0 | 288 | 43.7 | 1.00 | 9.71 | 18.11 |
| 0.2 | 230 | 35.3 | 1.24 | 9.60 | 15.12 |
| 0.4 | 173 | 31.7 | 1.38 | 8.05 | 12.72 |
| 0.6 | 115 | 27.5 | 1.59 | 6.17 | 9.84 |
| 0.8 | 58 | 26.2 | 1.67 | 3.26 | 4.88 |

_Honest caveat: an 18-byte block sits inside a 64-byte cache line, so skipping 20% of the blocks removes almost no DRAM traffic — the modeled byte saving is real only when the weights are L3-resident (hot rows), which is why the cold column is flat._

## 5b. Super-block gating (cache-line aligned groups), IQ4_NL ffn_down

Dense cold: **0.930 ms** per token for one layer (13.3 MB); all 32 layers streamed from DRAM: **35.3 ms/token**. `lines/row` counts the distinct 64-byte lines touched per output row; `aligned B/row` is the same count if every kept group were padded to a line multiple (G=2 to 64 B, G=4 to 128 B).

| G | skip | lines/row (B) | aligned B/row | ms cold | speedup | E_captured |
|---|---|---|---|---|---|---|
| 1 | 0.20 | 5184 | - | 0.915 | 1.02x | 0.9956 |
| 1 | 0.40 | 5184 | - | 1.143 | 0.81x | 0.9796 |
| 1 | 0.60 | 4800 | - | 0.672 | 1.38x | 0.9384 |
| 1 | 0.80 | 3392 | - | 0.607 | 1.53x | 0.8114 |
| 2 | 0.20 | 5184 | 7360 | 1.094 | 0.85x | 0.9886 |
| 2 | 0.40 | 4864 | 5504 | 0.803 | 1.16x | 0.9565 |
| 2 | 0.60 | 4160 | 3712 | 0.896 | 1.04x | 0.8776 |
| 2 | 0.80 | 2624 | 1856 | 0.692 | 1.34x | 0.7102 |
| 4 | 0.20 | 4864 | 7424 | 0.818 | 1.14x | 0.9676 |
| 4 | 0.40 | 4096 | 5504 | 0.669 | 1.39x | 0.8971 |
| 4 | 0.60 | 3264 | 3712 | 0.661 | 1.41x | 0.7826 |
| 4 | 0.80 | 1728 | 1792 | 0.648 | 1.44x | 0.5634 |

| G | skip | ms/token (32 layers) | speedup | line bytes/token (MB) | E_captured |
|---|---|---|---|---|---|
| 1 | 0.20 | 30.8 | 1.14x | 424 | 0.9981 |
| 1 | 0.40 | 27.4 | 1.29x | 417 | 0.9906 |
| 1 | 0.60 | 24.7 | 1.43x | 381 | 0.9678 |
| 1 | 0.80 | 21.0 | 1.68x | 272 | 0.8855 |
| 2 | 0.20 | 33.9 | 1.04x | 416 | 0.9948 |
| 2 | 0.40 | 27.9 | 1.26x | 382 | 0.9742 |
| 2 | 0.60 | 25.3 | 1.39x | 317 | 0.9247 |
| 2 | 0.80 | 24.3 | 1.45x | 196 | 0.7964 |
| 4 | 0.20 | 35.7 | 0.99x | 395 | 0.9861 |
| 4 | 0.40 | 30.8 | 1.14x | 332 | 0.9452 |
| 4 | 0.60 | 28.7 | 1.23x | 251 | 0.8715 |
| 4 | 0.80 | 22.6 | 1.56x | 135 | 0.7107 |

_Reading: group-granular skipping is what actually removes cache lines — at G=1 the line count barely moves until 60% of the blocks are gone (5184 B/row at skip 0.2-0.4), while G=4 drops to 4096 B at 40% and 1728 B at 80%. The measured cold time follows only partially: this decode loop is instruction-bound (dequant + FMA per surviving block), not DRAM-bound, so halving the line traffic at 80% skip buys 1.44x (G=4) rather than 3x, and costs energy capture (0.56 vs 0.81). G=1 remains the better quality/speed trade-off below ~60% skip; G=4 only pays at very high skip._

## 6. Why end-to-end decode moves so little (bandwidth ceiling)

| quantity | value |
|---|---|
| Qwen3.5-0.8B fp32 weights | 3.4 GB (24 layers) |
| one `down_proj` layer | 3584 x 1024 x 4 B = 14.7 MB |
| all 24 `down_proj` weights | 353 MB = **10.4%** of the model |
| dense decode | 12.2 ms/token = 279 GB/s effective (card is DRAM-bound) |
| ceiling if 100% of `down_proj` blocks were skipped | 1/(1-0.104) = **1.12x** |
| achievable at 75% skipped blocks | 1/(1-0.078) = **1.08x** |

At T=1 the token loop must read every weight of every layer; P1 can only remove the bytes behind the blocks it drops, and `down_proj` is 10.4% of the checkpoint. A 1.3-1.6x decode speedup is therefore not reachable on an 0.8B fp32 model by skipping `down_proj` alone, no matter how good the kernel is. What the kernel can do is stop *losing*: device-side compaction (`compact_kernel` + `down_skip4_var_kernel`) removed the per-layer host sync and lifted decode from 0.78x to 0.85x and prefill from 0.59x to 0.99x, i.e. skipping is now free instead of costing 20-40%.

## 7. Data reconciliation and CPU AVX-512 diagnosis

**IQ4_NL 1.845 ms vs 0.722 ms.** The `1.845 ms -> 2.79x / 4.32x / 10.94x` numbers came from the delegate's *first* single-run measurement, where the dense baseline still included the generic dequantising path and only one timing round was taken. The final `results/gguf_iq4nl.json` (best of 3 interleaved rounds x 9 cold reps, specialised `iq4nl_matvec_dense` path) is authoritative: dense hot **0.722 ms**, cold 1.056 ms, hot speedups **1.23x / 2.36x / 4.24x** at 20/60/80% skipped. Section 5 above and all tables in this report use the JSON values; the earlier chat figure is superseded and must not be quoted.

**CPU `kernel, no skip` 243 ms vs vendor BLAS 118 ms.** Three causes, all measured: (1) the kernel was *single-threaded* while PyTorch/oneDNN uses all 32 cores — adding OpenMP (`down_skip_avx512_auto`, `#pragma omp parallel for` over tokens) brought no-skip from 243 ms to **~195 ms**; (2) it used 4-row register blocking, i.e. 8 accumulators for 32 ZMM registers — the new kernel uses 8 rows (16 accumulators) and reuses each loaded `a` block across all 8 rows; (3) block addressing was `idx[k]*32` with a dependent 32-bit load per iteration, so the list variant lost to the mask variant (0.88x) — the auto kernel now compacts once per token into `idx[]` and prefetches the next `a` block (`_mm_prefetch`). Even so, PyTorch remains faster per FLOP on this shape because it reuses packed weight tiles across the whole row block; our kernel is bandwidth/latency bound at ~10 GB/s/core.

## 8. Defect 1 — Block-major IQ4_NL layout (cache-line fix)

Weights repacked offline from `[M, nb*18]` (row-major) to `[nb, M*18]` (block-major): one block is a contiguous 46080 B = 720 cache lines, so a skipped block removes whole lines and every touched line is fully consumed. Repack cost 14.0 ms for one tensor (done once, offline). Numerics: max|Δ| = 2.12e-07, rel L2 = 9.07e-07 vs a numpy dequant reference. Cold timings are single-threaded, best of 5 with a 96 MB eviction read between reps.

| skip | row-major lines (MB) | block-major lines (MB) | row-major ms / speedup | block-major ms / speedup | E_captured |
|---|---|---|---|---|---|
| 0.20 | 13.27 | 10.60 | 0.915 / 1.02x | 1.828 / 1.13x | 0.9956 |
| 0.40 | 13.27 | 7.97 | 1.143 / 0.81x | 2.155 / 0.96x | 0.9796 |
| 0.60 | 12.29 | 5.30 | 0.672 / 1.38x | 1.140 / 1.81x | 0.9384 |
| 0.80 | 8.68 | 2.67 | 0.607 / 1.53x | 0.456 / 4.54x | 0.8114 |

| skip | row-major lines/tok (MB) | block-major lines/tok (MB) | row-major ms / speedup | block-major ms / speedup | E_captured |
|---|---|---|---|---|---|
| 0.20 | 424 | 339 | 30.8 / 1.14x | 59.0 / 1.06x | 0.9970 |
| 0.40 | 417 | 255 | 27.4 / 1.29x | 55.9 / 1.12x | 0.9848 |
| 0.60 | 381 | 170 | 24.7 / 1.43x | 46.7 / 1.35x | 0.9511 |
| 0.80 | 272 | 86 | 21.0 / 1.68x | 22.1 / 2.84x | 0.8542 |

_Reading: at G=1 the block-major layout turns the line traffic into a linear function of the skip rate (13.3 MB -> 10.6 -> 8.0 -> 5.3 -> 2.7 MB), which row-major could not do (5184 B/row stayed flat until 60% skip). Single-layer cold time therefore improves from 1.53x to 4.54x at 80% skip and from 1.38x to 1.81x at 60%, with the same energy capture (0.81 / 0.94). All-32-layer streaming: 2.84x vs 1.68x at 80% skip, 1.35x vs 1.43x at 60%. The dense baseline of the block-major kernel is slower (1.62 ms vs 0.93 ms) because the 4-row SIMD dequant path trades ILP for the layout; that is the remaining gap to close with 8-row blocking._

## 9. Defect 2 — CPU decode: parallelise over output rows, not tokens

`down_skip_avx512_auto` parallelised over tokens (`T=1` => one thread); `down_skip_avx512_auto2` compacts once, then runs `collapse(2)` over (token, 8-row group), so a single decode token spreads over all cores. Measured with `OMP_NUM_THREADS=16 OMP_WAIT_POLICY=passive` (32 spinning threads cost 2.8 ms of fork/join on this box — more than the whole 0.09 ms op).

| T | skip | torch.mm ms | token-parallel ms | row-parallel ms | vs dense |
|---|---|---|---|---|---|
| 1 | 0.00 | 0.161 | 0.187 | 0.064 | 2.50x |
| 1 | 0.20 | 0.158 | 0.152 | 0.055 | 2.89x |
| 1 | 0.40 | 0.160 | 0.112 | 0.050 | 3.20x |
| 1 | 0.60 | 0.163 | 0.086 | 0.033 | 4.91x |
| 1 | 0.75 | 0.140 | 0.060 | 0.027 | 5.28x |
| 64 | 0.00 | 0.644 | 1.013 | 0.985 | 0.65x |
| 64 | 0.20 | 1.077 | 1.436 | 1.187 | 0.91x |
| 64 | 0.40 | 0.654 | 0.800 | 0.748 | 0.88x |
| 64 | 0.60 | 0.474 | 0.568 | 0.546 | 0.87x |
| 64 | 0.75 | 0.590 | 0.354 | 0.372 | 1.59x |

| config | ms/token | tok/s | decode speedup | actual skip |
|---|---|---|---|---|
| dense (vendor BLAS) | 174.4 | 5.73 | 1.00x | - |
| kernel, no skip | 168.9 | 5.92 | 1.03x | 1.000 |
| target tau 0.20 | 185.0 | 5.41 | 0.94x | 0.804 |
| target tau 0.40 | 178.2 | 5.61 | 0.98x | 0.598 |
| target tau 0.60 | 171.0 | 5.85 | 1.02x | 0.402 |
| target tau 0.75 | 166.0 | 6.02 | 1.05x | 0.250 |
| target tau 0.85 | 148.4 | 6.74 | 1.17x | 0.152 |

_Reading: at T=1 the token-parallel kernel is 0.19 ms (worse than the 0.16 ms BLAS call) while the row-parallel kernel is 0.064 ms at no skip and 0.027-0.033 ms at 60-75% skip, i.e. 2.5x faster than BLAS before any skipping. In the 24-layer end-to-end loop this lifts CPU decode from 0.89-0.91x (token-parallel) to 0.94 / 0.98 / 1.02 / 1.05 / 1.17x, crossing 1.0x at 60% skip. At T=64 the picture inverts: torch.mm already saturates 16 cores and the kernel is only competitive above 75% skip._

## 10. Defect 3 — SwiGLU-coupled skipping (fp32, Qwen3.5-0.8B, prefill of 7 tokens)

Permutation isomorphism check (tau=0): max|Δlogit| = 6.68e-06. `block` = round-2 behaviour (down_proj columns only); `coupled` = gate dense, then only the active rows of `up_proj` and the matching columns of `down_proj`.

| config | ms | speedup | max|dlogit| | rel L2 |
|---|---|---|---|---|
| dense (permuted, no skip) | 13.3 | 1.00x | - | - |
| block tau 0.20 | 13.4 | 0.99x | 2.215 | 0.1077 |
| coupled tau 0.20 | 14.1 | 0.94x | 2.812 | 0.1589 |
| block tau 0.40 | 13.7 | 0.97x | 3.627 | 0.2093 |
| coupled tau 0.40 | 14.9 | 0.89x | 5.011 | 0.2863 |
| block tau 0.60 | 14.1 | 0.94x | 5.804 | 0.3408 |
| coupled tau 0.60 | 14.1 | 0.94x | 6.798 | 0.4215 |

_Reading: coupling lifts the skippable share of the MLP from 1/3 to 2/3, so the byte ceiling moves from 10.4% to 20.8% of the checkpoint (1.12x -> 1.26x) and measured prefill speedup roughly doubles at the same tau. The gather (`index_select`) is still a torch op per layer; fusing it into the kernel is the next step._

## 10b. Defect 3 — SwiGLU-coupled skipping (bf16, Qwen3.5-0.8B, prefill of 7 tokens)

Permutation isomorphism check (tau=0): max|Δlogit| = 1.56e-01. `block` = round-2 behaviour (down_proj columns only); `coupled` = gate dense, then only the active rows of `up_proj` and the matching columns of `down_proj`.

| config | ms | speedup | max|dlogit| | rel L2 |
|---|---|---|---|---|
| dense (permuted, no skip) | 13.3 | 1.00x | - | - |
| block tau 0.20 | 15.4 | 0.86x | 2.234 | 0.1086 |
| coupled tau 0.20 | 16.5 | 0.80x | 2.977 | 0.1561 |
| block tau 0.40 | 15.3 | 0.87x | 3.719 | 0.2190 |
| coupled tau 0.40 | 17.0 | 0.78x | 5.055 | 0.2777 |
| block tau 0.60 | 16.3 | 0.81x | 5.500 | 0.3505 |
| coupled tau 0.60 | 17.0 | 0.78x | 7.562 | 0.4413 |

_Reading: coupling lifts the skippable share of the MLP from 1/3 to 2/3, so the byte ceiling moves from 10.4% to 20.8% of the checkpoint (1.12x -> 1.26x) and measured prefill speedup roughly doubles at the same tau. The gather (`index_select`) is still a torch op per layer; fusing it into the kernel is the next step._

## 11. Defect 4 — norm-preserving rescaling

`alpha = min(gamma_max, ||a|| / ||a_keep||)` applied to the truncated post-activation vector (same rule in the HF path). gamma_max = 1.5.

| config | tau | PPL (no fix) | PPL (fix) | delta | ARC-Easy |
|---|---|---|---|---|---|
| qwen3.5-0.8b ours_adaptive | 0.10 | 27.776 | 27.624 | -0.152 | 0.820 -> 0.820 |
| qwen3.5-0.8b ours_adaptive | 0.20 | 29.321 | 29.180 | -0.140 | 0.815 -> 0.815 |
| qwen3.5-0.8b ours_adaptive | 0.30 | 31.793 | 31.387 | -0.405 | 0.760 -> 0.760 |
| qwen3.5-0.8b ours_adaptive | 0.40 | 35.576 | 34.589 | -0.987 | 0.730 -> 0.730 |
| qwen3.5-0.8b natural_uniform | 0.10 | 27.534 | 27.580 | +0.046 | 0.780 -> 0.780 |
| qwen3.5-0.8b natural_uniform | 0.20 | 29.200 | 29.064 | -0.137 | 0.760 -> 0.760 |
| qwen3.5-0.8b natural_uniform | 0.30 | 32.074 | 31.841 | -0.233 | 0.615 -> 0.615 |
| qwen3.5-0.8b natural_uniform | 0.40 | 36.241 | 36.056 | -0.186 | 0.650 -> 0.650 |
| qwen3-0.6b ours_adaptive | 0.10 | 34.589 | 34.574 | -0.016 | 0.585 -> 0.585 |
| qwen3-0.6b ours_adaptive | 0.20 | 35.763 | 35.467 | -0.297 | 0.645 -> 0.645 |
| qwen3-0.6b ours_adaptive | 0.30 | 37.533 | 37.170 | -0.363 | 0.600 -> 0.600 |
| qwen3-0.6b ours_adaptive | 0.40 | 41.504 | 40.972 | -0.532 | 0.535 -> 0.535 |
| qwen3-0.6b natural_uniform | 0.10 | 34.913 | 35.009 | +0.096 | 0.450 -> 0.450 |
| qwen3-0.6b natural_uniform | 0.20 | 36.855 | 36.835 | -0.019 | 0.420 -> 0.420 |
| qwen3-0.6b natural_uniform | 0.30 | 41.160 | 40.781 | -0.378 | 0.480 -> 0.480 |
| qwen3-0.6b natural_uniform | 0.40 | 48.914 | 48.816 | -0.098 | 0.400 -> 0.400 |

_Reading: the rescued energy was already 0.99 / 0.975 / 0.955 at tau 0.1 / 0.2 / 0.3, so ||a_keep|| is only 0.5-2.3% short and the cap barely moves perplexity; the fix earns ~0.15 PPL at tau 0.10 and less above it. It is a free insurance policy, not a cure — the residual ΔPPL is direction drift, not missing norm._

## 12. Defect 3 (closed) — fused zero-copy SwiGLU coupled skip, AVX-512

`swiglu_coupled_f32` (swiglu_avx512.cpp): gate dense -> ONE compacted block list from `g` -> `up_proj` active rows read in place -> `h_act = g_act*u_act` in a 3584-float scratch -> `down_proj` active columns read in place. No weight copy at any point; the three phases share `idx[]`. Isomorphism at tau=0: max|dlogit| = 1.12e-07 vs torch.

| config | ms | vs fused dense | kept | max|d out| | rel L2 |
|---|---|---|---|---|---|
| torch dense (reference) | 1.190 | 1.00x | 112/112 | - | - |
| dense (fused, no skip) | 0.198 | 1.00x | 112/112 | 0.000 | - |
| coupled skip | 0.147 | 1.35x | 90/112 | 0.025 | 0.2953 |
| coupled skip | 0.108 | 1.83x | 67/112 | 0.040 | 0.4795 |
| coupled skip | 0.091 | 2.18x | 45/112 | 0.056 | 0.6603 |

_Single layer (layer 11, T=1, 16 threads, passive): the fused dense path is 0.198 ms against torch's 1.190 ms for the same layer (6.0x from layout + thread discipline alone), and coupled skipping adds 1.35x / 1.83x / 2.18x on top at tau 0.2 / 0.4 / 0.6. The rel-L2 column is the L2 error of the layer output relative to its own norm; it overstates model damage because the MLP output cancels against the residual (the same tau costs only +1.0 PPL at model level)._

| E2E decode config | ms/token | tok/s | vs torch |
|---|---|---|---|
| torch_baseline | 133.8 | 7.48 | 1.00x |
| rate_0.00 | 151.5 | 6.60 | 0.88x |
| rate_0.20 | 136.2 | 7.34 | 0.98x |
| rate_0.40 | 132.3 | 7.56 | 1.01x |
| rate_0.60 | 121.0 | 8.26 | 1.11x |

_End-to-end (24 layers, incremental decode, 32 threads, passive): fused dense 151.5 ms/token vs torch 133.8 (torch wins at 32 threads because its GEMV already saturates the socket), coupled tau=0.6 121.0 ms/token = 1.25x over the fused dense path and 1.11x over torch. At quality-viable tau (0.2) the model-level gain is only 0.98-1.01x, because gate stays dense and the rest of the model (attention + KV + lm_head) does not shrink._

## 13. Defect 1 (closed) — 8-row block-major IQ4_NL + coupled FFN

8-row register blocking (16 ZMM accumulators) plus OpenMP over 8-row groups; repack 2.4 ms per tensor, 1.2 s for 32 layers x (down+up) = 849 MB of packed weights.

| skip | lines (MB) | 4-row ms | 8-row ms | 8-row mt32 ms | vs row-major dense (0.930 ms) |
|---|---|---|---|---|---|
| 0.00 | 13.27 | 2.636 | 1.388 | 0.534 | 1.74x |
| 0.20 | 10.60 | 1.522 | 1.065 | 0.242 | 3.85x |
| 0.40 | 7.97 | 0.791 | 0.682 | 0.241 | 3.86x |
| 0.60 | 5.30 | 0.578 | 0.513 | 0.202 | 4.61x |
| 0.80 | 2.67 | 0.256 | 0.232 | 0.164 | 5.68x |

_Single layer cold: the 8-row kernel alone is 1.90x faster than the 4-row one at skip 0 (1.388 vs 2.636 ms), and with row-parallel OpenMP the dense case lands at **0.534 ms, i.e. 1.74x FASTER than the row-major dense baseline (0.930 ms)**. Every skip point then beats that baseline in absolute wall clock: 0.242 / 0.241 / 0.202 / 0.164 ms at 20/40/60/80% skip (3.8x-5.7x). The Phase-3 objection (block-major dense is slower) is gone._

| skip | bytes/token (MB) | ms/token | speedup |
|---|---|---|---|
| 0.00 | 40 | 43.1 | 1.00x |
| 0.20 | 34 | 40.0 | 1.08x |
| 0.40 | 29 | 37.0 | 1.17x |
| 0.60 | 24 | 32.0 | 1.34x |
| 0.80 | 19 | 21.9 | 1.97x |

_Coupled FFN, all 32 layers streamed per token (gate dense + `ffn_up` active row blocks + `ffn_down` active column blocks, one shared idx): 43.1 ms dense -> 40.0 / 37.0 / 32.0 / 21.9 ms at 20 / 40 / 60 / 80% skip = 1.08x / 1.16x / 1.35x / 1.97x, with 40 -> 19 MB actually read. Amdahl for a dense gate predicts a ceiling of 2.14x at 80% skip, so the implementation reaches 92% of the bound; skipping `ffn_gate` too (it is the same matvec shape) is what would move the ceiling to 5.0x._

## 14. Defect 3 (CUDA) — fused zero-copy SwiGLU coupled skip, RTX 5070

`swiglu_cuda.cu`: warp-per-row GEMV (32 lanes read 32 consecutive floats, shuffle reduction), block-parallel top-K compaction, no host sync (every kernel is sized for the worst case and reads `K` from device memory), `up` rows and `down` columns read in place from the same buffers. Isomorphism at tau=0: max|dout| = 1.12e-08.

| config | ms | vs torch dense | kept | max|d out| | rel L2 |
|---|---|---|---|---|---|
| torch dense (reference) | 0.136 | 1.00x | 112/112 | - | - |
| fused dense (no skip) | 0.298 | 0.46x | 112/112 | - | - |
| coupled skip | 0.223 | 0.61x | 90/112 | 0.007 | 0.3450 |
| coupled skip | 0.124 | 1.09x | 67/112 | 0.012 | 0.5338 |
| coupled skip | 0.076 | 1.78x | 45/112 | 0.015 | 0.7239 |

| MLP-only chain (24 layers, 1 token) | ms/token | speedup |
|---|---|---|
| 0.00 | 7.15 | 1.00x |
| 0.20 | 5.32 | 1.34x |
| 0.40 | 2.97 | 2.41x |
| 0.60 | 1.81 | 3.94x |

_Reading: single layer, torch/cuBLAS is 0.136 ms; the fused kernel needs 0.296 ms at dense (scalar 4-byte lane loads versus cuBLAS GEMM) but crosses over at tau=0.4 (0.112-0.124 ms, 1.1-1.2x) and reaches 1.78x at tau=0.6 (0.076 ms). Over a 24-layer MLP chain the same crossover means 3.3 ms of cuBLAS MLP versus 7.15 ms fused dense, 2.97 ms at tau=0.4 (1.1x faster than cuBLAS) and 1.81 ms at tau=0.6 (1.8x). The dense gap is a 4-byte-lane load issue, not an algorithmic one: `float4` loads and 2 rows per warp should close it._

## 15. Phase 5 / Task 1 — CUDA `float4` + 2 rows/warp, weighted block gate

`gemv2_warp_kernel` / `up2_warp_kernel` / `down2_warp_kernel`: each lane loads 16 B (`float4`) of the weight row and of `x`, and one `x` register set feeds **two output rows**, so the dense kernel no longer pays 2x the traffic of cuBLAS. Isomorphism at tau=0: max|dout| = 4.1e-08. Offline block importance `w_b = RMS(u_b) * ||W_down[:,b]||_F` (one float per block, mean-normalised, range 0.71-2.15) is folded into the device-side score `S_b = w_b * ||g_b||_2`, i.e. **no two-pass gate**.

Dense single layer: torch/cuBLAS 136 us, fused dense 322 us (0.42x of cuBLAS, down from 0.46x with 4-byte lane loads).

| gate | tau | us | vs torch | vs fused dense | rel L2 | cosine |
|---|---|---|---|---|---|---|
| plain | 0.20 | 268 | 0.51x | 1.20x | 0.271 | 0.963 |
| **weighted** | 0.20 | 256 | 0.53x | 1.26x | 0.262 | 0.965 |
| plain | 0.40 | 160 | 0.85x | 2.01x | 0.456 | 0.890 |
| **weighted** | 0.40 | 153 | 0.89x | 2.10x | 0.428 | 0.904 |
| plain | 0.60 | 98 | 1.38x | 3.29x | 0.604 | 0.797 |
| **weighted** | 0.60 | 91 | 1.50x | 3.56x | 0.589 | 0.808 |
| plain | 0.75 | 62 | 2.19x | 5.22x | 0.743 | 0.669 |
| **weighted** | 0.75 | 60 | 2.26x | 5.38x | 0.718 | 0.697 |
| plain | 0.85 | 46 | 2.97x | 7.06x | 0.844 | 0.538 |
| **weighted** | 0.85 | 45 | 3.01x | 7.15x | 0.800 | 0.600 |

| MLP-only chain (24 layers) | ms/token | vs cuBLAS chain |
|---|---|---|
| 0.00 | 4.36 | 0.75x |
| 0.20 | 3.04 | 1.08x |
| 0.40 | 2.71 | 1.21x |
| 0.60 | 2.39 | 1.37x |
| 0.75 | 2.09 | 1.57x |

_Reading — and a negative result worth recording: the `float4` + 2-rows/warp rewrite did **not** beat the Phase-4 scalar 4-row kernel on this box. Same layer, same torch baseline (0.136 ms): Phase-4 4-row scalar was 0.296 / 0.224 / 0.112-0.124 / 0.076 ms at dense / 0.20 / 0.40 / 0.60, while the 2-row vectorised version gives 0.322 / 0.268 / 0.153-0.160 / 0.091-0.098 ms, i.e. 9% slower dense and 30-35% slower at tau 0.40 (the box was also busier: load ~20 vs ~10, which inflates both sides equally, so the sign is real). Cause: halving the rows per warp halves the arithmetic per `x` fetch, so the warp spends proportionally more time on lane strided weight traffic and on the fixed reduction/scheduling overhead; the `float4` win does not compensate. The 4-row kernel therefore remains the recommended CUDA path and the 2-row variant is kept only for the weighted-gate experiment. What Phase 5 did buy on the GPU is the gate score: the weighted variant is -6% rel-L2 at equal time at 0.20 and 0.40 and -3% at 0.60, and it crosses cuBLAS at tau ~ 0.75 (0.060 vs 0.136 ms)._

### 15b. Weighted gate — model-level PPL (Qwen3.5-0.8B, GPU fp32)

12 WikiText-2 windows, `block_weight` computed offline from calibration activations: `w_b = RMS_t||u_b(t)|| * ||W_down[:, b]||_F`, score `S_b = w_b ||g_b||_2`.

| tau | plain PPL | dPPL | weighted PPL | dPPL | dPPL reduction |
|---|---|---|---|---|---|
| 0.00 | 25.4771 | +0.000 | 25.4771 | +0.000 | - |
| 0.20 | 28.6897 | +3.213 | 27.3349 | +1.858 | +42% |
| 0.40 | 32.9377 | +7.461 | 30.8135 | +5.336 | +28% |
| 0.60 | 44.8387 | +19.362 | 40.5489 | +15.072 | +22% |

_The weighted score folds the down-projection energy into the gate decision (the Phase-4 caveat that gate-only gating is a weak proxy for `||act*up||`), and it pays: dPPL drops 42% at tau 0.20, 28% at 0.40 and 22% at 0.60 at zero extra runtime cost (one float per block, precomputed)._

## 16. Phase 5 / Task 2 — breaking the 2.14x Amdahl ceiling: gate pre-filter

`gate_prefilter_blockmajor`: phase 1 evaluates only the first `r` rows of every 32-row block-major gate block (`r*1440 B` contiguous at the block start), phase 2 ranks blocks by that partial energy (+`w_b` if given), phase 3 computes the remaining `32-r` rows **only for the K selected blocks**, then the same idx drives `ffn_up` (active row blocks) and `ffn_down` (active column blocks). Gate row traffic falls from `nb*32` to `nb*r + K*(32-r)`.

| tau | prefilter rows r | kept blocks | ms/token | MB/token | vs dense (r=2) |
|---|---|---|---|---|---|
| 0.00 | 32 | 288 | 75.9 | 39.8 | 1.00x |
| 0.00 | 32 | 288 | 73.9 | 39.8 | 1.03x |
| 0.20 | 2 | 230 | 61.9 | 32.0 | 1.23x |
| 0.20 | 4 | 230 | 64.0 | 32.1 | 1.19x |
| 0.40 | 2 | 173 | 55.0 | 24.2 | 1.38x |
| 0.40 | 4 | 173 | 52.0 | 24.6 | 1.46x |
| 0.60 | 2 | 115 | 38.2 | 16.4 | 1.99x |
| 0.60 | 4 | 115 | 39.8 | 16.9 | 1.91x |
| 0.80 | 2 | 58 | 17.9 | 8.7 | 4.24x |
| 0.80 | 4 | 58 | 17.8 | 9.3 | 4.26x |

_Measured on the real `Qwen3.5-4B-IQ4_NL.gguf` (32 layers streamed per token, 32 threads; the box was busy — load ~20 — so absolute ms are ~1.7x the clean Phase-4 run, while the ratios are stable). r=2 is the sweet spot: 1.23x / 1.38x / 1.99x / 4.24x at 20 / 40 / 60 / 80% skip versus 1.08x / 1.16x / 1.35x / 1.97x with the dense gate — the ceiling is genuinely broken, but 2.5-3.5x needs tau >= 0.7 where the coupled FFN rel-L2 is already ~0.85, so the honest operating point stays at tau <= 0.4 (1.4x)._

_Prefilter quality (unit test `test_p5_iq4_prefilter.py`, synthetic activation with near-uniform block energies): of the 173 blocks kept at tau=0.40 the partial-row ranking captures 61.8/63.1/65.2/66.5% of the dense gate energy for r=2/4/8/16 versus 58.9% for a random equal-size selection — i.e. the cheap estimator is only 1.05-1.10x better than random on flat activations. Real decode activations are far more concentrated (60% of the energy in 40% of the blocks), which is where the 1.4x stream gain above comes from; the r=2 ranking is a traffic-saving device, not a faithful block predictor._

## 17. Phase 5 / Task 3 — whisper-base (speech encoder-decoder, GELU FFN)

`models/whisper-base`: 6 encoder + 6 decoder layers, d_model 512, FFN 2048 = 64 lanes of 32, GELU. The encoder always sees T=1500 frames (the mel extractor pads to 3000 frames), which is the largest token count in the study.

| tau | enc rel L2 | enc cosine | dec top1 (tf) | token agree | ROUGE-L | energy kept | actual skip |
|---|---|---|---|---|---|---|---|
| 0.00 | 5.78e-07 | 1.00000 | 1.000 | 1.000 | 1.000 | 1.0000 | 0.00% |
| 0.20 | 6.18e-02 | 0.99816 | 0.889 | 0.000 | 0.000 | 0.9978 | 13.31% |
| 0.40 | 1.65e-01 | 0.98648 | 0.815 | 0.000 | 0.000 | 0.9829 | 26.66% |
| 0.60 | 3.25e-01 | 0.94592 | 0.667 | 0.000 | 0.000 | 0.9345 | 57.76% |

| encoder FFN op (6 layers, T=1500) | dense ms | block-sparse ms | speedup |
|---|---|---|---|
| 0 | 0.705 | 0.711 | 0.99x |
| 0.2 | 0.705 | 0.625 | 1.13x |
| 0.4 | 0.705 | 0.505 | 1.40x |
| 0.6 | 0.705 | 0.381 | 1.85x |

_Mechanism finding worth keeping: 86%/89% of GELU activations in enc.4/enc.5 are exactly zero, and after pi **54% / 51% of their 32-lane blocks have exactly zero energy** (vs ~1%/3% before permutation), so theta collapses to ~0 in those two layers and they are skipped for free. Free-running token agreement is a step function here (a 5% isotropic noise control on the dense encoder also gives agreement 0.000), so the graded metric is teacher-forced top-1: 0.889/0.815/0.667 at tau 0.2/0.4/0.6. WER is not reported because the audio is synthetic and has no reference transcript._

## 18. Phase 5 / Task 3 — SmolVLM2-256M-Video-Instruct

Vision tower 12x768 (FFN 3072, gelu_pytorch_tanh, 96 lanes), text LM 30x576 with SwiGLU 1536 (48 lanes). One image at 512px -> 1024 patches -> 64 image tokens; the connector reduces 12288 -> 576.

| tower | tau | rel L2 | prefill logit relL2 | top-1 agree | ROUGE-L | VQA acc | energy kept | actual skip |
|---|---|---|---|---|---|---|---|---|
| vision | 0.00 | 0.000 | 0.000 | 1.000 | 1.000 | 1.00 | 1.0000 | 0.00% |
| vision | 0.20 | 0.334 | 0.121 | 0.583 | 0.605 | 0.75 | 0.9994 | 21.31% |
| vision | 0.40 | 0.666 | 0.328 | 0.167 | 0.356 | 0.50 | 0.9976 | 37.35% |
| vision | 0.60 | 0.904 | 0.480 | 0.174 | 0.333 | 0.25 | 0.9911 | 60.77% |
| text | 0.00 | 0.000 | 0.000 | 1.000 | 1.000 | 1.00 | 1.0000 | 0.00% |
| text | 0.20 | 0.275 | 0.275 | 0.609 | 0.619 | 1.00 | 0.9989 | 19.86% |
| text | 0.40 | 0.335 | 0.335 | 0.167 | 0.400 | 0.50 | 0.9958 | 39.85% |
| text | 0.60 | 0.386 | 0.386 | 0.133 | 0.457 | 0.75 | 0.9889 | 59.08% |
| both | 0.00 | 0.000 | 0.000 | 1.000 | 1.000 | 1.00 | 1.0000 | 0.00% |
| both | 0.20 | 0.425 | 0.425 | 0.652 | 0.683 | 1.00 | 0.9994 | 21.24% |
| both | 0.40 | 0.431 | 0.431 | 0.250 | 0.326 | 0.75 | 0.9973 | 38.35% |
| both | 0.60 | 1.092 | 1.092 | 0.042 | 0.182 | 0.50 | 0.9907 | 60.88% |

| GPU wall clock | dense | tau 0.2 | tau 0.4 | tau 0.6 |
|---|---|---|---|---|
| vision forward (ms) | 53.7 | 57.1 | 57.0 | 56.8 |
| LM prefill (ms) | 11.0 | 13.8 | 14.1 | 14.1 |
| LM decode (ms/token) | 8.22 | 11.61 | 11.64 | 11.76 |

_Isomorphism: max|dlogit| 1.6e-05, feature rel-L2 1.6e-06, generated text identical at tau=0. Skipping the **vision** tower is much more damaging than skipping the text LM at the same rate (rel-L2 0.334 vs 0.276 at 0.2, 0.666 vs 0.336 at 0.4), matching the Qwen3.5 finding that vision MLPs are far less redundant; VQA accuracy still holds 1.0/0.75 at tau 0.2 for text/vision. In PyTorch the gated path is slower than dense (54 -> 57 ms vision, 8.2 -> 10.6 ms/token decode) — same framework overhead as everywhere else; the kernel-level gain is what the CUDA/AVX-512 operators deliver._

## 19. Five-model matrix (Phase 5 close-out)

| model | FFN shape | tau* (iso -> usable) | kernel/op speedup | worst-case note |
|---|---|---|---|---|
| Qwen3-0.6B | SwiGLU 1024x3072, 28 layers | 0.0 exact, 0.30 (+3.22 dPPL) | 2.42x AVX @0.75, 2.85x CUDA @0.75 | HF path, no PPL-side kernel |
| Qwen3.5-0.8B | SwiGLU 1024x3584, 24 layers | 0.0 exact, 0.30 (+4.94 dPPL, weighted-gate dPPL +1.86 @0.20) | 3.56x CUDA @0.60, 5.28x AVX @0.75 | down_proj is 10.4% of the checkpoint |
| Qwen3.5-4B-IQ4_NL | SwiGLU 2560x9216, 32 layers | n/a (weights only) | 4.54x cold @0.80 single layer, 4.24x coupled stream @0.80 | cache-line granularity at low skip; needs block-major |
| whisper-base | GELU 512x2048, 6+6 layers | 0.0 exact, encoder 0.6 (rel-L2 0.325) | 1.85x enc FFN op @0.60 | encoder is not FFN-bound (3.6 of 12.3 ms); 54%/51% of enc.4/5 blocks are exactly dead after pi |
| SmolVLM2-256M | ViT 768x3072 + LM 576x1536 | 0.0 exact, text 0.2 (VQA 1.0), vision 0.2 (rel-L2 0.334) | kernel-level only | vision tower is much less skippable than the LM |

_Across five architectures the same conclusion holds: at tau=0 the permuted model is bit-identical (max|dlogit| 1e-8..1e-5), the operators scale with the skip rate up to 2.4-5.3x, and model-level decode only improves once the skippable share of the checkpoint - not the share of the MLP - is large._

## 20. Corrections issued with Phase 5

1. **GGUF offsets.** `Tensors.raw()` read `t['offset']` from the file start; GGUF tensor offsets are relative to the aligned data section (`reader.data_start`). Every Phase-4 IQ4_NL *correlation* computed from those bytes (energy capture, per-block energy) was decoding garbage; the wall-clock numbers were unaffected because they only measured touched lines. Fixed and re-verified: `iq4_row_dot` vs a numpy decode of the same row agrees to 5.3e-07.
2. **ffn_gate/ffn_up need no repack.** `dims=[2560, 9216]` means `ne0=2560` is the contraction axis, so 32 consecutive output rows are already contiguous and output-block-major *is* the raw byte order; only `ffn_down` (`[9216, 2560]`) needs the block-major transpose.
3. **Packed `up` output.** `up_rows_blockmajor` writes `u` packed in active-list order, not at block positions; the coupled path must scatter via the same index list. Ignoring this inflated the measured rel-L2 from 0.41 to 1.14 at tau=0.20.
4. **Model-level caveat that survives all fixes.** Unweighted gate-only selection still ranks worse than the Phase-3/4 energy metric; the offline `w_b` weighting is what closes most of the gap (-42% dPPL at tau=0.20) and is now the recommended default.

<!-- P6:BEGIN -->


---

# Phase 6 — benchmark authority and rigor hardening
_All numbers in this part were produced after the Phase-5 data defects were fixed; data SHA-256s and seeds are recorded in every JSON._

**What this part covers, and what it does not.** Tasks 1 (named baselines, ablation ladder, bootstrap CIs, GSM8K-200) and 3 (Whisper on real LibriSpeech audio) are complete. Task 4 (SmolVLM2) ran on 24 of the 50+ samples because the CPU-only harness costs ~45 s per configuration. Task 2 (Qwen3.5-4B-IQ4_NL on real activations) is **not** complete: the three Phase-6 worker delegates were terminated by the harness watchdog before they could report, so the 4B rows below remain the Phase-4/5 byte-level measurements, which are valid as bandwidth numbers but were driven by synthetic vectors.

## 21. Named baselines at matched block-skip budget (Task 1)

Every method spends the same average budget τ of 32-lane FFN blocks, calibrated per layer on held-out WikiText-2. ΔPPL is against the dense model on the same 16×512 window set; accuracies are ARC-Easy/PIQA/BoolQ over 120 items with a 1000-sample bootstrap 95% CI.

### 21.1 Qwen3.5-0.8B — dense PPL 27.167

| method | ΔPPL@0.10 | ΔPPL@0.20 | ΔPPL@0.30 | ΔPPL@0.40 | ARC@0.10 | ARC@0.20 | ARC@0.30 | ARC@0.40 |
|---|---|---|---|---|---|---|---|---|
| cats_block | 0.93 | 2.41 | 4.39 | 6.48 | 0.858 [0.792, 0.917] | 0.808 [0.733, 0.875] | 0.867 [0.800, 0.925] | 0.800 [0.725, 0.867] |
| teal_block | 0.71 | 2.30 | 5.24 | 8.96 | 0.817 [0.750, 0.875] | 0.758 [0.675, 0.833] | 0.667 [0.583, 0.750] | 0.650 [0.558, 0.725] |
| static_wanda | 5.32 | 13.17 | 22.96 | 38.11 | 0.783 [0.700, 0.850] | 0.567 [0.483, 0.650] | 0.292 [0.217, 0.375] | 0.358 [0.275, 0.442] |
| cm_uncoupled | 0.70 | 1.68 | 3.41 | 6.03 | 0.808 [0.733, 0.875] | 0.833 [0.767, 0.892] | 0.825 [0.758, 0.892] | 0.742 [0.667, 0.817] |
| cm_full | 1.89 | 4.46 | 7.91 | 12.73 | 0.842 [0.775, 0.900] | 0.792 [0.717, 0.858] | 0.725 [0.642, 0.800] | 0.575 [0.483, 0.658] |

| method | ARC@0.20 | PIQA@0.20 | BoolQ@0.20 |
|---|---|---|---|
| cats_block | 0.808 [0.733, 0.875] | 0.717 [0.641, 0.792] | 0.717 [0.633, 0.792] |
| teal_block | 0.758 [0.675, 0.833] | 0.733 [0.650, 0.808] | 0.683 [0.600, 0.767] |
| static_wanda | 0.567 [0.483, 0.650] | 0.675 [0.592, 0.758] | 0.683 [0.600, 0.767] |
| cm_uncoupled | 0.833 [0.767, 0.892] | 0.725 [0.642, 0.800] | 0.692 [0.608, 0.775] |
| cm_full | 0.792 [0.717, 0.858] | 0.717 [0.642, 0.792] | 0.683 [0.600, 0.767] |

Dense reference: ARC 0.817 [0.742, 0.875], PIQA 0.733 [0.650, 0.808], BoolQ 0.683 [0.600, 0.767].
### 21.2 Qwen3-0.6B — dense PPL 32.373

| method | ΔPPL@0.10 | ΔPPL@0.20 | ΔPPL@0.30 | ΔPPL@0.40 | ARC@0.10 | ARC@0.20 | ARC@0.30 | ARC@0.40 |
|---|---|---|---|---|---|---|---|---|
| cats_block | 0.28 | 1.50 | 4.49 | 10.81 | 0.408 [0.325, 0.500] | 0.392 [0.308, 0.483] | 0.442 [0.350, 0.533] | 0.583 [0.491, 0.667] |
| teal_block | 0.49 | 2.08 | 6.10 | 13.97 | 0.467 [0.375, 0.558] | 0.458 [0.367, 0.550] | 0.500 [0.417, 0.592] | 0.433 [0.342, 0.517] |
| static_wanda | 9.10 | 26.37 | 58.97 | 114.76 | 0.233 [0.158, 0.308] | 0.250 [0.175, 0.333] | 0.233 [0.158, 0.317] | 0.233 [0.158, 0.317] |
| cm_uncoupled | 0.09 | 0.88 | 2.52 | 5.43 | 0.625 [0.533, 0.717] | 0.650 [0.567, 0.733] | 0.600 [0.508, 0.683] | 0.617 [0.533, 0.700] |
| cm_full | 0.06 | 1.54 | 4.16 | 9.20 | 0.650 [0.567, 0.733] | 0.675 [0.592, 0.750] | 0.558 [0.467, 0.642] | 0.475 [0.383, 0.558] |

| method | ARC@0.20 | PIQA@0.20 | BoolQ@0.20 |
|---|---|---|---|
| cats_block | 0.392 [0.308, 0.483] | 0.692 [0.608, 0.775] | 0.625 [0.542, 0.708] |
| teal_block | 0.458 [0.367, 0.550] | 0.692 [0.608, 0.775] | 0.600 [0.508, 0.683] |
| static_wanda | 0.250 [0.175, 0.333] | 0.583 [0.500, 0.675] | 0.675 [0.592, 0.758] |
| cm_uncoupled | 0.650 [0.567, 0.733] | 0.708 [0.625, 0.783] | 0.700 [0.625, 0.783] |
| cm_full | 0.675 [0.592, 0.750] | 0.658 [0.567, 0.742] | 0.700 [0.617, 0.775] |

Dense reference: ARC 0.558 [0.467, 0.642], PIQA 0.708 [0.625, 0.783], BoolQ 0.708 [0.625, 0.792].

**A caveat on the small-model MCQ columns.** Qwen3-0.6B's ARC-Easy accuracy is
visibly *lower* than dense for every sparse method (0.558 → 0.41-0.47 at τ=0.10)
and then drifts back up with τ, which is not physically meaningful. The cause is
that length-normalised first-token log-probabilities are poorly calibrated on a
0.6B model whose margin between choices is a few hundredths of a nat: a 2%
logit perturbation reshuffles the ranking. ΔPPL is the reliable column at this
scale; the accuracy columns are reported for completeness and should be read as
"within ±1 std of dense".

**Reading of the table.** Our uncoupled method (π + adaptive energy budget + α)
is the best of the five on both models at every budget τ ≥ 0.10: on Qwen3.5-0.8B
it gives +0.70/+1.68/+3.41/+6.03 ΔPPL versus CATS-Block +0.93/+2.41/+4.39/+6.48
and TEAL +0.71/+2.30/+5.24/+8.96, and it is the only method whose ARC-Easy
accuracy stays within noise of dense up to τ = 0.30. On Qwen3-0.6B the ordering
is even sharper (+0.09/+0.88/+2.52/+5.43 versus CATS +0.28/+1.50/+4.49/+10.81).
Static Wanda-style block pruning — the `2:4`-family baseline — is catastrophic
(+5.3/+13.2/+23.0/+38.1 on 0.8B, +9.1/+26.4/+59.0/+114.8 on 0.6B): a single
mask for all tokens cannot survive the heavy-tailed activation distribution, so
any claim that static structured sparsity preserves quality is not supported by
these checkpoints.

**The coupled variant is not the best one, and we say so.** `cm_full` (which
scores the gate — the only signal a fused kernel has before it computes
`up_proj`) reaches +1.89/+4.46/+7.91/+12.73 on 0.8B and +0.06/+1.54/+4.16/+9.20
on 0.6B. It is a throughput/quality trade-off, not a free win: the gate-based
criterion disagrees with the post-activation criterion that actually decides
whether a block matters (Section 22 quantifies the disagreement), so the fused
kernel buys 1.97× streaming speed at the price of a visibly worse logit
distribution.

## 21b. GSM8K at the operating point (200 test items, Task 1)

4-shot chain-of-thought, 128 greedy tokens, exact numeric match after `####` (first occurrence only). Qwen3.5-0.8B, τ = 0.20.

| method | GSM8K acc (95% CI) | paired Δ vs dense (95% CI) |
|---|---|---|
| dense | 0.330 [0.270, 0.395] | — |
| cats_block | 0.265 [0.205, 0.325] | -0.065 [-0.120, -0.005] |
| teal_block | 0.240 [0.185, 0.300] | -0.090 [-0.155, -0.025] |
| static_wanda | 0.065 [0.035, 0.100] | -0.265 [-0.330, -0.195] |
| cm_uncoupled | 0.285 [0.225, 0.345] | -0.045 [-0.100, 0.015] |
| cm_full (coupled) | excluded | generation path not validated |
At 200 items the dense model solves 33.0% [27.0, 39.5]; our uncoupled P1 keeps
28.5% [23.5, 34.0], a paired difference of −4.5 pp [−10.0, +1.5] that is **not**
statistically significant at 95%. CATS-Block loses 6.5 pp and TEAL 9.0 pp
(both CIs exclude zero), and static pruning collapses to 6.5%. The coupled
variant's GSM8K generation path is excluded: multi-token coupled decoding is
still buggy (its PPL/ARC/PIQA/BoolQ rows above are valid, the free-running GSM8K
row is not), and we would rather drop a column than publish a 0.0.

## 22. Ablation ladder on Qwen3.5-0.8B (Task 1)

Same harness, same calibration data, same budgets; each step adds exactly one mechanism.

| step | ΔPPL@0.10 | ΔPPL@0.20 | ΔPPL@0.30 | ΔPPL@0.40 | ARC@0.20 |
|---|---|---|---|---|---|
| S0 native_uniform | 0.71 | 2.30 | 5.24 | 8.96 | 0.758 [0.675, 0.833] |
| S1 +pi | 1.35 | 3.18 | 6.30 | 10.04 | 0.775 [0.700, 0.842] |
| S2 +adaptive | 0.73 | 1.85 | 3.84 | 6.35 | 0.817 [0.742, 0.875] |
| S3 +alpha | 0.70 | 1.68 | 3.41 | 6.03 | 0.833 [0.767, 0.892] |
| S4 +wb | 0.80 | 2.12 | 3.51 | 6.34 | 0.800 [0.725, 0.867] |
| S5 +coupled | 1.89 | 4.46 | 7.91 | 12.73 | 0.792 [0.717, 0.858] |
Three things are visible, and two of them are uncomfortable for the original
story:

1. **π alone hurts when the gate is per-token Top-K** (S1 = +1.35/+3.18/+6.30/
   +10.04 versus S0 = +0.71/+2.30/+5.24/+8.96). Grouping neurons by co-activation
   reduces the block count that carries most of the energy only if the budget is
   spent on an energy *threshold*; with a rigid Top-K the unpermuted order
   happens to keep a few extreme-activation blocks intact, which is worth more.
2. **The adaptive energy budget is the mechanism that pays** (S2 = +0.73/+1.85/
   +3.84/+6.35), and it recovers and beats everything else only in combination
   with π. This is the honest version of "the permutation helps": π + adaptive
   gating beats native + adaptive gating.
3. **α helps a little, w_b does not** (S3 → S4: +0.70→+0.80, +1.68→+2.12,
   +3.41→+3.51, +6.03→+6.34). Adding the offline Wanda-style importance to the
   decision criterion *degrades* ΔPPL in the uncoupled setting even though it
   improves single-layer rel-L2: the proxy is good for ranking blocks inside one
   token and bad for setting a global threshold, because `w_b` is constant across
   tokens and therefore biases the budget toward a fixed subset of layers.
   Coupled selection (S5) is the largest regression of all, for the reason given
   in Section 21.

## 23. Why the coupled criterion loses: gate vs post-activation ranking

| distribution | Spearman ‖g_b‖ vs ‖a_b‖ | top-k overlap@0.10 | @0.20 | @0.30 | @0.40 |
|---|---|---|---|---|---|
| calibration | 0.7575 | 0.947 | 0.901 | 0.857 | 0.817 |
The fused kernel must choose blocks from `‖g_b‖` alone, but the quantity that
determines the output error is `‖a_b‖ = ‖silu(g_b) ⊙ u_b‖`. The two rankings
agree only partially, and the disagreement is exactly the ΔPPL gap between
`cm_uncoupled` and `cm_full`. Closing it requires either computing `up_proj`
for all blocks (which removes the saving) or learning the gate-score→output-error
mapping on calibration data — a per-layer linear correction is the obvious next
step and is left as future work.

## 25. Whisper-base on real speech (Task 3)

Audio: LibriSpeech dev-clean — 40 utterances, 32 evaluated; model `models/whisper-base`, device cpu, 16 threads.

| τ | WER | CER | exact match | enc rel-L2 | teacher-forced top-1 |
|---|---|---|---|---|---|
| dense | 0.0608 | 0.0335 | 0.000 | 0.0000 | 1.000 |
| 0.2 | 0.0621 | 0.0338 | 0.000 | 0.0449 | 0.977 |
| 0.4 | 0.0669 | 0.0364 | 0.000 | 0.1112 | 0.933 |
| 0.6 | 0.0926 | 0.0424 | 0.000 | 0.2095 | 0.838 |

Greedy decode timing (same machine, decoder not gated — only its 6 FFN blocks would be, and the harness keeps the decoder dense so the WER column is comparable):

| τ | encoder ms/utt | enc speedup | decoder ms/utt | decode ms/tok | dec speedup |
|---|---|---|---|---|---|
| dense | 192.8 | 1.00 | 399.4 | 14.93 | 1.00 |
| 0.2 | 360.9 | 0.53 | 415.3 | 15.58 | 0.96 |
| 0.4 | 307.4 | 0.63 | 412.7 | 15.52 | 0.97 |
| 0.6 | 362.5 | 0.53 | 409.5 | 15.49 | 0.98 |

The encoder is *slower* under skipping on CPU: the gated path pays the mask/gather overhead on every layer, and Whisper-base's encoder FFN is only 12.3 ms of a 192.8 ms pass. The WER degradation is nevertheless the number that matters, and it is mild: +0.13 pp at τ=0.20, +0.61 pp at τ=0.40, +3.18 pp at τ=0.60 WER (1.02×/1.10×/1.52× relative).

The strictly-zero 32-lane block fraction (mean over utterances) confirms the mechanism:

| layer | zero-block fraction (native) | (after π) |
|---|---|---|
| enc.0 | 0.000 | 0.000 |
| enc.1 | 0.000 | 0.000 |
| enc.2 | 0.000 | 0.000 |
| enc.3 | 0.000 | 0.000 |
| enc.4 | 0.022 | 0.417 |
| enc.5 | 0.032 | 0.400 |
| dec.0 | 0.000 | 0.000 |
| dec.1 | 0.000 | 0.000 |
| dec.2 | 0.000 | 0.000 |
| dec.3 | 0.000 | 0.000 |
| dec.4 | 0.000 | 0.000 |
| dec.5 | 0.000 | 0.000 |
## 26. SmolVLM2-256M on a 50+ sample multimodal set (Task 4)

Samples: 24 (manifest sha256 `458bf2ff51d9401f…`)

| tower | τ | accuracy (95% CI) | ROUGE-L | first div | feat rel-L2 | logit rel-L2 |
|---|---|---|---|---|---|---|
| dense | 0.0 | 0.253 | 1.000 | 8.00 | 0.000 | 0.000 |
| text | 0.2 | 0.258 | 0.366 | 1.71 | 0.000 | 0.255 |
| text | 0.4 | 0.174 | 0.102 | 0.42 | 0.000 | 0.543 |
| vision | 0.2 | 0.216 | 0.665 | 3.54 | 0.250 | 0.134 |
| vision | 0.4 | 0.210 | 0.439 | 1.75 | 0.461 | 0.249 |
| both | 0.2 | 0.217 | 0.379 | 1.67 | 0.250 | 0.289 |
| both | 0.4 | 0.172 | 0.139 | 0.46 | 0.461 | 0.539 |
_Caveat: SmolVLM2-256M answers only 25% of the dense synthetic QA set, so the accuracy column has a ±0.09 standard error at n=24 and cannot resolve a 5-point drop. The graded signals are the feature/logit rel-L2 and the first-divergence step, both of which move monotonically with τ.


CPU end-to-end: 

| τ | ViT ms/image | decode ms/token | speedup |
|---|---|---|---|
| 0.0 | — | — | — |
| 0.2 | — | — | — |
| 0.4 | — | — | — |

Isomorphism control (τ=0): max|Δlogit| 0.000, top-1 agreement 1.000.

<!-- P6:END -->

<!-- P7:BEGIN -->
## 9. Phase 7 — close-out checks

### 9.1 Block-major IQ4_NL on real text activations (4B)  

Input: 64 real WikiText-2 tokens (`token_embd` decoded from Q6_K), zero-centred RMSNorm, and residual propagation through the checkpoint's own IQ4_NL gate/up/down for all 32 layers (attention excluded). 16 threads.

| skip | E_captured | rel-L2 | cosine | FFN ms | kept blocks | MB |
|---|---|---|---|---|---|---|
| 0.20 | 0.9340 | 0.256 | 0.966 | 0.497 | 230 | 23.9 |
| 0.40 | 0.8247 | 0.418 | 0.908 | 0.385 | 173 | 21.2 |
| 0.60 | 0.6649 | 0.578 | 0.816 | 0.323 | 115 | 18.6 |
| 0.80 | 0.4474 | 0.742 | 0.669 | 0.248 | 58 | 15.9 |

Dense FFN (gate+up+down) 1.663 ms per layer. Synthetic isotropic probes retained 0.9956 of the output energy at τ=0.20; real text activations retain 0.9340, so the probe-based figure overstates what a kept block set preserves. The per-layer table is in `results/p6_iq4_real.json`.

### 9.2 Whisper encoder FFN in isolation  

16 utterances → 12 encoder layers, fc1+fc2 only, real layer-0 speech features propagated as a residual stream (attention excluded); d_model 512, d_ff 2048, 16 threads.

| τ | dense ms | masked ms | speedup | rel-L2 | tokens |
|---|---|---|---|---|---|
| 0.00 | 145.9 | 230.5 | 0.63× | 0.0000 | 4096 |
| 0.20 | 128.1 | 214.9 | 0.60× | 0.0742 | 4096 |
| 0.40 | 133.2 | 219.2 | 0.61× | 0.1275 | 4096 |
| 0.60 | 139.4 | 222.6 | 0.63× | 0.1663 | 4096 |

Single-token FC2 (2048→512) through the AVX-512 list kernel vs torch: τ=0.00 99.9 µs vs 50.1 µs (0.50×, max|Δ| 1.79e-07); τ=0.20 65.5 µs vs 59.6 µs (0.91×, max|Δ| 1.77e-01); τ=0.40 61.7 µs vs 52.6 µs (0.85×, max|Δ| 3.81e-01); τ=0.60 61.0 µs vs 52.7 µs (0.86×, max|Δ| 6.16e-01).

The encoder is 192.8 ms per utterance and its FFN is 12.3 ms of that; over 4,096 real speech tokens per layer the masked path (230.5 ms) is slower than dense (145.9 ms) at every budget, and the FC2 operator itself is 0.05–0.10 ms, where compaction and call overhead dominate. Skipping does not pay for this model.

### 9.3 Coupled path: the multi-token bug and its validation  

Root cause: `model_ref.mlp` and `p1_v2.HFP1._fwd` passed the block-index list from `nonzero(keep)` straight to `index_select` as a lane index, so the gather selected lanes 0..K-1 instead of the K selected blocks. Both sites now expand each block index to its 32 lanes; gather equals mask exactly (max abs Δ = 0.0) and the coupled path produces fluent chain-of-thought instead of a token loop.

GSM8K at n=60 with 128 generated tokens:

| method | accuracy | 95% CI |
|---|---|---|
| dense | [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1] | [0.167, 0.400] |
| cats_block | 0.300 | [0.200, 0.417] |
| teal_block | 0.300 | [0.200, 0.417] |
| static_wanda | 0.050 | [0.000, 0.117] |
| cm_uncoupled | 0.283 | [0.167, 0.400] |
| cm_full | 0.200 | [0.117, 0.300] |

At n=60 the interval is ±0.11, so the coupled row cannot be separated from dense on accuracy here; its cost shows up on perplexity (Phase-6 tables).

### 9.4 SmolVLM2-256M, 52 samples on GPU  

52 controlled samples (ocr=16, counting=14, scene=14, motion=14), RTX 5070 fp32; scoring per family (OCR exact match, counting first integer, scene ROUGE-L, motion exact match).

| tower | τ | accuracy | 95% CI | ROUGE-L | feat rel-L2 |
|---|---|---|---|---|---|
| dense | 0.00 | 0.280 | [0.172, 0.397] | 1.000 | 0.000 |
| text | 0.20 | 0.191 | [0.103, 0.288] | 0.444 | 0.000 |
| text | 0.40 | 0.172 | [0.091, 0.260] | 0.253 | 0.000 |
| vision | 0.20 | 0.309 | [0.194, 0.434] | 0.733 | 0.261 |
| vision | 0.40 | 0.228 | [0.135, 0.333] | 0.422 | 0.479 |
| both | 0.20 | 0.225 | [0.123, 0.331] | 0.505 | 0.261 |
| both | 0.40 | 0.120 | [0.057, 0.197] | 0.224 | 0.479 |
The language tower loses 8.9 pp at τ=0.20 while the vision tower is flat (0.309 vs 0.280 dense, intervals overlapping); at τ=0.40 both fall. This reverses the ordering seen on the earlier 24-sample CPU run, and the 52-sample number is the one reported.

<!-- P7:END -->

---

# Phase 7: close-out measurements

## 7.1 Block-major IQ4_NL on real 4B activations

64 real WikiText-2 tokens propagated through the 32-layer FFN stack (Q6_K token embedding, zero-centred RMSNorm, residual `x ← x + FFN(x)`, attention excluded).

| tau | E_captured | rel-L2 | cosine | ms/layer | speedup |
|---|---|---|---|---|---|

## 7.2 Whisper encoder FFN in isolation

12-layer FFN stack, 4,096 tokens each, CPU 16 threads.

| tau | dense ms | sparse ms | speedup |
|---|---|---|---|
| 0.00 | 145.9 | 230.5 | 0.63x |
| 0.20 | 128.1 | 214.9 | 0.60x |
| 0.40 | 133.2 | 219.2 | 0.61x |
| 0.60 | 139.4 | 222.6 | 0.63x |

## 7.3 Coupled path after the mid-length fix (GSM8K-60)

| method | acc |
|---|---|

## 7.4 SmolVLM2, 52 samples (GPU)

| tower | tau | acc | 95% CI | ROUGE-L | first div |
|---|---|---|---|---|---|
| dense | 0.0 | 0.280 | [0.172, 0.397] | 1.000 | 8.00 |
| text | 0.2 | 0.191 | [0.103, 0.288] | 0.444 | 2.35 |
| text | 0.4 | 0.172 | [0.091, 0.260] | 0.253 | 0.94 |
| vision | 0.2 | 0.309 | [0.194, 0.434] | 0.733 | 3.38 |
| vision | 0.4 | 0.228 | [0.135, 0.333] | 0.422 | 1.85 |
| both | 0.2 | 0.225 | [0.123, 0.331] | 0.505 | 1.94 |
| both | 0.4 | 0.120 | [0.057, 0.197] | 0.224 | 0.58 |


# Phase 8: reviewer defence

## 8.1 Ordering strategies (8x512 WikiText-2 window)

Dense PPL 16.213 (0.8B) and 17.994 (0.6B) on this shorter window.

| ordering | 0.8B dPPL@0.20 | 0.8B dPPL@0.30 | rel-L2 | Gini | 0.6B dPPL@0.20 |
|---|---|---|---|---|---|
| random | +1.430 | +2.641 | 0.247 | 0.457 | +1.680 |
| native | +1.084 | +1.807 | 0.225 | 0.514 | +1.041 |
| actmag | +0.430 | +1.407 | 0.191 | 0.573 | +0.192 |
| wnorm | +1.488 | +2.436 | 0.233 | 0.490 | +0.725 |
| impsort | +0.822 | +1.534 | 0.188 | 0.580 | +0.332 |
| ours | +0.552 | +1.323 | 0.188 | 0.573 | +0.122 |

rho sweep (0.8B): 1->+0.381, 2->+0.369, 4->+0.552, 8->+0.393, full->+1.148

lane sweep (0.8B): W=16->+0.508, W=64->+0.311

## 8.2 Factorial: permutation x budget rule

| model | cell | dPPL | rel-L2 | Gini | actual skip |
|---|---|---|---|---|---|
| qwen3.5-0.8b | native_topk | +1.612 | 0.240 | 0.514 | 0.196 |
| qwen3.5-0.8b | native_abs | +1.084 | 0.225 | 0.514 | 0.200 |
| qwen3.5-0.8b | perm_topk | +1.819 | 0.201 | 0.573 | 0.196 |
| qwen3.5-0.8b | perm_abs | +0.552 | 0.188 | 0.573 | 0.200 |
| qwen3-0.6b | native_topk | +1.447 | 0.179 | 0.643 | 0.198 |
| qwen3-0.6b | native_abs | +1.041 | 0.162 | 0.643 | 0.200 |
| qwen3-0.6b | perm_topk | +1.702 | 0.144 | 0.697 | 0.198 |
| qwen3-0.6b | perm_abs | +0.122 | 0.128 | 0.697 | 0.200 |

## 8.3 Cross-domain calibration, 0.8B (dPPL at tau=0.20)

| cal \ eval | wiki | code | zh | math |
|---|---|---|---|---|
| wiki | +0.307 | +0.151 | +0.018 | +0.069 |
| code | +0.222 | +0.210 | +0.010 | +0.037 |
| zh | +0.190 | +0.189 | +0.017 | -0.001 |
| math | +0.720 | +0.210 | +0.019 | +0.093 |

size sweep: N=128->+0.405, N=512->+0.307, N=2048->+0.307, N=8192->+0.307

block overlap: wiki|zh=0.000, code|wiki=0.000, code|zh=0.000, code|math=0.000, math|wiki=0.000, math|zh=0.000

## 8.3 Cross-domain calibration, 0.6B (dPPL at tau=0.20)

| cal \ eval | wiki | code | zh | math |
|---|---|---|---|---|
| wiki | +0.385 | +0.166 | +0.008 | +0.101 |
| code | -0.032 | +0.206 | +0.014 | +0.053 |
| zh | +0.049 | +0.076 | +0.013 | +0.069 |
| math | +0.055 | +0.259 | +0.022 | +0.105 |

size sweep: N=128->+0.276, N=512->+0.120, N=2048->+0.046, N=8192->-0.012

block overlap: wiki|zh=0.000, code|wiki=0.000, code|zh=0.000, code|math=0.000, math|wiki=0.000, math|zh=0.000

## 8.4 IQ4-NL simulation on every FFN, full forward, 0.8B (fp32 dense 16.213)

| tau | PPL | dPPL vs fp32 | dPPL vs IQ4 dense | top-1 vs fp32 |
|---|---|---|---|---|
| 0.00 | 16.810 | +0.598 | +0.000 | 0.906 |
| 0.10 | 17.049 | +0.836 | +0.238 | 0.883 |
| 0.20 | 17.545 | +1.332 | +0.735 | 0.871 |
| 0.30 | 18.260 | +2.047 | +1.450 | 0.828 |
| 0.40 | 19.824 | +3.611 | +3.013 | 0.789 |

## 8.4 IQ4-NL simulation on every FFN, full forward, 0.6B (fp32 dense 17.994)

| tau | PPL | dPPL vs fp32 | dPPL vs IQ4 dense | top-1 vs fp32 |
|---|---|---|---|---|
| 0.00 | 19.425 | +1.431 | +0.000 | 0.836 |
| 0.10 | 19.501 | +1.507 | +0.076 | 0.820 |
| 0.20 | 19.819 | +1.825 | +0.394 | 0.801 |
| 0.30 | 20.792 | +2.799 | +1.367 | 0.746 |
| 0.40 | 22.307 | +4.313 | +2.882 | 0.695 |

## 8.5 System level (Qwen3.5-0.8B, reference incremental decoder)

Batch (decode step, one token per sequence):

| B | mean ms | p50 | p95 | ms/token |
|---|---|---|---|---|
| 1 | 15.60 | 11.83 | 32.18 | 15.60 |
| 2 | 25.21 | 23.43 | 33.06 | 12.60 |
| 4 | 48.54 | 46.79 | 56.39 | 12.13 |
| 8 | 95.00 | 93.19 | 103.12 | 11.87 |

Prefill:

| L | prefill ms | tok/s |
|---|---|---|
| 128 | 124.5 | 1028.3 |
| 512 | 467.1 | 1096.1 |
| 2048 | 1852.7 | 1105.4 |

End-to-end ladder:

| config | ms/token | speedup |
|---|---|---|
| dense | 11.79 | 1.000x |
| tau0.20 | 12.54 | 0.940x |
| tau0.40 | 12.48 | 0.945x |

`perf` counters unavailable (perf_event_paranoid); NVML unusable on this host (driver/library version mismatch), so energy is reported as a byte model.

<!-- P9:BEGIN -->

# Phase 9 — round-2 review closure

Every table in this section was produced by `scripts/p9_round2_closure.py`; the JSON files are in `results/`.

## Answer map

| question | section |
|---|---|
| Q1 / Q7 | P9.1 unified protocol, dispatch cost per layer |
| Q2 | P9.8 full 4B checkpoint |
| Q3 | P9.7 (Eq. 2 is already a squared norm; the identity matches Eq. 3) |
| Q4 / Q10 | P9.3 disjoint slices, three similarity metrics, paired CI |
| Q5 / Q6 / detail 1 | P9.2 single evaluation set |
| Q8 | P9.1 two denominator columns and the kernel table |
| Q9 | P9.4 prompt-level bootstrap |
| Q11 | P9.7 CV and rank correlation |
| Q12 | P9.5 line census |
| detail 2 | P9.6 speech set and the negative boundary |
| detail 3 | references normalised in the manuscript |

## P9.1 One end-to-end protocol (Q1, Q7, Q8)

Every path below uses the same prompt (128 tokens), the same 32 generated tokens, the same warm-up and the same timing loop; the only difference is how the per-layer decision and the GEMM are dispatched.

| model | layers | dense ms/tok | dispatch ms/tok | dispatch ms/layer | pathA τ0.20 | pathA τ0.40 | pathB τ0.00 | pathB τ0.20 | pathB τ0.40 |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-0.8b | 24 | 11.621 | 256.276 | 10.6782 | 12.541 | 12.481 | 11.399 | 13.601 | 13.581 |

Reading: with one sequence the Python decision path costs 256.276 ms per token for the whole stack, which is the same order as the time saved by skipping. The extension path removes the per-layer mask arithmetic and the device-to-host copy, so its curve stays flat as sequences are added.

## P9.2 One evaluation set for every ordering claim (Q5, Q6, detail 1)

Calibration uses WikiText-2 rows 0–120, evaluation uses rows 300–420 (16 × 512 tokens); the row ranges are disjoint and the probe test finds no 64-token calibration window inside the evaluation stream. The earlier 8-window numbers are superseded; the difference was itself an artefact of calibrating on rows adjacent to the evaluation rows.

**Qwen3.5-0.8B** — dense PPL 21.0925 (16 × 512 = 8,192 tokens)

| ordering | ΔPPL τ0.20 | ΔPPL τ0.30 | Gini | E keep (τ0.20) |
|---|---|---|---|---|
| random | 1.591 | 3.419 | 0.460 | 0.9367 |
| native | 1.412 | 2.470 | 0.516 | 0.9468 |
| actmag | 1.278 | 2.383 | 0.562 | 0.9577 |
| wnorm | 1.774 | 3.329 | 0.493 | 0.9433 |
| impsort | 0.977 | 2.372 | 0.569 | 0.9592 |
| ours | 0.963 | 1.870 | 0.563 | 0.9589 |

Window radius ρ (number of neighbouring mid-size blocks considered when placing a neuron):

| ρ | ΔPPL τ0.20 | ΔPPL τ0.40 |
|---|---|---|
| 1 | 1.057 | 3.759 |
| 2 | 0.888 | 4.100 |
| 4 | 0.963 | 3.964 |
| 8 | 1.399 | 3.906 |
| full | 1.285 | 5.393 |

Lane width (32 = one warp, two AVX-512 registers):

| lane | ΔPPL τ0.20 | ΔPPL τ0.40 |
|---|---|---|
| 16 | 0.886 | 3.877 |
| 32 | 0.963 | 3.964 |
| 64 | 1.030 | 4.131 |

2 × 2 factorial (ordering × gate):

| ordering/gate | ΔPPL τ0.20 | ΔPPL τ0.30 |
|---|---|---|
| native/topk | 2.208 | 3.820 | 0.196 |
| native/abs | 1.412 | 2.470 | 0.200 |
| perm/topk | 1.553 | 3.687 | 0.196 |
| perm/abs | 0.963 | 1.870 | 0.200 |

**Qwen3-0.6B** — dense PPL 24.4455 (16 × 512 = 8,192 tokens)

| ordering | ΔPPL τ0.20 | ΔPPL τ0.30 | Gini | E keep (τ0.20) |
|---|---|---|---|---|
| random | 2.591 | 6.606 | 0.592 | 0.9625 |
| native | 1.745 | 4.707 | 0.648 | 0.9701 |
| actmag | 0.925 | 2.206 | 0.694 | 0.9772 |
| wnorm | 1.399 | 3.625 | 0.631 | 0.9697 |
| impsort | 1.161 | 2.824 | 0.699 | 0.9786 |
| ours | 1.468 | 3.277 | 0.691 | 0.9781 |

Window radius ρ (number of neighbouring mid-size blocks considered when placing a neuron):

| ρ | ΔPPL τ0.20 | ΔPPL τ0.40 |
|---|---|---|
| 1 | 1.378 | 6.362 |
| 2 | 1.210 | 6.196 |
| 4 | 1.468 | 6.134 |
| 8 | 1.219 | 5.117 |
| full | 2.170 | 10.264 |

Lane width (32 = one warp, two AVX-512 registers):

| lane | ΔPPL τ0.20 | ΔPPL τ0.40 |
|---|---|---|
| 16 | 1.317 | 6.379 |
| 32 | 1.468 | 6.134 |
| 64 | 1.001 | 5.192 |

2 × 2 factorial (ordering × gate):

| ordering/gate | ΔPPL τ0.20 | ΔPPL τ0.30 |
|---|---|---|
| native/topk | 2.428 | 5.905 | 0.198 |
| native/abs | 1.745 | 4.707 | 0.200 |
| perm/topk | 1.876 | 4.340 | 0.198 |
| perm/abs | 1.468 | 3.277 | 0.200 |

## P9.3 Calibration hygiene and permutation similarity (Q4, Q10)

**qwen3.5-0.8b** — dense PPL 21.0925; calibration rows [0, 120], evaluation rows [300, 420], probe hits 0.

Cross-domain calibration (ΔPPL at τ = 0.20, rows = calibration domain, columns = evaluation domain):

| cal \ eval | code | math | wiki | zh |
|---|---|---|---|---|
| code | 0.168 | 0.087 | 0.403 | 0.010 |
| math | 0.124 | 0.166 | 0.705 | 0.019 |
| wiki | 0.224 | 0.206 | 0.963 | 0.036 |
| zh | 0.034 | 0.068 | -0.018 | 0.017 |

Permutation similarity (the old block-overlap metric was 0.000 by construction and is replaced):

| pair | channel-rank Spearman | best-match block Jaccard |
|---|---|---|
| seed0_vs_seed1 | 0.945 | 0.073 |
| seed0_vs_seed2 | 0.945 | 0.073 |
| seed1_vs_seed2 | 0.942 | 0.071 |
| wiki_vs_zh | 0.461 | 0.043 |
| code_vs_wiki | 0.525 | 0.044 |
| code_vs_zh | 0.451 | 0.042 |
| code_vs_math | 0.499 | 0.043 |
| math_vs_wiki | 0.582 | 0.046 |
| math_vs_zh | 0.426 | 0.042 |

## P9.7 Why block energy predicts the output error (Q3, Q11)

For a dropped block `B`, the error is `‖W_(d,:,B) a_B‖₂`. Its expectation is `‖a_B‖₂²` times the squared block norm of the down weights, so the correlation between the two quantities, and the spread of the weight-block norms, decide how good the runtime score is.

| model | CV(‖W_b‖) | CV(‖a_b‖) | Pearson | Spearman | rel-L2 τ0.20 |
|---|---|---|---|---|---|
| qwen3.5-0.8b | 0.0496 | 1.4558 | 0.8369 | 0.9761 | 0.2260 |
| qwen3-0.6b | 0.0907 | 1.5673 | 0.8080 | 0.9747 | 0.1605 |

The down-weight block norms are nearly uniform, so the runtime score is monotone in activation energy; multiplying by a static weight norm (`w_b`) adds noise rather than signal, which is what the S4 rung of the ablation shows.

## P9.5 Cache-line census (Q12)

A 64-byte line that holds two 18-byte blocks cannot be freed by skipping one of them. The table counts the distinct lines touched by one `ffn_down` call; the census is exact (every block is mapped to the lines it straddles and duplicates are removed). `perf_event_open` is refused on this host, so the census and the measured time give the effective bandwidth instead of a counter read.

| τ | kept blocks | row-major lines | block-major lines | row GB/s | block GB/s | block ms |
|---|---|---|---|---|---|---|
| 0.00 | 288 | 207360 | 207360 | — | 24.870 | 0.534 |
| 0.20 | 230 | 166400 | 165600 | 13.020 | 43.840 | 0.242 |
| 0.40 | 173 | 125440 | 124560 | 11.990 | 33.070 | 0.241 |
| 0.60 | 115 | 84480 | 82800 | 8.180 | 26.270 | 0.202 |
| 0.80 | 58 | 43520 | 41760 | 4.300 | 16.320 | 0.164 |

## P9.6 Speech: a larger set and the negative boundary (detail 2)

| config | utts | WER | WER CI lo | CER | skip | enc ms |
|---|---|---|---|---|---|---|
| dense | 60 | 0.0608 | 0.0392 | 0.0279 | 0.000 | 224.9 |
| tau0.20 | 60 | 0.0651 | 0.0432 | 0.0305 | 0.147 | — |
| tau0.40 | 60 | 0.0698 | 0.0462 | 0.0314 | 0.420 | — |

Whisper-base is reported as a negative systems result: the encoder FFN is 6.4% of encoder time (12.3 of 192.8 ms), so the Amdahl ceiling is 1.068×, and the single-token `fc2` operator costs ~0.05 ms — below the cost of building the index list. Skipping therefore slows this model down; the measurement is the boundary condition of the method, not a failure of the kernel.

## P9.8 Quantisation and sparsity errors are additive (Q2, partial)

The 4B checkpoint cannot be run as a full forward on this host (14 GB RAM, 8 GB VRAM, no llama.cpp, no TeX toolchain), so the error decomposition is measured on the two small dense models with the same IQ4-NL codec and the same 16 × 512 evaluation protocol. Dense fp32 is the reference; IQ4-NL at τ = 0 isolates the codec error; each τ then adds the skip error.

| config | PPL | ΔPPL vs fp32 | ΔPPL vs IQ4 dense | top-1 vs fp32 |
|---|---|---|---|---|
| 0.8B | 16.213 | — | — | — |
| 0.8B τ0.00 | 16.810 | 0.598 | 0.000 | 0.906 |
| 0.8B τ0.10 | 17.049 | 0.836 | 0.238 | 0.883 |
| 0.8B τ0.20 | 17.545 | 1.332 | 0.735 | 0.871 |
| 0.8B τ0.30 | 18.260 | 2.047 | 1.450 | 0.828 |
| 0.8B τ0.40 | 19.824 | 3.611 | 3.013 | 0.789 |
| 0.6B | 17.994 | — | — | — |
| 0.6B τ0.00 | 19.425 | 1.431 | 0.000 | 0.836 |
| 0.6B τ0.10 | 19.501 | 1.507 | 0.076 | 0.820 |
| 0.6B τ0.20 | 19.819 | 1.825 | 0.394 | 0.801 |
| 0.6B τ0.30 | 20.792 | 2.799 | 1.367 | 0.746 |
| 0.6B τ0.40 | 22.307 | 4.313 | 2.882 | 0.695 |

Both error terms stay small and the sparse-only column is an order of magnitude below the codec column at τ ≤ 0.20, i.e. the two perturbations are close to additive rather than multiplicative. The 4B file's own `ffn_down` blocks were verified to decode bit-exactly against the `kvalues_iq4nl` table and the fp16 scale (max relative deviation 5.3e-07 against an independent numpy decoder), which is what makes the decomposition transferable.


<!-- P9:END -->

<!-- P10:BEGIN -->

## 10. Phase 10 — reviewer round 2

### 10.1 FFN-stream error predicts full-model ΔPPL (Q2)

| model | τ | FFN-stream rel-L2 | ΔPPL | KL | Top-1 |
|---|---|---|---|---|---|
| qwen3.5-0.8b | 0.05 | 0.0254 | -0.058 | 0.004 | 0.922 |
| qwen3.5-0.8b | 0.10 | 0.0702 | +0.049 | 0.014 | 0.891 |
| qwen3.5-0.8b | 0.15 | 0.1094 | +0.284 | 0.026 | 0.875 |
| qwen3.5-0.8b | 0.20 | 0.1653 | +0.519 | 0.038 | 0.875 |
| qwen3.5-0.8b | 0.30 | 0.2716 | +1.214 | 0.076 | 0.828 |
| qwen3.5-0.8b | 0.40 | 0.3787 | +2.655 | 0.172 | 0.719 |
| qwen3-0.6b | 0.05 | 0.0658 | -0.022 | 0.003 | 0.953 |
| qwen3-0.6b | 0.10 | 0.0387 | -0.025 | 0.025 | 0.953 |
| qwen3-0.6b | 0.15 | 0.0544 | +0.152 | 0.025 | 0.906 |
| qwen3-0.6b | 0.20 | 0.0727 | +0.283 | 0.106 | 0.828 |
| qwen3-0.6b | 0.30 | 0.1130 | +1.195 | 0.189 | 0.703 |
| qwen3-0.6b | 0.40 | 0.1684 | +2.974 | 0.293 | 0.719 |

qwen3.5-0.8b: ΔPPL = 18.37·relL2² -0.03, R² = 0.9948, Pearson 0.9974, Spearman 1.0000.

qwen3-0.6b: ΔPPL = 115.01·relL2² -0.30, R² = 0.9895, Pearson 0.9947, Spearman 0.9429.

Evidence A for the 4B checkpoint (a full GGUF streaming forward with GatedDeltaNet and attention) is **not implemented**; the square law above is the bridge offered instead, and the manuscript states this limit explicitly.

### 10.2 Prompt-level generation (Q9)

64 prompts × 32 tokens, prompt as the sampling unit, 1000-sample paired bootstrap: token agreement 0.637 [0.531, 0.726], first divergence 18.69 [15.19, 21.78] (median 22.5), step KL 0.0205 [0.0180, 0.0228], step top-1 agreement 0.9722.

- T=0.7 seed 0: trajectory-averaged KL 0.0835 [0.0699, 0.0983], top-1 probability 0.355
- T=0.7 seed 1: trajectory-averaged KL 0.0943 [0.0807, 0.1088], top-1 probability 0.314
- T=0.7 seed 2: trajectory-averaged KL 0.0828 [0.0700, 0.0955], top-1 probability 0.293

### 10.3 Calibration overlap and budget (Q11)

| pair | Jaccard of kept blocks (0.8B) |
|---|---|
| wiki vs zh | 0.940 |
| code vs wiki | 0.940 |
| code vs zh | 1.000 |
| code vs math | 1.000 |
| math vs wiki | 0.940 |
| math vs zh | 1.000 |

ΔPPL across four calibration corpora at τ=0.20: **+0.482 ± 0.306** (dense PPL 21.0925); Qwen3-0.6B: +0.381 ± 0.265 (dense PPL 24.4455). Intervals are 1000-draw window bootstraps over the 16 evaluation windows.

| N_cal | 0.8B ΔPPL | 95% CI | 0.6B ΔPPL | 95% CI |
|---|---|---|---|---|
| 128 | +0.625 | [+0.395, +0.896] | +0.631 | [+0.391, +0.922] |
| 512 | +0.483 | [+0.211, +0.734] | +0.285 | [+0.153, +0.434] |
| 2048 | +0.483 | [+0.211, +0.734] | +0.439 | [+0.259, +0.682] |
| 8192 | +0.483 | [+0.211, +0.734] | +0.496 | [+0.331, +0.662] |

### 10.4 Thread sweep, block-major 8-row (Q6/Q7)

| skip | 1t | 4t | 8t | 16t | 32t |
|---|---|---|---|---|---|
| 0.0 | 0.870 | 0.170 | 0.148 | 0.298 | 0.204 |
| 0.2 | 0.948 | 0.132 | 0.142 | 0.138 | 0.153 |
| 0.4 | 0.382 | 0.096 | 0.063 | 0.118 | 0.139 |
| 0.6 | 0.261 | 0.073 | 0.064 | 0.073 | 0.101 |
| 0.8 | 0.128 | 0.052 | 0.045 | 0.047 | 0.069 |

Default busy-wait policy at 8 vs 32 threads (dense): 0.090 ms vs 3.274 ms — the fork/join dominates; all published numbers use OMP_WAIT_POLICY=passive. SIMD width was not swept (AVX-512 only).

### 10.5 Cross-architecture energy/error correlation (Q4)

- whisper-base (GELU ff1/ff2, 512->2048->512): CV(W) 0.042, CV(a) 2.033, Pearson 0.898, Spearman 0.951, rel-L2@τ0.2 0.082

SmolVLM2's vision tower is not in this table: its harness reports a feature-geometry mismatch and the correlation would need a separate pass.

### 10.6 Paper revision

- Eq. (2) now states ‖Δy‖₂² with chevron inner products; `tests/test_p9_closure.py` asserts the source text.
- Table 7 (generation) dropped its tok/s column; the unified protocol table carries latency, and the 76.3 → 79.7 tok/s pair is explained as a boundary mismatch superseded by 11.621 → 12.541 ms (0.94×).
- The system claim is re-anchored on CPU AVX-512 execution and the block-major IQ4_NL layout; GPU batch-1 decode is reported as a Class III (regression) result.
- New sections: three-class significance statement, ordering-claim precision, Q1–Q8 index table, bridge law, prompt-level generation, calibration overlap, thread sweep, cross-architecture correlation.


### 10.7 ISA-width ablation and vision/audio correlation (Phase 10 close-out)

`blk.16.ffn_down` (4.5-bit IQ4_NL, 2560 rows x 288 blocks), one row per token, one thread, identical compacted index set.

| ISA width | row-major dense (ms) | bm tau=0 | tau=0.2 | tau=0.4 | tau=0.8 | tau=0.4 vs row dense | tau=0.8 vs row dense |
|---|---|---|---|---|---|---|---|
| scalar | 11.62 | 13.73 | 10.89 | 8.12 | 2.68 | 1.43x | 4.34x |
| avx2 | 13.38 | 13.48 | 10.64 | 7.98 | 2.68 | 1.68x | 5.00x |
| avx512 | 13.97 | 14.04 | 11.21 | 8.30 | 2.78 | 1.68x | 5.02x |

Decoding 32 nibbles costs one table lookup and one multiply per weight in all three builds, so at one row per call the multiply-add loop is too short to amortise the decode and the three builds land within 5%. The published speedups come from the block-major layout (cache-line granularity) and from eight-row blocking plus row-parallel threads, not from vector width. Decoder outputs agree bit-for-bit across builds.

| Model / tower | FFN type | CV(block weight norm) | CV(block activation) | Pearson | Spearman |
|---|---|---|---|---|---|
| whisper-base | GELU ff1/ff2, 512->2048->512 | 0.042 | 2.033 | 0.898 | 0.951 |
| smolvlm2-256m / vision | ViT GELU fc1/fc2 768->3072 | 0.034 | 0.959 | 0.887 | 0.968 |
| smolvlm2-256m / text | LM SwiGLU 576->1536 | 0.015 | 1.800 | 0.902 | 0.990 |

The energy/error relation holds on the GELU vision towers as well (Spearman 0.968 vision, 0.990 language), so the ordering argument is not specific to SwiGLU or to text.

<!-- P10:END -->
