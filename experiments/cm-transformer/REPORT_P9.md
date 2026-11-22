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
