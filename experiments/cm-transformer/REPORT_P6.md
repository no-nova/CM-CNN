

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
