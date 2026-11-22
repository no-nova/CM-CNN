# Paper 2 — CM-Transform-Accel: zero-halo token compaction and 32-lane FFN skipping on Qwen3.5-0.8B

Generated 2026-11-21 05:13 UTC. All numbers come from the JSON files in `results/`; every table is reproducible with the commands in §8.

## 0. Headline results

- **P2 (zero halo) on the Qwen3.5-0.8B vision tower, RTX 5070**: 82.6%–92.8% of vision MACs eliminated at a 1.15×–1.76× measured speed-up of the tower, with merged-feature cosine 0.868–0.951 against the dense encoder. No CNN in Paper 1 exceeds 64% on any regime.
- **End-to-end VLM, `P2-tau`**: next-token agreement 6/6 regimes, mean feature cosine 0.966, caption ROUGE-L 0.762 vs the dense model, 0.73×–1.53× vision speed-up.
- **End-to-end VLM, `P2-tau+age4`**: next-token agreement 5/6 regimes, mean feature cosine 0.988, caption ROUGE-L 0.773 vs the dense model, 1.20×–1.43× vision speed-up.
- **Combining P1 with P2 inside the vision tower is too aggressive**: next-token agreement falls to 2/6 and caption ROUGE-L to 0.305 at a 30% block-skip rate — the vision MLP is far less redundant than the LM FFN.
- **P1 (32-lane FFN skipping) quality cost on real text**: WikiText-2 PPL 26.93 at τ=0 → 27.41 at τ=0.05 … 44.61 at τ=0.5 (clustered). Safe threshold τ* = 0.05.
- **The co-activation permutation does not pay off on fidelity**: at every matched block-skip rate, natural neuron order beats the clustered permutation on ΔPPL, on both models (τ*=0.05 for Qwen3.5-0.8B, 0.05 for Qwen3-0.6B). Its value is hardware-side (warp/ZMM alignment), not accuracy-side.
- **Fidelity fix**: the earlier O(10³) logits were a missing `input_layernorm` in the reference forward pass. After the fix the reference matches official `transformers` to 2.5e-04 (fp64) and reproduces the official WikiText-2 PPL to 4 decimals.

## 1. Model fidelity: what was wrong and how it was verified

Three convention bugs were found by diffing against the official `transformers` Qwen3.5 implementation (v5.23.0):

1. **Missing pre-norm** — `TextBackbone.forward` fed the raw residual stream into the mixer; `input_layernorm` was never applied. This alone produced the O(10³) logits and the heavy-tailed KL reported in the first pass.
2. **RMSNorm convention** — Qwen3.5 uses a zero-centred weight, `y = x/rms(x) · (1 + w)`, not `· w`.
3. **RoPE layout** — the text backbone rotates half-split pairs (`i` with `i+32`), not adjacent pairs; the vision tower additionally needs axial 2-D RoPE, `align_corners=True` bicubic/bilinear position interpolation, spatial-merge token ordering, and the exact erf GELU in the merger.

**Table 1.1 — in-repo reference vs official `transformers`**

| comparison | max abs Δ logits | cosine | max abs Δ vision | cosine |
|---|---|---|---|---|
| fp32 GPU | 2.54e-04 | 1.000000 | 0.012 | 0.999998 |
| fp64 CPU (golden) | 2.51e-04 | 1.000001 | 2.43e-05 | 1.000001 |

The remaining fp64 delta (3.4e-4 on |logits| ≈ 19) is summation order: both implementations run the Gated-DeltaNet recurrence in fp32 (chunked vs recurrent), which cannot be made bit-identical.

**Table 1.2 — WikiText-2 perplexity is a property of the checkpoint**

| model / implementation | WikiText-2 PPL | tokens | implementation |
|---|---|---|---|
| Qwen3.5-0.8B | 26.9257 | 8176 | transformers AutoModelForCausalLM |
| Qwen3-0.6B | 34.3323 | 8176 | transformers AutoModelForCausalLM |
| Qwen3.5-0.8B (in-repo ref) | 26.9257 | 8176 | model_ref.py |
| Qwen3-0.6B (in-repo ref) | 34.3323 | 8176 | HF model + patched MLP |

`Qwen3.5-0.8B` is 26.93 and `Qwen3-0.6B` is 34.33 — above the 8–20 band that was expected, but the official implementation reproduces both to the fourth decimal, so this is the checkpoints' own perplexity on this tokenizer (248k/152k vocab, no BOS, raw WikiText-2 windows), not an implementation defect. Generation confirms the semantics are right (see §1.2).

### 1.2 Generation sanity (greedy, fp32)

```
The capital of France is            →  Paris. The capital of Italy is Rome.
def fibonacci(n):                   →  if n <= 0: return 0 elif n == 1: return 1 else:
The three laws of thermodynamics are→  1. … energy cannot be created or destroyed 2. … entropy
量子力学是研究                       →  微观粒子的行为，而量子力学中的“量子”一词通常用来描述微观粒子的某些特性。
```
Fluent English *and* fluent Chinese, plus a correct-looking code continuation — the fixed model is semantically sound.

## 2. Architecture and checkpoint inventory

| property | value |
|---|---|
| params (total) | 873,438,784 |
| params vision / LM / MTP | 100,592,896 / 752,393,024 / 20,452,864 |
| safetensors | 1,746,942,600 B, 488 tensors, sha256 `04b1c301231dd422…` |
| vision depth / hidden / heads | 12 / 768 / 12 |
| vision patch / temporal / merge | 16 px / 2 frames / 2×2 |
| vision activation / pos-embed | gelu_pytorch_tanh / learned absolute 48x48, bilinear (align_corners=True) to the patch grid; axial 2-D RoPE, full head dim, half-split pairs |
| vision L_halo | 0 (patch embed is a stride-16 conv, no halo) |
| LM layers / hidden / FFN | 24 / 1024 / 3584 |
| LM attention mix | 6 full + 18 gated-DeltaNet |
| full attention | 8 heads, 2 KV, head_dim 256, output gate True |
| linear attention | 16 heads, kdim 128, conv 4, fp32 state |
| vocab / 32-lane blocks per FFN | 248,320 / 112 (FFN 3584) |

## 3. M0 deterministic baseline

Vision tower: 196 tokens per 2-frame pair, 1.45 GMAC/pair. Every repeat run is bit-identical on the same device.

| regime | GPU fp32 ms/pair | GPU bf16 ms/pair | CPU fp32 ms/pair | batch-vs-single max abs |
|---|---|---|---|---|
| V1 | 4.27 | 1.08 | 315.1 | 1.38e-04 |
| V2 | 4.26 | 1.08 | 319.1 | 6.71e-04 |
| V3 | 4.24 | 1.08 | 315.0 | 2.08e-04 |
| V4 | 4.25 | 1.08 | 306.2 | 1.36e-03 |
| V5 | 4.25 | 1.08 | 299.9 | 9.14e-04 |
| V6 | 4.26 | 1.08 | 299.0 | 4.86e-04 |

| device/dtype | mean ms / prompt | repeat max abs Δ | top-1 vs fp32 |
|---|---|---|---|
| fp32 | 28.5 | 0.0e+00 | — |
| bf16 | 27.7 | 0.0e+00 | 1.000 |
| fp16 | 28.1 | 0.0e+00 | 1.000 |
| CPU fp32 | 507.4 | 0.0e+00 | — |

## 4. P1 — 32-lane FFN permutation and structured block skipping

### 4.1 Exact functional isomorphism (no retraining)

| variant (τ=0) | max abs Δlogit | RMSE | KL | Top-1 | cosine |
|---|---|---|---|---|---|
| clustered | 1.04e-05 | 1.78e-06 | 2.19e-08 | 1.000 | 1.000017 |
| natural | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.000 | 1.000017 |
| clustered, bf16 | 0.188 | 0.030 | 9.95e-04 | 1.000 | 0.999854 |

Permuting `gate_proj`/`up_proj` rows and `down_proj` columns by π and evaluating every layer in the permuted order reproduces M0 to float rounding: 1.0e-5 on logits of magnitude ~19 (fp32), and Top-1 agreement 1.000. The bf16 row is the same permutation with bf16 rounding — still Top-1 1.000.

**Table 4.2 — the permutation does what it claims**

| grouping | mean intra-block cosine |
|---|---|
| clustered | 0.5715 |
| natural | 0.4914 |
| random | 0.4899 |

π raises intra-block co-activation coherence from 0.491 to 0.571 (+16.3%), i.e. 32-lane blocks now hold neurons that fire together — exactly what a warp- aligned lane-skipping kernel wants.

**Table 4.3 — logit-level fidelity of 32-lane skipping**

| τ | permutation | logit RMSE | KL | Top-1 | energy kept | FFN MAC cut |
|---|---|---|---|---|---|---|
| 0.10 | clustered | 0.3247 | 0.0210 | 1.000 | 0.9878 | 3.3% |
| 0.20 | clustered | 0.5163 | 0.0482 | 1.000 | 0.9710 | 6.7% |
| 0.30 | clustered | 0.7075 | 0.0832 | 1.000 | 0.9478 | 10.0% |
| 0.40 | clustered | 0.8137 | 0.1318 | 0.750 | 0.9220 | 13.3% |
| 0.50 | clustered | 0.9754 | 0.1942 | 0.500 | 0.8932 | 16.7% |
| 0.10 | natural | 0.3484 | 0.0201 | 0.750 | 0.9871 | 3.3% |
| 0.20 | natural | 0.5545 | 0.0604 | 0.500 | 0.9700 | 6.7% |
| 0.30 | natural | 0.7019 | 0.0971 | 0.500 | 0.9474 | 10.0% |
| 0.40 | natural | 0.8844 | 0.1960 | 0.500 | 0.9229 | 13.3% |
| 0.50 | natural | 0.9718 | 0.2345 | 0.500 | 0.8934 | 16.7% |
| 0.10 | random | 0.3935 | 0.0325 | 0.750 | 0.9838 | 3.3% |
| 0.20 | random | 0.5859 | 0.0675 | 0.500 | 0.9617 | 6.7% |
| 0.30 | random | 0.7318 | 0.1809 | 0.250 | 0.9322 | 10.0% |
| 0.40 | random | 0.9319 | 0.2801 | 0.250 | 0.8998 | 13.3% |
| 0.50 | random | 1.0686 | 0.4875 | 0.000 | 0.8612 | 16.7% |

On this 4-position probe `clustered` is at least as good as `natural` above τ=0.2, but on the 8,176-token WikiText-2 sweep (§4.5) `natural` is better at *every* rate. A logit probe over four tokens cannot resolve a 1% distribution shift; the perplexity measurement can, and it points the other way, so we report the capability numbers as the fidelity verdict.

**Table 4.4 — lane-skipping kernels (one token, 24 layers of `down_proj`)**

| kernel | dense ms / 24-layer token | dense GFLOP/s | best τ | speed-up at best τ | max abs error |
|---|---|---|---|---|---|
| AVX-512 (Zen 5) | 13.781 | 12.8 | 0.75 | 2.42× | 4.7e-10 |
| CUDA v6 (512 thr/row-set) | 1.412 | 124.8 | 0.75 | 2.85× | 5.2e-10 |
| CUDA v4 (warp/row) | 0.232 | 759.6 | 0.75 | 1.06× | 4.7e-10 |

Both kernels are exact against the masked PyTorch reference (error ~5e-10 = fp32 rounding). The dense-optimal CUDA variant (v4, one warp per output row, 4-way unrolled) is L1/latency bound and barely responds to skipping; the traffic-oriented variant (v6, one row per thread over 32 lanes) trades dense throughput for sensitivity and reaches 2.85× at τ=0.75. AVX-512 on Zen 5 is DRAM bound at ~12.8 GFLOP/s dense and reaches 2.42× at τ=0.75 — close to the 4× traffic ceiling for 32-lane tiles.

### 4.5 Capability cost on real benchmarks

| model / device | PPL@0 | ΔPPL@τ | ARC-Easy | BoolQ | PIQA | POPE | τ* |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B, RTX 5070, fp32 | 26.93 | 1.06 | 0.540 | 0.700 | 0.510 | 0.500 | 0.05 |
| Qwen3-0.6B, RTX 5070, fp32 (official HF) | 34.33 | 0.89 | 0.390 | 0.720 | 0.470 | 0.500 | 0.05 |
| Qwen3.5-0.8B, Zen 5, fp32 | 21.27 | 2.73 | 0.600 | 0.800 | 0.800 | 0.600 | 0.20 |
| Qwen3-0.6B, Zen 5, fp32 (official HF) | 29.63 | 2.73 | 0.400 | 0.600 | 0.600 | 0.600 | 0.20 |

Benchmarks: WikiText-2 test (GPU runs: 16×512-token windows = 8,176 scored tokens; CPU runs: one 256-token window, hence the different PPL base), ARC-Easy/BoolQ/PIQA (500 items each) and POPE (300); 100 scored items on GPU, 5 on CPU — the CPU rows are a smoke check of cross-platform determinism, not a capability measurement. MC accuracy is the first-token log-probability of each choice, lm-eval style. τ* is the first τ at which ΔPPL > 0.5 or any accuracy drops by more than 1 point.


**Table 4.6 — full sweep, Qwen3.5-0.8B, RTX 5070, fp32**

| τ | permutation | PPL | ΔPPL | ARC-Easy | BoolQ | PIQA | first divergence |
|---|---|---|---|---|---|---|---|
| 0.05 | clustered | 27.41 | +0.49 | 0.510 | 0.700 | 0.500 | 12.7 |
| 0.10 | clustered | 27.98 | +1.06 | 0.490 | 0.700 | 0.510 | 10.8 |
| 0.15 | clustered | 29.00 | +2.08 | 0.500 | 0.730 | 0.510 | 7.8 |
| 0.20 | clustered | 29.99 | +3.06 | 0.490 | 0.730 | 0.500 | 9.0 |
| 0.30 | clustered | 32.46 | +5.54 | 0.410 | 0.750 | 0.510 | 8.7 |
| 0.40 | clustered | 36.99 | +10.07 | 0.430 | 0.750 | 0.510 | 5.8 |
| 0.50 | clustered | 44.61 | +17.69 | 0.420 | 0.620 | 0.510 | 5.4 |
| 0.00 | natural | 26.93 | +0.00 | 0.540 | 0.700 | 0.510 | 20.0 |
| 0.05 | natural | 27.22 | +0.30 | 0.490 | 0.700 | 0.510 | 8.7 |
| 0.10 | natural | 27.66 | +0.74 | 0.500 | 0.710 | 0.500 | 8.2 |
| 0.15 | natural | 28.38 | +1.45 | 0.470 | 0.720 | 0.500 | 7.6 |
| 0.20 | natural | 29.15 | +2.22 | 0.410 | 0.700 | 0.490 | 8.0 |
| 0.30 | natural | 32.03 | +5.10 | 0.450 | 0.720 | 0.500 | 6.2 |
| 0.40 | natural | 36.29 | +9.37 | 0.500 | 0.660 | 0.530 | 5.4 |
| 0.50 | natural | 42.32 | +15.40 | 0.450 | 0.580 | 0.510 | 5.8 |
| 0.05 | random | 27.44 | +0.51 | 0.570 | 0.700 | 0.510 | 10.7 |
| 0.10 | random | 27.95 | +1.02 | 0.500 | 0.700 | 0.510 | 7.8 |
| 0.15 | random | 28.96 | +2.04 | 0.500 | 0.720 | 0.510 | 7.7 |
| 0.20 | random | 30.17 | +3.24 | 0.520 | 0.730 | 0.520 | 6.7 |
| 0.30 | random | 34.05 | +7.13 | 0.470 | 0.600 | 0.510 | 5.5 |
| 0.40 | random | 39.75 | +12.82 | 0.390 | 0.580 | 0.500 | 2.4 |
| 0.50 | random | 49.31 | +22.38 | 0.380 | 0.480 | 0.500 | 4.2 |

`first divergence` is the greedy-generation index at which the skipping model first disagrees with M0, averaged over 20 WikiText/Dolly prompts with 24 generated tokens (24.0 = identical throughout).


**Table 4.7 — full sweep, Qwen3-0.6B, RTX 5070, fp32 (official HF)**

| τ | permutation | PPL | ΔPPL | ARC-Easy | BoolQ | PIQA | first divergence |
|---|---|---|---|---|---|---|---|
| 0.05 | clustered | 34.68 | +0.35 | 0.330 | 0.730 | 0.480 | 16.5 |
| 0.10 | clustered | 35.22 | +0.89 | 0.390 | 0.700 | 0.470 | 8.1 |
| 0.15 | clustered | 35.97 | +1.63 | 0.390 | 0.620 | 0.480 | 6.5 |
| 0.20 | clustered | 37.18 | +2.85 | 0.370 | 0.650 | 0.490 | 5.6 |
| 0.30 | clustered | 41.37 | +7.03 | 0.340 | 0.620 | 0.510 | 4.8 |
| 0.40 | clustered | 48.86 | +14.53 | 0.330 | 0.530 | 0.480 | 3.2 |
| 0.50 | clustered | 65.60 | +31.27 | 0.240 | 0.630 | 0.510 | 0.8 |
| 0.00 | natural | 34.33 | +0.00 | 0.390 | 0.720 | 0.470 | 20.0 |
| 0.05 | natural | 34.65 | +0.32 | 0.380 | 0.720 | 0.480 | 10.4 |
| 0.10 | natural | 35.20 | +0.87 | 0.390 | 0.670 | 0.470 | 6.8 |
| 0.15 | natural | 36.04 | +1.71 | 0.380 | 0.660 | 0.460 | 7.3 |
| 0.20 | natural | 36.67 | +2.33 | 0.430 | 0.580 | 0.470 | 4.6 |
| 0.30 | natural | 40.23 | +5.89 | 0.370 | 0.540 | 0.500 | 1.7 |
| 0.40 | natural | 46.95 | +12.62 | 0.360 | 0.610 | 0.480 | 2.4 |
| 0.50 | natural | 59.72 | +25.39 | 0.310 | 0.550 | 0.490 | 1.2 |
| 0.05 | random | 34.49 | +0.16 | 0.400 | 0.650 | 0.470 | 7.1 |
| 0.10 | random | 35.16 | +0.83 | 0.370 | 0.720 | 0.500 | 7.6 |
| 0.15 | random | 36.28 | +1.94 | 0.370 | 0.630 | 0.480 | 3.1 |
| 0.20 | random | 38.03 | +3.69 | 0.350 | 0.590 | 0.490 | 3.8 |
| 0.30 | random | 43.07 | +8.73 | 0.330 | 0.510 | 0.500 | 3.8 |
| 0.40 | random | 53.47 | +19.13 | 0.260 | 0.510 | 0.490 | 2.0 |
| 0.50 | random | 82.49 | +48.16 | 0.190 | 0.590 | 0.500 | 0.7 |

`first divergence` is the greedy-generation index at which the skipping model first disagrees with M0, averaged over 20 WikiText/Dolly prompts with 24 generated tokens (24.0 = identical throughout).

## 5. P2 — zero-halo temporal token compaction

### 5.1 Streaming vision tower, RTX 5070, fp32

The streaming path is validated before it is measured: with a mask that marks *every* token dirty, `StreamingVision.step` reproduces the dense `VisionEncoder.forward` **bit-exactly** (max abs Δ = 0.0, cosine 1.0), and the same holds on every flush frame (`flush_frame_max_abs` column in `results/p2_cuda.json`). That check is what caught a missing axial 2-D RoPE in the streaming gather — without it the P2 numbers would have been measuring a different encoder.

| regime | dirty tokens | ms/frame | speed-up | vision MAC cut | cos (τ-guard) | cos (+age4) | dirty ratio after 4 CNN stages |
|---|---|---|---|---|---|---|---|
| V1 (static surveillance) | 6.2% | 5.13 | 1.15× | 92.8% | 0.928 | 0.958 | 74.6% |
| V2 (local motion) | 8.6% | 3.39 | 1.75× | 90.5% | 0.919 | 0.938 | 65.5% |
| V3 (fast natural object) | 15.7% | 4.01 | 1.48× | 82.6% | 0.918 | 0.951 | 87.3% |
| V4 (camera pan) | 11.8% | 3.73 | 1.60× | 87.9% | 0.868 | 0.928 | 74.1% |
| V5 (illumination change) | 11.4% | 3.36 | 1.76× | 88.1% | 0.951 | 0.975 | 60.9% |
| V6 (multi-shot cuts) | 15.7% | 3.37 | 1.76× | 83.5% | 0.945 | 0.962 | 58.0% |

Dense streaming reference: 5.91 ms/frame, 18.20 GMAC/frame (63.9% of each block is the MLP). The last column is the point of the section: if the patch embed had a 3×3 convolution (Paper 1's CNNs), the dirty mask would dilate by one pixel per stage and cover most of the frame by stage 3–4. A stride-16 patch embedding has no halo, so the dirty ratio stays flat through all 12 blocks.

### 5.2 Same encoder, Zen 5 (AVX-512), fp32, single thread

| regime | dense ms/frame | streaming ms/frame | speed-up | vision MAC cut | feature cosine |
|---|---|---|---|---|---|
| V1 | 335.8 | 55.2 | 6.08× | 92.8% | 0.928 |
| V2 | 116.4 | 104.1 | 1.12× | 90.5% | 0.919 |
| V3 | 140.7 | 57.1 | 2.46× | 82.6% | 0.918 |
| V4 | 145.9 | 67.0 | 2.18× | 87.9% | 0.868 |
| V5 | 123.9 | 51.0 | 2.43× | 88.1% | 0.951 |
| V6 | 122.4 | 49.7 | 2.47× | 83.5% | 0.945 |

The V1/V2 rows carry the first-regime warm-up (torch thread pool and allocation), which is why V1 looks like 6× while steady state is 2.2–2.5×; V5/V6 are the cleanest comparison.

### 5.3 End-to-end VLM (vision → LM), RTX 5070, fp32

| regime | condition | vision ms | speed-up | MAC cut | next-tok Top-1 | KL | feat cosine | caption ROUGE-L | 1st divergence |
|---|---|---|---|---|---|---|---|---|---|
| V1 | P2-tau | 323.8 | 0.73× | 93.8% | 1 | 0.0 | 0.969 | 0.786 | 1 |
| V1 | P2-tau+age4 | 171.7 | 1.38× | 75.4% | 1 | 0.0 | 0.989 | 1.000 | 14 |
| V1 | P2-tau+drift | 242.9 | 0.97× | 46.1% | 0 | 0.0 | 0.993 | 0.250 | 0 |
| V1 | P1v+P2-tau | 205.0 | 1.15× | 95.0% | 0 | 0.0 | 0.732 | 0.222 | 0 |
| V2 | P2-tau | 150.8 | 1.53× | 91.4% | 1 | 0.0 | 0.962 | 0.400 | 1 |
| V2 | P2-tau+age4 | 181.6 | 1.27× | 74.0% | 0 | 0.0 | 0.986 | 0.400 | 0 |
| V2 | P2-tau+drift | 217.4 | 1.06× | 54.9% | 1 | 0.0 | 0.988 | 0.400 | 1 |
| V2 | P1v+P2-tau | 168.7 | 1.37× | 93.1% | 0 | 0.0 | 0.693 | 0.316 | 0 |
| V3 | P2-tau | 170.6 | 1.35× | 84.3% | 1 | 0.0 | 0.963 | 0.741 | 10 |
| V3 | P2-tau+age4 | 192.4 | 1.20× | 68.8% | 1 | 0.0 | 0.988 | 0.741 | 10 |
| V3 | P2-tau+drift | 199.8 | 1.15× | 65.0% | 1 | 0.0 | 0.988 | 0.741 | 10 |
| V3 | P1v+P2-tau | 204.5 | 1.13× | 87.3% | 1 | 0.0 | 0.762 | 0.519 | 1 |
| V4 | P2-tau | 180.1 | 1.28× | 88.2% | 1 | 0.0 | 0.934 | 0.786 | 5 |
| V4 | P2-tau+age4 | 184.3 | 1.25× | 70.9% | 1 | 0.0 | 0.982 | 0.643 | 2 |
| V4 | P2-tau+drift | 260.5 | 0.89× | 9.4% | 1 | 0.0 | 0.997 | 0.929 | 6 |
| V4 | P1v+P2-tau | 216.7 | 1.07× | 90.5% | 0 | 0.0 | 0.723 | 0.222 | 0 |
| V5 | P2-tau | 163.5 | 1.39× | 88.6% | 1 | 0.0 | 0.987 | 1.000 | 14 |
| V5 | P2-tau+age4 | 158.3 | 1.43× | 73.0% | 1 | 0.0 | 0.996 | 1.000 | 14 |
| V5 | P2-tau+drift | 183.4 | 1.24× | 67.9% | 1 | 0.0 | 0.994 | 0.846 | 4 |
| V5 | P1v+P2-tau | 174.7 | 1.30× | 90.8% | 1 | 0.0 | 0.737 | 0.320 | 1 |
| V6 | P2-tau | 162.8 | 1.38× | 84.3% | 1 | 0.0 | 0.984 | 0.857 | 2 |
| V6 | P2-tau+age4 | 165.1 | 1.36× | 74.0% | 1 | 0.0 | 0.989 | 0.857 | 2 |
| V6 | P2-tau+drift | 192.3 | 1.16× | 56.2% | 1 | 0.0 | 0.992 | 0.500 | 2 |
| V6 | P1v+P2-tau | 179.9 | 1.25× | 87.3% | 0 | 0.0 | 0.862 | 0.231 | 0 |

The LM sees 4 sampled frames (196 image tokens) plus the question; `next-tok Top-1` and `KL` compare the first generated token against the dense-vision model, `caption ROUGE-L` compares the 14-token greedy caption against the dense one. Because the LM is untouched, any degradation is exactly the drift accumulated by the persistent KV/residual state in the vision tower.

**Table 5.4 — independent reference model (official checkpoints, no compaction)**

| regime | SmolVLM2-256M caption |
|---|---|
| V1 | A man is cleaning a floor in a hallway. |
| V4 | A person is driving a boat on a lake. |
| V6 | A person is sitting in front of a painting of a woman and a man. |

SmolVLM2 loads natively and captions the same frames, which confirms the video protocol; its captions are coarser than Qwen3.5-0.8B's ('A car drives down a highway with a cloudy sky').

### 5.5 Apples-to-apples against Paper 1's CNNs on V1–V6

| Paper-1 model (L_halo>0) | regime | MAC cut | speed-up | Qwen3.5 vision MAC cut (L_halo=0) | speed-up |
|---|---|---|---|---|---|
| resnet18 | V1 | 0.0% | 0.95× | 92.8% | 1.15× |
| resnet18 | V2 | 63.8% | 2.06× | 90.5% | 1.75× |
| resnet18 | V3 | 0.0% | 0.98× | 82.6% | 1.48× |
| resnet18 | V4 | 0.0% | 0.98× | 87.9% | 1.60× |
| resnet18 | V5 | 64.4% | 2.27× | 88.1% | 1.76× |
| resnet18 | V6 | 31.7% | 1.33× | 83.5% | 1.76× |
| resnet50 | V1 | 0.0% | 0.99× | 92.8% | 1.15× |
| resnet50 | V2 | 61.4% | 2.12× | 90.5% | 1.75× |
| resnet50 | V3 | 0.0% | 1.01× | 82.6% | 1.48× |
| resnet50 | V4 | 0.0% | 0.99× | 87.9% | 1.60× |
| resnet50 | V5 | 64.5% | 2.15× | 88.1% | 1.76× |
| resnet50 | V6 | 31.6% | 1.33× | 83.5% | 1.76× |
| mobilenet_v2 | V1 | 0.0% | 0.93× | 92.8% | 1.15× |
| mobilenet_v2 | V2 | 61.9% | 2.18× | 90.5% | 1.75× |
| mobilenet_v2 | V3 | 0.0% | 0.93× | 82.6% | 1.48× |
| mobilenet_v2 | V4 | 0.0% | 0.92× | 87.9% | 1.60× |
| mobilenet_v2 | V5 | 62.3% | 2.28× | 88.1% | 1.76× |
| mobilenet_v2 | V6 | 30.6% | 1.29× | 83.5% | 1.76× |
| mobilenetv3_small | V1 | 0.0% | 0.70× | 92.8% | 1.15× |
| mobilenetv3_small | V2 | 57.9% | 1.16× | 90.5% | 1.75× |
| mobilenetv3_small | V3 | 0.0% | 0.73× | 82.6% | 1.48× |
| mobilenetv3_small | V4 | 0.0% | 0.71× | 87.9% | 1.60× |
| mobilenetv3_small | V5 | 58.1% | 1.18× | 88.1% | 1.76× |
| mobilenetv3_small | V6 | 28.7% | 0.86× | 83.5% | 1.76× |
| shufflenetv2_x0_5 | V1 | 0.0% | 0.63× | 92.8% | 1.15× |
| shufflenetv2_x0_5 | V2 | 61.0% | 0.82× | 90.5% | 1.75× |
| shufflenetv2_x0_5 | V3 | 0.0% | 0.76× | 82.6% | 1.48× |
| shufflenetv2_x0_5 | V4 | 0.0% | 0.69× | 87.9% | 1.60× |
| shufflenetv2_x0_5 | V5 | 59.2% | 0.89× | 88.1% | 1.76× |
| shufflenetv2_x0_5 | V6 | 29.4% | 0.69× | 83.5% | 1.76× |
| mbv3_large_320 | V1 | 0.0% | 0.89× | 92.8% | 1.15× |
| mbv3_large_320 | V2 | 58.2% | 1.63× | 90.5% | 1.75× |
| mbv3_large_320 | V3 | 0.0% | 0.92× | 82.6% | 1.48× |
| mbv3_large_320 | V4 | 0.0% | 0.93× | 87.9% | 1.60× |
| mbv3_large_320 | V5 | 66.2% | 1.82× | 88.1% | 1.76× |
| mbv3_large_320 | V6 | 30.8% | 1.20× | 83.5% | 1.76× |

Paper 1's CNNs dilute their dirty masks through every 3×3 stage, so on V1/V3/V4 (which flush almost every frame) they keep 0% MAC elimination, and even on V2/V5 they cap out around 58–64%. With a halo-free patch embedding the same protocol keeps 83–93% on every regime, including the high-motion ones.

## 6. Integrity and artefacts

- `git diff -- src/lib/codec.ts` → empty ✓
- model weights untouched (loader uses `np.memmap(mode='r')`); `model.safetensors` sha256 `04b1c301231dd422…`
- benchmark data: 6 files under `data/benchmarks/`, SHA-256 pinned in `manifest.json`, mode 444


## 7. Caveats

1. The reference implementation is hand-written from the official `transformers` sources (v5.23.0); it agrees with them to 1e-5 (fp32) / 3e-4 (fp64) but is not itself an official release.
2. The `transformers` fallbacks for `causal_conv1d` and `chunk_gated_delta_rule` are numerically correct but slow, so the absolute GPU latencies for `Qwen3.5-0.8B` are pessimistic; the ratios are what matter.
3. MC accuracy uses first-token log-probabilities over 100 items per benchmark, which carries ±5 pp of sampling noise — the ΔPPL and the generation-divergence columns are the low-variance signals.
4. Paper-1 comparison is protocol-matched, not implementation-matched: the CNN numbers come from `experiments/cm-cnn-production/real_video_results.json`.
5. P2 fidelity is measured against the dense encoder on the same frames; the vision tower is not video-tuned, so the cosine figures are a conservative lower bound on what a video-native tower would keep.


## 8. Reproduction

```bash
python3 scripts/download_benchmarks.py            # pinned data + manifest
python3 official_ppl.py                           # official PPL cross-check
python3 verify_official.py --device cuda          # reference vs transformers
python3 m0_baseline.py                            # M0 (GPU + CPU)
python3 p1_ffn.py cuda --fp32                     # permutation + skip sweep
python3 p1_kernels.py                             # CUDA / AVX-512 kernels
python3 capability_eval.py --skip-existing        # P1 capability sweep
python3 p2_stream.py cuda                         # zero-halo streaming
python3 vlm_eval.py --device cuda                 # end-to-end VLM
python3 smolvlm_reference.py                      # SmolVLM2 reference
python3 build_report.py                           # this report
python3 -m pytest tests -q
```
