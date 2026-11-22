<!-- P7:BEGIN -->
## 9. Phase 7 — close-out checks

### 9.1 Block-major IQ4_NL on real text activations (4B)  

Input: 64 real WikiText-2 tokens (`token_embd` decoded from Q6_K), zero-centred RMSNorm, and residual propagation through the checkpoint's own IQ4_NL gate/up/down for all 32 layers (real WikiText-2 token embeddings, RMSNormed, propagated through the real FFN residual stream (no attention); attention excluded). 16 threads.

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
