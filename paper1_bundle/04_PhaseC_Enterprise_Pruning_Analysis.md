# Phase C: Enterprise Hybrid — Structured Channel Pruning + CM Dynamic Spatial-Tile Gating

Question: **when enterprise-grade preprocessing (structured pruning + short fine-tune) is allowed, can CM's
content-adaptive `2x2` spatial-tile skipping beat both the original dense network (M0) and ALCS/AMC-style static
pruning (S1) on real single-core AVX-512 wall clock?**

Answer, measured: **static pruning yes, spatial-tile gating no.** Structured pruning removes 52.05% / 52.29% of
ResNet-18 / MobileNetV2 MACs and turns that straight into 2.40x / 1.70x single-core wall-clock speedup
with *better* accuracy than the teacher. The CM tile gate finds 9.7% of ResNet-18's tiles it dares to skip
inside a 0.35 pp dev budget (+1.30 pp of MAC cut, +2.0% wall clock), and on MobileNetV2 it calibrates
to "skip nothing". The reason is structural (sections 4 and 5), not an implementation defect, and it is measured from
two independent directions: the C++ gate calibration and a NumPy analysis of the branch tensors.

Recorded data: `results.json` (this directory). Engine: `native/cm_enterprise_avx512.cpp`. Weights:
`pruned_{resnet18,mobilenetv2}_rgb.bin` (chmod 444, SHA-256 in `results.json`), produced by
`scripts/prepare_enterprise_backbones.py` (`prune_report.json`).

---

## 1. Why this phase abandons scalar gather entirely

Phase B established that exact per-scalar input compaction costs ~2 load uops per FMA while the register-tiled dense
kernel costs 0.5, so compaction never wins on this core. Phase C therefore keeps **every** executed MAC on the
full-speed kernel:

* pruning keeps every width a multiple of 16 (depthwise 8), so all conv and 1x1 layers keep the `co % 64 == 0` path of
  `conv_tile` (2x2 pixels x 4 output groups = 16 resident ZMM accumulators, one 64 B weight row reused by 16 FMAs);
* the only "sparsity" left is *structural*: whole `2x2` tiles of residual branches that the gate decides not to
  compute. A skipped tile costs zero weight loads and zero FMAs, so the MAC cut translates into wall clock
  proportionally — provided the tile is genuinely skippable.

## 2. Structured pruning pipeline (`scripts/prepare_enterprise_backbones.py`, 41.6 s + 13.3 s on the RTX 5070)

1. Importances per channel *space* (stem, stage buses, BasicBlock internal axes, `InvertedResidual` expansions, head),
   aggregated over each space's producer and consumer convolutions (L2 of the incoming and outgoing weights, the
   Taylor-style proxy used by AMC).
2. Widths `al(w * alpha)` (multiple of 16) with `alpha` bisected on the exported graph's MAC count to hit a 50% target:
   ResNet-18 alpha 0.6719 -> widths 48/80/176/352; MobileNetV2 alpha 0.7047 -> 16-aligned stem/expansion/
   projection widths.
3. 12 epochs AdamW + cosine, batch 128, KD (`KL(log_softmax(student/2), softmax(teacher/2))`, weight 0.6) plus a
   `2x2`-tile sparsity term `4e-4 * mean_tile |branch output|` that teaches the residual branches to stay near zero on
   smooth tiles — i.e. we *trained* the network to be tile-skippable, and still measured what that bought.
4. BN folded into the preceding conv, exported as `CMUNIV02`, `chmod 444`, SHA-256 pinned, and re-executed from the
   exported bytes in NumPy to prove the file matches PyTorch.

| model | alpha | teacher test | pruned test | delta | MACs (teacher -> pruned) | cut |
|---|---|---|---|---|---|---|
| ResNet-18 | 0.6719 | 97.78% | 98.33% | +0.56 pp | 555,422,720 -> 266,309,056 | 52.05% |
| MobileNetV2 | 0.7047 | 96.94% | 98.33% | +1.39 pp | 52,302,848 -> 24,952,416 | 52.29% |

Pruning at 50% is *free* here, and the distillation term more than compensates the capacity loss.

## 3. The ablation waterfall (single core, `OMP_NUM_THREADS=1`, 10 reps, modes interleaved)

A skipped tile writes the dev-calibrated **mean** of the branch output (not zeros), so the residual add still gets a
calibrated constant; the work order allows this ("`y_tile = shortcut_tile` (or a calibrated mean bias)"). The random
control uses the per-block skip rate that S2 *actually realised on dev*, so the comparison is rate-for-rate.

ResNet-18:

| mode | Top-1 | MACs/img | cut | ms/img | speedup | logit RMSE vs M0 | tile skip |
|---|---|---|---|---|---|---|---|
| M0 original | 97.78% | 555,422,720 | 0.00% | 7.686 | 1.00x | 0.000e+00 | 0.00% |
| S1 static-pruned | 98.33% | 266,309,056 | 52.05% | 3.200 | 2.40x | 1.489e+00 | 0.00% |
| S2 + CM tile gate | 97.78% | 259,094,208 | 53.35% | 3.136 | 2.45x | 1.546e+00 | 9.72% |
| S2-random (equal rate) | 96.39% | 258,984,589 | 53.37% | 3.128 | 2.46x | 1.607e+00 | 9.78% |

MobileNetV2:

| mode | Top-1 | MACs/img | cut | ms/img | speedup | logit RMSE vs M0 | tile skip |
|---|---|---|---|---|---|---|---|
| M0 original | 96.94% | 52,302,848 | 0.00% | 0.623 | 1.00x | 0.000e+00 | 0.00% |
| S1 static-pruned | 98.33% | 24,952,416 | 52.29% | 0.367 | 1.70x | 1.360e+00 | 0.00% |
| S2 + CM tile gate | 98.33% | 24,952,416 | 52.29% | 0.366 | 1.70x | 1.360e+00 | 0.00% |
| S2-random (equal rate) | 98.33% | 24,952,416 | 52.29% | 0.365 | 1.70x | 1.360e+00 | 0.00% |

Two things to read carefully:

* S2 is **better than the equal-rate random control** (97.78% vs 96.39% at 9.7% vs 9.78%
  skip, logit RMSE 1.546 vs 1.607): the context table does pick the tiles whose branch output is
  closest to its mean. The gate works; there is almost nothing to pick.
* S2 is only 1.020x faster than S1 (2.0%), so the spatial gate cannot be the headline. Static
  pruning is.

## 4. Why the gate can only skip ~10% of tiles

The calibration is honest about what it can afford: for each block, deepest first, it takes the loosest
visit-weighted quantile of the learned per-context mean deviation whose dev accuracy stays within budget
(`max(S1_dev - 0.4 pp, M0_dev - 0.6 pp)`), and then re-validates on the whole dev set, disabling the most aggressive
block until the budget holds. Measured rejections (60-image dev subset, same binary):

| block | tiles | grid | accepted rate | accuracy at q=0.90 / 0.75 / 0.60 / 0.30 |
|---|---|---|---|---|
| 0 (nodes 1-3) | 16x16 | 8x8 | 0.0% | 31.7 / 55.0 / 66.7 / 78.3 |
| 1 (nodes 5-6) | 16x16 | 8x8 | 10% | 86.7 / 91.7 / 95.0 / 95.0 |
| 2 (nodes 8-9) | 8x8 | 4x4 | 0.0% | 86.7 / 91.7 / 95.0 / 95.0 |
| 3 (12-13) | 8x8 | 4x4 | 12% | 85.0 / 88.3 / 95.0 / 98.3 |
| 4 (15-16) | 4x4 | 2x2 | 0.0% | 90.0 / 93.3 / 91.7 / 96.7 |
| 5 (19-20) | 4x4 | 2x2 | 0.0% | 85.0 / 86.7 / 93.3 / 93.3 |
| 6 (22-23) | 2x2 | 1x1 | 0.0% | 83.3 / 93.3 / 93.3 / 95.0 |
| 7 (26-27) | 2x2 | 1x1 | 0.0% | 96.7 (accepted on the subset, rejected on the full dev set) |

The first block tolerates nothing because its residual branch carries the stem-level detail; the deep `2x2` blocks
tolerate nothing because each of their four tiles sees the entire image. Only the two middle blocks give up 10%
and 12% of their own tiles, and those blocks are a small part of the MAC budget, hence +1.30 pp end to
end.

## 5. Independent NumPy evidence: the branch output is never locally flat

Executed against the exported pruned weights (per-tile deviation `max_c |F[c] - mean_c F|`, one value per 2x2 tile):

| residual branch | mean abs output | tile deviation p10 / p50 / p90 | tiles below 0.5x mean |
|---|---|---|---|
| tail 3 (stage 1) | 0.600 | 0.87 / 0.95 / 1.34 | 16.0% |
| tail 6 | 0.626 | 0.68 / 0.74 / 1.07 | 35.0% |
| tail 9 | 0.648 | 0.99 / 1.02 / 1.44 | 40.1% |
| tail 13 | 0.672 | 1.31 / 1.34 / 1.53 | 27.7% |
| tail 16 | 0.730 | 1.45 / 2.36 / 3.29 | 15.8% |
| tail 20 | 0.736 | 2.11 / 2.56 / 3.54 | 8.2% |
| tail 23 | 0.744 | 2.01 / 3.36 / 4.00 | 0.0% |
| tail 27 | 0.756 | 2.22 / 3.36 / 4.70 | 0.0% |

The deviation is of the same order as the signal itself everywhere, and in the deep blocks it is *larger*. Replacing a
tile by a constant is therefore not a small perturbation; the residual stream is genuinely distributed over the whole
canvas.

## 6. Temporal mode (S3)

Stream accuracy is dominated by the out-of-distribution moving-digit dataset (M0 scores 28.56% on the 2,160
frames), so S3 is compared to the stream's own dense baseline, never to static accuracy. ResNet-18: S1-stream
2.878 ms/frame at 28.56% -> S3 3.224 ms/frame at 28.29% (0.89x, MAC cut
3.36%, argmax agreement 93.06%). The sticky rule (skip when the static gate skips, or when the tile was
skipped last frame and is temporally stable) inherits the static gate's ceiling, so temporal mode cannot rescue what
the static gate cannot find. MobileNetV2's stream runs at 1.00x with zero skips.

## 7. What would actually be needed

1. **Sparsity the network did not have to be taught.** Tile skipping pays when branches are *structurally* flat (e.g.
   segmentation backbones on sparse scenes, or VGG-4 in Phase A with 78% zero input pixels). On a 32x32 full-canvas
   digit task the receptive fields are global by the third stage.
2. **Finer granularity with a real predictor.** Instead of predicting "this 2x2 tile contributes little", predict the
   threshold-crossing risk per output channel and skip only channels that stay within a margin — but that is back to
   per-channel branching at 16 lanes, which Phase B measured as a loss.
3. **Bigger inputs.** At 224x224 the deep feature maps stay 14x14 and a 3x3 conv sees 1/16 of the canvas: that is the
   regime where background tiles exist and where the ALCS/AMC + CM hybrid would be worth re-measuring. Everything in
   this repository is 32x32, where the receptive-field argument dominates.

## 8. Reproduce

```bash
python3 scripts/prepare_enterprise_backbones.py            # ~55 s on the RTX 5070 (cached datasets)
g++ -O3 -march=native -mavx512f -mavx512vl -mavx512bw -mfma -std=c++20 \
    native/cm_enterprise_avx512.cpp -o experiments/cm-cnn-enterprise/cm_enterprise_avx512
CM_MODEL=resnet18    npm run bench:cm-cnn:enterprise      # -> /tmp/ent_resnet18.json
CM_MODEL=mobilenetv2 npm run bench:cm-cnn:enterprise      # -> /tmp/ent_mobilenetv2.json
npm run test:cm-cnn:enterprise
```

Calibration touches `dev.bin` only; `test.bin` is used solely for measurement.

## 9. v3: rank-16 sub-branch gating (what changed, what it bought)

The v2 gate had one atom: the whole residual branch of a `2x2` tile. On a `4x4` or `2x2` feature
map that means 176-352 channels at once, and dev calibration refused it everywhere. v3 splits the
mid axis into 16-channel sub-branches and lets the gate switch individual ones off, with a
closed-form bias compensation so a switched-off part still contributes its mean:

    y = x + b2 + sum_{g live} F_g(x) + sum_{g off} db_g,    db_g = W2[:, 16g..] . mean(h_g)

Implementation (all in `native/cm_enterprise_avx512.cpp`, which is generated by `/tmp/gen_ent.py`
from fragment files): per node a role (producer = skip output groups, consumer = skip input blocks,
consumer+compensation = also add `db_g`), a per-tile `uint64` mask of dead sub-branches, a per-block
`G x co` compensation table built from the dev mean activations, and a per-(block, sub-branch)
threshold calibrated on dev by greedy benefit/cost (`dMACs / mean deviation`). Skipping is exact for
the *kept* work: the packed tile list never emits a dead 16-channel block, so no weight row is loaded
and no FMA is executed; the only approximation is the bias compensation itself.

Measured (8 reps, interleaved, single core):

| mode | ResNet-18 acc | MAC cut | ms/img | x vs M0 | MobileNetV2 acc | MAC cut | ms/img | x vs M0 |
|---|---|---|---|---|---|---|---|---|
| `M0` | 97.78% | 0.00% | 8.475 | 1.00x | 96.94% | 0.00% | 0.621 | 1.00x |
| `S1` | 98.33% | 52.05% | 3.141 | 2.70x | 98.33% | 52.29% | 0.374 | 1.66x |
| `S2` rank-16 gate | 96.11% | 54.80% | 3.179 | 2.67x | 98.33% | 53.61% | 0.409 | 1.52x |
| `S2-random` equal rate | 94.17% | 54.79% | 2.883 | 2.94x | 97.78% | 53.62% | 0.409 | 1.52x |

Findings:

1. **Dev cannot see the damage.** `S2` scores exactly the `S1` accuracy on dev (98.33% -> 98.33%) and
   loses 2.22 pp on test. With 360 dev images and a 0.35 pp budget, per-sub-branch approximations
   are below the dev noise floor. Any future gate must be validated on a larger calibration split.
2. **Context beats random, modestly.** At an identical 25.37% tile skip rate the CM context loses
   1.94 pp less accuracy than random (96.11% vs 94.17%), and 0.15 lower logit RMSE. The signal is
   real but small.
3. **Deep blocks stay closed.** Realised skip rates: 36%/24%/46%/15% in stages 1-2, then 8.6%/5.5%/
   11.3%/4.5% in stages 3-4. Rank-16 lowered the floor from "all 352 channels" to "16 channels", but
   the deviation of a deep sub-branch is still ~1x its mean.
4. **Time does not follow MACs.** `S2` executes 2.75 pp fewer MACs than `S1` and is 0.04 ms slower:
   the savings sit in the cheap early blocks, the mask lookup and compensation add are paid
   everywhere.
5. **The patch-packed zero-tap prefilter is a net loss.** It eliminates 33% of executed MACs
   (`555.4M -> 374M`) but the per-channel dependent mask walk halves the FMA rate (83 -> 38 GMAC/s),
   so the dense path ends up ~1.4x slower. Kept in the tree as a measured negative result.

The recommended operating point is therefore `S1`: 2.70x (ResNet-18) / 1.66x (MobileNetV2) with
accuracy *above* the original model, and the rank-16 gate documented as the measured ceiling of
content-adaptive skipping on this workload.
