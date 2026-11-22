# Universal CM-CNN on ResNet-18 / MobileNetV2 — measurement analysis

This directory holds the Phase B "universal" engine: a DAG-level AVX-512
inference engine that runs *any* exported CNN graph, plus the exact-sparse and
co-activation-clustered variants requested in the work order.

Everything below is measured on the AMD Ryzen 9 9955HX (Zen 5) with
`OMP_NUM_THREADS=1`, `g++ -O3 -march=native -mfma -mavx512f -mavx512vl
-mavx512bw -std=c++20`, 10 timing repetitions on the 360 static test images and
2 repetitions over the 2160-frame stream (v3; v2 used 20/3). Raw numbers:
`results.json`.

## 1. What works (verified, not asserted)

| check | ResNet-18 | MobileNetV2 |
|---|---|---|
| exported file == PyTorch fused model | `max_diff 4.6e-06`, acc `97.78%` | `max_diff 3.0e-06`, acc `96.94%` |
| `M0` dense engine == PyTorch test accuracy | `97.7778%` vs `97.7778%` | `96.9444%` vs `96.9444%` |
| `M0` per-layer MACs == analytic total | `555,422,720` | `52,302,848` |
| `M2` clustered == `M0` (0.00 pp) | 100% argmax, logit RMSE `2.0e-06` | 100% argmax, logit RMSE `1.4e-06` |
| dev-calibrated dead zone vs dense Top-1 | `97.78%` (= dense) | `96.67%` (-0.28 pp, budget 0.35) |
| stream `M0` == PyTorch on the same frames | `29.26%` == `29.26%` | `50.05%` == `50.05%` |

The isomorphism is exact because clustering is a pure channel relabelling of a
union-find partition of channel spaces (`build_spaces`): every conv weight is
permuted `w'[t][i'][o'] = w[t][π⁻¹(i')][π⁻¹(o')]`, biases/FC rows follow the same
permutation, and depthwise weights are permuted once per coupled axis. Residual
buses get a *shared* stage permutation so `ReLU(πA + πB) = πReLU(A + B)`.

## 2. Why the speedup targets cannot be met on these checkpoints

Zero skipping only removes work in proportion to zeros that are *structurally*
absent. With exact scalar compaction the per-layer MAC count is

```
MACs = H·W·Σ_o Σ_taps (# non-zero input scalars) = H·W·K·ρ
```

where `ρ` is the input density. Adding 8-lane fiber skipping with all-dead rate
`δ8` gives

```
cut = 1 − ρ·(1 − δ8)
```

Measured `ρ` (post-ReLU density of the layer inputs, test set):

| model | ρ (mean over conv/dw layers) | ceiling `1 - ρ` | measured cut (M1) | ρ with dead zone | measured cut (M2z) | cut needed for the gate |
|---|---|---|---|---|---|---|
| ResNet-18 | 0.519 | 48.1% | 0.0% | 0.391 | 7.2% | 75% |
| MobileNetV2 | 0.648 | 35.2% | 2.8% | 0.576 | 2.8% | 70% |

The measured cut sits far below the `1 - ρ` ceiling because the compacted gather costs
~2 load uops per FMA against 0.5 for the tile kernel (see the bullet below): the
dispatcher spends the MAC budget where it is cheapest, not where the cut is largest.

To reach a 75% cut on ResNet-18 with the measured `δ8 ≈ 0.2` one would need
`ρ ≤ 0.31`; to reach 70% on MobileNetV2 at any `δ8` one would need `ρ ≤ 0.30`.
MobileNetV2 is *architecturally* dense: ReLU6 expansion layers keep ~91% of
their channels positive, so there is nothing to skip. This is a property of the
frozen weights, not of the engine.

Two further consequences, both measured:

* **The CM gate has almost nothing to learn.** With dev-prewarmed tables, the
  shortcut-activity context and a 99.5% precision target, *no* layer on either
  model can safely skip: predicted-dead fibers are not predictable from a
  4-neighbour + energy context. At a 98% target the gate does fire (~1.9% of
  ResNet-18 fibers) but costs 0.83 pp of accuracy - over the 0.35 pp budget - so
  0.995 is the honest knob and at that setting the gate self-disables.
* **Exact compaction is load bound, not FMA bound.** Every executed MAC needs a
  64 B weight-row load plus a scalar broadcast, i.e. ~2 load uops per FMA against
  0.5 for the patch-packed tile kernel (`conv_tile`). ResNet-18's `conv4_1` weights
  are 9·512·512·4 = 9.4 MB and the gathered rows are visited in pixel order, so the
  stream is L3-resident. Measured: dense 6.839 ms/img = 81 GMAC/s on ResNet-18, and
  the compacted path has never beaten it on any layer of either model by more than
  the dispatch hysteresis margin.

## 3. What the numbers say

v3 numbers, `results.json`, 10 static reps / 2 stream reps, single core, modes interleaved.

* Exact scalar compaction (`M1`) and clustering (`M2`) are *lossless*: identical
  accuracy, identical argmax, logit RMSE at FP-rounding level (`2.03e-06` on
  ResNet-18, `1.43e-06` on MobileNetV2).
* **The dense engine is 1.76x / 1.21x faster than v2** (`12.030 -> 6.839` ms/img on
  ResNet-18, `0.742 -> 0.615` on MobileNetV2). This is the patch-packed shared-weight
  (`conv_tile`) kernel: 2x2 pixels x 4 output groups = 16 resident ZMM accumulators,
  0.5 load uops per FMA instead of 1.125. Nothing to do with sparsity.
* **Clustering pays inside the packed kernel.** With the clustered channel order the
  tile kernel skips all-zero `(tap, 16-channel block)` windows and removes `2.91%` of
  ResNet-18's MACs; with the unclustered order it removes `0.00%` (M1 cut `0.0%`). The
  8-lane co-activation grouping is what makes a 2x2x16 window go to zero *together*.
* **Per-layer calibration beats a global dead zone.** The greedy search over
  `eta_l` (deepest-first, 0.35 pp dev budget, only accepted when it buys >= 0.5% MACs)
  drops ResNet-18 input density `0.519 -> 0.391` and raises the cut to `7.20%` with
  **no test-accuracy change at all** (97.78%). On MobileNetV2 it drops density
  `0.648 -> 0.576` for `2.75%` cut at `-0.28 pp` (96.67%).
* **Per-layer kernel dispatch is a wash on these two models.** Chosen on dev by
  measured per-node time with a 15% hysteresis margin: ResNet-18 `compacted 5, tile-dense
  16, packed 9`; MobileNetV2 `compacted 20, tile-dense 25`. The compacted kernel is
  never 15% faster than the tile kernel on MobileNetV2 yet the dispatcher still picks it
  for half the layers, and the result is that M1/M2/M3 land at `0.78x`-`0.97x` of dense
  instead of at parity. Honest reading: the dispatcher's dev-time signal is too noisy at
  0.01 ms/layer, and `M2`'s `2.91%` cut does not pay for its gather on this core.
* The CM gate (`M3`) adds nothing at >= 85% precision on either model: every layer
  calibrates to "gate off", fiber skip rate `0.000`. Consequently the random controls
  are no-ops (their logit RMSE equals M3's) and the gate report marks
  `random-separation` as *not binding* rather than as a pass.
* **Coupled Expand -> DW -> Project pruning is implemented but inert here.** The 1x1
  Expand publishes per-fiber dead bits, `3x3` Depthwise reads them and skips the whole
  8-lane fiber (bias-only store), and the 1x1 Project's packed tile drops it as an
  all-zero tap. Because the calibrated gate threshold is 0 on every MobileNetV2 layer,
  no fiber is ever marked dead (`fiber_skip_rate 0.0000`); forcing a loose gate
  (`CM_SKIP_CAL=1`, threshold 0.84) keeps accuracy at 96.94% with a 25.1% MAC cut, so
  the machinery is wired and correct, just never triggered by these weights.
* Temporal (`M4`) removes `1.26%` of ResNet-18 MACs (`0.94x` of its own dense pass) and
  `2.69%` on MobileNetV2 (`0.61x`); the stream's own accuracy is unchanged (`0.2926` /
  `0.5005`).

## 4. What would actually make this fast

1. **Patch-packed shared-weight kernels - done in this revision, and it worked.**
   Time went to 0.5 load uops per FMA by holding 16 accumulators in registers
   (`conv_tile`): `1.76x` on ResNet-18 dense, `1.21x` on MobileNetV2 dense. Failed
   variants (stack-staged weights = store-to-load forwarding stalls; generic
   `for g < ng` loops = all 16 accumulators spilled; a union-of-rows list = spilled
   again) are documented in `docs/CM_CNN_ACCELERATION.md` §12.3.
2. **The compacted gather is the wrong shape for this core.** Per live scalar it needs
   a compacted-value broadcast plus a gathered weight row; the tile kernel needs 0.5
   loads. That is why a 55% MAC cut bought nothing in v2 and why per-layer dispatch now
   converges on the tile kernels. Exploiting sparsity profitably needs a *structured*
   zero pattern that removes whole weight rows, not isolated scalars.
3. **The clustering payoff in v3 is structural, not scalar**: it converts 2.9% of
   ResNet-18's tiles into all-zero tiles. Raising that number requires clustering the
   *tile* pattern (a joint 2x2x16 sparsity objective), not just per-channel co-activation.
4. **Larger MAC cuts require structural sparsity in training** (channel gating,
   structured pruning, or quantization with zero run-length). The dead zone is the
   empirical form of this: `7.2%` of ResNet-18's MACs for 0.00 pp here, and further
   truncation costs accuracy fast.

## 5. Reproduce

```bash
g++ -O3 -march=native -mavx512f -mavx512vl -mavx512bw -mfma -std=c++20 \
    native/cm_universal_avx512.cpp -o experiments/cm-cnn-universal/cm_universal_avx512

python3 scripts/prepare_pretrained_backbones.py        # re-export + verify (cached in /tmp/ckpt_cmuniv)

OMP_NUM_THREADS=1 CM_PREC=0.995 \
  ./experiments/cm-cnn-universal/cm_universal_avx512 \
  --model resnet18 \
  --weights experiments/cm-cnn-universal/pretrained_resnet18_rgb.bin \
  --train experiments/cm-cnn-rgb-v4/dataset/train.bin \
  --test  experiments/cm-cnn-rgb-v4/dataset/test.bin \
  --dev   experiments/cm-cnn-rgb-v4/dataset/dev.bin \
  --stream experiments/cm-cnn-rgb-v4/dataset/stream.bin \
  --reps 20 --reps-stream 3 --json /tmp/resnet18.json
```

`node src/lib/vision/cm-cnn-universal.ts` runs both models and prints the gate
report; `node --test src/__tests__/cm-cnn-universal.test.ts` checks the recorded
results.

## 6. Notes on the stream split

The 2160-frame RGB stream is *out of distribution* for these checkpoints: frames
drift the digit by up to 6 px per frame at 18–22 px scale with per-frame texture
noise, while the backbones were trained on static centred renders. Both PyTorch
and the engine score `29.26%` (ResNet-18) / `50.05%` (MobileNetV2) on it — the
agreement to 4 decimal places is itself the fidelity check; the absolute level
says nothing about the engine.
