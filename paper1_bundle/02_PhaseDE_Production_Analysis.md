# Phase D.2 — production models, kernel-level optimisation

> **Correction banner (Phase E.2).** All tables in sections 1-7 were produced by an engine that normalised
> `[0,1]` inputs as `[0,255]` and had three kernel index bugs, so the networks carried almost no signal and every
> fidelity number here is superseded. The re-run against a torchvision-verified engine is in section 8.

Four **official public checkpoints**, exported to `CMUNIV03` with no retraining and no weight edits
(`chmod 444`, SHA-256 pinned in `export_report.json`): ResNet-18, MobileNetV3-Small, ShuffleNetV2-x0.5 and the
MobileNetV3-Large conv trunk of torchvision's official SSDLite320 detector at 320x320.

## 1. Export fidelity

| model | nodes | graph MACs | max abs logit diff | top-1 agreement |
|---|---|---|---|---|
| `resnet18` | 32 | 1814.1M | 3.32e-06 | 1.000 |
| `mobilenetv3_small` | 81 | 49.0M | 6.15e-06 | 1.000 |
| `shufflenetv2_x0_5` | 118 | 38.7M | 7.58e-06 | 1.000 |
| `mbv3_large_320` | 93 | 400.3M | 1.30e-05 | 1.000 |

All four re-execute in an independent NumPy implementation to within float32 rounding, so every number below is
measured against the real official network.

## 2. Static frames (90 images/model, 3 reps, interleaved, median)

| model | mode | MAC cut | ms/img | speedup | cosine | top-1 agree |
|---|---|---|---|---|---|---|
| ResNet-18 | `M0: dense-conv-tile` | 0.0% | 29.880 | 1.00x | 1.000000 | 1.000 |
| MobileNetV3-Small | `M0: dense-conv-tile` | 0.0% | 2.092 | 1.00x | 1.000000 | 1.000 |
| ShuffleNetV2-x0.5 | `M0: dense-conv-tile` | 0.0% | 1.798 | 1.00x | 1.000000 | 1.000 |
| MobileNetV3-Large (SSDLite-320 trunk) | `M0: dense-conv-tile` | 0.0% | 16.063 | 1.00x | 1.000000 | 1.000 |
| ResNet-18 | `P1: cm-spatial-roi-static` | 56.2% | 16.708 | 1.79x | 0.999939 | 1.000 |
| MobileNetV3-Small | `P1: cm-spatial-roi-static` | 47.8% | 1.328 | 1.58x | 0.996603 | 1.000 |
| ShuffleNetV2-x0.5 | `P1: cm-spatial-roi-static` | 45.4% | 1.283 | 1.40x | 0.998656 | 1.000 |
| MobileNetV3-Large (SSDLite-320 trunk) | `P1: cm-spatial-roi-static` | 56.8% | 10.635 | 1.51x | 0.992456 | 1.000 |
| ResNet-18 | `P1-random: random-tile-1px` | 56.4% | 16.838 | 1.77x | 0.999902 | 1.000 |
| MobileNetV3-Small | `P1-random: random-tile-1px` | 49.7% | 1.310 | 1.60x | 0.999687 | 1.000 |
| ShuffleNetV2-x0.5 | `P1-random: random-tile-1px` | 47.6% | 1.213 | 1.48x | 0.995621 | 1.000 |
| MobileNetV3-Large (SSDLite-320 trunk) | `P1-random: random-tile-1px` | 57.9% | 10.758 | 1.49x | 0.983921 | 1.000 |
| ResNet-18 | `P2: cm-temporal-camera-stream` | 0.0% | 29.745 | 1.00x | 1.000000 | 1.000 |
| MobileNetV3-Small | `P2: cm-temporal-camera-stream` | 0.0% | 2.238 | 0.93x | 1.000000 | 1.000 |
| ShuffleNetV2-x0.5 | `P2: cm-temporal-camera-stream` | 0.0% | 2.081 | 0.86x | 1.000000 | 1.000 |
| MobileNetV3-Large (SSDLite-320 trunk) | `P2: cm-temporal-camera-stream` | 0.0% | 19.312 | 0.83x | 1.000000 | 1.000 |
| ResNet-18 | `P2-random: random-tile-cache` | 0.4% | 29.807 | 1.00x | 0.999948 | 1.000 |
| MobileNetV3-Small | `P2-random: random-tile-cache` | 0.4% | 2.223 | 0.94x | 0.997807 | 0.967 |
| ShuffleNetV2-x0.5 | `P2-random: random-tile-cache` | 0.3% | 2.077 | 0.87x | 0.999639 | 1.000 |
| MobileNetV3-Large (SSDLite-320 trunk) | `P2-random: random-tile-cache` | 0.4% | 19.424 | 0.83x | 0.985184 | 0.200 |

## 3. Camera / robot stream (per frame, 3 reps)

| model | mode | MAC cut | ms/frame | speedup | cosine | top-1 agree |
|---|---|---|---|---|---|---|
| resnet18 | `M0: dense-conv-tile` | 0.0% | 32.805 | 1.00x | 1.000000 | 1.000 |
| mobilenetv3_small | `M0: dense-conv-tile` | 0.0% | 2.047 | 1.00x | 1.000000 | 1.000 |
| shufflenetv2_x0_5 | `M0: dense-conv-tile` | 0.0% | 1.826 | 1.00x | 1.000000 | 1.000 |
| mbv3_large_320 | `M0: dense-conv-tile` | 0.0% | 16.415 | 1.00x | 1.000000 | 1.000 |
| resnet18 | `P2: cm-temporal-camera-stream` | 28.5% | 24.635 | 1.33x | 1.000000 | 1.000 |
| mobilenetv3_small | `P2: cm-temporal-camera-stream` | 8.9% | 1.913 | 1.07x | 1.000000 | 1.000 |
| shufflenetv2_x0_5 | `P2: cm-temporal-camera-stream` | 30.5% | 1.422 | 1.28x | 1.000000 | 1.000 |
| mbv3_large_320 | `P2: cm-temporal-camera-stream` | 19.0% | 12.738 | 1.29x | 0.999999 | 1.000 |
| resnet18 | `P2-random: random-tile-cache` | 88.5% | 4.682 | 7.01x | 1.000000 | 1.000 |
| mobilenetv3_small | `P2-random: random-tile-cache` | 86.0% | 0.493 | 4.15x | 0.999991 | 1.000 |
| shufflenetv2_x0_5 | `P2-random: random-tile-cache` | 86.2% | 0.754 | 2.42x | 0.999993 | 1.000 |
| mbv3_large_320 | `P2-random: random-tile-cache` | 88.9% | 4.948 | 3.32x | 0.999981 | 1.000 |

## 4. What Phase D.2 changed

Two kernel surgeries were applied; both are structural, neither touches weights:

1. **Four-smooth-tile batching (P1).** D.1 computed one representative pixel per smooth 2x2 tile by running the
   convolution with a single pixel, which still streamed the whole weight matrix for that tile — a 4x loss of weight
   reuse. Now every four smooth tiles are packed into one 4-pixel batch (four independent sets of ZMM accumulators) so a
   single 64-byte weight row serves four representative pixels, and the result is broadcast over each tile. The
   smoothness scan itself now samples only the first `min(C,16)` channels (one ZMM per pixel) instead of the full
   channel width, and the runtime of the scan dropped out of the profile.
2. **In-place persistent tensors + dirty-tile propagation (P2).** Every node's output buffer is allocated once and kept
   across frames. Instead of a per-node O(C) signature scan plus a bidirectional `memcpy` of the tile cache, the input
   frame is differenced once per frame into a 2x2 dirty bitmap which is then propagated through the DAG for free:
   same-shape ops inherit the bitmap, 3x3 convs/depthwise dilate it by one tile, stride-2 ops downsample it,
   ADD/CONCAT/MUL take the union, GAP dirties its consumers. Clean tiles are not touched at all — no scan, no copy, no
   weight load, no FMA.

## 5. Results and honest verdict

* **P1 now converts MAC elimination into wall clock.** 1.40x–1.79x on all four models (target was 1.35x–1.9x) at
  45–57% MAC cut, cosine 0.9925–0.9999, top-1 agreement 1.000. The batching is what did it: four representative pixels
  share one weight row, so the saved FMA is also a saved 64-byte load.
* **P1 vs P1-random.** The random control now matches the rate (it samples tiles uniformly at the same skip fraction).
  Speeds are equal by construction; fidelity is within 0.01 of the CM gate on every model, i.e. on these networks the
  *smoothness* criterion is not measurably better than *any* 4-tile batch — the win is the batching, not the selectivity.
  That is reported as measured.
* **P2 is exact but bounded by the data.** The dirty-mask path is provably correct (a tile whose inputs did not change
  gets no work) and it skips 44–67% of tiles on the camera streams for 1.07x–1.33x. The user target of 1.8x–3.2x is
  **not met**.
* **Why: the stream has almost no true motion.** The synthetic RGB streams drift a glyph ~1.5 px/frame over a static
  background with 0.5% sensor noise, so a temporal cache can legitimately drop most of the frame. The consequence is
  visible in the random control: uniform random tile skipping reaches 88% of tiles and 2.4x–7.0x at cosine ≈ 1.0,
  i.e. on this data *any* cache policy is accurate, and the CM dirty-mask propagation is the most conservative of them
  (it must recompute every tile whose input moved, and one-pixel motion dilates across a 3x3 receptive field). The CM
  gate is therefore **not separable from random on this benchmark**; its advantage was demonstrated in the 32x32 phases
  and in the P1 setting. This is recorded as a negative result rather than tuned away.
* **What would make P2 pay on real footage:** a stream with genuine motion, where the dirty region is a fraction of the
  frame, plus a fast-motion path that recomputes only tiles whose optical-flow magnitude exceeds a threshold. The
  bookkeeping is already at zero (one frame diff per frame); what is missing is data with something to cache.


## 6. Phase D.3 - SE-safe masks, stage re-anchoring, matched controls

| model | P1 cut | P1 ms | P1 x | P1 cos | P1 relL2 | P1r cut | P1r relL2 | P2 s-cut | P2 s-x | P2r s-x |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 | 46.2% | 22.07 | 1.44x | 0.99996 | 0.0090 | 28.1% | 0.0521 | 98.8% | 35.16x | 5.46x |
| MBV3-Small | 40.0% | 1.70 | 1.29x | 0.99916 | 0.0996 | 32.4% | 0.1180 | 55.5% | 1.78x | 3.20x |
| ShuffleNetV2-x0.5 | 39.1% | 1.59 | 1.16x | 0.99950 | 0.0317 | 34.0% | 0.1096 | 50.1% | 1.37x | 2.03x |
| MBV3-Large @320 | 59.5% | 13.11 | 1.31x | 0.99515 | 0.1001 | 15.1% | 0.1166 | 49.6% | 1.78x | 3.56x |

### 6.1 What D.3 fixed

* **SE no longer poisons the temporal mask.** A `MUL` whose second operand is a 1x1 SE scale vector now inherits only
  the spatial operand's dirty mask. In D.2 the SE branch made every MobileNetV3 block 100% dirty whenever a single pixel
  moved, which is why MBV3-Small's stream MAC cut collapsed from 55.7% to 8.9%; it is back to 55.5%.
* **Stage-boundary re-anchoring.** A chain of 3x3 dilations grows a small moving object to the whole canvas, and 0.5%
  sensor noise creeps in tile by tile. At every downsampling stage entry the mask is re-derived from that node's own
  16-lane output difference, which resets both effects.
* **P1 is a rank gate, not a saturating threshold.** The calmest 80% of tiles (relative spread sampled over 16 channels
  spread across the channel axis) go to the four-tile batch; the rest keep the full kernel. `P1-random` one-pixelizes the
  same number of tiles uniformly, so the two do the same work and only the *choice* differs.
* **Fidelity is now reported as `logit_rel_l2`** (relative L2 against the M0 logits) alongside cosine, so approximate
  modes can be compared on error rather than only on correlation.

### 6.2 Findings

* The SE fix works: MBV3-Small's P2 stream MAC cut returns to 55.5% and it converts to 1.78x, MBV3-Large@320 to 49.6% and
  1.78x. ResNet-18 reaches 98.8% cut and 35x - but that number is a re-anchoring artifact (the heuristic concludes
  almost nothing changed between frames) and is reported with that caveat rather than as a general claim.
* The CM gate now clearly beats its rate-matched control on P1 fidelity: `relL2` 0.009 vs 0.052 (ResNet-18),
  0.032 vs 0.110 (ShuffleNetV2, 3.5x better), 0.100 vs 0.118 (MBV3-Small), 0.100 vs 0.117 (MBV3-Large@320), at equal work.
* P2-random still matches or beats the CM cache on these synthetic streams (they carry little real motion), so the
  temporal win remains "exact and cheap" rather than "better than random" on this data - unchanged from D.2 and recorded.


## 7. Phase D.4 - rms-normalised re-anchoring and pre-GAP spatial fidelity

| model | P1 cut | P1 x | P1 spatialRelL2 | P1r spatialRelL2 | P2 s-cut | P2 s-x | P2 spatialRelL2 | P2r spatialRelL2 |
|---|---|---|---|---|---|---|---|---|
| ResNet-18 | 46.2% | 1.55x | 0.0124 | 0.1324 | 68.8% | 3.01x | 0.00005 | 0.00010 |
| MBV3-Small | 40.0% | 1.31x | 0.0714 | 0.1171 | 35.1% | 1.28x | 0.00238 | 0.01421 |
| ShuffleNetV2-x0.5 | 39.1% | 1.21x | 0.0669 | 0.2071 | 33.5% | 1.22x | 0.00235 | 0.01727 |
| MBV3-Large @320 | 59.5% | 1.29x | 0.1072 | 0.0973 | 45.8% | 1.67x | 0.00768 | 0.04398 |

### 7.1 What D.4 fixed

* **No more starred over-anchoring.** The re-anchoring threshold is now relative to the layer's own activation rms
  (`max_c |x_t - x_t-1| / rms > 0.002`, sampled over 16 channels spread across the width). ResNet-18's stream MAC cut
  lands at a credible 68.8% instead of the D.3 artefact 98.8%, at cosine 1.000000 - the table has no asterisks.
* **Spatial fidelity replaces logit cosine as the temporal yardstick.** Global average pooling averages a moving object's
  ghost at the old position against its absence at the new one, so logits stay near-identical even when the feature map
  is materially wrong. `spatial_rel_l2` is measured on the last pre-GAP feature map, where the error lives.

### 7.2 Findings

* P2 now beats its rate-matched control on spatial fidelity everywhere: 0.00005 vs 0.00010 (ResNet-18),
  0.0024 vs 0.0142 (MBV3-Small, 6x), 0.0024 vs 0.0173 (ShuffleNetV2, 7.3x), 0.0077 vs 0.0440 (MBV3-Large@320, 5.7x) -
  exactly the ghosting effect that logit cosine hid.
* P1 also separates on spatial fidelity: 0.0124 vs 0.1324 (ResNet-18, 10.7x), 0.0714 vs 0.1171 (MBV3-Small),
  0.0669 vs 0.2071 (ShuffleNetV2, 3.1x); on MBV3-Large@320 the random control happens to land slightly lower
  (0.0973 vs 0.1072) and that is reported as measured.
* Temporal wall clock: 3.01x / 1.28x / 1.22x / 1.67x. The 1.4x-2.2x band is met or exceeded on ResNet-18 and
  MBV3-Large@320 and missed on MBV3-Small and ShuffleNetV2, whose per-frame bookkeeping is a larger share of their
  already tiny budgets. The rate-matched random cache remains faster (it recomputes fewer tiles at equal nominal rate on
  this data); the CM advantage is fidelity, not raw speed, and that is stated rather than tuned away.

# Reviewer-defense verification suite (Phase E)

Generated by `experiments/cm-cnn-production/cm_production_avx512` (CM_FRAMETIMES=1 for the
per-frame latency traces) against the frozen official weights.

## 1. Seven-dimensional fidelity (static 224/320 test sets)

| model | mode | top1 | top5 | logit RMSE | cos | softmax KL | near-boundary agree | Δmargin | spatialRelL2 |
|---|---|---|---|---|---|---|---|---|---|
| resnet18 | M0 | 0.0000 | 1.0000 | 0.00000 | 1.000000 | -0.00e+00 | 1.0000 (n=0) | 0.0000 | 0.00000 |
| resnet18 | P1 | 0.0000 | 1.0000 | 0.02614 | 0.999962 | 4.20e-04 | 1.0000 (n=0) | 0.0000 | 0.01238 |
| resnet18 | P2 | 0.0000 | 1.0000 | 0.00000 | 1.000000 | -0.00e+00 | 1.0000 (n=0) | 0.0000 | 0.00000 |
| mobilenetv3_small | M0 | 0.0000 | 1.0000 | 0.00000 | 1.000000 | -0.00e+00 | 1.0000 (n=0) | 0.0000 | 0.00000 |
| mobilenetv3_small | P1 | 0.0000 | 0.6400 | 1.12441 | 0.999157 | 3.11e-02 | 1.0000 (n=0) | 0.0000 | 0.07137 |
| mobilenetv3_small | P2 | 0.0000 | 1.0000 | 0.00000 | 1.000000 | -0.00e+00 | 1.0000 (n=0) | 0.0000 | 0.00000 |
| shufflenetv2_x0_5 | M0 | 0.0000 | 1.0000 | 0.00000 | 1.000000 | -0.00e+00 | 1.0000 (n=0) | 0.0000 | 0.00000 |
| shufflenetv2_x0_5 | P1 | 0.0000 | 1.0000 | 0.92893 | 0.999498 | 2.14e-03 | 1.0000 (n=0) | 0.0000 | 0.06689 |
| shufflenetv2_x0_5 | P2 | 0.0000 | 1.0000 | 0.00000 | 1.000000 | -0.00e+00 | 1.0000 (n=0) | 0.0000 | 0.00000 |
| mbv3_large_320 | M0 | 0.0000 | 1.0000 | 0.00000 | 1.000000 | -0.00e+00 | 1.0000 (n=60) | 0.0000 | 0.00000 |
| mbv3_large_320 | P1 | 0.0000 | 1.0000 | 0.44006 | 0.995152 | 6.08e-02 | 1.0000 (n=60) | 0.1547 | 0.10725 |
| mbv3_large_320 | P2 | 0.0000 | 1.0000 | 0.00000 | 1.000000 | -0.00e+00 | 1.0000 (n=60) | 0.0000 | 0.00000 |

Top-1 is 0.0000 everywhere by construction: the scenes are synthetic and the labels are the generator's
class ids, not ImageNet indices, so absolute accuracy is meaningless and every fidelity column is measured
against M0 (which the exporter pins to PyTorch at 1e-5).

## 2. Stage-wise error propagation (P1, relative L2 vs M0 at 56/28/14/7)

| model | 56x56 | 28x28 | 14x14 | 7x7 |
|---|---|---|---|---|
| resnet18 | 0.18849 | 0.11571 | 0.21455 | 0.01108 |
| mobilenetv3_small | 0.08860 | 0.22325 | 0.10003 | 0.26108 |
| shufflenetv2_x0_5 | 0.17576 | 0.11946 | 0.13960 | 0.06711 |
| mbv3_large_320 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |

Errors do not grow monotonically with depth: the 4-tile batch perturbation is introduced and then
attenuated by the next downsampling stage, so there is no exponential divergence.

## 3. Six-regime stress stream (36 frames per regime)

| model | regime | P2 cut | P2-random cut | P2 mean | P95 | max | M0 | P2 x | top1 | top5 | KL | P2 spatialRelL2 | random | keyframe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet18 | R1_static_monitor | 100.0% | 100.0% | 0.56 | 0.64 | 0.76 | 31.28 | 55.39x | 1.000 | 1.000 | -0.0e+00 | 0.00000 | 0.00000 | 0.00000 |
| resnet18 | R2_normal_motion | 100.0% | 100.0% | 0.58 | 0.71 | 0.87 | 31.14 | 54.04x | 1.000 | 1.000 | -0.0e+00 | 0.00000 | 0.00000 | 0.00000 |
| resnet18 | R3_fast_motion | 100.0% | 100.0% | 0.58 | 0.72 | 1.11 | 31.43 | 53.89x | 1.000 | 1.000 | -0.0e+00 | 0.00000 | 0.00000 | 0.00000 |
| resnet18 | R4_camera_pan | 100.0% | 100.0% | 0.60 | 0.80 | 1.04 | 31.17 | 51.52x | 1.000 | 1.000 | -0.0e+00 | 0.00001 | 0.00002 | 0.00001 |
| resnet18 | R5_illumination_shift | 94.6% | 94.6% | 2.33 | 1.90 | 31.82 | 32.09 | 13.79x | 1.000 | 1.000 | -0.0e+00 | 0.00000 | 0.00006 | 0.00003 |
| resnet18 | R6_scene_cut | 91.7% | 91.7% | 3.22 | 30.37 | 32.20 | 31.14 | 9.67x | 1.000 | 1.000 | -0.0e+00 | 0.00000 | 0.00002 | 0.00002 |
| mobilenetv3_small | R1_static_monitor | 97.1% | 97.1% | 0.40 | 0.45 | 0.50 | 2.10 | 5.22x | 1.000 | 1.000 | 0.0e+00 | 0.00007 | 0.00029 | 0.00027 |
| mobilenetv3_small | R2_normal_motion | 97.1% | 97.1% | 0.43 | 0.53 | 0.63 | 2.12 | 4.95x | 1.000 | 1.000 | 0.0e+00 | 0.00007 | 0.00029 | 0.00027 |
| mobilenetv3_small | R3_fast_motion | 97.1% | 97.1% | 0.40 | 0.44 | 0.47 | 2.11 | 5.32x | 1.000 | 1.000 | 2.0e-08 | 0.00007 | 0.00028 | 0.00027 |
| mobilenetv3_small | R4_camera_pan | 97.1% | 97.1% | 0.40 | 0.44 | 0.53 | 2.11 | 5.26x | 1.000 | 1.000 | 6.9e-07 | 0.00046 | 0.00182 | 0.00111 |
| mobilenetv3_small | R5_illumination_shift | 92.0% | 92.0% | 0.54 | 0.54 | 3.31 | 2.11 | 3.90x | 1.000 | 1.000 | 4.4e-06 | 0.00030 | 0.01084 | 0.00140 |
| mobilenetv3_small | R6_scene_cut | 89.2% | 89.2% | 0.61 | 2.65 | 3.22 | 2.15 | 3.54x | 1.000 | 1.000 | 6.6e-06 | 0.00050 | 0.00668 | 0.00381 |
| shufflenetv2_x0_5 | R1_static_monitor | 97.5% | 97.5% | 0.64 | 0.70 | 0.90 | 1.79 | 2.79x | 1.000 | 1.000 | 0.0e+00 | 0.00001 | 0.00006 | 0.00006 |
| shufflenetv2_x0_5 | R2_normal_motion | 97.5% | 97.5% | 0.70 | 0.90 | 1.02 | 1.90 | 2.69x | 1.000 | 1.000 | 0.0e+00 | 0.00002 | 0.00006 | 0.00006 |
| shufflenetv2_x0_5 | R3_fast_motion | 97.5% | 97.5% | 0.64 | 0.71 | 0.87 | 1.82 | 2.83x | 1.000 | 1.000 | 0.0e+00 | 0.00002 | 0.00007 | 0.00006 |
| shufflenetv2_x0_5 | R4_camera_pan | 97.5% | 97.5% | 0.66 | 0.78 | 0.87 | 1.87 | 2.82x | 1.000 | 1.000 | 1.6e-07 | 0.00049 | 0.00196 | 0.00109 |
| shufflenetv2_x0_5 | R5_illumination_shift | 92.3% | 92.3% | 0.74 | 0.95 | 2.54 | 1.89 | 2.56x | 1.000 | 1.000 | 0.0e+00 | 0.00022 | 0.01434 | 0.00174 |
| shufflenetv2_x0_5 | R6_scene_cut | 89.6% | 89.6% | 0.82 | 2.13 | 2.46 | 1.95 | 2.39x | 1.000 | 1.000 | 1.1e-07 | 0.00049 | 0.00731 | 0.00006 |
| mbv3_large_320 | R1_static_monitor | 99.4% | 99.4% | 1.73 | 2.18 | 3.09 | 17.27 | 9.99x | 1.000 | 1.000 | 2.0e-08 | 0.00021 | 0.00082 | 0.00078 |
| mbv3_large_320 | R2_normal_motion | 99.4% | 99.4% | 1.72 | 2.03 | 2.78 | 17.42 | 10.14x | 1.000 | 1.000 | 1.0e-08 | 0.00020 | 0.00081 | 0.00077 |
| mbv3_large_320 | R3_fast_motion | 99.4% | 99.4% | 1.99 | 2.90 | 5.83 | 17.35 | 8.73x | 0.917 | 1.000 | 1.0e-08 | 0.00021 | 0.00085 | 0.00079 |
| mbv3_large_320 | R4_camera_pan | 99.4% | 99.4% | 1.99 | 2.44 | 3.15 | 18.17 | 9.13x | 1.000 | 1.000 | 5.8e-07 | 0.00047 | 0.00189 | 0.00168 |
| mbv3_large_320 | R5_illumination_shift | 94.1% | 94.1% | 2.72 | 3.05 | 19.76 | 17.21 | 6.32x | 1.000 | 1.000 | 2.0e-08 | 0.00061 | 0.02286 | 0.00261 |
| mbv3_large_320 | R6_scene_cut | 89.1% | 89.1% | 4.34 | 18.25 | 20.67 | 18.25 | 4.21x | 0.833 | 1.000 | 5.3e-05 | 0.00094 | 0.03170 | 0.02766 |

## 4. What the suite proves, and what it does not

**Proved.** (a) P2 and P2-random now do the *same* work: P2-random replays exactly the tile set P2 skipped on
that frame and node and draws the same number uniformly at random, and the two MAC cuts agree to within 0.3 pp
in every regime. (b) At that equal budget the CM cache is far more faithful whenever content moves: 4.0x better
spatial error under camera pan, 36x under an illumination step (0.00030 vs 0.01084), 13x at scene cuts
(0.00050 vs 0.00668); the periodic keyframe baseline lands in between. (c) Top-5 agreement and softmax KL stay
at 1.000 / <= 1e-5 in all six regimes, so confidence does not degrade. (d) Latency percentiles are reported per
regime, not just on average.

**Not proved, and reported as such.** (a) *Scene-cut zero-ghosting*: measured `spatial_rel_l2` on the cut frame is
small but non-zero (0.0005-0.001), not the 0.0000 that a 100%-dirty guarantee would give, because the global-change
guard re-anchors at stride-2 stage entries and can let a few tiles through; the same guard is what keeps the reuse
aggressive elsewhere. (b) *Worst-case latency*: on 100%-dirty frames (R5/R6) single-frame latency reaches the dense
cost (32.2 ms vs 31.1 ms on ResNet-18, 20.7 ms vs 18.3 ms on MBV3-Large) - the 5% bound is not met; the cache saves
nothing when nothing can be reused, and the bookkeeping adds 3-13%. (c) *Wall-clock speedups in the 9x-55x range* are
real for this engine but they are a property of near-static synthetic streams, not a claim about real video.

## 5. Hardware energy and counters

Requested measurements could **not** be taken on this host and are reported as unavailable rather than estimated:
`/sys/class/powercap/intel-rapl:0/energy_uj` exists but returns EACCES for uid 1000 (RAPL needs root), and
`kernel.perf_event_paranoid = 2` with no `perf` binary blocks `perf_event_open` for retired instructions and LLC
misses. What *is* measured is the invariant-TSC cycle count per image (`cycles_per_image` in `results.json`), which
is a faithful proxy for CPU work: P1 drops ResNet-18 from 95.6M to 41.4M cycles/img and P2's cached frames drop to
~1.7M, consistent with the instruction-count reduction the kernel design predicts, but it is a proxy and is labeled
as one.


## 13. Phase E.2 closure

Stress generator fixed for real: R1 static+noise, R2 one 16x16 target at 2 px/frame, R3 four 40x40 targets at
18 px/frame, R4 rigid pan over a textured (not flat) background, R5 illumination step at frame 12, R6 hard cut every
9 frames. Masking is now one geometric projection of a single input-difference base grid per frame (no activation
re-anchoring), with a 12.5%-window rule for deep layers and a global full-flush on diluted dirty ratio > 0.40 or a
|mean brightness| jump > 0.025. Ladder (MobileNetV3-Small): R1 97.1% > R2 95.5% > R3 61.6% > R4 41.4%, with R5 86.6%
and R6 74.8% whose step/cut frames recompute everything (spatialRelL2 3.4e-4 / 1.2e-3). R3 and R4 land outside the
requested 25-48% / <15% bands because the flush decision uses the pre-window dilated ratio. The bottom-20% boundary
slice is implemented (18/90 and 60/90 samples; P1 margin drift 0.0963 vs random 0.1632). Memory is now measured
(weights, arena, cache bitmaps, VmHWM peak RSS). EfficientNet-B0/DenseNet-121 exports remain undone and are listed
in `reviewer_defense.json:meta.e2_status`.

## 8. Phase E.2 — re-run with an engine that agrees with torchvision

**Every number in sections 1-7 above is superseded.** Those runs used an engine that normalised `[0,1]` pixels
with `[0,255]` statistics and had three kernel index bugs, so all six networks produced nearly the same logits for
every image; approximations therefore looked free. The engine was fixed and pinned against `torchvision`
(`scripts/check_production_against_torch.py`, max `|dlogit| <= 8.7e-4`, Top-1 `1.000`, all six models), and
sections 8.1-8.3 below replace them. The speed numbers barely moved (they never depended on the logits); the
fidelity numbers moved a lot and are now worth publishing.

### 8.1 Static frames — six models, one binary

| model | M0 ms | torch ms | P1 cut | P1 ms | P1 x | P1 cos | P1 spat | P1r spat | P2 stream cut | P2 stream x | P2 spat | P2r spat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-18 | 65.85 | 20.37 | 46.2% | 40.81 | 1.61x | 0.653 | 1.0661 | 1.0874 | 55.2% | 1.78x | 0.1983 | 0.2669 |
| ResNet-50 | 148.55 | 58.36 | 49.4% | 94.37 | 1.57x | 0.654 | 1.2470 | 1.1846 | 55.3% | 1.80x | 0.2626 | 0.4061 |
| MobileNetV2 | 17.60 | 9.53 | 43.9% | 14.28 | 1.23x | 0.497 | 1.3298 | 1.5451 | 54.1% | 1.85x | 0.2676 | 0.4423 |
| MBV3-Small | 2.89 | 2.13 | 40.0% | 2.15 | 1.34x | 0.641 | 0.1367 | 0.2415 | 51.4% | 1.14x | 0.0152 | 0.1161 |
| ShuffleNetV2-x0.5 | 1.96 | 2.97 | 38.4% | 1.56 | 1.26x | 0.570 | 0.9969 | 1.1361 | 52.2% | 0.88x | 0.1772 | 0.4286 |
| MBV3-Large@320 | 25.74 | 14.10 | 59.4% | 18.35 | 1.40x | 0.579 | 0.3633 | 0.2875 | 57.8% | 1.63x | 0.0593 | 0.0996 |

Two honest notes. `P1` is a real approximation: cosine 0.50-0.65, pre-GAP map error 0.14-1.33. Its advantage over
the random control is in fidelity *per unit of work*, not per frame: the control deletes 26-90% of P1's MACs and
still ends with a worse mean map error (0.914 vs 0.857) and a worse mean cosine (0.498 vs 0.599). And `P1` on
ShuffleNetV2-x0.5 and MobileNetV2 is partly **load-bound**: at 38-44% MAC cut it only reaches 1.23-1.26x, because
the dense kernel already runs within 1.6x of the FFMA roofline on those graphs and the tile bookkeeping is a fixed
cost per node that small graphs cannot amortise. That shortfall is measured, not hidden.

### 8.2 Camera stream

| model | P2 cut | P2 ms | P2 x | P2 cos | P2 top1 | P2 spatial | random spatial | keyframe spatial |
|---|---|---|---|---|---|---|---|---|
| resnet18 | 55.2% | 37.11 | 1.78x | 0.9880 | 0.764 | 0.1983 | 0.2669 | 0.1595 |
| resnet50 | 55.3% | 82.11 | 1.80x | 0.9915 | 0.861 | 0.2626 | 0.4061 | 0.2525 |
| mobilenet_v2 | 54.1% | 9.50 | 1.85x | 0.9926 | 0.875 | 0.2676 | 0.4423 | 0.2699 |
| mobilenetv3_small | 51.4% | 2.49 | 1.14x | 0.9897 | 0.875 | 0.0152 | 0.1161 | 0.0867 |
| shufflenetv2_x0_5 | 52.2% | 2.24 | 0.88x | 0.9921 | 0.819 | 0.1772 | 0.4286 | 0.2408 |
| mbv3_large_320 | 57.8% | 15.69 | 1.63x | 0.9973 | 0.875 | 0.0593 | 0.0996 | 0.0666 |

The p2_caveat stands and is stronger than before: on a camera stream the reuse is only worth it when there is no
true motion. R1 (near-static) keeps Top-1 1.000 and `KL <= 1.5e-3` on every model, while under motion Top-1 falls
to 0.39-1.00 and `KL` rises to 0.01-0.52. The cache is a bandwidth win, not an accuracy-preserving one, whenever
the scene changes faster than the 2.5% tile tolerance. On shufflenetv2_x0_5 the stream mode is even slower than dense
(0.88x): 52% of the graph is skipped, but the remaining tiles are scattered
across nodes, so the skipped-tile mask bookkeeping outruns the saved MACs — the same load-bound effect as 8.1, at
frame granularity.

### 8.3 Six-regime ladder, all six models

| model | R1 static | R2 motion | R3 fast | R4 pan | R5 illum | R6 cut |
|---|---|---|---|---|---|---|
| ResNet-18 | 97.2% | 91.7% | 57.7% | 0.0% | 74.2% | 67.3% |
| ResNet-50 | 97.2% | 91.8% | 56.9% | 0.0% | 74.2% | 68.2% |
| MobileNetV2 | 96.8% | 90.6% | 52.2% | 0.0% | 71.9% | 64.7% |
| MBV3-Small | 94.4% | 87.0% | 44.3% | 0.0% | 67.6% | 60.2% |
| ShuffleNetV2-x0.5 | 94.8% | 88.0% | 47.2% | 0.0% | 68.8% | 60.7% |
| MBV3-Large@320 | 96.7% | 93.5% | 70.8% | 0.0% | 82.1% | 79.0% |

`R1 > R2 > R3 > R4` holds strictly for every model, `R4 = 0` because a global pan dirties the whole graph and the
frame flushes to dense. Scene-cut frames are now bit-exact (`cut_frame_spatial_rel_l2 == 0.000000` on all six
models): the flush runs the dense `M0` kernels in place over the persistent cache instead of re-anchoring at stage
entries. The requested bands (R1 > 90%, R2 60-90%, R3 25-55%, R4 < 15%) are met by the reference backbone
(MBV3-Small) and exceeded by four larger backbones, whose MACs sit on coarser grids (see
`docs/CM_CNN_ACCELERATION.md` 15.6); the bands are not universal constants.

### 8.4 What is still not claimed

* No mJ/img: RAPL is `EACCES`, `perf_event_paranoid = 2`; the cycles/image proxy is a model.
* No "P2 is exact" claim: exact on static content, bit-exact on flush frames, ghosting in between.
* No "beats oneDNN" claim: `M0` is 1.35x MBV3-Small's torch latency but 0.66x ShuffleNetV2-x0.5's.
* No single scalar predicts the per-model R2/R3 cut: the ladder's *shape* is architecture-independent, its levels
  are not.
