# CM-CNN-Accel

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24435068.svg)](https://doi.org/10.5281/zenodo.24435068)
[![release](https://img.shields.io/badge/release-paper1--v1.0.0-blue)](https://github.com/no-nova/CM-CNN/releases/tag/paper1-v1.0.0)
[![licence](https://img.shields.io/badge/licence-MPL--2.0-green)](LICENSE)

**Microarchitecture-aware spatial and temporal computation skipping for CNN inference on
AVX-512 CPUs.** Technical report 1, version 2.0, November 2026 ·
[manuscript (PDF, 31 pages)](paper1_bundle/CM_CNN_Accel_Paper1_Final.pdf) ·
[Chinese edition](paper1_bundle/CM_CNN_Accel_Paper1_ZH.pdf) ·
[archived at Zenodo](https://doi.org/10.5281/zenodo.24435068)

AVX-512 kernels, benchmark drivers, per-phase result files and the reproduction recipe behind
the paper.

---

## The question

Spatial and temporal pruning papers usually report MAC reductions and stop there. On a
general-purpose core the MAC count is not the binding constraint: what matters is how many
*load µops* each FMA needs, whether a skipping decision keeps vector loads contiguous, and how
much arithmetic one decision controls. This repository measures which forms of redundancy
actually convert into wall-clock time, and which ones are defeated by gather throughput or by
memory traffic that never shrinks.

The organising quantity is the **tile arithmetic density model**,

```
α_l = MACs_l / (Tiles_l · C_in,l) ≈ 4 · C_out,l · K_l² / g_l
```

which predicts, per layer, whether the reduction in MACs can pay for the gathers it adds.

## Headline numbers

Single AMD Ryzen 9 9955HX core (Zen 5, AVX-512F/BW/DQ/VL/VNNI), `OMP_NUM_THREADS=1`,
g++ 16.2.1, `-O3 -march=native -mavx512f -mfma -std=c++20`.

| Result | Value | File |
|---|---|---|
| `conv_tile` dense microkernel (2×2 pixel × 4 output groups in 16 ZMM registers) | **0.5 load µops per FMA, 81 GMAC/s**; the scalar-gather sparse kernels the paper evaluates need ≈2.0 | `paper1_bundle/09_PhaseA_VGG4_Results.json` |
| 16-lane-aligned structured pruning over a union-find channel permutation (`S1`) | **2.70×** ResNet-18 (+0.56 pp), **1.66–1.70×** MobileNetV2 (+1.39 pp) | `paper1_bundle/13_PhaseC_Enterprise_Pruning_Results.json` |
| VGG-4 on naturally sparse input (78% exact-zero pixels), 8-lane co-activation clustering | 84.14% of MACs removed, **4.33×**, 95.00% accuracy | `paper1_bundle/09_PhaseA_VGG4_Results.json`, `10_PhaseA_VGG4_Ablation.json` |
| Temporal skipping (`P2`), dense in-place fallback above a 0.50 MAC-weighted dirty ratio | 51.3–75.4% MAC elimination (0.81–2.47×) on fixed-camera V1/V2/V3/V5; 33.7–37.4% on hard cuts (V6); **bit-exact on scene-cut frames** | `experiments/cm-cnn-production/real_video_results.json` |
| Numerical alignment with PyTorch | `max|Δlogit| ≤ 8.66e-4` static, `≤ 1.44e-3` over 216 real-video frames, Top-1 agreement 0.995–1.000 | `paper1_bundle/07_PhaseE_PyTorch_Agreement.json` |
| Weight export integrity | SHA-256 pinned per model and per exported tensor | `paper1_bundle/08_PhaseE_Model_Export_SHA256.json` |

Two negative results are part of the argument: fine-grained dynamic scalar skipping **loses** to
the dense microkernel (0.61–0.97×) because gather throughput, not MAC count, sets the limit; and
temporal reuse collapses under halo percolation when inter-frame change is spatially scattered,
which is why `P2` adds a pre-dilation 3×3 majority filter and a global-change fallback.

## What is in here

| Path | Contents |
|---|---|
| [`paper1_bundle/`](paper1_bundle) | manuscript PDFs (EN/ZH), Markdown source, figures, and the 14 result files behind the tables |
| [`native/`](native) | AVX-512 kernels: `conv_tile`, structured pruning, production and RGB pipelines |
| [`experiments/cm-cnn-accel/`](experiments/cm-cnn-accel) · [`enterprise/`](experiments/cm-cnn-enterprise) · [`production/`](experiments/cm-cnn-production) · [`rgb-v4/`](experiments/cm-cnn-rgb-v4) · [`universal/`](experiments/cm-cnn-universal) | compiled drivers, datasets, per-phase result JSONs, training logs |
| [`src/lib/vision/`](src/lib/vision) | TypeScript models and dataset generators used by the drivers |
| [`src/__tests__/cm-cnn-*.test.ts`](src/__tests__) | acceptance tests that pin the published numbers |
| [`scripts/`](scripts) | dataset generation, video stream building, PyTorch cross-checks |
| [`docs/CM_CNN_ACCELERATION.md`](docs/CM_CNN_ACCELERATION.md) | methodology, phase notes, reproduction recipe |
| [`file/cm_cnn_accel_paper/`](file/cm_cnn_accel_paper) | paper working directory: Markdown source, figure scripts, PDF build |
| [`data/real_videos/`](data/real_videos) | real-video frame manifests (frames are fetched, not redistributed) |

## Which file backs which claim

| Claim in the paper | Result file |
|---|---|
| Dense microkernel reaches 0.5 load µops per FMA | `paper1_bundle/09_PhaseA_VGG4_Results.json` |
| Co-activation clustering on natural input | `paper1_bundle/09_…` + `10_PhaseA_VGG4_Ablation.json` |
| Universal zero-modification deployment | `paper1_bundle/12_PhaseB_Universal_ZeroMod_Results.json` |
| Enterprise pruning on MobileNetV2 / ResNet-18 | `paper1_bundle/13_…` + `14_PhaseC_Prune_Report.json` |
| Production pipeline and real video | `experiments/cm-cnn-production/{results,real_video_results}.json`, `paper1_bundle/06_PhaseDE_Production_Results.json` |
| Six-model reviewer-defence suite | `paper1_bundle/05_PhaseE_Reviewer_Defense_6Models.json` |
| Agreement with PyTorch | `paper1_bundle/07_PhaseE_PyTorch_Agreement.json` |
| Spatial heatmaps (figure 5) | `paper1_bundle/11_PhaseA_Spatial_Heatmaps.txt` |

## Quick start

The kernels build with g++ and the drivers are committed as compiled binaries:

```bash
g++ -O3 -march=native -mavx512f -mavx512bw -mfma -std=c++20 \
    native/cm_production_avx512.cpp -o experiments/cm-cnn-production/cm_production_avx512

# dense microkernel benchmark on the VGG-4 weights
OMP_NUM_THREADS=1 experiments/cm-cnn-accel/cm_cnn_avx512 \
    --model experiments/cm-cnn-accel/pretrained_vgg4.bin --reps 10

# production pipeline on the real-video manifest
OMP_NUM_THREADS=1 experiments/cm-cnn-production/cm_production_avx512 \
    --manifest experiments/cm-cnn-production/real_video_manifest.json --reps 3
```

Acceptance tests (Node 22 or newer, no bundler; `npm` aliases mirror these):

```bash
node --test src/__tests__/cm-cnn-accel.test.ts
node --test src/__tests__/cm-cnn-production.test.ts
npm run test:cm-cnn:all
```

Files larger than 5 MB (exported weights, `dataset/*.bin` streams) are not committed:
`paper1_bundle/08_PhaseE_Model_Export_SHA256.json` and the per-phase `weights.sha256` pin their
digests, and the generators under `scripts/` rebuild the derived ones.

## How the numbers were produced

- **Platform.** One Zen 5 core, `OMP_NUM_THREADS=1`, so thread scheduling is out of the picture;
  SIMD flags and compiler version are recorded in `docs/CM_CNN_ACCELERATION.md`.
- **Cross-checks.** Every kernel result is compared against a PyTorch reference on the same
  inputs; the acceptance tests fail if a published number drifts.
- **Video.** Six real sequences (V1–V6) with different motion and camera behaviour; scene cuts
  are handled by the dense fallback and verified to reproduce bit-exactly.
- **Weights.** Read-only, SHA-256 pinned before and after each run.

## Scope

The paper is about a single core and about batch-1 inference; it does not claim multi-core
scaling or GPU results. Its temporal mechanism assumes a partially static scene, and the
measurements quantify what happens at cuts rather than assuming them away.

Companion report: **CM-Transform** (paper 2), co-activation permuted block sparsity and
cache-line aligned execution for transformers —
[github.com/no-nova/CM-Transformer](https://github.com/no-nova/CM-Transformer),
[doi:10.5281/zenodo.24435107](https://doi.org/10.5281/zenodo.24435107).

## Citation

```
CM-CNN-Accel: Microarchitecture-Aware Spatial and Temporal Computation Skipping for CNN
Inference on AVX-512 CPUs. Technical Report 1, version 2.0, November 2026.
doi:10.5281/zenodo.24435068
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff) · licence:
[MPL-2.0](LICENSE).
