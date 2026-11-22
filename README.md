# CM-CNN-Accel

Microarchitecture-aware spatial and temporal computation skipping for CNN inference on
AVX-512 CPUs. Paper 1, version 2.0, November 2026.

The manuscript argues that theoretical MAC reductions rarely become wall-clock speedups on
general-purpose CPUs, and shows which forms of redundancy do. The work is organised around
three coupled factors: SIMD execution granularity (does skipping keep vector loads
contiguous?), load-port throughput (load µops per FMA), and convolution topology (how much
arithmetic one skipping decision controls).

**Manuscript:** [`paper1_bundle/CM_CNN_Accel_Paper1_Final.pdf`](paper1_bundle/CM_CNN_Accel_Paper1_Final.pdf)
(31 pages, English) · Chinese edition `CM_CNN_Accel_Paper1_ZH.pdf` · TeX-free build via
`paper1_bundle/` sources and `file/cm_cnn_accel_paper/build_paper_pdf.sh`.

**Release:** [paper1-v1.0.0](https://github.com/no-nova/CM-CNN/releases/tag/paper1-v1.0.0)
carries the compiled PDF as a standalone asset.

## What the paper reports

Measured on a single AMD Ryzen 9 9955HX core (Zen 5, AVX-512F/BW/DQ/VL/VNNI), `OMP_NUM_THREADS=1`,
g++ 16.2.1 with `-O3 -march=native -mavx512f -mfma -std=c++20`.

| Finding | Result |
|---|---|
| `conv_tile` dense microkernel (2×2 pixel × 4 output groups resident in 16 ZMM registers) | 0.5 load µops per FMA, 81 GMAC/s; scalar-gather sparse kernels need ≈2.0 |
| `S1` 16-lane-aligned structured pruning over a union-find channel permutation | 2.70× on ResNet-18 (+0.56 pp accuracy), 1.66–1.70× on MobileNetV2 (+1.39 pp) |
| VGG-4 on naturally sparse input (78% exact-zero pixels) with 8-lane co-activation clustering | 84.14% of MACs removed, 4.33× at 95.00% accuracy |
| `P2` temporal skipping with a dense in-place fallback above a 0.50 MAC-weighted dirty ratio | 51.3–75.4% MAC elimination (0.81–2.47×) on fixed-camera V1/V2/V3/V5, 33.7–37.4% on hard cuts (V6), bit-exact on scene-cut frames |
| Tile Arithmetic Density Model `α_l = MACs_l / (Tiles_l · C_in,l) ≈ 4·C_out,l·K_l² / g_l` | explains why ResNet-50 gains 1.80× on a stream where ShuffleNetV2-x0.5 loses 12% to scheduling |

Numerical alignment with PyTorch: `max |Δlogit| ≤ 8.66e-4` on static images, `≤ 1.44e-3`
across 216 real-video frames, Top-1 agreement 0.995–1.000.

Two negative results are part of the argument rather than an appendix: fine-grained dynamic
scalar skipping *loses* to the dense microkernel (0.61–0.97×) because gather throughput, not
MAC count, sets the limit; and temporal reuse collapses under halo percolation when
inter-frame change is spatially scattered, which is why the paper adds a pre-dilation 3×3
majority filter and a global-change fallback.

## Layout

| Path | Contents |
|---|---|
| `paper1_bundle/` | manuscript (EN/ZH, Markdown + PDF), figures, and the 14 result JSONs behind the tables |
| `native/` | AVX-512 kernels: `conv_tile`, structured pruning, production and RGB pipelines |
| `experiments/cm-cnn-{accel,enterprise,production,rgb-v4,universal}/` | compiled drivers, datasets, per-phase result JSONs, training logs |
| `src/lib/vision/` | TypeScript models and dataset generators used by the benchmark drivers |
| `src/__tests__/cm-cnn-*.test.ts` | acceptance tests that pin the published numbers |
| `scripts/` | dataset generation (`gen-cm-cnn-dataset.ts`, `gen-cm-rgb-dataset.ts`), video stream building, PyTorch cross-checks |
| `docs/CM_CNN_ACCELERATION.md` | methodology, phase notes, and the reproduction recipe |
| `file/cm_cnn_accel_paper/` | paper working directory: Markdown sources, figure scripts, PDF build |
| `data/real_videos/` | real-video frame manifests (frames themselves are fetched) |

## Running things

The kernels build with g++ and the drivers are committed as compiled binaries:

```bash
g++ -O3 -march=native -mavx512f -mavx512bw -mfma -std=c++20 \
    native/cm_production_avx512.cpp -o experiments/cm-cnn-production/cm_production_avx512

# dense microkernel benchmark on VGG-4 weights
OMP_NUM_THREADS=1 experiments/cm-cnn-accel/cm_cnn_avx512 \
    --model experiments/cm-cnn-accel/pretrained_vgg4.bin --reps 10

# acceptance tests (Node 22+ with type stripping; no bundler needed)
node --test src/__tests__/cm-cnn-accel.test.ts
node --test src/__tests__/cm-cnn-production.test.ts
```

`package.json` carries the same entry points (`bench:cm-cnn`, `test:cm-cnn`,
`bench:cm-cnn:universal`, `bench:cm-cnn:enterprise`, plus the `*:universal` / `*:enterprise`
test aliases). Its `name` field and the Vite/React scripts belong to the UCZ project described
below, not to this paper.

Weight and dataset files larger than 5 MB are not committed (exported `.bin` weights,
`dataset/*.bin` streams): `paper1_bundle/08_PhaseE_Model_Export_SHA256.json` and the per-phase
`weights.sha256` files pin their digests, and the generators under `scripts/` rebuild the
derived ones.

## Other projects in this repository

Two unrelated projects were developed in the same tree and are still here, which is why the
root `README.md` used to describe a compression engine:

| Project | Where | What it is |
|---|---|---|
| **UCZ** (Universal Context-Mixing, Zero-Prior) | `src/` (React app + `src/lib/*.ts` codec), `tests/*.test.ts`, `docs/UCZ_V*.md`, `BENCHMARKS.md`, `CONSTITUTION.md`, `PRIOR_REGISTRY.md`, `index.html`, `vite.config.ts`, `benchmark/` | TypeScript zero-prior compression engine |
| **SNN** | `src/lib/snn/`, `src/lib/sequence/`, `docs/SNN_*.md`, `tests/snn.test.ts` | spiking-neural-network experiments |

Neither is used by the CM-CNN numbers. `src/lib/codec.ts` belongs to UCZ; the CM-CNN
experiments never modify it, and `docs/CM_CNN_ACCELERATION.md` asserts its SHA-256 as a
provenance check. Splitting these two projects into their own repositories is pending; until
then the file lists above are the boundary.

## Citation

```
CM-CNN-Accel: Microarchitecture-Aware Spatial and Temporal Computation Skipping for CNN
Inference on AVX-512 CPUs. Technical Report 1, version 2.0, November 2026.
```

Related work in this series: *CM-Transform* (Paper 2), co-activation permuted block sparsity
with cache-line aligned execution, at
[no-nova/CM-Transformer](https://github.com/no-nova/CM-Transformer).

## Licence

Mozilla Public License 2.0 — see [`LICENSE`](LICENSE).
