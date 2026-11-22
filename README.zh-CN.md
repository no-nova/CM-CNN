# CM-CNN-Accel

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24435068.svg)](https://doi.org/10.5281/zenodo.24435068)
[![release](https://img.shields.io/badge/release-paper1--v1.0.0-blue)](https://github.com/no-nova/CM-CNN/releases/tag/paper1-v1.0.0)
[![licence](https://img.shields.io/badge/licence-MPL--2.0-green)](LICENSE)

**面向 AVX-512 CPU 的卷积网络推理加速：微架构感知的空间与时间计算跳过。**
技术报告 1，v2.0，2026 年 11 月 ·
[手稿（英文，31 页）](paper1_bundle/CM_CNN_Accel_Paper1_Final.pdf) ·
[中文版](paper1_bundle/CM_CNN_Accel_Paper1_ZH.pdf) ·
[Zenodo 归档](https://doi.org/10.5281/zenodo.24435068)

本仓库包含论文背后的 AVX-512 内核、基准驱动、各阶段结果文件与复现步骤。

---

## 要回答的问题

空间/时间剪枝的论文通常只报 MAC 削减量。在通用 CPU 上，MAC 数并不是约束：真正决定墙钟时间的是
**每次 FMA 需要多少次加载 µop**、跳过决策是否破坏连续向量访存、以及**一次决策能控制多少算术量**。
本仓库测量哪一类冗余真的能转成时间，哪一类被 gather 吞吐或"不缩小的内存流量"打败。

贯穿全篇的量是 **Tile Arithmetic Density Model**：

```
α_l = MACs_l / (Tiles_l · C_in,l) ≈ 4 · C_out,l · K_l² / g_l
```

它按层预测 MAC 削减能否抵掉新增的 gather 开销。

## 主要结果

平台：单核 AMD Ryzen 9 9955HX（Zen 5，AVX-512F/BW/DQ/VL/VNNI），`OMP_NUM_THREADS=1`，
g++ 16.2.1，`-O3 -march=native -mavx512f -mfma -std=c++20`。

| 结果 | 数值 | 结果文件 |
|---|---|---|
| `conv_tile` 稠密微内核（2×2 像素 × 4 输出组常驻 16 个 ZMM 寄存器） | **每次 FMA 仅 0.5 次加载 µop，81 GMAC/s**；论文评估的标量 gather 稀疏内核约需 2.0 | `paper1_bundle/09_PhaseA_VGG4_Results.json` |
| 并查集通道置换 + 16-lane 对齐结构化剪枝（`S1`） | ResNet-18 **2.70×**（+0.56 pp），MobileNetV2 **1.66–1.70×**（+1.39 pp） | `paper1_bundle/13_PhaseC_Enterprise_Pruning_Results.json` |
| VGG-4 + 8-lane 共激活聚类（输入 78% 精确零像素） | 消除 **84.14%** MAC，**4.33×**，精度 95.00% | `paper1_bundle/09_…`、`10_PhaseA_VGG4_Ablation.json` |
| 时间跳过 `P2`（脏比 >0.50 时原地稠密回退） | 固定机位 V1/V2/V3/V5 消除 51.3–75.4% MAC（0.81–2.47×）；硬切 V6 33.7–37.4%；**切镜帧逐位精确** | `experiments/cm-cnn-production/real_video_results.json` |
| 与 PyTorch 的数值对齐 | 静态图像 `max|Δlogit| ≤ 8.66e-4`，216 帧真实视频 `≤ 1.44e-3`，Top-1 一致率 0.995–1.000 | `paper1_bundle/07_PhaseE_PyTorch_Agreement.json` |
| 权重导出完整性 | 逐模型、逐张量 SHA-256 | `paper1_bundle/08_PhaseE_Model_Export_SHA256.json` |

两个负面结果同样是论证的一部分：细粒度动态标量跳过**打不过**稠密微内核（0.61–0.97×），因为瓶颈是
gather 吞吐而非 MAC 数；帧间变化在空间上散乱时会触发 halo percolation，使时间复用退化，因此 `P2`
加入膨胀前的 3×3 多数投票滤波与全局变化回退。

## 目录

| 路径 | 内容 |
|---|---|
| [`paper1_bundle/`](paper1_bundle) | 手稿 PDF（中英文）、Markdown 源、图，以及支撑表格的 14 份结果文件 |
| [`native/`](native) | AVX-512 内核源码：`conv_tile`、结构化剪枝、production 与 RGB 流水线 |
| [`experiments/cm-cnn-accel/`](experiments/cm-cnn-accel) · [`enterprise/`](experiments/cm-cnn-enterprise) · [`production/`](experiments/cm-cnn-production) · [`rgb-v4/`](experiments/cm-cnn-rgb-v4) · [`universal/`](experiments/cm-cnn-universal) | 已编译驱动、数据集、分阶段结果 JSON、训练日志 |
| [`src/lib/vision/`](src/lib/vision) | 驱动使用的 TypeScript 模型与数据集生成器 |
| [`src/__tests__/cm-cnn-*.test.ts`](src/__tests__) | 锁定论文数字的验收测试 |
| [`scripts/`](scripts) | 数据集生成、视频流构建、PyTorch 交叉校验 |
| [`docs/CM_CNN_ACCELERATION.md`](docs/CM_CNN_ACCELERATION.md) | 方法学、阶段记录、复现步骤 |
| [`file/cm_cnn_accel_paper/`](file/cm_cnn_accel_paper) | 论文工作目录：Markdown 源、绘图脚本、PDF 构建 |
| [`data/real_videos/`](data/real_videos) | 真实视频帧清单（帧本身不随仓库分发） |

## 哪个文件支撑哪条结论

| 论文中的结论 | 结果文件 |
|---|---|
| 稠密微内核 0.5 加载 µop/FMA | `paper1_bundle/09_PhaseA_VGG4_Results.json` |
| 自然输入上的共激活聚类 | `paper1_bundle/09_…` + `10_PhaseA_VGG4_Ablation.json` |
| 通用零改动部署 | `paper1_bundle/12_PhaseB_Universal_ZeroMod_Results.json` |
| MobileNetV2 / ResNet-18 企业剪枝 | `paper1_bundle/13_…` + `14_PhaseC_Prune_Report.json` |
| 生产流水线与真实视频 | `experiments/cm-cnn-production/{results,real_video_results}.json`、`paper1_bundle/06_PhaseDE_Production_Results.json` |
| 六模型评审防守套件 | `paper1_bundle/05_PhaseE_Reviewer_Defense_6Models.json` |
| 与 PyTorch 一致 | `paper1_bundle/07_PhaseE_PyTorch_Agreement.json` |
| 空间热力图（图 5） | `paper1_bundle/11_PhaseA_Spatial_Heatmaps.txt` |

## 快速开始

内核用 g++ 构建，驱动以已编译二进制提交：

```bash
g++ -O3 -march=native -mavx512f -mavx512bw -mfma -std=c++20 \
    native/cm_production_avx512.cpp -o experiments/cm-cnn-production/cm_production_avx512

OMP_NUM_THREADS=1 experiments/cm-cnn-accel/cm_cnn_avx512 \
    --model experiments/cm-cnn-accel/pretrained_vgg4.bin --reps 10

OMP_NUM_THREADS=1 experiments/cm-cnn-production/cm_production_avx512 \
    --manifest experiments/cm-cnn-production/real_video_manifest.json --reps 3
```

验收测试（Node 22+，无需打包器；`npm` 别名与之一致）：

```bash
node --test src/__tests__/cm-cnn-accel.test.ts
node --test src/__tests__/cm-cnn-production.test.ts
npm run test:cm-cnn:all
```

超过 5 MB 的权重与数据集（导出的 `.bin`、`dataset/*.bin`）不入库：摘要记录在
`paper1_bundle/08_PhaseE_Model_Export_SHA256.json` 与各阶段 `weights.sha256` 中，派生数据由
`scripts/` 下的生成器重建。

## 数字是怎么测出来的

- **平台**：单核 Zen 5，`OMP_NUM_THREADS=1`，排除线程调度因素；SIMD 开关与编译器版本记录在
  `docs/CM_CNN_ACCELERATION.md`。
- **交叉校验**：每个内核结果都与同输入的 PyTorch 参考对比，验收测试在这些数字漂移时报错。
- **视频**：六段真实序列（V1–V6）覆盖不同运动与机位；切镜交给稠密回退，并验证逐位一致。
- **权重**：只读，运行前后各校验一次 SHA-256。

## 边界

论文只讨论单核、batch-1 推理，不声称多核扩展或 GPU 结果。时间复用机制假设场景部分静止，报告的是
切镜时实际发生的情况，而不是把它假设掉。

同系列：**CM-Transform**（论文 2，共激活置换块稀疏与缓存行对齐执行），
[github.com/no-nova/CM-Transformer](https://github.com/no-nova/CM-Transformer)，
[doi:10.5281/zenodo.24435107](https://doi.org/10.5281/zenodo.24435107)。

## 引用

```
CM-CNN-Accel: Microarchitecture-Aware Spatial and Temporal Computation Skipping for CNN
Inference on AVX-512 CPUs. Technical Report 1, version 2.0, November 2026.
doi:10.5281/zenodo.24435068
```

机器可读元数据：[`CITATION.cff`](CITATION.cff) · 许可：[MPL-2.0](LICENSE)。
