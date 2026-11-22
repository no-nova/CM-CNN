# CM-CNN-Accel

面向 AVX-512 CPU 的卷积神经网络推理加速：微架构感知的空间与时间计算跳过。论文 1，v2.0，2026 年 11 月。

核心论点是：在通用 CPU 上，理论 MAC 削减很少等比转化为墙钟加速；本文用三个相互耦合的因素解释
原因——SIMD 执行粒度（跳过是否破坏连续向量访存）、加载端口吞吐（每次 FMA 的加载 µop 数）、
以及卷积拓扑（一次跳过决策能控制多少算术量），并给出在这些约束下确实能跑快的方案。

**手稿：** [`paper1_bundle/CM_CNN_Accel_Paper1_Final.pdf`](paper1_bundle/CM_CNN_Accel_Paper1_Final.pdf)
（31 页英文）· 中文版 `paper1_bundle/CM_CNN_Accel_Paper1_ZH.pdf` · Markdown 源 `CM_CNN_Accel_Paper1_v2.md`

**Release：** [paper1-v1.0.0](https://github.com/no-nova/CM-CNN/releases/tag/paper1-v1.0.0)，PDF 作为独立附件。

## 主要结果

测试平台：单核 AMD Ryzen 9 9955HX（Zen 5，AVX-512F/BW/DQ/VL/VNNI），`OMP_NUM_THREADS=1`，
g++ 16.2.1，`-O3 -march=native -mavx512f -mfma -std=c++20`。

| 内容 | 结果 |
|---|---|
| `conv_tile` 稠密微内核（2×2 像素 × 4 输出组常驻 16 个 ZMM 寄存器） | 每次 FMA 仅 0.5 次加载 µop，81 GMAC/s；标量 gather 稀疏内核约需 2.0 |
| `S1` 基于并查集通道置换的 16-lane 对齐结构化剪枝 | ResNet-18 2.70×（精度 +0.56 pp），MobileNetV2 1.66–1.70×（+1.39 pp） |
| VGG-4 + 8-lane 共激活聚类（输入 78% 为精确零像素） | 消除 84.14% MAC，4.33×，精度 95.00% |
| `P2` 时间跳过 + 脏比 >0.50 时原地稠密回退 | 固定机位 V1/V2/V3/V5 消除 51.3–75.4% MAC（0.81–2.47×），硬切 V6 33.7–37.4%，切镜帧逐位精确 |
| Tile Arithmetic Density Model `α_l = MACs_l / (Tiles_l · C_in,l) ≈ 4·C_out,l·K_l² / g_l` | 解释为何同一视频流 ResNet-50 得 1.80×，而 ShuffleNetV2-x0.5 因调度开销反而慢 12% |

与 PyTorch 的数值对齐：静态图像 `max|Δlogit| ≤ 8.66e-4`，216 帧真实视频 `≤ 1.44e-3`，Top-1 一致率 0.995–1.000。

两个负面结果同样是论证的一部分：细粒度动态标量跳过**打不过**稠密微内核（0.61–0.97×），因为瓶颈是
gather 吞吐而非 MAC 数；帧间变化在空间上散乱时会触发 halo percolation，使时间复用退化，因此需要
膨胀前的 3×3 多数投票滤波与全局变化回退。

## 目录

| 路径 | 内容 |
|---|---|
| `paper1_bundle/` | 手稿 PDF（中英文）、Markdown 源、图、以及支撑各表格的 14 份结果 JSON |
| `native/` | AVX-512 内核源码：`conv_tile`、结构化剪枝、production 与 RGB 流水线 |
| `experiments/cm-cnn-{accel,enterprise,production,rgb-v4,universal}/` | 已编译驱动、数据集、分阶段结果 JSON、训练日志 |
| `src/lib/vision/` | 基准驱动使用的 TypeScript 模型与数据集生成器 |
| `src/__tests__/cm-cnn-*.test.ts` | 锁定论文数字的验收测试 |
| `scripts/` | 数据集生成、视频流构建、PyTorch 交叉校验 |
| `docs/CM_CNN_ACCELERATION.md` | 方法学、阶段记录与复现步骤 |
| `file/cm_cnn_accel_paper/` | 论文工作目录：Markdown 源、绘图脚本、PDF 构建 |
| `data/real_videos/` | 真实视频帧清单（帧本身需另行获取） |

## 复现

```bash
g++ -O3 -march=native -mavx512f -mavx512bw -mfma -std=c++20 \
    native/cm_production_avx512.cpp -o experiments/cm-cnn-production/cm_production_avx512

OMP_NUM_THREADS=1 experiments/cm-cnn-accel/cm_cnn_avx512 \
    --model experiments/cm-cnn-accel/pretrained_vgg4.bin --reps 10

node --test src/__tests__/cm-cnn-accel.test.ts
node --test src/__tests__/cm-cnn-production.test.ts
```

`package.json` 提供同样的入口：`bench:cm-cnn`、`bench:cm-cnn:universal`、`bench:cm-cnn:enterprise`、
`test:cm-cnn`、`test:cm-cnn:universal`、`test:cm-cnn:production`、`test:cm-cnn:enterprise`、
`test:cm-cnn:all`。没有运行时依赖，TypeScript 源码直接依赖 Node 的类型剥离运行。

超过 5 MB 的权重与数据集（导出的 `.bin`、`dataset/*.bin`）不入库：摘要记录在
`paper1_bundle/08_PhaseE_Model_Export_SHA256.json` 与各阶段的 `weights.sha256` 中，派生数据由
`scripts/` 重建。

## 引用

```
CM-CNN-Accel: Microarchitecture-Aware Spatial and Temporal Computation Skipping for CNN
Inference on AVX-512 CPUs. Technical Report 1, version 2.0, November 2026.
```

同系列：*CM-Transform*（论文 2，共激活置换块稀疏与缓存行对齐执行），
见 [no-nova/CM-Transformer](https://github.com/no-nova/CM-Transformer)。

## 许可

Mozilla Public License 2.0 — 见 [`LICENSE`](LICENSE)。
