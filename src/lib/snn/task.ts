// SPDX-License-Identifier: MPL-2.0
//
// 任务与读出层：一个自造的、零下载的"事件相机"方向分类集。
//
// 之所以不用 MNIST/CIFAR：这个仓库的宪法禁止外部语料与预训练权重，而本实验
// 唯一要证明的事是"门控前后网络算出来的东西差多少"，任务的绝对精度并不重要，
// 可复现与零依赖才重要。刺激的形态（一条移动的亮条 + 背景散粒噪声）足够让
// 脉冲卷积核产生方向选择性，于是后续比较才有区分度。

import { splitmix32, type Rng } from './rand'

export interface EventSample {
  /** T × 1 × H × W 的 0/1 事件帧，按 (t, y, x) 排布。 */
  frames: Uint8Array
  label: number
}

export interface Dataset {
  T: number
  H: number
  W: number
  classes: number
  train: EventSample[]
  test: EventSample[]
}

/**
 * 生成"移动条"方向分类集。
 *
 * - 一条宽 `barW` 的亮条以 1 px/步 匀速移动，条上像素以概率 `pBar` 发事件；
 * - 背景以概率 `pNoise` 抛散粒噪声（事件相机的暗噪声）；
 * - 4 个类别：右 / 左 / 下 / 上。
 */
export function makeMovingBarDataset(opts: {
  perClass: number; testPerClass: number; T: number; H: number; W: number
  barW?: number; pBar?: number; pNoise?: number; seed?: number
}): Dataset {
  const { perClass, testPerClass, T, H, W } = opts
  const barW = opts.barW ?? 3
  const pBar = opts.pBar ?? 0.55
  const pNoise = opts.pNoise ?? 0.004
  const seed = opts.seed ?? 0x9e3779b9
  const classes = 4

  const make = (label: number, idx: number): EventSample => {
    const rng: Rng = splitmix32((seed ^ (label * 0x9e3779b1) ^ (idx * 0x85ebca6b)) >>> 0)
    const frames = new Uint8Array(T * H * W)
    // 起始位置随机，避免所有样本的条都从同一处出发（否则网络只需看第一帧）
    const start = Math.floor(rng() * (H - barW))
    for (let t = 0; t < T; t++) {
      const off = t * H * W
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          let p = pNoise
          if (label === 0) { // →
            const cx = (start + t) % W
            if (x >= cx && x < cx + barW) p = pBar
          } else if (label === 1) { // ←
            const cx = (W - 1 - ((start + t) % W) - barW + W * 2) % W
            if (x >= cx && x < cx + barW) p = pBar
          } else if (label === 2) { // ↓
            const cy = (start + t) % H
            if (y >= cy && y < cy + barW) p = pBar
          } else { // ↑
            const cy = (H - 1 - ((start + t) % H) - barW + H * 2) % H
            if (y >= cy && y < cy + barW) p = pBar
          }
          frames[off + y * W + x] = rng() < p ? 1 : 0
        }
      }
    }
    return { frames, label }
  }

  const train: EventSample[] = []
  const test: EventSample[] = []
  for (let c = 0; c < classes; c++) {
    for (let i = 0; i < perClass; i++) train.push(make(c, i))
    for (let i = 0; i < testPerClass; i++) test.push(make(c, perClass + i))
  }
  return { T, H, W, classes, train, test }
}

/**
 * 生成**非平稳突发流**（实验二用）。
 *
 * 审稿人的质疑是："固定 top-B 30% 在全黑的一帧上浪费、在爆闪的一帧上不够。"
 * 要回答它就得先造出这种数据：一个样本内部**逐时间步**剧烈改变活跃度——
 *
 *   - 前 `q1` 步：只有 1 条移动条（安静）；
 *   - 中间 `burst` 步：突然 4 条条同时出现 + 背景噪声上调（爆发）；
 *   - 后 `q2` 步：又回到 1 条（安静）。
 *
 * 同时返回逐时间步的活跃度，好让报告直接把"预算该跟着变"这件事画出来。
 */
export function makeBurstDataset(opts: {
  perClass: number; testPerClass: number; T: number; H: number; W: number
  seed?: number
}): Dataset & { burstFrom: number; burstTo: number; perStepDensity: Float64Array } {
  const { perClass, testPerClass, T, H, W } = opts
  const seed = opts.seed ?? 0x1f2e3d4c
  const burstFrom = Math.floor(T * 0.35)
  const burstTo = Math.floor(T * 0.65)
  const classes = 4

  const make = (label: number, idx: number): EventSample => {
    const rng: Rng = splitmix32((seed ^ (label * 0x9e3779b1) ^ (idx * 0x85ebca6b)) >>> 0)
    const frames = new Uint8Array(T * H * W)
    const start = Math.floor(rng() * (H - 3))
    for (let t = 0; t < T; t++) {
      const off = t * H * W
      const bursting = t >= burstFrom && t < burstTo
      const pBar = bursting ? 0.75 : 0.55
      const pNoise = bursting ? 0.05 : 0.004
      const nBars = bursting ? 4 : 1
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          let p = pNoise
          for (let b = 0; b < nBars; b++) {
            // 爆发时 4 条条互相错开 1/4 个周期，覆盖整幅画面
            const shift = (b * W) >> 2
            let cx: number, cy: number
            if (label === 0) { cx = (start + t + shift) % W; cy = -1 }
            else if (label === 1) { cx = (W - 1 - ((start + t + shift) % W) - 3 + W * 2) % W; cy = -1 }
            else if (label === 2) { cy = (start + t + shift) % H; cx = -1 }
            else { cy = (H - 1 - ((start + t + shift) % H) - 3 + H * 2) % H; cx = -1 }
            if (cx >= 0) { if (x >= cx && x < cx + 3) p = pBar }
            else if (y >= cy && y < cy + 3) p = pBar
          }
          frames[off + y * W + x] = rng() < p ? 1 : 0
        }
      }
    }
    return { frames, label }
  }

  const train: EventSample[] = []
  const test: EventSample[] = []
  for (let c = 0; c < classes; c++) {
    for (let i = 0; i < perClass; i++) train.push(make(c, i))
    for (let i = 0; i < testPerClass; i++) test.push(make(c, perClass + i))
  }
  // 逐时间步的输入密度（在测试集上量）——这就是"预算该长什么样"的真值
  const perStepDensity = new Float64Array(T)
  for (const s of test) for (let t = 0; t < T; t++) {
    let n = 0
    const off = t * H * W
    for (let i = 0; i < H * W; i++) n += s.frames[off + i]
    perStepDensity[t] += n / (H * W)
  }
  for (let t = 0; t < T; t++) perStepDensity[t] /= Math.max(1, test.length)
  return { T, H, W, classes, train, test, burstFrom, burstTo, perStepDensity }
}

/**
 * 生成**带分布突变（regime shift）的流**（实验三用）。
 *
 * 前 `shiftAt` 个样本是一种规律，之后突然换成另一种（默认：移动条变成**斜向**
 * 且速度翻倍）。用来看在线学习器的熵尖峰与重新收敛速度——与 UCZ 里
 * `CounterModel` 的 `MAX_COUNT` 对折老化是同一套机制。
 */
/** 突变后（第 `shiftAt` 个样本起）换成哪一种新规律。 */
export type RegimeB =
  /** 画面转 90°：逐像素密度完全不变，只是位置全换了（纯协变量漂移）。 */
  | 'transpose'
  /** 同密度均匀噪声：物体不见了，空间相关结构消失。 */
  | 'noise'
  /** 条变宽 3 → 9，亮度不变：输入量三倍。 */
  | 'wide'
  /** 条变实（p 0.55 → 1.0）且速度翻倍：抖动消失、时间相关结构改变。 */
  | 'solid'

export function makeRegimeShiftDataset(opts: {
  count: number; T: number; H: number; W: number; shiftAt?: number; seed?: number
  regimeB?: RegimeB
}): { samples: EventSample[]; T: number; H: number; W: number; shiftAt: number } {
  const { count, T, H, W } = opts
  const rb: RegimeB = opts.regimeB ?? 'noise'
  const shiftAt = opts.shiftAt ?? Math.floor(count / 2)
  const seed = opts.seed ?? 0x5a5a5a5a
  const samples: EventSample[] = []
  if (H !== W) throw Error('转置需要方阵输入，先保证 H === W')
  const BAR = 3

  /** 与"一条宽 BAR 的条"等密度的均匀噪声：BAR/边长 × 亮度 + 背景。 */
  const NOISE_P = (BAR / H) * 0.55 + (1 - BAR / H) * 0.004
  /** 宽 3×BAR 的条。 */
  const barPWide = (px: number, py: number, step: number, label: number): number =>
    barP(px, py, step, label) > 0 || barP(px, py, step - BAR, label) > 0 || barP(px, py, step - 2 * BAR, label) > 0
      ? 0.55 : 0

  /** 一条宽 BAR 的匀速移动条在像素 (px,py) 上的发放概率（step 已含起点）。 */
  const barP = (px: number, py: number, step: number, label: number): number => {
    if (label === 0) {
      const cx = step % W
      return px >= cx && px < cx + BAR ? 0.55 : 0
    }
    if (label === 1) {
      const cx = (W - 1 - (step % W) - BAR + W * 2) % W
      return px >= cx && px < cx + BAR ? 0.55 : 0
    }
    if (label === 2) {
      const cy = step % H
      return py >= cy && py < cy + BAR ? 0.55 : 0
    }
    const cy = (H - 1 - (step % H) - BAR + H * 2) % H
    return py >= cy && py < cy + BAR ? 0.55 : 0
  }

  for (let i = 0; i < count; i++) {
    const regime = i < shiftAt ? 0 : 1
    const rng: Rng = splitmix32((seed ^ (i * 0x85ebca6b) ^ (regime * 0xc2b2ae35)) >>> 0)
    const frames = new Uint8Array(T * H * W)
    const label = i & 3
    const start = Math.floor(rng() * (H - BAR))
    for (let t = 0; t < T; t++) {
      const off = t * H * W
      // 规律 A：轴对齐匀速移动条，1 px/步。
      // 规律 B：**整幅画面转 90°（转置）+ 速度翻倍**。
      //   转置保持每个像素的"亮/暗"分布完全不变（整帧密度一模一样），却把每条
      //   移动条搬到了完全不同的位置——于是每个神经元积累的输入历史全部作废，
      //   而"发放率变了"这个混淆因素被精确地钉住了。这正是协变量漂移的定义，
      //   也是在线模型最难、最值得看的一种突变。
      const step = regime === 0 ? start + t : start + t * 2
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          let q = 0.004
          if (regime === 0) q = barP(x, y, step, label)
          else if (rb === 'transpose') q = barP(y, x, step, label)
          else if (rb === 'noise') q = NOISE_P
          else if (rb === 'wide') q = barPWide(x, y, step, label)
          else q = barP(x, y, step, label) > 0 ? 1 : 0
          frames[off + y * W + x] = rng() < (q > 0 ? q : 0.004) ? 1 : 0
        }
      }
    }
    samples.push({ frames, label })
  }
  return { samples, T, H, W, shiftAt }
}

// ===== 读出层：多项式 logistic 回归 =====
//
// 读出层**只在稠密参考跑出来的特征上训练一次**，然后原样套到所有门控跑出来的
// 特征上。这样准确率的变化就纯粹反映"门控改了网络的表达"，而不是"每种门控
// 各自重新拟合了一次读出"。

/**
 * 把每一行就地 L2 归一：`x ← x / ‖x‖₂`。
 *
 * **为什么要它**：门控漏掉的脉冲会让特征向量的**模长**整体缩水（同样算 30% 的
 * 神经元，漏 34% 的脉冲 ⟹ 总发放数少三分之一），而线性读出看的是
 * `w·x`——模长一缩，所有类的分数一起缩，阈值边界就全歪了。归一之后读出只看
 * 各通道脉冲的**相对比例（方向）**，不看绝对数量：只要空间分布没变，哪怕漏掉
 * 35% 的脉冲，分类器照样认得出这个数字。
 *
 * 这是一个**后处理**步骤，作用在读出层的输入上，不改网络、不改门控、不改账本。
 * 没有可调参数，不是"再训一个读出层"。
 */
export function l2NormalizeRows(X: Float64Array, n: number, d: number): void {
  for (let i = 0; i < n; i++) {
    const b = i * d
    let s = 0
    for (let j = 0; j < d; j++) s += X[b + j] * X[b + j]
    if (!(s > 0)) continue // 全零行（一个脉冲都没发）保持原样
    const inv = 1 / Math.sqrt(s)
    for (let j = 0; j < d; j++) X[b + j] *= inv
  }
}

export interface Readout {
  classes: number
  d: number
  W: Float64Array // classes × (d+1)，最后一列是偏置
  mean: Float64Array
  inv: Float64Array
  /** 训练前是否对每行做过 L2 归一（记录下来，好在报告里说清口径）。 */
  l2Norm: boolean
}

export function trainReadout(
  X: Float64Array, y: Int32Array, n: number, d: number, classes: number,
  epochs = 400, lr = 0.5, l2 = 1e-4, seed = 12345, l2Norm = false,
  // Optional convention: distribute the L2 penalty over n samples, keeping λ independent of dataset size.
  // The default preserves historical experiments, where the penalty was applied n times per epoch.
  scalePenaltyBySamples = false,
): Readout {
  if (scalePenaltyBySamples && (!Number.isInteger(n) || n < 1)) throw Error('regularized readout requires samples')
  if (l2Norm) l2NormalizeRows(X, n, d)
  const mean = new Float64Array(d)
  const inv = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i * d + j]
  for (let j = 0; j < d; j++) mean[j] /= Math.max(1, n)
  let vars = new Float64Array(d)
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) {
    const t = X[i * d + j] - mean[j]
    vars[j] += t * t
  }
  for (let j = 0; j < d; j++) inv[j] = 1 / Math.sqrt(vars[j] / Math.max(1, n) + 1e-12)

  const W = new Float64Array(classes * (d + 1))
  const rng: Rng = splitmix32(seed)
  for (let i = 0; i < W.length; i++) W[i] = (rng() - 0.5) * 0.01
  const z = new Float64Array(n * (d + 1))
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) z[i * (d + 1) + j] = (X[i * d + j] - mean[j]) * inv[j]
    z[i * (d + 1) + d] = 1
  }
  const prob = new Float64Array(classes)
  for (let e = 0; e < epochs; e++) {
    const step = lr / (1 + e * 0.01)
    for (let i = 0; i < n; i++) {
      const b = i * (d + 1)
      let mx = -1e9
      for (let c = 0; c < classes; c++) {
        let s = 0
        for (let j = 0; j <= d; j++) s += W[c * (d + 1) + j] * z[b + j]
        prob[c] = s
        if (s > mx) mx = s
      }
      let sum = 0
      for (let c = 0; c < classes; c++) { prob[c] = Math.exp(prob[c] - mx); sum += prob[c] }
      for (let c = 0; c < classes; c++) {
        const g = (prob[c] / sum - (y[i] === c ? 1 : 0)) / n
        for (let j = 0; j <= d; j++) W[c * (d + 1) + j] -= step * (g * z[b + j] + (scalePenaltyBySamples ? l2 / n : l2) * W[c * (d + 1) + j])
      }
    }
  }
  return { classes, d, W, mean, inv, l2Norm }
}

export function readoutAccuracy(r: Readout, X: Float64Array, y: Int32Array, n: number): number {
  const d = r.d, classes = r.classes
  // 训练时归一过，评分时也要归一（幂等：已经是单位向量的行不会再变）
  if (r.l2Norm) l2NormalizeRows(X, n, d)
  let ok = 0
  for (let i = 0; i < n; i++) {
    let best = 0, bestScore = -1e18
    for (let c = 0; c < classes; c++) {
      let s = r.W[c * (d + 1) + d]
      for (let j = 0; j < d; j++) s += r.W[c * (d + 1) + j] * ((X[i * d + j] - r.mean[j]) * r.inv[j])
      if (s > bestScore) { bestScore = s; best = c }
    }
    if (best === y[i]) ok++
  }
  return ok / Math.max(1, n)
}

// ===== 真实数据：手写数字（sklearn `digits`，8×8）=====
//
// 报告第 7 节（现在的第 9 节）里写过"数据是合成的移动条，不是实拍数据"。为了不
// 让这句话一直挂着，这里引入一小份**真实**手写数字：
//
//   data/digits-8x8.csv.gz —— 57 KB，1797 张 8×8 灰度手写数字（0-9），
//   来自 scikit-learn 自带的 `digits`，其源头是 UCI 的 optdigits（43 位书写者
//   的真实笔迹，不是程序画的）。
//
// 之所以选它而不选 MNIST：MNIST 的一万张测试图是 7.5 MB，与本仓库"轻量、零
// 臃肿依赖"的原则冲突；而 8×8 / 57 KB 这个尺寸既能真跑起来，又足够"真"——
// 它有笔画结构、类内书写差异、以及 3/5、4/9 这种真的会认错的相邻类。
//
// 读取只用 node 内置的 zlib/fs，不引入任何 npm 依赖。

import { readFileSync, existsSync } from 'node:fs'
import { gunzipSync } from 'node:zlib'

export interface DigitsData {
  /** n × 64，逐行排布，取值 0..16。 */
  px: Uint8Array
  labels: Uint8Array
  n: number
  side: number
}

let digitsCache: DigitsData | null = null

/** 从仓库根往上找 `data/digits-8x8.csv.gz`。 */
function findDigitsCsv(): string | null {
  const rel = 'data/digits-8x8.csv.gz'
  let dir = process.cwd()
  for (let i = 0; i < 6; i++) {
    const p = `${dir}/${rel}`
    if (existsSync(p)) return p
    const up = dir.lastIndexOf('/')
    if (up <= 0) break
    dir = dir.slice(0, up)
  }
  return null
}

/** 读入手写数字集。文件不存在时返回 null（报告会明确写出"这一节被跳过"）。 */
export function loadDigits(path?: string): DigitsData | null {
  if (digitsCache && !path) return digitsCache
  const p = path ?? findDigitsCsv()
  if (!p) return null
  let text: string
  try {
    text = gunzipSync(readFileSync(p)).toString('utf8')
  } catch {
    return null
  }
  const lines = text.trim().split('\n')
  const n = lines.length
  const px = new Uint8Array(n * 64)
  const labels = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    const parts = lines[i].split(',')
    labels[i] = Number(parts[64]) | 0
    for (let j = 0; j < 64; j++) px[i * 64 + j] = Number(parts[j]) | 0
  }
  const d: DigitsData = { px, labels, n, side: 8 }
  if (!path) digitsCache = d
  return d
}

/** 8×8 → side×side 的最近邻放大（放大到网络输入的尺寸）。 */
function upscale(src: Uint8Array, from: number, to: number, out: Uint8Array): void {
  for (let y = 0; y < to; y++) {
    const sy = Math.min(from - 1, (y * from / to) | 0)
    for (let x = 0; x < to; x++) {
      const sx = Math.min(from - 1, (x * from / to) | 0)
      out[y * to + x] = src[sy * from + sx]
    }
  }
}

/** Explicit source-index dataset for non-contiguous disjoint splits. */
export function makeIndexedDigitsDataset(opts: {
  indices: number[]; T: number; H: number; W: number; encoding: 'poisson' | 'saccade'; seed?: number; path?: string
}): EventSample[] | null {
  const d = loadDigits(opts.path)
  if (!d) return null
  if (!opts.indices.every(i => Number.isInteger(i) && i >= 0 && i < d.n) || new Set(opts.indices).size !== opts.indices.length)
    throw Error('数字样本索引必须唯一且有效')
  const make = digitsEncoder(d, opts)
  return opts.indices.map(make)
}

function digitsEncoder(d: DigitsData, opts: {
  T: number; H: number; W: number; encoding: 'poisson' | 'saccade'; seed?: number
}): (index: number) => EventSample {
  const { T, H, W } = opts
  const seed = opts.seed ?? 0x2b2b2b2b
  const inCh = opts.encoding === 'saccade' ? 2 : 1
  const make = (idxIn: number): EventSample => {
    const idx = idxIn % d.n
    const big = new Uint8Array(H * W)
    upscale(d.px.subarray(idx * 64, idx * 64 + 64), d.side, H, big)
    // 归一化到 0..1（原图取值 0..16）
    const norm = new Float32Array(H * W)
    for (let i = 0; i < H * W; i++) norm[i] = big[i] / 16

    const frames = new Uint8Array(T * inCh * H * W)
    if (opts.encoding === 'poisson') {
      const rng: Rng = splitmix32((seed ^ (idx * 0x9e3779b1)) >>> 0)
      const pMax = 0.35
      for (let t = 0; t < T; t++) {
        const off = t * H * W
        for (let i = 0; i < H * W; i++) frames[off + i] = rng() < norm[i] * pMax ? 1 : 0
      }
      return { frames, label: d.labels[idx] }
    }
    // saccade：漂移 + 一次眼跳，取相邻帧差分的两个极性
    const rng: Rng = splitmix32((seed ^ (idx * 0x85ebca6b) ^ 0x5bd1e995) >>> 0)
    let vx = [-1, 0, 1][(rng() * 3) | 0]
    let vy = [-1, 0, 1][(rng() * 3) | 0]
    if (vx === 0 && vy === 0) vx = 1
    const margin = 2
    let ox = margin + ((rng() * (W - H - 2 * margin + 1)) | 0)
    let oy = margin + ((rng() * (H - H - 2 * margin + 1)) | 0)
    if (W - H - 2 * margin + 1 <= 0) ox = Math.max(0, (W - H) >> 1)
    if (H - H - 2 * margin + 1 <= 0) oy = 0
    const saccadeAt = (T * 0.5) | 0
    // 源图是 H×W（已放大到画布尺寸），位移靠裁剪窗口实现
    const cur = new Float32Array(H * W)
    const prev = new Float32Array(H * W)
    const place = (dst: Float32Array): void => {
      dst.fill(0)
      for (let y = 0; y < H; y++) {
        const sy = y - oy
        if (sy < 0 || sy >= H) continue
        for (let x = 0; x < W; x++) {
          const sx = x - ox
          if (sx < 0 || sx >= W) continue
          dst[y * W + x] = norm[sy * W + sx]
        }
      }
    }
    place(prev)
    for (let t = 0; t < T; t++) {
      if (t > 0) {
        ox += vx; oy += vy
        if (ox < 0) { ox = 0; vx = -vx }
        if (oy < 0) { oy = 0; vy = -vy }
        if (t === saccadeAt) { ox += (rng() < 0.5 ? -3 : 3); oy += (rng() < 0.5 ? -3 : 3) }
        if (ox < 0) ox = 0
        if (oy < 0) oy = 0
      }
      place(cur)
      const off = t * inCh * H * W
      if (t === 0) continue // 第一帧没有"变化"可言
      for (let i = 0; i < H * W; i++) {
        const dd = cur[i] - prev[i]
        if (dd > 0.05) frames[off + i] = 1 // ON
        else if (dd < -0.05) frames[off + H * W + i] = 1 // OFF
      }
      prev.set(cur)
    }
    return { frames, label: d.labels[idx] }
  }
  return make
}

/**
 * 把手写数字转成 T 步事件流（实验四用）。
 *
 * 两种编码，对应真实神经形态传感器的两种主流做法：
 *
 *  - `poisson`：**泊松速率编码**——每个像素每步以 `p ∝ 亮度` 独立发事件。
 *    常见于把静态图喂给 SNN 的基准做法。
 *  - `saccade`：**眼跳编码**——数字在视野里漂移（每步 1 px，中途一次眼跳），
 *    事件 = 相邻两帧的亮度变化（ON/OFF 两极性，两个输入通道）。这正是 DVS
 *    相机在眼动时看到的东西：只有运动边缘会产生事件。
 */
export function makeDigitsDataset(opts: {
  nTrain: number; nTest: number; T: number; H: number; W: number
  encoding: 'poisson' | 'saccade'; seed?: number; path?: string
  /** 可选的明确测试起始索引：不允许环绕或覆盖训练区。 */
  testStart?: number
}): Dataset | null {
  const d = loadDigits(opts.path)
  if (!d) return null
  const { nTrain, nTest, T, H, W } = opts
  const total = nTrain + nTest
  if (![nTrain, nTest].every(v => Number.isInteger(v) && v >= 0)) throw Error('样本数必须为非负整数')
  if (total > d.n) throw Error(`只要得起 ${d.n} 张，收到 ${total}`)
  if (opts.testStart !== undefined && (!Number.isInteger(opts.testStart) || opts.testStart < nTrain || opts.testStart + nTest > d.n))
    throw Error('测试区间必须在训练区之外且不能越界或环绕')

  const make = digitsEncoder(d, opts)

  const train: EventSample[] = []
  const test: EventSample[] = []
  for (let i = 0; i < nTrain; i++) train.push(make(i))
  // 保留历史 500/200 的索引，但当偏移环绕到训练区时改用剩余连续区间。
  // 两个区间必须真正不交，不能只依赖 total <= n。
  const start = (nTrain + 977) % d.n
  const legacyDisjoint = Array.from({ length: nTest }, (_, i) => (start + i) % d.n)
    .every(i => i >= nTrain)
  for (let i = 0; i < nTest; i++) test.push(make(opts.testStart !== undefined ? opts.testStart + i : legacyDisjoint ? start + i : nTrain + i))
  return { T, H, W, classes: 10, train, test }
}
