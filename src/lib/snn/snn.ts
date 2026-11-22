// SPDX-License-Identifier: MPL-2.0
//
// 脉冲卷积网络 + 四种"下一刻该算哪些神经元"的门控策略。
//
// 这个文件是整个实验的主角。它把对话里的两个主张都变成了可以跑的代码：
//
//   可能性 1（时间 = 一串矩阵）：`stepLayer` 每步推进一次，膜电位是递归状态。
//   可能性 2（时间 = 矩阵的一个维度）：`psnToeplitz` / `psnScan` 把 T 个时间步
//     折进一次带因果掩码的矩阵乘法里，不再有逐步依赖。
//
//   小型 CNN 预测下一刻要用的神经元：`stepLayer` 的 `cm-freeze` / `cm-cons`
//     两种模式，候选集来自 `predict.ts` 的上下文混合预测器。
//
// 名词约定：
//   * **fanin** = inCh*k*k，一个神经元每步的乘加次数——这就是对话里"总矩阵"的一行。
//   * **candidate / 候选集** = 本步真正做乘加的神经元集合。
//   * **证书（certificate）** = 一个**可证明**的上界：v̂ ≥ v 恒成立，于是
//     v̂ < θ 时"这个神经元一定不会发放"是可以被证明的，不是猜的。

import { makeGauss, splitmix32, type Rng } from './rand'

// ===== 层定义 =====

export interface LayerSpec {
  inCh: number
  outCh: number
  inH: number
  inW: number
  k: number
  stride: number
  pad: number
  /** 膜电位泄漏系数 λ ∈ (0,1)。 */
  decay: number
  /** 发放阈值 θ。 */
  thresh: number
  /** 权重初始化尺度。 */
  scale: number
  seed: number
  /**
   * 发放后是否把膜电位硬重置为 0。
   *
   * 这个开关是本实验的一条主线：重置是一步强非线性，它同时 (a) 让整条时间线
   * 折不进一次矩阵乘（LIF 折不了，无重置的 PSN 能折），(b) 破坏"跳过者取上界
   * ⟹ 状态仍是真值上界"这个单调性（见 experiment.ts 的定理验证）。
   */
  hardReset: boolean
}

export type GateMode =
  | 'dense' // 基线：全算
  | 'event' // 盒内零输入跳过（逐位无损）
  | 'cert' // 上界证书：跳过者取电流上界
  | 'cm-decay' // 预测器 top-B；跳过者只泄漏
  | 'cm-bound' // 预测器 top-B；跳过者取上界
  | 'cm-thresh' // 预测器**阈值**门控：P(spike) > τ 就算（不定预算、不排序）
  | 'random' // 对照：同预算随机选
export type BoundMode = 'perCh' | 'pooled'

export interface Layer {
  spec: LayerSpec
  outH: number
  outW: number
  /** 神经元数 = outCh*outH*outW。 */
  N: number
  fanin: number
  W: Float32Array
  v: Float32Array
  /** 证书用：每个 (输出通道, 输入通道) 的正权重降序前缀和。 */
  topPerCh: Float32Array
  /** 证书用：每个输出通道、跨所有输入通道的正权重降序前缀和（更松但更便宜）。 */
  topPooled: Float32Array
  spk: Uint8Array
  /** 可塑性稳态阈值偏移，按输出通道；默认为零。 */
  thresholdOffsets: Float32Array
}

export function outSize(inSize: number, k: number, stride: number, pad: number): number {
  return Math.floor((inSize + 2 * pad - k) / stride) + 1
}

export function buildLayer(spec: LayerSpec): Layer {
  const outH = outSize(spec.inH, spec.k, spec.stride, spec.pad)
  const outW = outSize(spec.inW, spec.k, spec.stride, spec.pad)
  if (outH <= 0 || outW <= 0) throw Error(`卷积输出尺寸非正: ${JSON.stringify(spec)}`)
  const N = spec.outCh * outH * outW
  const kk = spec.k * spec.k
  const rng: Rng = splitmix32(spec.seed)
  const gauss = makeGauss(rng)
  const W = new Float32Array(spec.outCh * spec.inCh * kk)
  for (let i = 0; i < W.length; i++) W[i] = gauss() * spec.scale

  const L: Layer = {
    spec, outH, outW, N, fanin: spec.inCh * kk, W,
    v: new Float32Array(N), topPerCh: new Float32Array(spec.outCh * spec.inCh * (kk + 1)),
    topPooled: new Float32Array(spec.outCh * (spec.inCh * kk + 1)), spk: new Uint8Array(N),
    thresholdOffsets: new Float32Array(spec.outCh),
  }
  rebuildWeightBounds(L)
  return L
}

/** 权重改变后必须重建证书上界，不能继续使用旧权重的缓存。 */
export function rebuildWeightBounds(L: Layer): void {
  const { spec, W, topPerCh, topPooled } = L
  const kk = spec.k * spec.k
  const tmp = new Float64Array(kk)
  for (let oc = 0; oc < spec.outCh; oc++) {
    for (let ic = 0; ic < spec.inCh; ic++) {
      let m = 0
      for (let j = 0; j < kk; j++) {
        const w = W[(oc * spec.inCh + ic) * kk + j]
        if (w > 0) tmp[m++] = w
      }
      const slice = Array.from(tmp.subarray(0, m)).sort((a, b) => b - a)
      const base = (oc * spec.inCh + ic) * (kk + 1)
      let acc = 0
      topPerCh[base] = 0
      for (let j = 0; j < kk; j++) {
        if (j < m) acc += slice[j]
        topPerCh[base + j + 1] = acc
      }
    }
    // pooled：跨通道合并的正权重降序前缀和（只按"盒内总脉冲数"查一次表）
    let m = 0
    const big = new Float64Array(spec.inCh * kk)
    for (let j = 0; j < spec.inCh * kk; j++) {
      const w = W[oc * spec.inCh * kk + j]
      if (w > 0) big[m++] = w
    }
    const slice = Array.from(big.subarray(0, m)).sort((a, b) => b - a)
    const base = oc * (spec.inCh * kk + 1)
    let acc = 0
    topPooled[base] = 0
    for (let j = 0; j < spec.inCh * kk; j++) {
      if (j < m) acc += slice[j]
      topPooled[base + j + 1] = acc
    }
  }
}

export function resetLayer(L: Layer): void {
  L.v.fill(0)
  L.spk.fill(0)
}

// ===== 盒计数：小型 CNN 的第一层 =====
//
// "下一刻有多少脉冲会落进这个神经元的感受野"——这是一个**盒滤波**，也就是一个
// 卷积核全为 1 的 CNN。用积分图做，代价与核大小无关；多尺度就是多几个核。
// 这一步的输出同时喂给 (a) 证书的上界表 和 (b) 上下文混合预测器。

export function boxCountScratch(inCh: number, inH: number, inW: number): Int32Array {
  return new Int32Array(inCh * (inH + 1) * (inW + 1))
}

/** 建积分图：每个输入通道一张，(inH+1)×(inW+1)。 */
export function buildIntegral(
  S: Uint8Array, inCh: number, inH: number, inW: number, scratch: Int32Array,
): number {
  const iw = inW + 1
  const plane = iw * (inH + 1)
  let ops = 0
  for (let ic = 0; ic < inCh; ic++) {
    const base = ic * inH * inW
    const planeBase = ic * plane
    scratch.fill(0, planeBase, planeBase + iw) // 第 0 行（含第 0 列）为 0
    for (let y = 0; y < inH; y++) {
      let sum = 0
      const o = planeBase + (y + 1) * iw
      scratch[o] = 0
      const rb = base + y * inW
      for (let x = 0; x < inW; x++) {
        sum += S[rb + x]
        scratch[o + x + 1] = scratch[o - iw + x + 1] + sum
      }
      ops += inW
    }
  }
  return ops
}

/** 从积分图里查盒和：4 次取数换一个任意大小的盒，与核大小无关。 */
export function boxCountsFromII(
  scratch: Int32Array, inCh: number, inH: number, inW: number,
  k: number, stride: number, pad: number, outH: number, outW: number, out: Int32Array,
): number {
  const iw = inW + 1
  const plane = iw * (inH + 1)
  let ops = 0
  for (let ic = 0; ic < inCh; ic++) {
    const ii = ic * plane
    for (let oy = 0; oy < outH; oy++) {
      let y0 = oy * stride - pad
      let y1 = y0 + k
      if (y0 < 0) y0 = 0
      if (y1 > inH) y1 = inH
      for (let ox = 0; ox < outW; ox++) {
        let x0 = ox * stride - pad
        let x1 = x0 + k
        if (x0 < 0) x0 = 0
        if (x1 > inW) x1 = inW
        const c = y1 <= y0 || x1 <= x0
          ? 0
          : scratch[ii + y1 * iw + x1] - scratch[ii + y0 * iw + x1] - scratch[ii + y1 * iw + x0] + scratch[ii + y0 * iw + x0]
        out[(oy * outW + ox) * inCh + ic] = c
        ops += 4
      }
    }
  }
  return ops
}

/**
 * 一步到位：建积分图 + 查盒和。
 *
 * `pad` 可以大于卷积核的 padding：把 box 放大到 K 并以同一中心对齐，等价于
 * `pad' = pad + (K-k)>>1, k' = K`，于是"多尺度盒滤波"就是换一次 padding 参数，
 * 积分图只建一次。
 *
 * @returns 本次的加/取数次数（用于把门控开销老实算进总成本）
 */
export function boxCounts(
  S: Uint8Array, inCh: number, inH: number, inW: number,
  k: number, stride: number, pad: number, outH: number, outW: number,
  scratch: Int32Array, out: Int32Array,
): number {
  return buildIntegral(S, inCh, inH, inW, scratch) +
    boxCountsFromII(scratch, inCh, inH, inW, k, stride, pad, outH, outW, out)
}

// ===== 一步推进 =====

export interface StepOptions {
  mode: GateMode
  bound: BoundMode
  /**
   * 预测器给出的发放概率（0..1），cm-* 模式用。
   *
   * **长度是 N / group 而不是 N**：`group` 个连续神经元共享一个概率。
   */
  probs?: Float32Array
  /** 预算：本步最多算多少个**单元**（不是神经元）。 */
  budget?: number
  /** 复用的选择缓冲（长度 ≥ N/group），由调用方提供以避免每步分配。 */
  select?: Uint8Array
  /** 复用的"最终要算"缓冲（长度 ≥ N/group）：`forced` 与 `select` 的并集。 */
  take?: Uint8Array
  /** 复用的直方图缓冲（长度 128；只有 top-B 模式用）。 */
  hist?: Int32Array
  /**
   * 结构化稀疏的粒度：每 `group` 个连续神经元打包成一个"密集小块"，
   * 一起算或一起跳。1 = 逐神经元（非结构化，GPU 上会碎片化）。
   */
  group?: number
  /** `cm-thresh` 的概率阈值：P(spike) > τ 才计算。 */
  tau?: number
  /** 随机模式用：本步"可能有输入"的神经元掩码（与预测器的前置过滤一致）。 */
  eligible?: Uint8Array
  rng?: Rng
}

export interface StepStats {
  computed: number
  skipped: number
  spikes: number
  /** 真实乘加次数（"总矩阵"里实际被乘的那部分）。 */
  macs: number
  /** 门控本身的代价（积分图 + 盒查询 + 上界查表 + 选取）。 */
  gateOps: number
  /**
   * **连续段的个数**：把"要不要算"按神经元编号排成 0/1 串，从 0 跳到 1 的次数。
   *
   * 这是"非结构化稀疏会不会把访存打碎"的可测代理量：逐神经元选点时
   * `runs ≈ 被选中的神经元数`（几乎每段长度 1），纤维模式下 `runs ≈ 被选中的块数`
   * 而每段长度 ≥ group。GPU/SIMD 需要的是**少而长的段**。
   */
  runs: number
  /** 平均段长 = computed / runs（越长越coalesce）。 */
  runLen: number
  /** 被选中的小块数（单元数）。 */
  tiles: number
}

/**
 * 按概率直方图选出"预算内最可能发放的 B 个神经元"。
 *
 * 用直方图而不是排序：`O(N)` 而不是 `O(N log N)`。选出来的是**阈值以上全体**，
 * 所以实际个数可能略多于 B（预算是下界）。
 *
 * `p[i] < 0` 表示该神经元被前置过滤器判定为"本步不可能发放"（见 predict.ts），
 * 不参与排序、也不占用预算。
 */
export function selectTopB(p: Float32Array, n: number, B: number, out: Uint8Array, hist: Int32Array): number {
  out.fill(0, 0, n)
  let eligible = 0
  for (let i = 0; i < n; i++) if (p[i] >= 0) eligible++
  if (B >= eligible) {
    for (let i = 0; i < n; i++) out[i] = p[i] >= 0 ? 1 : 0
    return eligible
  }
  hist.fill(0)
  const BINS = hist.length
  for (let i = 0; i < n; i++) {
    if (p[i] < 0) continue
    let b = (p[i] * BINS) | 0
    if (b < 0) b = 0
    else if (b >= BINS) b = BINS - 1
    hist[b]++
  }
  let acc = 0, thr = BINS - 1
  for (let b = BINS - 1; b >= 0; b--) {
    acc += hist[b]
    if (acc >= B) { thr = b; break }
  }
  // 临界桶只取"还差的那几个"：概率大量并列时（同一上下文的神经元共享一个预测值）
  // 整桶收下会让实际候选数远超预算，与 random 对照就不公平了。
  let quota = B - (acc - hist[thr])
  let count = 0
  for (let i = 0; i < n; i++) {
    if (p[i] < 0) continue
    let b = (p[i] * BINS) | 0
    if (b < 0) b = 0
    else if (b >= BINS) b = BINS - 1
    if (b > thr) { out[i] = 1; count++ }
    else if (b === thr && quota > 0) { out[i] = 1; quota--; count++ }
  }
  return count
}

/**
 * **阈值门控**：`P(spike) > τ` 就算，否则跳过。
 *
 * 与 top-B 的三点区别，每一点都是审稿人问的"能不能再省"：
 *   1. **不排序**：单遍扫描，O(N) 且常数是 1（top-B 要建直方图 + 阈值搜索 + 再扫一遍）。
 *   2. **预算是数据决定的**：全黑的一帧自然一个都不算，爆闪的一帧自然多算。
 *      固定 top-B 在两端都错——安静时浪费、爆发时不够。
 *   3. 之所以敢这么做，是因为 APM 校准层让概率是**绝对校准**的：`P = 0.2` 真的
 *      意味着 20% 会发放（报告 A 节的校准表）。没校准的概率做阈值门控没有意义。
 *
 * `p[i] < 0`（前置过滤判定"不可能发放"）一律跳过，不看 τ。
 */
export function selectThresh(p: Float32Array, n: number, tau: number, out: Uint8Array): number {
  let count = 0
  for (let i = 0; i < n; i++) {
    const on = p[i] > tau
    out[i] = on ? 1 : 0
    if (on) count++
  }
  return count
}

/**
 * 随机选 B 个（对照：证明"选得准"比"算得少"更关键）。
 * `eligible` 为 null 时全部可选；否则只在有输入的神经元里抽。
 */
export function selectRandom(n: number, B: number, out: Uint8Array, rng: Rng, eligible?: Uint8Array): number {
  out.fill(0, 0, n)
  let m = n
  if (eligible) {
    m = 0
    for (let i = 0; i < n; i++) if (eligible[i]) m++
  }
  if (B >= m) {
    for (let i = 0; i < n; i++) out[i] = eligible && !eligible[i] ? 0 : 1
    return m
  }
  // 无放回地正好抽 B 个（选择抽样），保证与预测器门控的候选数严格可比
  let count = 0, remaining = m
  for (let i = 0; i < n; i++) {
    if (eligible && !eligible[i]) continue
    if (rng() * remaining < B - count) { out[i] = 1; count++ }
    remaining--
  }
  return count
}

/**
 * 证书上界：盒内已有 n 个（按通道统计的）脉冲时，突触电流**最多**能有多大。
 * 这是"最坏情况下所有脉冲都落在最强的那几个突触上"，因此是真实电流的严格上界。
 */
function boundOf(L: Layer, oc: number, counts: Int32Array, cbase: number, nTot: number, mode: BoundMode): number {
  const { inCh, k } = L.spec
  const kk = k * k
  if (mode === 'pooled') {
    const cap = inCh * kk
    return L.topPooled[oc * (cap + 1) + (nTot > cap ? cap : nTot)]
  }
  let s = 0
  for (let ic = 0; ic < inCh; ic++) {
    const n = counts[cbase + ic]
    const base = (oc * inCh + ic) * (kk + 1)
    s += L.topPerCh[base + (n > kk ? kk : n)]
  }
  return s
}

/**
 * 直方图桶数。128 桶对 [0,1) 的概率足够，而且**桶数本身也是成本**：top-B 每步要
 * 清零 + 反向搜索这 128 个桶，所以它不是越大越好。
 */
export const TOPB_BINS = 128

export function stepLayer(
  L: Layer, S: Uint8Array, counts: Int32Array, opts: StepOptions, out: Uint8Array, st: StepStats,
): void {
  const { inCh, outCh, k, stride, pad, decay, thresh, inH, inW } = L.spec
  const kk = k * k
  const v = L.v, W = L.W
  const N = L.N
  const mode = opts.mode
  const group = opts.group ?? 1
  const nG = N / group
  const take = opts.take! // 长度 ≥ N/group：forced ∪ select，最终判决
  const sel = opts.select! // 长度 ≥ N/group：预测器/随机选出来的那一半
  let computed = 0, spikes = 0, macs = 0, runs = 0, tiles = 0, gateOps = 0

  // ===== ① "必须算"的判定：不看概率，只靠定理 =====
  // `forced` 对**所有**模式都成立：泄漏后的电位（＋电流上界）已经过阈，说明这个
  // 神经元哪怕一个输入都没有也会发放——跳过它漏掉的是定理内的脉冲，不是近似。
  // 按**块**取并：块里只要有一个神经元 forced，整块就得算。这正是结构化稀疏的
  // 代价（会比逐神经元多选一点），换来的是块内连续、块间不碎片。
  const useBound = mode === 'cert' || mode === 'cm-bound'
  for (let g = 0; g < nG; g++) take[g] = 0
  for (let oy = 0; oy < L.outH; oy++) {
    for (let ox = 0; ox < L.outW; ox++) {
      const p = oy * L.outW + ox
      const cbase = p * inCh
      let nTot = 0
      for (let ic = 0; ic < inCh; ic++) nTot += counts[cbase + ic]
      for (let oc = 0; oc < outCh; oc++) {
        const i = p * outCh + oc
        const g = (i / group) | 0
        if (take[g] && mode !== 'cert') continue // 已经要算了（cert 不 care，它只看上界）
        if (mode === 'dense') { take[g] = 1; continue }
        const bound = useBound ? boundOf(L, oc, counts, cbase, nTot, opts.bound) : 0
        // event：盒内有输入就必须算（跳过会让携带状态不再等于真值）
        if ((mode === 'event' && nTot > 0) || decay * v[i] + bound >= thresh + L.thresholdOffsets[oc]) take[g] = 1
      }
    }
  }

  // ===== ② 预测器 / 随机选取（按单元，不是按神经元）=====
  if (mode === 'cm-decay' || mode === 'cm-bound' || mode === 'cm-thresh' || mode === 'random') {
    const B = opts.budget ?? nG
    if (mode === 'random') {
      selectRandom(nG, B, sel, opts.rng ?? splitmix32(1), opts.eligible)
      gateOps += nG // 选择抽样：单遍
    } else if (mode === 'cm-thresh') {
      if (opts.probs) selectThresh(opts.probs, nG, opts.tau ?? 0.05, sel)
      else sel.fill(1, 0, nG)
      gateOps += nG // 阈值扫描：单遍，没有直方图、没有排序
    } else {
      if (opts.probs) selectTopB(opts.probs, nG, B, sel, opts.hist!)
      else sel.fill(1, 0, nG)
      // top-B 的真实成本：建直方图一遍 + 反向搜阈值 + 再扫一遍 + 直方图清零
      gateOps += 2 * nG + 2 * TOPB_BINS
    }
    for (let g = 0; g < nG; g++) if (sel[g]) take[g] = 1
  }
  for (let g = 0; g < nG; g++) if (take[g]) tiles++

  // ===== ③ 计算：以块为单位，块内是密集的小矩阵 =====
  let prev = 0
  for (let oy = 0; oy < L.outH; oy++) {
    const y0 = oy * stride - pad
    const ky0 = Math.max(0, -y0), ky1 = Math.min(k, inH - y0)
    for (let ox = 0; ox < L.outW; ox++) {
      const x0 = ox * stride - pad
      const kx0 = Math.max(0, -x0), kx1 = Math.min(k, inW - x0)
      const p = oy * L.outW + ox
      const cbase = p * inCh
      let nTot = 0
      for (let ic = 0; ic < inCh; ic++) nTot += counts[cbase + ic]
      for (let oc = 0; oc < outCh; oc++) {
        const i = p * outCh + oc
        if (!take[(i / group) | 0]) {
          // 跳过：不做 fanin 次乘加。状态怎么推进由模式决定——这是各模式之间
          // 唯一影响"未来"的差别，也是它们精度不同的根源。
          //   event / cm-decay / cm-thresh：I 按 0 处理（泄漏照做）
          //   cert / cm-bound             ：I 取严格上界（保守，偏多不偏少）
          const useBound = mode === 'cert' || mode === 'cm-bound'
          const bound = useBound ? boundOf(L, oc, counts, cbase, nTot, opts.bound) : 0
          v[i] = decay * v[i] + bound
          out[i] = 0
          prev = 0
          continue
        }
        if (!prev) runs++
        prev = 1
        computed++
        let acc = 0
        for (let ic = 0; ic < inCh; ic++) {
          if (mode !== 'dense' && counts[cbase + ic] === 0) continue // 该通道盒内无脉冲：整块贡献为 0
          const wbase = (oc * inCh + ic) * kk
          const sbase = ic * inH * inW
          for (let ky = ky0; ky < ky1; ky++) {
            const rowW = wbase + ky * k
            const rowS = sbase + (y0 + ky) * inW + x0
            for (let kx = kx0; kx < kx1; kx++) acc += W[rowW + kx] * S[rowS + kx]
          }
          macs += (ky1 - ky0) * (kx1 - kx0)
        }
        const nv = decay * v[i] + acc
        if (nv >= thresh + L.thresholdOffsets[oc]) {
          out[i] = 1
          v[i] = L.spec.hardReset ? 0 : nv // 硬重置 / 不重置（PSN 家族）
          spikes++
        } else {
          out[i] = 0
          v[i] = nv
        }
      }
    }
  }
  st.computed = computed
  st.skipped = N - computed
  st.spikes = spikes
  st.macs = macs
  st.gateOps = gateOps
  st.runs = runs
  st.runLen = runs > 0 ? computed / runs : 0
  st.tiles = tiles
}

export interface PsnOut { v: Float32Array; s: Uint8Array; flops: number; depth: number }

/** 逐步递归（有重置 = LIF；无重置 = PSN）。 */
export function lifSequential(I: Float32Array, T: number, N: number, decay: number, thresh: number, reset = true): PsnOut {
  const v = new Float32Array(T * N)
  const s = new Uint8Array(T * N)
  const cur = new Float32Array(N)
  for (let t = 0; t < T; t++) {
    const o = t * N
    for (let i = 0; i < N; i++) {
      let nv = decay * cur[i] + I[o + i]
      let sp = 0
      if (nv >= thresh) { sp = 1; nv = reset ? 0 : nv }
      cur[i] = nv
      v[o + i] = nv
      s[o + i] = sp
    }
  }
  return { v, s, flops: T * N * 2, depth: T }
}

/** 一次 Toeplitz 矩阵乘：V = L·I，深度 1，代价 T²N/2。 */
export function psnToeplitz(I: Float32Array, T: number, N: number, decay: number, thresh: number): PsnOut {
  const v = new Float32Array(T * N)
  const s = new Uint8Array(T * N)
  // 预计算衰减幂
  const pow = new Float64Array(T)
  for (let i = 0; i < T; i++) pow[i] = i === 0 ? 1 : pow[i - 1] * decay
  for (let i = 0; i < N; i++) {
    for (let t = 0; t < T; t++) {
      let acc = 0
      for (let tau = 0; tau <= t; tau++) acc += pow[t - tau] * I[tau * N + i]
      v[t * N + i] = acc
      s[t * N + i] = acc >= thresh ? 1 : 0
    }
  }
  return { v, s, flops: (T * (T + 1) / 2) * N * 2, depth: 1 }
}

/**
 * 结合扫描（Hillis–Steele）：同样的线性递归，但保持 O(TN) 的乘加量，
 * 深度降到 O(log T)。v(t) = a·v(t-1) + I(t) 的转移是仿射的，仿射映射对
 * 复合封闭，所以可以两两合并。
 */
export function psnScan(I: Float32Array, T: number, N: number, decay: number, thresh: number): PsnOut {
  const v = new Float32Array(T * N)
  const s = new Uint8Array(T * N)
  const A = new Float64Array(T)
  const B = new Float64Array(T)
  let flops = 0
  for (let i = 0; i < N; i++) {
    for (let t = 0; t < T; t++) { A[t] = decay; B[t] = I[t * N + i] }
    for (let off = 1; off < T; off <<= 1) {
      for (let t = T - 1; t >= off; t--) {
        // 先发生的是 [t-off]，后发生的是 [t]：复合 (a1,b1)∘(a2,b2) = (a1a2, a2b1 + b2)
        const a = A[t - off] * A[t]
        const b = A[t] * B[t - off] + B[t]
        A[t] = a
        B[t] = b
        flops += 3
      }
    }
    for (let t = 0; t < T; t++) {
      v[t * N + i] = B[t]
      s[t * N + i] = B[t] >= thresh ? 1 : 0
    }
  }
  return { v, s, flops, depth: Math.ceil(Math.log2(Math.max(2, T))) }
}

/** 把 T 帧输入一次性卷成 T×N 的电流矩阵（"时间作为一个维度"的那一半）。 */
export function convTimeStacked(S: Uint8Array, T: number, L: Layer): Float32Array {
  const { inCh, outCh, k, stride, pad, inH, inW } = L.spec
  const kk = k * k
  const I = new Float32Array(T * L.N)
  for (let t = 0; t < T; t++) {
    const Sin = S.subarray(t * inCh * inH * inW, (t + 1) * inCh * inH * inW)
    for (let oy = 0; oy < L.outH; oy++) {
      const y0 = oy * stride - pad
      for (let ox = 0; ox < L.outW; ox++) {
        const x0 = ox * stride - pad
        const pbase = (oy * L.outW + ox) * outCh
        for (let oc = 0; oc < outCh; oc++) {
          let acc = 0
          for (let ic = 0; ic < inCh; ic++) {
            const wbase = (oc * inCh + ic) * kk
            const sbase = ic * inH * inW
            for (let ky = 0; ky < k; ky++) {
              const y = y0 + ky
              if (y < 0 || y >= inH) continue
              for (let kx = 0; kx < k; kx++) {
                const x = x0 + kx
                if (x < 0 || x >= inW) continue
                acc += L.W[wbase + ky * k + kx] * Sin[sbase + y * inW + x]
              }
            }
          }
          I[(t * L.outH * L.outW + (oy * L.outW + ox)) * outCh + oc] = acc
        }
        void pbase
      }
    }
  }
  return I
}
