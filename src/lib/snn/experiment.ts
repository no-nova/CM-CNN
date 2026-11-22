// SPDX-License-Identifier: MPL-2.0
//
// 实验：把"用小型 CNN 预测 SNN 下一刻要用的神经元，从而只算一个很小的'总矩阵'"
// 这件事，变成一组可复核的数字。
//
// 三块内容，对应对话里提出的三个主张：
//
//   A. 预测器本身够不够好（压缩视角）
//      把脉冲张量当成待压缩的比特流，用 UCZ 的 CM 预测器 + 区间编码器真的压一遍，
//      与"逐神经元自适应码""全局频率码""原样存"对比。码长的下界，就是**任何
//      无损门控都必须检查的神经元比例**——这是整套方法的信息论天花板。
//
//   B. 门控能不能省下真金白银（计算视角）
//      稠密 / 事件驱动 / 上界证书 / 预测器预算 / 随机预算 五种策略，比较：
//      乘加数、候选比例、漏脉冲率、额外脉冲率、任务准确率。
//
//   C. 时间到底能不能当成矩阵的一维（可能性 2）
//      把 T 步递归折成一次带因果掩码的矩阵乘，验证与逐步递归一致，并量出
//      "带重置的 LIF 折不了"到底差多少。

import { bitCost, PSCALE, RangeEncoder } from './cm'
import { DEFAULT_PRED, SpikePredictor, type PredConfig } from './predict'
import {
  boxCountScratch, boxCountsFromII, buildIntegral, buildLayer, convTimeStacked, lifSequential,
  psnScan, psnToeplitz, resetLayer, stepLayer, TOPB_BINS, type BoundMode, type GateMode, type Layer,
  type LayerSpec, type StepStats,
} from './snn'
import {
  makeBurstDataset, makeDigitsDataset, makeMovingBarDataset, makeRegimeShiftDataset,
  l2NormalizeRows, readoutAccuracy, trainReadout, type EventSample, type RegimeB,
} from './task'
import { splitmix32, type Rng } from './rand'

// ===== 配置 =====

/** 网络的"宽度"：决定 L2 的 fanin = ch1 × k2 × k2。 */
export interface NetShape {
  /** 输入通道数（1 = 单极性事件；2 = ON/OFF 双极性，眼跳编码需要）。 */
  in1: number
  /** L1 的输出通道数 = L2 的输入通道数。 */
  ch1: number
  /** L2 的卷积核边长。 */
  k2: number
  /** L2 的输出通道数（默认 8；宽通道实验里会放大，给随机储层更多特征冗余）。 */
  ch2: number
}

export interface ExpConfig {
  T: number
  H: number
  W: number
  trainPerClass: number
  testPerClass: number
  /** cm-* / random 模式的预算（候选神经元占全体的比例）。 */
  budgets: number[]
  seed: number
  bound: BoundMode
  pred: PredConfig
  /**
   * 网络宽度。默认 `{ch1: 4, k2: 5}` ⟹ L2 fanin = 100；
   * `{ch1: 16, k2: 5}` ⟹ 400，`{ch1: 16, k2: 6}` ⟹ 576（真实视觉网络的量级）。
   */
  shape: NetShape
}

export const DEFAULT_EXP: ExpConfig = {
  T: 20,
  H: 24,
  W: 24,
  trainPerClass: 32,
  testPerClass: 32,
  budgets: [0.5, 0.3, 0.2, 0.1, 0.05],
  seed: 0x5eed1234,
  bound: 'perCh',
  pred: DEFAULT_PRED,
  shape: { in1: 1, ch1: 4, k2: 5, ch2: 8 },
}

/**
 * 两层脉冲卷积网络。
 *
 * **权重尺度按 1/√fanin 归一**：随机权重下 `std(Σ w·x) = √fanin × scale`，所以
 * 换 fanin 时如果不跟着调 scale，输入电流的量级会跟着变，发放率、准确率全都
 * 变了——那样"换 fanin 之后门控变得更划算"和"网络工作点变了"就混在一起，A/B
 * 没法比。归一之后 fanin 是**唯一变量**。
 */
const FANIN1_REF = 25 // L1: 1×5×5
const FANIN2_REF = 100 // L2: 4×5×5（默认形状）

export function defaultLayers(
  H: number, W: number, hardReset = true, shape: NetShape = DEFAULT_EXP.shape,
): LayerSpec[] {
  const k1 = 5
  const { in1, ch1, k2, ch2 } = shape
  const fanin1 = in1 * k1 * k1
  const fanin2 = ch1 * k2 * k2
  return [
    {
      inCh: in1, outCh: ch1, inH: H, inW: W, k: k1, stride: 2, pad: 2, decay: 0.85, thresh: 0.6,
      scale: 0.30 * Math.sqrt(FANIN1_REF / fanin1), seed: 11, hardReset,
    },
    {
      inCh: ch1, outCh: ch2, inH: H >> 1, inW: W >> 1, k: k2, stride: 1, pad: k2 >> 1,
      decay: 0.85, thresh: 0.9, scale: 0.12 * Math.sqrt(FANIN2_REF / fanin2), seed: 22, hardReset,
    },
  ]
}

/** 门控粒度：`'fiber'` 展开成"该层的全部输出通道打包成一块"。 */
function groupOf(opts: RunOptions, spec: LayerSpec): number {
  return opts.group === 'fiber' ? spec.outCh : (opts.group ?? 1)
}

// ===== 一次前向的账本 =====

export interface LayerBudget {
  macs: number
  gateOps: number
  predOps: number
  /** 层间布局转换的元素复制次数。 */
  bridgeOps: number
  neuronSteps: number
  computed: number
  spikes: number
  runs: number
  tiles: number
  /** 本层的门控粒度（每块几个神经元）。 */
  group: number
}

export interface Budget {
  /** 真实乘加次数——"总矩阵"里被乘到的那部分。 */
  macs: number
  /** 门控本身的开销：积分图 + 盒查询 + 上界查表 + 选取（含 top-B 排序）。 */
  gateOps: number
  /** 预测器开销。 */
  predOps: number
  /** 层间布局转换的元素复制次数。 */
  bridgeOps: number
  neuronSteps: number
  computed: number
  spikes: number
  /** 连续段个数（访存碎片化的代理量：越少越 coalesce）。 */
  runs: number
  /** 被选中的小块数。 */
  tiles: number
}

export interface RunOptions {
  mode: GateMode
  budgetFrac: number
  usePredictor: boolean
  trainPredictor: boolean
  record: boolean
  rng?: Rng
  /**
   * 结构化稀疏的粒度：每 `group` 个连续神经元打包成一个密集小块一起算。
   *
   * - `1`（默认）= 逐神经元（非结构化，GPU 上访存会碎片化）；
   * - 数字 n = 每 n 个连续通道一块；
   * - `'fiber'` = **逐层取该层的 outCh**，也就是"同一空间位置的全部通道"打包成
   *   一块——它们的感受野盒完全重合，于是共享一次 CM 预测，预测器开销降到
   *   1/outCh，而且块内访存连续、块与块之间不碎片。
   */
  group?: number | 'fiber'
  /** `cm-thresh` 的概率阈值 τ。 */
  tau?: number
  /** 读出特征的空域分块边长（1 = 只按通道归总，即主干实验的老行为）。 */
  pool?: number
  /**
   * 逐时间步的显微镜钩子：`onStep(层号, 步号, 预测概率, 实际脉冲, 该层神经元数)`。
   * 只有 `usePredictor` 为真时才调用；只读，不进账本。`probs[i] < 0` 表示被前置
   * 过滤器排掉的单元（这些单元没有被预测，统计时要跳过）。
   */
  onStep?: (li: number, t: number, probs: Float64Array | Float32Array, spk: Uint8Array, N: number) => void
  /**
   * 教师训练阶段用：同时把**纤维粒度**的预测器也训起来。
   *
   * 纤维预测器的标签是"这一整条纤维里有没有任何一个通道发放"，与逐神经元
   * 预测器的标签不同，所以它必须自己学一遍——用逐神经元那个的参数去初始化
   * 是错的（连表大小都不一样）。
   */
  trainFiber?: boolean
  /** 显式将层输出从位置交错转换为下一层所需的通道优先；旧模型默认 false。 */
  channelMajorBridge?: boolean
  /** 只读当前层的前、后脉冲，用于样本级局部可塑性统计；不得修改运行中状态。 */
  observeSpikes?: (li: number, input: Uint8Array, output: Uint8Array) => void
}

/** 网络 + 每层一个预测器 + 复用缓冲。 */
export class Net {
  readonly layers: Layer[]
  readonly preds: SpikePredictor[]
  /** 纤维粒度的预测器（`group = outCh`）：同一空间位置共享一次预测。 */
  readonly predsFiber: SpikePredictor[]
  /**
   * 结构化（成块）粒度的预测器，按粒度惰性创建：`'fiber'` = 每层的 outCh，
   * 数字 n = 每 n 个连续神经元一块。
   *
   * **为什么按粒度索引**：预测器的 `probs` 长度是 `N / group`，门控读的也是
   * `take[i / group]`。两者必须同一个 group，否则门控会读到数组外面去——
   * 之前把 `predsFiber` 硬编码成 outCh，一旦换成 group=8 就静默算出垃圾
   * （候选比例与预算脱钩）。现在粒度是键，错配在构造时就暴露。
   */
  private readonly predsByGroup = new Map<string, SpikePredictor[]>()
  private readonly scratch: Int32Array[]
  private readonly cnt1: Int32Array[]
  private readonly cnt2: Int32Array[]
  private readonly sel: Uint8Array[]
  private readonly take: Uint8Array[]
  private readonly elig: Uint8Array[]
  private readonly histBuf: Int32Array[]
  private readonly st: StepStats[]

  private readonly seedBase: number
  private readonly predCfg: PredConfig

  constructor(specs: LayerSpec[], pred: PredConfig = DEFAULT_PRED, seed = 777) {
    this.layers = specs.map(buildLayer)
    this.seedBase = seed
    this.predCfg = pred
    this.preds = specs.map((s, i) => new SpikePredictor(this.layers[i].N, s.outCh, s.inCh, (seed + i * 7919) >>> 0, pred))
    this.scratch = specs.map(s => boxCountScratch(s.inCh, s.inH, s.inW))
    const mkc = (s: LayerSpec, l: Layer) => new Int32Array(l.outH * l.outW * s.inCh)
    this.cnt1 = specs.map((s, i) => mkc(s, this.layers[i]))
    this.cnt2 = specs.map((s, i) => mkc(s, this.layers[i]))
    this.sel = this.layers.map(l => new Uint8Array(l.N))
    this.take = this.layers.map(l => new Uint8Array(l.N))
    this.elig = this.layers.map(l => new Uint8Array(l.N))
    this.histBuf = this.layers.map(() => new Int32Array(TOPB_BINS))
    this.st = this.layers.map(() => ({ computed: 0, skipped: 0, spikes: 0, macs: 0, gateOps: 0, runs: 0, runLen: 0, tiles: 0 }))
    // 纤维粒度是默认就支持的那一档，先建好，免得第一次用到时才惰性创建
    this.predsFiber = this.predsFor('fiber')
  }

  /**
   * 取（必要时惰性创建）某个门控粒度对应的预测器组。
   * `group = 1` 就是逐神经元的 `this.preds`。
   */
  predsFor(group: number | 'fiber'): SpikePredictor[] {
    if (!group || group === 1) return this.preds
    const key = String(group)
    let p = this.predsByGroup.get(key)
    if (!p) {
      p = this.layers.map((l, i) => {
        const g = group === 'fiber' ? l.spec.outCh : group
        return new SpikePredictor(l.N, l.spec.outCh, l.spec.inCh,
          (this.seedBase + i * 7919 + g * 104729) >>> 0, this.predCfg, g)
      })
      this.predsByGroup.set(key, p)
    }
    return p
  }

  /** 教师阶段要把哪些粒度的预测器也训一遍（不训练就会选出垃圾）。 */
  ensureGroups(groups: (number | 'fiber')[]): void {
    for (const g of groups) this.predsFor(g)
  }

  /** 已注册的、需要额外训练的成块预测器（不含逐神经元那一套）。 */
  private groupedPredictors(): SpikePredictor[][] {
    return [...this.predsByGroup.values()]
  }

  /**
   * 读出层的特征维数。
   *
   * `pool = 1`（默认）是"每层每个通道的总发放数"，维数 = Σ outCh；
   * `pool = k > 1` 是把每层的发放图再按 k×k 网格分块求和，维数 = Σ outCh·k²。
   * 分块只影响读出用的特征，**不碰前向、不碰门控、不碰账本**——它是线性读出
   * 层的输入分辨率，不是网络的一部分。默认 1 保证主干实验的数字不变。
   */
  featureDim(pool = 1): number {
    return this.layers.reduce((a, l) => a + l.spec.outCh * pool * pool, 0)
  }

  /**
   * 跑一个样本。
   *
   * 每一步对每个层做四件事：① 多尺度盒滤波（小型 CNN 的卷积部分）；
   * ② CM 预测器给出每个神经元的发放概率；③ 按门控策略选候选集并推进；
   * ④ 用真实发放更新预测器（在线学习，监督信号就是网络自己跑出来的脉冲）。
   */
  run(frames: Uint8Array, T: number, opts: RunOptions, coders?: Coder[]): {
    spikes: Uint8Array[]; features: Float64Array; budget: Budget; perLayer: LayerBudget[]
    /** 逐时间步的候选比例（突发流那一节要画"预算该不该跟着时间变"）。 */
    perStepCand: Float64Array
    ms: number
  } {
    const L = this.layers
    for (let i = 0; i < L.length; i++) resetLayer(L[i])
    const b: Budget = { macs: 0, gateOps: 0, predOps: 0, bridgeOps: 0, neuronSteps: 0, computed: 0, spikes: 0, runs: 0, tiles: 0 }
    const perLayer: LayerBudget[] = L.map(l => ({
      macs: 0, gateOps: 0, predOps: 0, bridgeOps: 0, neuronSteps: 0, computed: 0, spikes: 0, runs: 0, tiles: 0,
      group: groupOf(opts, l.spec),
    }))
    const rec: Uint8Array[] = []
    if (opts.record) for (const l of L) rec.push(new Uint8Array(T * l.N))
    const pool = opts.pool ?? 1
    const dims = this.featureDim(pool)
    const acc = new Float64Array(dims)
    const frameSize = L[0].spec.inCh * L[0].spec.inH * L[0].spec.inW
    if (L.some(l => l.N % groupOf(opts, l.spec) !== 0)) throw Error('group 不能整除某层的神经元数')

    const bridges = opts.channelMajorBridge ? L.slice(0, -1).map((l, i) => {
      const next = L[i + 1].spec
      if (next.inCh !== l.spec.outCh || next.inH !== l.outH || next.inW !== l.outW)
        throw Error('层间维度不匹配')
      return new Uint8Array(l.N)
    }) : []
    const perStepCand = new Float64Array(T)
    let stepComputed = 0, stepNeurons = 0
    const t0 = performance.now()
    for (let t = 0; t < T; t++) {
      let input = frames.subarray(t * frameSize, (t + 1) * frameSize)
      stepComputed = 0; stepNeurons = 0
      for (let li = 0; li < L.length; li++) {
        const l = L[li]
        const s = l.spec
        const usePred = opts.usePredictor
        const cnt1 = this.cnt1[li]
        const cnt2 = this.cnt2[li]
        const pl = perLayer[li] // 分层账本：fanin 越小，预测器的固定开销越难摊薄
        const gSpec = opts.group ?? 1 // 用户给的粒度（可能是 'fiber'）
        const group = groupOf(opts, s) // 本层的门控粒度（'fiber' 展开成 outCh）
        const needCounts = s.k > 0 && (opts.mode !== 'dense' || usePred)
        let gate = 0
        if (needCounts) {
          // ① 小型 CNN 的卷积部分：积分图建一次，多尺度盒查询各 4 次取数
          gate += buildIntegral(input, s.inCh, s.inH, s.inW, this.scratch[li])
          gate += boxCountsFromII(this.scratch[li], s.inCh, s.inH, s.inW, s.k, s.stride, s.pad, l.outH, l.outW, cnt1)
          if (usePred) {
            const K = s.k * 2 // 2× 尺度：同一中心，padding 放大 (K-k)/2
            gate += boxCountsFromII(this.scratch[li], s.inCh, s.inH, s.inW, K, s.stride, s.pad + ((K - s.k) >> 1), l.outH, l.outW, cnt2)
          }
          b.gateOps += gate
          pl.gateOps += gate
        }
        // ② CM 预测器（前置过滤器会跳过"盒内无脉冲"的单元，成本按实际算过的算）
        //    成块模式用 predsFor(gSpec)：整块共享一次预测，于是预测器开销
        //    除以 group——这正是"同一个位置的感受野盒完全重合"换来的。
        let pred = 0
        if (usePred) {
          const P = this.predsFor(gSpec)[li]
          const evaluated = P.predict(cnt1, cnt2, l.v, s.thresh, t)
          pred = evaluated * P.opsPerNeuron()
          b.predOps += pred
          pl.predOps += pred
        }
        // ③ 门控 + 推进
        const st = this.st[li]
        const nUnits = l.N / group
        if (opts.mode === 'random' && usePred) {
          // 随机对照也要用同一套前置过滤，否则比的是"过滤"而不是"排序"。
          // 用独立缓冲：sel 马上要被 selectRandom 覆写，不能兼任掩码。
          const pr = this.predsFor(gSpec)[li].probs
          for (let i = 0; i < nUnits; i++) this.elig[li][i] = pr[i] >= 0 ? 1 : 0
        }
        stepLayer(l, input, cnt1, {
          mode: opts.mode,
          bound: 'perCh',
          probs: usePred ? this.predsFor(gSpec)[li].probs : undefined,
          budget: opts.mode === 'cm-thresh' ? nUnits : Math.max(1, Math.round(opts.budgetFrac * nUnits)),
          select: this.sel[li],
          take: this.take[li],
          hist: this.histBuf[li],
          eligible: opts.mode === 'random' && usePred ? this.elig[li] : undefined,
          group,
          tau: opts.tau,
          rng: opts.rng,
        }, l.spk, st)
        b.macs += st.macs
        b.gateOps += st.gateOps
        b.runs += st.runs
        b.tiles += st.tiles
        pl.macs += st.macs
        pl.gateOps += st.gateOps
        pl.computed += st.computed
        pl.spikes += st.spikes
        pl.runs += st.runs
        pl.tiles += st.tiles
        b.computed += st.computed
        b.spikes += st.spikes
        b.neuronSteps += l.N
        pl.neuronSteps += l.N
        stepComputed += st.computed
        stepNeurons += l.N
        // 显微镜钩子：把本步"预测了什么 / 实际发了什么"交给外部（§实验三用它
        // 逐时间步读编码成本）。它只读不写，不进账本。
        if (opts.onStep && usePred) {
          opts.onStep(li, t, this.predsFor(gSpec)[li].probs, l.spk, l.N)
        }
        opts.observeSpikes?.(li, input, l.spk)
        // ④ 在线学习 / 在线编码
        if (usePred && (opts.trainPredictor || coders)) {
          const P = this.predsFor(gSpec)[li]
          P.learnWithCounts(l.spk, cnt1, cnt2, l.v, s.thresh, t, opts.trainPredictor)
        }
        // 教师阶段把所有**已注册的**成块预测器也训一遍（不算推理成本：它们是
        // 离线学出来的）。粒度是键：注册过什么粒度，这里就训什么粒度。
        if (usePred && opts.trainFiber && group <= 1) {
          for (const arr of this.groupedPredictors()) {
            const F = arr[li]
            F.predict(cnt1, cnt2, l.v, s.thresh, t)
            F.learnWithCounts(l.spk, cnt1, cnt2, l.v, s.thresh, t, true)
          }
        }
        if (coders && coders[li]) coders[li].codeStep(l.spk, this.preds[li])
        if (opts.record) {
          rec[li].set(l.spk, t * l.N)
          const off = this.layers.slice(0, li).reduce((a, x) => a + x.spec.outCh * pool * pool, 0)
          for (let i = 0; i < l.N; i++) {
            if (!l.spk[i]) continue
            const c = i % s.outCh
            if (pool === 1) acc[off + c]++
            else {
              const p = (i / s.outCh) | 0
              let gy = (((p / l.outW) | 0) * pool / l.outH) | 0
              let gx = (((p % l.outW) * pool / l.outW) | 0)
              if (gy >= pool) gy = pool - 1
              if (gx >= pool) gx = pool - 1
              acc[off + c * pool * pool + gy * pool + gx]++
            }
          }
        }
        if (opts.channelMajorBridge && li + 1 < L.length) {
          const bridge = bridges[li], pos = l.outH * l.outW
          for (let p = 0; p < pos; p++) for (let c = 0; c < s.outCh; c++)
            bridge[c * pos + p] = l.spk[p * s.outCh + c]
          b.bridgeOps += l.N
          pl.bridgeOps += l.N
          input = bridge
        } else input = l.spk
      }
      perStepCand[t] = stepComputed / Math.max(1, stepNeurons)
    }
    const ms = performance.now() - t0
    const f = new Float64Array(dims)
    if (opts.record) {
      let off = 0
      for (let li = 0; li < L.length; li++) {
        const l = L[li]
        const pos = l.outH * l.outW
        const posPerBin = (T * pos) / (pool * pool)
        for (let c = 0; c < l.spec.outCh * pool * pool; c++) f[off + c] = acc[off + c] / Math.max(1, posPerBin)
        off += l.spec.outCh * pool * pool
      }
    }
    return { spikes: rec, features: f, budget: b, perLayer, perStepCand, ms }
  }
}

// ===== 压缩视角 =====

export interface CodingResult {
  layer: number
  neuronSteps: number
  /** 原样存：每神经元步 1 比特。 */
  rawBits: number
  /** 全局单一自适应计数。 */
  globalBits: number
  /** 逐神经元自适应计数（知道身份，不知道上下文）。 */
  perNeuronBits: number
  /** CM 预测器的理论码长 Σ -log2(p)。 */
  cmIdealBits: number
  /** CM 预测器 + 真区间编码器压出来的比特数（含编码器开销）。 */
  cmCodedBits: number
  /** 编码开销：真编码 / 理论码长 − 1。 */
  coderOverhead: number
  /** 平均二元熵 H(p)：无损门控必须检查的神经元比例的信息论下界。 */
  entropyFloor: number
  bitsPerNeuronStep: number
  spikes: number
  bitsPerSpike: number
  /** 稳态（后半段）每神经元步码长：在线学习收敛后的水平。 */
  steadyBitsPerNeuronStep: number
  calibration: { lo: number; hi: number; pred: number; obs: number; n: number }[]
}

/**
 * 在线编码器：每步把一层的脉冲向量喂进区间编码器，同时统计两个基线的码长。
 *
 * 与压缩器同构——预测器在第 0 步只能给 p=0.5，一边编一边学，所以前半段的
 * 码长天然偏贵；报告里同时给"全程平均"和"稳态（后半段）平均"。
 */
export class Coder {
  private readonly rc = new RangeEncoder()
  private globalN0 = 1
  private globalN1 = 1
  private readonly c0: Uint16Array
  private readonly c1: Uint16Array
  private globalBits = 0
  private perNeuronBits = 0
  private cmIdeal = 0
  private ent = 0
  private spikes = 0
  private steps = 0
  private stepsLate = 0
  private bitsLate = 0
  private readonly nbins = 10
  private readonly binPred = new Float64Array(this.nbins)
  private readonly binObs = new Float64Array(this.nbins)
  private readonly binN = new Float64Array(this.nbins)
  /** 后半段起点（步号）。 */
  readonly half: number

  /** 到目前为止的累计码长（比特，理论值；实编输出与它差不到 0.2%）。 */
  get bits(): number {
    return this.cmIdeal
  }

  constructor(readonly N: number, T: number) {
    this.c0 = new Uint16Array(N).fill(1)
    this.c1 = new Uint16Array(N).fill(1)
    this.half = T >> 1
  }

  codeStep(spk: Uint8Array, pred: SpikePredictor): void {
    const p = pred.probs
    const N = this.N
    let stepBits = 0
    for (let i = 0; i < N; i++) {
      const y = spk[i] ? 1 : 0
      let p12 = Math.round(p[i] * PSCALE)
      if (p12 < 1) p12 = 1
      else if (p12 > PSCALE - 1) p12 = PSCALE - 1
      this.rc.encode(y, p12)
      const ideal = bitCost(p12, y)
      this.cmIdeal += ideal
      stepBits += ideal
      const pp = p12 / PSCALE
      this.ent += -(pp * Math.log2(Math.max(1e-9, pp)) + (1 - pp) * Math.log2(Math.max(1e-9, 1 - pp)))
      {
        const g = this.globalN1 / (this.globalN0 + this.globalN1)
        this.globalBits += bitCost(Math.round(g * PSCALE), y)
        if (y) this.globalN1++
        else this.globalN0++
        if (this.globalN0 + this.globalN1 > 4000) { this.globalN0 = (this.globalN0 >> 1) + 1; this.globalN1 = (this.globalN1 >> 1) + 1 }
      }
      {
        const a = this.c0[i], bb = this.c1[i]
        this.perNeuronBits += bitCost(Math.round((bb / (a + bb)) * PSCALE), y)
        if (y) { if (this.c1[i] < 60000) this.c1[i]++ } else if (this.c0[i] < 60000) this.c0[i]++
        if (a + bb > 4000) { this.c0[i] = (a >> 1) + 1; this.c1[i] = (bb >> 1) + 1 }
      }
      this.spikes += y
      // 前置过滤器判定"本步不可能发放"的神经元（probs = -1）不进校准统计：
      // 它们是定理送的，不是预测器猜的，混进来会把 calibration 表污染成负数。
      if (p[i] < 0) continue
      let bin = (p[i] * this.nbins) | 0
      if (bin < 0) bin = 0
      else if (bin >= this.nbins) bin = this.nbins - 1
      this.binPred[bin] += p[i]
      this.binObs[bin] += y
      this.binN[bin]++
    }
    this.steps += N
    if (this.steps / N >= this.half) { this.stepsLate += N; this.bitsLate += stepBits }
  }

  finish(layer: number): CodingResult {
    const bytes = this.rc.finish()
    const cal = []
    for (let b = 0; b < this.nbins; b++) {
      cal.push({
        lo: b / this.nbins, hi: (b + 1) / this.nbins,
        pred: this.binN[b] ? this.binPred[b] / this.binN[b] : 0,
        obs: this.binN[b] ? this.binObs[b] / this.binN[b] : 0,
        n: this.binN[b],
      })
    }
    return {
      layer,
      neuronSteps: this.steps,
      rawBits: this.steps,
      globalBits: this.globalBits,
      perNeuronBits: this.perNeuronBits,
      cmIdealBits: this.cmIdeal,
      cmCodedBits: bytes.length * 8,
      coderOverhead: this.cmIdeal ? (bytes.length * 8) / this.cmIdeal - 1 : 0,
      entropyFloor: this.ent / Math.max(1, this.steps),
      bitsPerNeuronStep: this.cmIdeal / Math.max(1, this.steps),
      spikes: this.spikes,
      bitsPerSpike: this.spikes ? this.cmIdeal / this.spikes : 0,
      steadyBitsPerNeuronStep: this.bitsLate / Math.max(1, this.stepsLate),
      calibration: cal,
    }
  }
}

// ===== 计算视角 =====

export interface RunResult {
  label: string
  mode: GateMode
  budgetFrac: number
  /** `cm-thresh` 的概率阈值 τ（其余模式为 undefined）。 */
  tau?: number
  /** 读出特征的空域分块边长（1 = 只按通道归总，即主干实验的老行为）。 */
  pool?: number
  /**
   * 逐时间步的显微镜钩子：`onStep(层号, 步号, 预测概率, 实际脉冲, 该层神经元数)`。
   * 只有 `usePredictor` 为真时才调用；只读，不进账本。`probs[i] < 0` 表示被前置
   * 过滤器排掉的单元（这些单元没有被预测，统计时要跳过）。
   */
  onStep?: (li: number, t: number, probs: Float64Array | Float32Array, spk: Uint8Array, N: number) => void
  /** 门控粒度：1 = 逐神经元，'fiber' = 整条纤维一块。 */
  group?: number | 'fiber'
  /** 平均连续段长度（访存碎片化的代理量：越长越 coalesce）。 */
  runLen: number
  /** 每步的连续段个数（按神经元步长归一）。 */
  runsPerStep: number
  /** 每步被选中的小块数（按神经元步长归一）。 */
  tilesPerStep: number
  /** 分层账本（fanin 不同的层，门控的性价比完全不同）。 */
  perLayer: LayerBudget[]
  totalOps: number
  macs: number
  gateOps: number
  predOps: number
  /** 总算术量 / 稠密乘加量（越小越好）。 */
  opsRatio: number
  /** 乘加量 / 稠密乘加量——"总矩阵"缩到了多大。 */
  macRatio: number
  candidateFrac: number
  spikeRate: number
  missRate: number
  extraRate: number
  accuracy: number
  ms: number
}

export interface TimeFoldResult {
  layer: number
  neurons: number
  T: number
  /** 带重置的 LIF 与无重置 PSN 的脉冲串差异（个）：重置到底改变了多少。 */
  lifVsPsnHamming: number
  psnSeqVsToeplitzMaxErr: number
  psnSeqVsToeplitzHamming: number
  psnSeqVsScanMaxErr: number
  psnSeqVsScanHamming: number
  seqFlops: number
  toeplitzFlops: number
  scanFlops: number
  seqDepth: number
  toeplitzDepth: number
  scanDepth: number
  msSequential: number
  msToeplitz: number
  msScan: number
}

/**
 * 定理边界验证：把"跳过者取电流上界 ⟹ 携带状态 v̂ 是真值 v 的上界 ⟹
 * v̂ < θ 时'本步一定不发'是被证明的"这条推理，逐层、逐神经元模型地验一遍。
 *
 * **必须在同一条输入流下比较**：如果直接比较"整网稠密跑"和"整网证书跑"，
 * 第 1 层多出来的脉冲会改变第 2 层的输入，于是第 2 层的差异已经不是门控本身的
 * 误差，而是误差传播。所以这里先用稠密跑录下每一层的输入流，然后把同一条流
 * 分别喂给"全算的那一层"和"证书门控的那一层"，只比这一层的输出。
 */
export interface TheoremRow {
  layer: number
  fanin: number
  /** 稠密发了、门控没发的脉冲数（定理声称：无重置时应恒为 0）。 */
  miss: number
  extra: number
  trueSpikes: number
  macRatio: number
  candidateFrac: number
}

export interface TheoremResult {
  samples: number
  noReset: TheoremRow[]
  hardReset: TheoremRow[]
}

export function runTheoremCheck(cfg: ExpConfig, samples: EventSample[]): TheoremResult {
  const denseOpts: RunOptions = { mode: 'dense', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true }
  const certOpts: RunOptions = { mode: 'cert', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true }

  const run = (hardReset: boolean): TheoremRow[] => {
    const specs = defaultLayers(cfg.H, cfg.W, hardReset, cfg.shape)
    const net = new Net(specs, cfg.pred, cfg.seed)
    // 每一层一个"单层网络"，复用同一套权重与规格
    const singles = specs.map(sp => new Net([sp], cfg.pred, cfg.seed))
    const rows: TheoremRow[] = specs.map((sp, li) => ({
      layer: li, fanin: sp.inCh * sp.k * sp.k, miss: 0, extra: 0, trueSpikes: 0, macRatio: 0, candidateFrac: 0,
    }))
    let denseMacs = specs.map(() => 0)
    let certMacs = specs.map(() => 0)
    let comp = specs.map(() => 0)
    let steps = specs.map(() => 0)
    for (const s of samples) {
      const d = net.run(s.frames, cfg.T, denseOpts)
      for (let li = 0; li < specs.length; li++) {
        const inStream = li === 0 ? s.frames : d.spikes[li - 1]
        const a = singles[li].run(inStream, cfg.T, denseOpts)
        const b = singles[li].run(inStream, cfg.T, certOpts)
        denseMacs[li] += a.budget.macs
        certMacs[li] += b.budget.macs
        comp[li] += b.budget.computed
        steps[li] += b.budget.neuronSteps
        const x = a.spikes[0], y = b.spikes[0]
        for (let i = 0; i < x.length; i++) {
          if (x[i]) rows[li].trueSpikes++
          if (x[i] && !y[i]) rows[li].miss++
          else if (!x[i] && y[i]) rows[li].extra++
        }
      }
    }
    for (let li = 0; li < rows.length; li++) {
      rows[li].macRatio = certMacs[li] / Math.max(1, denseMacs[li])
      rows[li].candidateFrac = comp[li] / Math.max(1, steps[li])
    }
    return rows
  }
  return { samples: samples.length, noReset: run(false), hardReset: run(true) }
}

export interface ExpResult {
  config: ExpConfig
  /** 实验二：非平稳突发流（固定预算 vs 自适应阈值）。 */
  burst?: BurstResult
  /** 实验三：冷启动 + 分布突变的在线轨迹。 */
  trajectory?: TrajectoryResult
  /** 实验四：真实手写数字。 */
  digits?: DigitsResult[]
  layers: { N: number; fanin: number; outH: number; outW: number; outCh: number }[]
  dataset: { train: number; test: number; T: number; H: number; W: number; classes: number }
  coding: CodingResult[]
  runs: RunResult[]
  theorem: TheoremResult
  timeFold: TimeFoldResult
  dense: { macs: number; spikes: number; neuronSteps: number; spikeRate: number; accuracy: number; ms: number }
  ms: number
}

const DENSE_OPTS: RunOptions = { mode: 'dense', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true }

export function runExperiment(cfgIn: Partial<ExpConfig> = {}): ExpResult {
  const cfg: ExpConfig = { ...DEFAULT_EXP, ...cfgIn }
  const t0 = performance.now()
  const specs = defaultLayers(cfg.H, cfg.W, true, cfg.shape)
  const data = makeMovingBarDataset({
    perClass: cfg.trainPerClass, testPerClass: cfg.testPerClass, T: cfg.T, H: cfg.H, W: cfg.W, seed: cfg.seed,
  })
  const all: EventSample[] = [...data.train, ...data.test]
  const nTrain = data.train.length
  const nTest = data.test.length
  const dims = specs.reduce((a, s) => a + s.outCh, 0)

  // ---- Pass A：在线编码 + 顺带把预测器训练起来（教师 = 稠密网络自己的脉冲）----
  const net = new Net(specs, cfg.pred, cfg.seed)
  const coders = net.layers.map(l => new Coder(l.N, cfg.T))
  // 逐神经元与纤维两个预测器都在这里训（同一个稠密教师，标签不同而已）
  for (const s of data.train) net.run(s.frames, cfg.T, { mode: 'dense', budgetFrac: 1, usePredictor: true, trainPredictor: true, trainFiber: true, record: false }, coders)
  const coding = coders.map((c, i) => c.finish(i))

  // ---- 稠密参考：所有样本（训练 + 测试）的脉冲张量与读出特征 ----
  const refSpikes: Uint8Array[][] = []
  const refFeat = new Float64Array(all.length * dims)
  const denseBudget: Budget = { macs: 0, gateOps: 0, predOps: 0, bridgeOps: 0, neuronSteps: 0, computed: 0, spikes: 0, runs: 0, tiles: 0 }
  let denseMs = 0
  for (let i = 0; i < all.length; i++) {
    const r = net.run(all[i].frames, cfg.T, DENSE_OPTS)
    refSpikes.push(r.spikes)
    refFeat.set(r.features, i * dims)
    denseBudget.macs += r.budget.macs
    denseBudget.neuronSteps += r.budget.neuronSteps
    denseBudget.spikes += r.budget.spikes
    denseMs += r.ms
  }

  // ---- 读出层：只在稠密特征上训练一次，之后所有门控共用 ----
  const yAll = new Int32Array(all.length)
  all.forEach((s, i) => { yAll[i] = s.label })
  const readout = trainReadout(refFeat.subarray(0, nTrain * dims), yAll.subarray(0, nTrain), nTrain, dims, 4, 600, 0.8)
  const denseAcc = readoutAccuracy(readout, refFeat.subarray(nTrain * dims), yAll.subarray(nTrain), nTest)

  // ---- Pass B：各种门控 ----
  const runs: RunResult[] = []
  const runOne = (label: string, opts: RunOptions, _selfTrain = false, netToUse: Net = net): void => {
    const b: Budget = { macs: 0, gateOps: 0, predOps: 0, bridgeOps: 0, neuronSteps: 0, computed: 0, spikes: 0, runs: 0, tiles: 0 }
    const plAll: LayerBudget[] = netToUse.layers.map(l => ({ macs: 0, gateOps: 0, predOps: 0, bridgeOps: 0, neuronSteps: 0, computed: 0, spikes: 0, runs: 0, tiles: 0, group: groupOf(opts, l.spec) }))
    const feat = new Float64Array(all.length * dims)
    let missed = 0, extra = 0, trueSpikes = 0, ms = 0
    for (let i = 0; i < all.length; i++) {
      const r = netToUse.run(all[i].frames, cfg.T, opts)
      b.macs += r.budget.macs; b.gateOps += r.budget.gateOps; b.predOps += r.budget.predOps; b.bridgeOps += r.budget.bridgeOps
      b.neuronSteps += r.budget.neuronSteps; b.computed += r.budget.computed; b.spikes += r.budget.spikes
      b.runs += r.budget.runs; b.tiles += r.budget.tiles
      ms += r.ms
      for (let li = 0; li < plAll.length; li++) {
        const src = r.perLayer[li], dst = plAll[li]
        dst.macs += src.macs; dst.gateOps += src.gateOps; dst.predOps += src.predOps; dst.bridgeOps += src.bridgeOps
        dst.neuronSteps += src.neuronSteps; dst.computed += src.computed; dst.spikes += src.spikes
        dst.runs += src.runs; dst.tiles += src.tiles
      }
      feat.set(r.features, i * dims)
      for (let li = 0; li < netToUse.layers.length; li++) {
        const a = refSpikes[i][li], c = r.spikes[li]
        for (let j = 0; j < a.length; j++) {
          const x = a[j], y = c[j]
          if (x) trueSpikes++
          if (x && !y) missed++
          else if (!x && y) extra++
        }
      }
    }
    // 准确率一律用**稠密特征上训出来的那个读出层**来评，各档之间才可比
    // （自训练那一档也只有预测器是自学的，读出层同样不许偷看）。
    const acc = readoutAccuracy(readout, feat.subarray(nTrain * dims), yAll.subarray(nTrain), nTest)
    runs.push({
      label, mode: opts.mode, budgetFrac: opts.budgetFrac, perLayer: plAll,
      tau: opts.tau, group: opts.group,
      runLen: b.computed / Math.max(1, b.runs),
      runsPerStep: b.runs / Math.max(1, b.neuronSteps),
      tilesPerStep: b.tiles / Math.max(1, b.neuronSteps),
      totalOps: b.macs + b.gateOps + b.predOps + b.bridgeOps,
      macs: b.macs, gateOps: b.gateOps, predOps: b.predOps,
      opsRatio: 0, macRatio: 0,
      candidateFrac: b.computed / Math.max(1, b.neuronSteps),
      spikeRate: b.spikes / Math.max(1, b.neuronSteps),
      missRate: trueSpikes ? missed / trueSpikes : 0,
      extraRate: trueSpikes ? extra / trueSpikes : 0,
      accuracy: acc, ms,
    })
  }

  runOne('dense（基线：老实算完整矩阵）', DENSE_OPTS)
  runOne('event（零输入跳过，无损）', { mode: 'event', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true })
  runOne('cert（上界证书：跳过者取电流上界）', { mode: 'cert', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true })
  for (const b of cfg.budgets) {
    runOne(`cm-bound ${(b * 100).toFixed(0)}%（跳过者取上界）`, { mode: 'cm-bound', budgetFrac: b, usePredictor: true, trainPredictor: false, record: true })
  }
  for (const b of cfg.budgets) {
    runOne(`cm-decay ${(b * 100).toFixed(0)}%（跳过者只泄漏）`, { mode: 'cm-decay', budgetFrac: b, usePredictor: true, trainPredictor: false, record: true })
  }
  // 结构化稀疏：同一空间位置的全部通道打包成一块（感受野盒完全重合 ⟹ 共享一次预测）
  for (const b of cfg.budgets) {
    if (b > 0.5) continue
    runOne(`cm-decay ${(b * 100).toFixed(0)}% 纤维块（group=outCh）`, { mode: 'cm-decay', budgetFrac: b, usePredictor: true, trainPredictor: false, record: true, group: 'fiber' })
  }
  // 自适应阈值门控：不设预算、不排序，P(spike) > τ 才算
  for (const tau of [0.02, 0.05, 0.12]) {
    runOne(`cm-thresh τ=${tau}（阈值门控）`, { mode: 'cm-thresh', budgetFrac: 1, usePredictor: true, trainPredictor: false, record: true, tau })
  }
  for (const b of [0.2, 0.1]) {
    runOne(`random ${(b * 100).toFixed(0)}%（对照：同样预算，随机选）`, { mode: 'random', budgetFrac: b, usePredictor: true, trainPredictor: false, record: true, rng: splitmix32(99) })
  }
  // 自训练：预测器从网络自己的门控输出里学，没有稠密教师
  {
    const selfNet = new Net(specs, cfg.pred, cfg.seed ^ 0xabcdef)
    runOne('cm-decay 20%（自训练，无教师）', { mode: 'cm-decay', budgetFrac: 0.2, usePredictor: true, trainPredictor: true, record: true }, true, selfNet)
  }

  for (const r of runs) {
    r.opsRatio = r.totalOps / Math.max(1, denseBudget.macs)
    r.macRatio = r.macs / Math.max(1, denseBudget.macs)
  }

  // ---- Pass E/F/G：三个追问（突发流 / 在线轨迹 / 真实数字）----
  // 默认规模下这三段加起来比 B 段还贵，所以只在非 quick 模式跑；
  // `--quick` 只跑主干，保证烟测仍然是一秒级。
  const extra = cfg.T >= 20
  const burst = extra ? runBurstExperiment(cfg) : undefined
  const trajectory = extra ? runTrajectoryExperiment(cfg, 96, 64) : undefined
  const digits = extra ? runDigitsExperiment(cfg) : undefined

  // ---- Pass C：定理边界（重置 vs 无重置）----
  const theorem = runTheoremCheck(cfg, data.test.slice(0, Math.min(8, nTest)))

  // ---- Pass D：时间折叠（可能性 1 vs 可能性 2）----
  const timeFold = runTimeFold(net.layers[0], data.test[0], cfg)

  return {
    config: cfg,
    burst, trajectory, digits,
    layers: net.layers.map(l => ({ N: l.N, fanin: l.fanin, outH: l.outH, outW: l.outW, outCh: l.spec.outCh })),
    dataset: { train: nTrain, test: nTest, T: cfg.T, H: cfg.H, W: cfg.W, classes: data.classes },
    coding,
    runs,
    theorem,
    timeFold,
    dense: {
      macs: denseBudget.macs, spikes: denseBudget.spikes, neuronSteps: denseBudget.neuronSteps,
      spikeRate: denseBudget.spikes / Math.max(1, denseBudget.neuronSteps), accuracy: denseAcc, ms: denseMs,
    },
    ms: performance.now() - t0,
  }
}

/** 时间折叠：把 T 步递归折成一次带因果掩码的矩阵乘。 */
export function runTimeFold(layer: Layer, sample: EventSample, cfg: ExpConfig): TimeFoldResult {
  const T = cfg.T
  const N = layer.N
  // 先把 T 帧一次性卷成 T×N 的电流矩阵——时间在这里真的是矩阵的一个下标
  const I = convTimeStacked(sample.frames, T, layer)
  const d = layer.spec.decay, th = layer.spec.thresh

  const t1 = performance.now()
  const lif = lifSequential(I, T, N, d, th, true)
  const msLIF = performance.now() - t1
  const t2 = performance.now()
  const seq = lifSequential(I, T, N, d, th, false)
  const msSequential = performance.now() - t2
  const t3 = performance.now()
  const toe = psnToeplitz(I, T, N, d, th)
  const msToeplitz = performance.now() - t3
  const t4 = performance.now()
  const scan = psnScan(I, T, N, d, th)
  const msScan = performance.now() - t4

  const hamming = (a: Uint8Array, b: Uint8Array) => {
    let n = 0
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++
    return n
  }
  const maxErr = (a: Float32Array, b: Float32Array) => {
    let m = 0
    for (let i = 0; i < a.length; i++) { const x = Math.abs(a[i] - b[i]); if (x > m) m = x }
    return m
  }
  void msLIF
  return {
    layer: 0, neurons: N, T,
    lifVsPsnHamming: hamming(lif.s, seq.s),
    psnSeqVsToeplitzMaxErr: maxErr(seq.v, toe.v),
    psnSeqVsToeplitzHamming: hamming(seq.s, toe.s),
    psnSeqVsScanMaxErr: maxErr(seq.v, scan.v),
    psnSeqVsScanHamming: hamming(seq.s, scan.s),
    seqFlops: seq.flops, toeplitzFlops: toe.flops, scanFlops: scan.flops,
    seqDepth: seq.depth, toeplitzDepth: toe.depth, scanDepth: scan.depth,
    msSequential: msLIF + msSequential, msToeplitz, msScan,
  }
}

// ===================== 实验二：非平稳突发流下的"固定预算 vs 自适应阈值" =====================

export interface BurstRow {
  label: string
  mode: GateMode
  budgetFrac: number
  tau?: number
  /** 逐时间步的候选比例（测试集均值）。 */
  perStepCand: number[]
  /** 逐时间步的漏脉冲率（测试集均值）。 */
  perStepMiss: number[]
  candidateFrac: number
  missRate: number
  opsRatio: number
  accuracy: number
}

export interface BurstResult {
  T: number
  /** 逐时间步的输入事件密度（测试集均值）。 */
  density: number[]
  burstFrom: number
  burstTo: number
  denseAcc: number
  /** 安静段 / 爆发段的实测输入密度。 */
  quietDensity: number
  burstDensity: number
  rows: BurstRow[]
}

/**
 * 突发流实验。
 *
 * 审稿人的第二个问题："全黑的一帧你还硬挑 30%；爆闪的一帧你只给 30%，不就漏光了？"
 * 这里构造一个**样本内部**逐时间步剧烈改变活跃度的流（安静 → 爆发 → 安静），比较
 * 固定 top-B 与自适应阈值门控。看两件事：
 *   1. 自适应模式是不是"安静段少算、爆发段多算"，而固定预算只能画一条直线；
 *   2. 换来换去之后，总账（总算术比 / 漏脉冲率）是更好还是更差。
 */
export function runBurstExperiment(cfg: ExpConfig): BurstResult {
  const bd = makeBurstDataset({
    perClass: 32, testPerClass: 16, T: cfg.T, H: cfg.H, W: cfg.W, seed: cfg.seed ^ 0x1234,
  })
  const specs = defaultLayers(cfg.H, cfg.W, true, cfg.shape)
  const net = new Net(specs, cfg.pred, cfg.seed)
  const nTrain = bd.train.length, nTest = bd.test.length
  const dims = net.featureDim()
  // 教师：稠密跑一遍**训练集**，把预测器训起来（测试集一次都不看）
  for (const s of bd.train) {
    net.run(s.frames, cfg.T, {
      mode: 'dense', budgetFrac: 1, usePredictor: true, trainPredictor: true,
      trainFiber: true, record: false,
    })
  }
  // 稠密参考：测试集的真值脉冲 + 特征
  const refSpikes: Uint8Array[][] = []
  const refFeat = new Float64Array(nTest * dims)
  let denseMacs = 0
  for (let i = 0; i < nTest; i++) {
    const r = net.run(bd.test[i].frames, cfg.T, {
      mode: 'dense', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true,
    })
    refSpikes.push(r.spikes); refFeat.set(r.features, i * dims); denseMacs += r.budget.macs
  }
  // 读出层只用**训练集**的稠密特征来训，测试集上评——与主干 B 段同一套做法
  const Xtr = new Float64Array(nTrain * dims)
  const ytr = new Int32Array(nTrain), yte = new Int32Array(nTest)
  bd.train.forEach((s, i) => {
    const r = net.run(s.frames, cfg.T, {
      mode: 'dense', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true,
    })
    Xtr.set(r.features, i * dims); ytr[i] = s.label
  })
  bd.test.forEach((s, i) => { yte[i] = s.label })
  const ro = trainReadout(Xtr, ytr, nTrain, dims, bd.classes, 600, 0.8)
  const denseAcc = readoutAccuracy(ro, refFeat, yte, nTest)

  const strategies: { label: string; opts: RunOptions }[] = [
    { label: 'cm-decay 固定 30%', opts: { mode: 'cm-decay', budgetFrac: 0.3, usePredictor: true, trainPredictor: false, record: true } },
    { label: 'cm-decay 固定 20%', opts: { mode: 'cm-decay', budgetFrac: 0.2, usePredictor: true, trainPredictor: false, record: true } },
    { label: 'cm-thresh τ=0.05', opts: { mode: 'cm-thresh', budgetFrac: 1, usePredictor: true, trainPredictor: false, record: true, tau: 0.05 } },
    { label: 'cm-thresh τ=0.12', opts: { mode: 'cm-thresh', budgetFrac: 1, usePredictor: true, trainPredictor: false, record: true, tau: 0.12 } },
  ]

  const rows: BurstRow[] = []
  for (const st of strategies) {
    const perStepCand = new Float64Array(cfg.T)
    const perStepMiss = new Float64Array(cfg.T)
    const perStepTrue = new Float64Array(cfg.T)
    let ops = 0, cand = 0, steps = 0, missed = 0, trueSpikes = 0
    const feat = new Float64Array(nTest * dims)
    for (let i = 0; i < nTest; i++) {
      const r = net.run(bd.test[i].frames, cfg.T, st.opts)
      ops += r.budget.macs + r.budget.gateOps + r.budget.predOps
      cand += r.budget.computed; steps += r.budget.neuronSteps
      feat.set(r.features, i * dims)
      for (let li = 0; li < net.layers.length; li++) {
        const a = refSpikes[i][li], c = r.spikes[li], N = net.layers[li].N
        for (let j = 0; j < a.length; j++) {
          if (!a[j]) continue
          const t = (j / N) | 0
          perStepTrue[t]++; trueSpikes++
          if (!c[j]) { perStepMiss[t]++; missed++ }
        }
      }
      for (let t = 0; t < cfg.T; t++) perStepCand[t] += r.perStepCand[t]
    }
    for (let t = 0; t < cfg.T; t++) {
      perStepCand[t] /= Math.max(1, nTest)
      perStepMiss[t] = perStepTrue[t] ? perStepMiss[t] / perStepTrue[t] : 0
    }
    rows.push({
      label: st.label, mode: st.opts.mode, budgetFrac: st.opts.budgetFrac, tau: st.opts.tau,
      perStepCand: Array.from(perStepCand), perStepMiss: Array.from(perStepMiss),
      candidateFrac: cand / Math.max(1, steps),
      missRate: trueSpikes ? missed / trueSpikes : 0,
      opsRatio: ops / Math.max(1, denseMacs),
      accuracy: readoutAccuracy(ro, feat, yte, nTest),
    })
  }
  // 输入密度：测试集逐时间步的输入事件密度
  const density = new Float64Array(cfg.T)
  for (let i = 0; i < nTest; i++) {
    const fr = bd.test[i].frames
    for (let t = 0; t < cfg.T; t++) {
      let c = 0
      const off = t * cfg.H * cfg.W
      for (let j = 0; j < cfg.H * cfg.W; j++) c += fr[off + j]
      density[t] += c / (cfg.H * cfg.W)
    }
  }
  for (let t = 0; t < cfg.T; t++) density[t] /= Math.max(1, nTest)
  const meanRange = (a: number, b: number) => {
    let s = 0
    for (let t = a; t < b; t++) s += density[t]
    return s / Math.max(1, b - a)
  }
  return {
    T: cfg.T, density: Array.from(density), burstFrom: bd.burstFrom, burstTo: bd.burstTo,
    denseAcc, rows,
    quietDensity: (meanRange(0, bd.burstFrom) + meanRange(bd.burstTo, cfg.T)) / 2,
    burstDensity: meanRange(bd.burstFrom, bd.burstTo),
  }
}

// ===================== 实验三：冷启动 + 分布突变（在线学习轨迹） =====================

export interface TrajectoryPoint {
  sample: number
  /** 该样本的编码成本（比特/神经元步）。 */
  bpb: number
  /** 同上，但除以"只告诉你这一帧总体发放率"所需的码长——残差熵显微镜的读数。 */
  bpbRel: number
  /** 该样本的门控漏脉冲率（对照稠密）。 */
  missRate: number
  /** 该样本的候选比例。 */
  cand: number
}

/** 一种突变方式的一次测量。 */
export interface RegimeSummary {
  regime: RegimeB
  /** 旧规律稳态：突变前最后一个样本的**逐时间步** rel 均值。 */
  preRel: number
  /** 突变样本自身的逐时间步 rel 峰值。 */
  peakRel: number
  /** 峰值出现在第几个时间步（样本内）。 */
  peakAtStep: number
  /** 尖峰倍数 = peakRel / preRel。 */
  spike: number
  /** 突变样本末尾（最后 5 步）的 rel —— 看"同一个样本内就地适应了多少"。 */
  tailRel: number
  /** 突变前 10 个样本的漏脉冲率均值。 */
  missPre: number
  /** 突变样本本身的漏脉冲率。 */
  missAt: number
  /** 最后 10 个样本的漏脉冲率均值。 */
  missPost: number
  /** 第几个样本恢复到"突变前 ×1.05"；`null` = 96 个样本内没恢复。 */
  recoveredAt: number | null
  /** 尖峰之后，同一个样本内第几个**时间步**落回"旧规律 ×1.1"；`null` = 本样本内没落回。 */
  recoveredStep: number | null
  /** 最后 10 个样本的 rel 均值。 */
  postRel: number
}

export interface TrajectoryResult {
  shiftAt: number
  T: number
  /**
   * 冷启动的**逐时间步**读数：`coldSteps[k].rel[t]` = 第 `coldAt[k]` 个样本
   * 第 t 步的相对编码成本。
   */
  coldAt: number[]
  coldSteps: { bpb: number[]; rel: number[] }[]
  /** 四种突变方式的汇总。 */
  regimes: RegimeSummary[]
  /** 逐时间步明细只给这一路（默认 `noise`，尖峰最明显）。 */
  detail: RegimeB
  shiftSamples: number[]
  shiftSteps: { sample: number; bpb: number[]; rel: number[] }[]
  /** `detail` 那一路的逐样本轨迹。 */
  points: TrajectoryPoint[]
}

/**
 * 冷启动 + 分布突变的在线轨迹。
 *
 * **读数 = 相对码长。** 分母是"只知道这一步总体发放率、其余一无所知"时的码长
 * 下界 H(r)：
 *
 *     rel = CM 码长 ÷ H(该步实测发放率)
 *     rel > 1 = 连发放率都还没学到；rel < 1 = 上下文真的抓到了结构。
 *
 * 用相对值而不用绝对值，是因为绝对码长会跟着发放率一起变，而换规律时发放率
 * 也变了——两个效应会混在一起分不开。
 *
 * **按时间步看，不按样本看。** 一步就是一整层的发放向量（L1 576 + L2 1152 个
 * 比特），预测器一步就更新一次。最初我按样本聚合读这条曲线，四种突变方式里
 * 最大的尖峰只有 1.03×，几乎看不见——模型在新规律的头几步之内就适应完了，样本
 * 尺度把尖峰平均掉了。换成步尺度，尖峰才出来。这是这一节最值得记住的教训。
 *
 * **四种突变方式都测。** 只报一种会让人以为"分布突变"是一个标量，其实它取决于
 * 你动的是哪一层结构：动**位置**（转 90°）、动**输入量**（条变宽）、动**时间
 * 抖动**（条变实）、动**空间相关性**（同密度噪声）。前三种基本不产生尖峰，第四种
 * 产生——因为预测器读的是"感受野盒计数 + 膜电位"，前三种都没改变"给定盒计数和
 * 膜电位，会不会发"这个条件分布，第四种把它彻底改写了。
 *
 * 成本不用区间编码器，直接对预测概率取 log-loss：真编码器与理论码长差不到 0.2%
 * （A 段已实测），这里要看的是**变化的形状**，差 0.2% 不影响结论，还省一半时间。
 */
export function runTrajectoryExperiment(
  cfg: ExpConfig, nSamples = 96, shiftAt = 64, detail: RegimeB = 'noise',
): TrajectoryResult {
  const T = cfg.T
  const specs = defaultLayers(cfg.H, cfg.W, true, cfg.shape)
  const hEnt = (r: number): number => {
    if (r <= 0 || r >= 1) return 0
    return -r * Math.log2(r) - (1 - r) * Math.log2(1 - r)
  }
  const mean = (a: number[]): number => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length)

  /** 逐时间步收集器：按步累加 log-loss 与发放数。 */
  const mkTally = (Tm: number) => {
    const bits = new Float64Array(Tm), ones = new Float64Array(Tm), n = new Float64Array(Tm)
    return {
      hook(li: number, t: number, probs: Float64Array | Float32Array, spk: Uint8Array, N: number): void {
        void li
        let b = 0, o = 0
        for (let i = 0; i < N; i++) {
          const p = probs[i]
          if (p < 0) continue // 前置过滤器排掉的单元：没有预测，不计入
          const q = p < 1e-6 ? 1e-6 : p > 1 - 1e-6 ? 1 - 1e-6 : p
          if (spk[i]) { b -= Math.log2(q); o++ } else b -= Math.log2(1 - q)
        }
        bits[t] += b; ones[t] += o; n[t] += N
      },
      series(): { bpb: number[]; rel: number[] } {
        const bpb: number[] = [], rel: number[] = []
        for (let t = 0; t < Tm; t++) {
          const v = n[t] ? bits[t] / n[t] : 0
          const r = n[t] ? ones[t] / n[t] : 0
          bpb.push(v)
          rel.push(hEnt(r) > 1e-9 ? v / hEnt(r) : 1)
        }
        return { bpb, rel }
      },
      total(): { bpb: number; rel: number } {
        let b = 0, o = 0, nn = 0
        for (let t = 0; t < Tm; t++) { b += bits[t]; o += ones[t]; nn += n[t] }
        const v = nn ? b / nn : 0
        const r = nn ? o / nn : 0
        return { bpb: v, rel: hEnt(r) > 1e-9 ? v / hEnt(r) : 1 }
      },
    }
  }

  const coldAt = [0, 1, 3, 7, 15, 31].filter(i => i < shiftAt)
  const coldSteps: { bpb: number[]; rel: number[] }[] = []
  const coldIdxOf = new Map<number, number>()
  coldAt.forEach((i, k) => coldIdxOf.set(i, k))

  const shiftSamples: number[] = []
  for (let k = -2; k <= 3; k++) {
    const i = shiftAt + k
    if (i >= 0 && i < nSamples) shiftSamples.push(i)
  }
  const shiftSet = new Set(shiftSamples)

  const regimes: RegimeSummary[] = []
  let shiftSteps: { sample: number; bpb: number[]; rel: number[] }[] = []
  let points: TrajectoryPoint[] = []

  for (const regime of ['transpose', 'wide', 'solid', 'noise'] as RegimeB[]) {
    const { samples } = makeRegimeShiftDataset({
      count: nSamples, T, H: cfg.H, W: cfg.W, shiftAt, regimeB: regime,
    })
    // 参考网：稠密，预测器自训练——编码成本就从它的预测概率里读出来
    const netRef = new Net(specs, cfg.pred, cfg.seed)
    // 门控网：cm-decay 30%，预测器也自训练（没有稠密教师）
    const netGated = new Net(specs, cfg.pred, cfg.seed)
    const pts: TrajectoryPoint[] = []
    const stepsOf: { sample: number; bpb: number[]; rel: number[] }[] = []

    for (let i = 0; i < samples.length; i++) {
      const tal = mkTally(T)
      const rRef = netRef.run(samples[i].frames, T, {
        mode: 'dense', budgetFrac: 1, usePredictor: true, trainPredictor: true, record: true,
        onStep: tal.hook,
      })
      const rG = netGated.run(samples[i].frames, T, {
        mode: 'cm-decay', budgetFrac: 0.3, usePredictor: true, trainPredictor: true, record: true,
      })
      let missed = 0, trueSpikes = 0
      for (let li = 0; li < netRef.layers.length; li++) {
        const a = rRef.spikes[li], c = rG.spikes[li]
        for (let j = 0; j < a.length; j++) {
          if (a[j]) trueSpikes++
          if (a[j] && !c[j]) missed++
        }
      }
      const tot = tal.total()
      const ser = tal.series()
      const ck = coldIdxOf.get(i)
      if (ck !== undefined && regime === detail) coldSteps[ck] = ser
      if (shiftSet.has(i)) stepsOf.push({ sample: i, bpb: ser.bpb, rel: ser.rel })
      pts.push({
        sample: i, bpb: tot.bpb, bpbRel: tot.rel,
        missRate: trueSpikes ? missed / trueSpikes : 0,
        cand: rG.budget.computed / Math.max(1, rG.budget.neuronSteps),
      })
    }

    if (regime === detail) { shiftSteps = stepsOf; points = pts }

    // 汇总：尖峰在**步尺度**上找
    const before = stepsOf.filter(x => x.sample < shiftAt).pop()
    const at = stepsOf.length ? stepsOf.find(x => x.sample === shiftAt) : undefined
    const preRel = before ? mean(before.rel) : 1
    let peakRel = -1, peakAtStep = 0
    if (at) at.rel.forEach((v, t) => { if (v > peakRel) { peakRel = v; peakAtStep = t } })
    const tailRel = at ? mean(at.rel.slice(-5)) : 1
    // 就地恢复：尖峰之后，同一个样本内第几个时间步落回"旧规律 ×1.1"
    let recoveredStep: number | null = null
    if (at) {
      for (let t = peakAtStep; t < at.rel.length; t++) {
        if (at.rel[t] <= preRel * 1.1) { recoveredStep = t; break }
      }
    }
    // 恢复：从突变样本起，第几个样本的**样本级** rel 落回突变前水平 ×1.05
    const preSampleRel = mean(pts.slice(Math.max(0, shiftAt - 10), shiftAt).map(p => p.bpbRel))
    let recoveredAt: number | null = null
    for (let i = shiftAt; i < pts.length; i++) {
      if (pts[i].bpbRel <= preSampleRel * 1.05) { recoveredAt = i; break }
    }
    regimes.push({
      regime,
      preRel,
      peakRel,
      peakAtStep,
      spike: peakRel / Math.max(1e-9, preRel),
      tailRel,
      missPre: mean(pts.slice(Math.max(0, shiftAt - 10), shiftAt).map(p => p.missRate)),
      missAt: pts[shiftAt] ? pts[shiftAt].missRate : 0,
      missPost: mean(pts.slice(Math.max(0, pts.length - 10)).map(p => p.missRate)),
      recoveredAt,
      postRel: mean(pts.slice(Math.max(0, pts.length - 10)).map(p => p.bpbRel)),
      recoveredStep,
    })
  }

  return { shiftAt, T, coldAt, coldSteps, regimes, detail, shiftSamples, shiftSteps, points }
}

// ===================== 实验四 B：真实数字上的"合体"实验 =====================

/** 一种配置组合。三条杠杆各是一个开关：**宽通道**、**纤维块**、**读出 L2 归一**。 */
export interface DigitsVariant {
  label: string
  /** 网络形状；不写的字段沿用默认（ch1=4, ch2=8, k2=5）。 */
  shape?: Partial<NetShape>
  /** 门控粒度：true = 纤维块（group = outCh），false = 逐神经元。 */
  fiber?: boolean
  /**
   * 门控粒度（覆盖 `fiber`）：`1` = 逐神经元，`n` = 每 n 个连续神经元一块，
   * `'fiber'` = 整条纤维（该层全部输出通道一块）。
   */
  group?: number | 'fiber'
  /** 读出特征是否做逐样本 L2 归一（只看各通道的**相对比例**，不看绝对发放数）。 */
  l2Norm?: boolean
}

export interface DigitsCell {
  variant: string
  shape: NetShape
  /** L2 的 fanin = ch1 × k2²。 */
  fanin2: number
  /** 该档网络的总神经元数。 */
  neurons: number
  fiber: boolean
  /** 实际用的门控粒度（'fiber' 会展开成该层的 outCh）。 */
  groupLabel: string
  l2Norm: boolean
  denseAcc: number
  denseSpikeRate: number
  denseMacs: number
  rows: DigitsRow[]
}

export interface DigitsMatrixResult {
  encoding: 'poisson' | 'saccade'
  nTrain: number
  nTest: number
  pool: number
  cells: DigitsCell[]
}

/**
 * 在真实手写数字上，把 §7（宽通道 / 大 fanin）、§8（纤维块）和"读出层 L2 归一"
 * 三条杠杆**合起来**跑一遍。
 *
 * 为什么必须合：§11 量到的是"fanin 100 时 CM 门控在真实数据上打不过无损的
 * `event`"。文档里当时只指了两个方向（把 fanin 提上去 / 把粒度换成纤维块），
 * 没给数字。这一节就是那两个方向的数字，外加第三条：
 *
 * **第三条杠杆（读出 L2 归一）。** 漏了 34% 的脉冲准确率为什么掉那么多？因为
 * 门控把特征向量的**模长**也一起缩了——总发放数少三分之一，线性读出的 `w·x`
 * 全线缩水，决策边界就歪了。把每个样本的特征向量除掉它自己的 ‖x‖₂，读出只看
 * 各通道脉冲的**相对比例（方向）**，不看绝对数量。只要空间分布没被门控打乱，
 * 漏 35% 的脉冲也不该影响分类。
 *
 * 这一条**不是**重新训一个读出层：读出权重仍然只在稠密特征上训一次，归一只是
 * 送进线性分类器之前的一步**无参数**后处理。所以 §6 那条"准确率只反映门控改了
 * 表达、不反映各自重新拟合"的口径没有被破坏。
 *
 * **实现上的一点讲究**：同一个 (编码, 形状) 下的教师训练、稠密参考、训练集特征
 * 是完全一样的，只跑一次，然后在多个档位之间复用——否则同一件事要重复七遍，
 * 这一节根本跑不完。
 */
export function runDigitsMatrix(
  cfg: ExpConfig,
  variants: DigitsVariant[],
  encodings: ('poisson' | 'saccade')[] = ['poisson', 'saccade'],
  nTrain = 500, nTest = 200, pool = 3,
): DigitsMatrixResult[] {
  const out: DigitsMatrixResult[] = []
  for (const encoding of encodings) {
    const data = makeDigitsDataset({ nTrain, nTest, T: cfg.T, H: cfg.H, W: cfg.W, encoding, seed: cfg.seed })
    if (!data) continue
    const cells: DigitsCell[] = []
    // 按 (ch1, ch2, k2) 分组：同组共用一次教师训练与稠密参考
    const groups = new Map<string, { key: string; shape: NetShape; vs: DigitsVariant[] }>()
    for (const v of variants) {
      const shape: NetShape = { ...cfg.shape, in1: encoding === 'saccade' ? 2 : 1, ...v.shape }
      const key = `${shape.in1}/${shape.ch1}/${shape.ch2}/${shape.k2}`
      let g = groups.get(key)
      if (!g) { g = { key, shape, vs: [] }; groups.set(key, g) }
      g.vs.push(v)
    }
    for (const g of groups.values()) {
      const specs = defaultLayers(cfg.H, cfg.W, true, g.shape)
      const net = new Net(specs, cfg.pred, cfg.seed)
      const dims = net.featureDim(pool)
      // 先把本形状组要用到的**所有**粒度注册好——教师阶段要把它们都训一遍，
      // 漏注册就会得到一个没训过的预测器，选出来的是噪声。
      net.ensureGroups(g.vs.map(v => v.group ?? (v.fiber ? ('fiber' as const) : 1)))
      // ---- 教师：稠密跑训练集，把逐神经元与成块两套预测器都训起来 ----
      for (const s of data.train) {
        net.run(s.frames, cfg.T, {
          mode: 'dense', budgetFrac: 1, usePredictor: true, trainPredictor: true,
          trainFiber: true, record: false,
        })
      }
      // ---- 稠密参考（测试集真值脉冲 + 特征）与训练集特征（只跑一次）----
      const refSpikes: Uint8Array[][] = []
      const refFeat0 = new Float64Array(nTest * dims)
      const Xtr0 = new Float64Array(nTrain * dims)
      const ytr = new Int32Array(nTrain), yte = new Int32Array(nTest)
      let denseMacs = 0, denseSpikes = 0, denseSteps = 0
      for (let i = 0; i < nTest; i++) {
        const r = net.run(data.test[i].frames, cfg.T, {
          mode: 'dense', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true, pool,
        })
        refSpikes.push(r.spikes); refFeat0.set(r.features, i * dims)
        denseMacs += r.budget.macs; denseSpikes += r.budget.spikes; denseSteps += r.budget.neuronSteps
        yte[i] = data.test[i].label
      }
      for (let i = 0; i < nTrain; i++) {
        const r = net.run(data.train[i].frames, cfg.T, {
          mode: 'dense', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true, pool,
        })
        Xtr0.set(r.features, i * dims); ytr[i] = data.train[i].label
      }

      for (const v of g.vs) {
        const group = v.group ?? (v.fiber ? ('fiber' as const) : 1)
        // 归一要就地改这两块矩阵，不同档位之间得各拿一份副本
        const Xtr = v.l2Norm ? Xtr0.slice() : Xtr0
        const refFeat = v.l2Norm ? refFeat0.slice() : refFeat0
        if (v.l2Norm) { l2NormalizeRows(Xtr, nTrain, dims); l2NormalizeRows(refFeat, nTest, dims) }
        const ro = trainReadout(Xtr, ytr, nTrain, dims, data.classes, 800, 0.8, 1e-4, 12345, !!v.l2Norm)
        const denseAcc = readoutAccuracy(ro, refFeat, yte, nTest)

        const strategies: { label: string; opts: RunOptions }[] = [
          { label: 'event（零输入跳过，无损）', opts: { mode: 'event', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true, pool, group } },
          { label: 'cm-decay 70%', opts: { mode: 'cm-decay', budgetFrac: 0.7, usePredictor: true, trainPredictor: false, record: true, pool, group } },
          { label: 'cm-decay 50%', opts: { mode: 'cm-decay', budgetFrac: 0.5, usePredictor: true, trainPredictor: false, record: true, pool, group } },
          { label: 'cm-decay 30%', opts: { mode: 'cm-decay', budgetFrac: 0.3, usePredictor: true, trainPredictor: false, record: true, pool, group } },
          { label: 'cm-decay 20%', opts: { mode: 'cm-decay', budgetFrac: 0.2, usePredictor: true, trainPredictor: false, record: true, pool, group } },
          { label: 'cm-thresh τ=0.05', opts: { mode: 'cm-thresh', budgetFrac: 1, usePredictor: true, trainPredictor: false, record: true, tau: 0.05, pool, group } },
          { label: 'random 50%（对照）', opts: { mode: 'random', budgetFrac: 0.5, usePredictor: true, trainPredictor: false, record: true, rng: splitmix32(98), pool, group } },
          { label: 'random 30%（对照）', opts: { mode: 'random', budgetFrac: 0.3, usePredictor: true, trainPredictor: false, record: true, rng: splitmix32(99), pool, group } },
        ]
        const rows: DigitsRow[] = []
        for (const st of strategies) {
          let ops = 0, macs = 0, cand = 0, steps = 0, missed = 0, extra = 0, trueSpikes = 0
          const feat = new Float64Array(nTest * dims)
          for (let i = 0; i < nTest; i++) {
            const r = net.run(data.test[i].frames, cfg.T, st.opts)
            ops += r.budget.macs + r.budget.gateOps + r.budget.predOps
            macs += r.budget.macs
            cand += r.budget.computed; steps += r.budget.neuronSteps
            feat.set(r.features, i * dims)
            for (let li = 0; li < net.layers.length; li++) {
              const a = refSpikes[i][li], c = r.spikes[li]
              for (let j = 0; j < a.length; j++) {
                if (a[j]) trueSpikes++
                if (a[j] && !c[j]) missed++
                else if (!a[j] && c[j]) extra++
              }
            }
          }
          rows.push({
            label: st.label,
            candidateFrac: cand / Math.max(1, steps),
            macRatio: macs / Math.max(1, denseMacs),
            opsRatio: ops / Math.max(1, denseMacs),
            missRate: trueSpikes ? missed / trueSpikes : 0,
            extraRate: trueSpikes ? extra / trueSpikes : 0,
            accuracy: readoutAccuracy(ro, feat, yte, nTest),
          })
        }
        cells.push({
          variant: v.label, shape: g.shape, fanin2: g.shape.ch1 * g.shape.k2 * g.shape.k2,
          neurons: net.layers.reduce((a, l) => a + l.N, 0),
          fiber: group === 'fiber', groupLabel: group === 'fiber' ? 'outCh' : String(group),
          l2Norm: !!v.l2Norm,
          denseAcc, denseSpikeRate: denseSpikes / Math.max(1, denseSteps), denseMacs, rows,
        })
      }
    }
    // 保持与 DIGITS_VARIANTS 相同的顺序，方便对照
    const order = new Map(variants.map((v, i) => [v.label, i]))
    cells.sort((a, b) => (order.get(a.variant) ?? 0) - (order.get(b.variant) ?? 0))
    out.push({ encoding, nTrain, nTest, pool, cells })
  }
  return out
}

/**
 * §11 合体实验用的七个档位。
 *
 * 排列是**逐条加杠杆**的，所以相邻两行的差可以归因到刚加的那一条：
 *   ①→② 只动读出归一；①→③ 只动粒度；①→④ 只动宽度；
 *   ④→⑤ 在宽通道上加粒度；⑤→⑥ 再加读出归一（**合体**）；
 *   ①→⑦ 是"不换宽度、只把归一和粒度都加上"，用来看宽度到底是不是必需的；
 *   ⑥→⑧ 只把粒度从"整条纤维（32 通道一块）"改细成"8 通道一块"，用来看粒度
 *        是不是越粗越好（答案：不是，见报告）。
 */
export const DIGITS_VARIANTS: DigitsVariant[] = [
  { label: '① 基线 ch1=4,ch2=8 · 逐神经元 · 无归一' },
  { label: '② +读出 L2 归一', l2Norm: true },
  { label: '③ +纤维块', fiber: true },
  { label: '④ 宽通道 ch1=16,ch2=32', shape: { ch1: 16, ch2: 32 } },
  { label: '⑤ 宽通道 + 纤维块', shape: { ch1: 16, ch2: 32 }, fiber: true },
  { label: '⑥ 宽通道 + 纤维块 + 归一（合体）', shape: { ch1: 16, ch2: 32 }, fiber: true, l2Norm: true },
  { label: '⑦ 不换宽度 + 纤维块 + 归一', fiber: true, l2Norm: true },
  // ⑧ 是"粒度到底该多粗"的探针：宽通道下整条纤维是 32 个通道一块，太粗了，
  //   一次预测错了 32 个通道全错。改成 8 个通道一块，看看是不是更划算。
  { label: '⑧ 宽通道 + 纤维块(group=8) + 归一', shape: { ch1: 16, ch2: 32 }, group: 8, l2Norm: true },
]

// ===================== 实验四：真实手写数字（眼跳 / 泊松编码） =====================

export interface DigitsRow {
  label: string
  candidateFrac: number
  macRatio: number
  opsRatio: number
  missRate: number
  extraRate: number
  accuracy: number
}

export interface DigitsResult {
  encoding: 'poisson' | 'saccade'
  nTrain: number
  nTest: number
  denseAcc: number
  denseSpikeRate: number
  /** 读出特征的空域分块边长（见 `Net.featureDim` 的注释）。 */
  pool: number
  rows: DigitsRow[]
}

/**
 * 在**真实**手写数字上跑同一套门控。
 *
 * 数据：scikit-learn 内置的 `digits` —— 1797 张真实手写数字（8×8，源头是 UCI
 * optdigits，43 位书写者的真笔迹），57 KB，读取只依赖 node 的 zlib/fs。
 * 不再是程序画出来的移动条：它有笔画结构、类内书写差异，也有 3/5、4/9 这种
 * 真的会认错的相邻类。
 *
 * 两种编码都跑：`poisson` 是率编码（亮的地方随机发），`saccade` 是差分事件编码
 * （数字漂移 + 一次眼跳，取相邻帧亮度变化的 ON/OFF 两个极性，两个输入通道）。
 * 后者发放率只有前者一半左右——正好能看清一条边界：发放率越低，"零输入跳过"
 * 就越占便宜，CM 门控能再加的就越少。
 */
export function runDigitsExperiment(
  cfg: ExpConfig, nTrain = 600, nTest = 200, pool = 3,
): DigitsResult[] {
  const out: DigitsResult[] = []
  for (const encoding of ['poisson', 'saccade'] as const) {
    const data = makeDigitsDataset({
      nTrain, nTest, T: cfg.T, H: cfg.H, W: cfg.W, encoding, seed: cfg.seed,
    })
    if (!data) continue
    const shape = { ...cfg.shape, in1: encoding === 'saccade' ? 2 : 1 }
    const specs = defaultLayers(cfg.H, cfg.W, true, shape)
    const net = new Net(specs, cfg.pred, cfg.seed)
    const dims = net.featureDim(pool)
    // 教师：稠密跑**训练集**，把预测器训起来（测试集一次都不看）
    for (const s of data.train) {
      net.run(s.frames, cfg.T, {
        mode: 'dense', budgetFrac: 1, usePredictor: true, trainPredictor: true,
        trainFiber: true, record: false,
      })
    }
    const refSpikes: Uint8Array[][] = []
    const refFeat = new Float64Array(nTest * dims)
    const Xtr = new Float64Array(nTrain * dims)
    const ytr = new Int32Array(nTrain), yte = new Int32Array(nTest)
    let denseMacs = 0, denseSpikes = 0, denseSteps = 0
    for (let i = 0; i < nTest; i++) {
      const r = net.run(data.test[i].frames, cfg.T, {
        mode: 'dense', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true, pool,
      })
      refSpikes.push(r.spikes); refFeat.set(r.features, i * dims)
      denseMacs += r.budget.macs; denseSpikes += r.budget.spikes; denseSteps += r.budget.neuronSteps
      yte[i] = data.test[i].label
    }
    for (let i = 0; i < nTrain; i++) {
      const r = net.run(data.train[i].frames, cfg.T, {
        mode: 'dense', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true, pool,
      })
      Xtr.set(r.features, i * dims); ytr[i] = data.train[i].label
    }
    const ro = trainReadout(Xtr, ytr, nTrain, dims, data.classes, 800, 0.8)
    const denseAcc = readoutAccuracy(ro, refFeat, yte, nTest)

    const strategies: { label: string; opts: RunOptions }[] = [
      { label: 'event（零输入跳过，无损）', opts: { mode: 'event', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true, pool } },
      { label: 'cm-decay 50%', opts: { mode: 'cm-decay', budgetFrac: 0.5, usePredictor: true, trainPredictor: false, record: true, pool } },
      { label: 'cm-decay 30%', opts: { mode: 'cm-decay', budgetFrac: 0.3, usePredictor: true, trainPredictor: false, record: true, pool } },
      { label: 'cm-decay 20%', opts: { mode: 'cm-decay', budgetFrac: 0.2, usePredictor: true, trainPredictor: false, record: true, pool } },
      { label: 'cm-thresh τ=0.05', opts: { mode: 'cm-thresh', budgetFrac: 1, usePredictor: true, trainPredictor: false, record: true, tau: 0.05, pool } },
      { label: 'random 30%（对照）', opts: { mode: 'random', budgetFrac: 0.3, usePredictor: true, trainPredictor: false, record: true, rng: splitmix32(99), pool } },
    ]
    const rows: DigitsRow[] = []
    for (const st of strategies) {
      let ops = 0, macs = 0, cand = 0, steps = 0, missed = 0, extra = 0, trueSpikes = 0
      const feat = new Float64Array(nTest * dims)
      for (let i = 0; i < nTest; i++) {
        const r = net.run(data.test[i].frames, cfg.T, st.opts)
        ops += r.budget.macs + r.budget.gateOps + r.budget.predOps
        macs += r.budget.macs
        cand += r.budget.computed; steps += r.budget.neuronSteps
        feat.set(r.features, i * dims)
        for (let li = 0; li < net.layers.length; li++) {
          const a = refSpikes[i][li], c = r.spikes[li]
          for (let j = 0; j < a.length; j++) {
            if (a[j]) trueSpikes++
            if (a[j] && !c[j]) missed++
            else if (!a[j] && c[j]) extra++
          }
        }
      }
      rows.push({
        label: st.label,
        candidateFrac: cand / Math.max(1, steps),
        macRatio: macs / Math.max(1, denseMacs),
        opsRatio: ops / Math.max(1, denseMacs),
        missRate: trueSpikes ? missed / trueSpikes : 0,
        extraRate: trueSpikes ? extra / trueSpikes : 0,
        accuracy: readoutAccuracy(ro, feat, yte, nTest),
      })
    }
    out.push({
      encoding, nTrain, nTest, denseAcc,
      denseSpikeRate: denseSpikes / Math.max(1, denseSteps), pool, rows,
    })
  }
  return out
}

