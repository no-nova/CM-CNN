// SPDX-License-Identifier: MPL-2.0
//
// "小型 CNN 预测 SNN 下一刻会用到哪些神经元"——这个预测器本身是什么结构？
//
// 直接照搬 UCZ 的分层：
//
//   ① 固定的卷积特征库（"CNN 的卷积部分"，不需要训练）
//       · 多尺度盒滤波：1×、2×、4× 感受野里的脉冲计数（积分图 → 与核大小无关的
//         O(1) 查询）。这就是一个核全为 1 的卷积层，多尺度 = 多通道。
//       · 逐神经元静态哈希（身份）、时间相位。
//   ② 上下文哈希（CM 的 hh()）：把上面这些弱特征两两组合压成定长表下标。
//      组合出来的联合分布有信息量，而单独的每个特征都很弱——这正是 context
//      mixing 的老本行。
//   ③ 自适应混合器（"CNN 的最后一层"，在线训练）
//      每个粗上下文一套权重的 logistic 混合，学习率与 UCZ 的字节模型完全相同。
//   ④ APM/SSE 校准层：把混合器输出再过一张按上下文分开的自适应概率映射表。
//
// ③ ④ 的训练信号是"这个神经元刚才到底发没发放"——网络自己跑出来的脉冲，
// 不需要任何外部训练集。这就是 UCZ 宪法里"零预训练权重"的同一条约束，
// 被搬到了 SNN 上。

import { APM, CounterModel, Mixer, PSCALE, hh } from './cm'
import { splitmix32, type Rng } from './rand'
import { packIntegers, unpackIntegers } from './cm-state'

/** 盒内脉冲数的量化档（3 位）。_counts 本身可以很大，但有判别力的是"几乎没有/有几个/很多"。 */
const COUNT_QT = [1, 2, 3, 5, 8, 12, 20]

export function quantCount(n: number): number {
  let q = 0
  for (const t of COUNT_QT) if (n >= t) q++
  return q & 7
}

export interface PredConfig {
  /** 上下文表位数。 */
  bits: number
  /** 混合器权重集个数（粗上下文）。 */
  mixerSets: number
  /** 是否启用 APM 校准层。 */
  apmOn: boolean
  /** APM 上下文个数。 */
  apmCtx: number
  /**
   * 是否把"膜电位离阈多远"（vQ）也塞进上下文哈希。
   *
   * 关掉（默认）时上下文只由**输入**侧的盒计数构成，于是即使网络被门控改了
   * 状态，特征分布也不会漂移；打开后纯预测（压缩）指标略好，但门控下会变差——
   * 因为被跳过的神经元膜电位本身就是脏的。见 docs/SNN_CM.md 的 A/B。
   */
  stateContexts: boolean
}

/** 默认值即"严格零先验"：表空、权重 0，唯一的先验是常数项 256（p=0.5）。 */
export const DEFAULT_PRED: PredConfig = { bits: 18, mixerSets: 16, apmOn: true, apmCtx: 16, stateContexts: false }

/** Snapshot at a sample/step boundary, never between predict and learn. */
export interface PredictorState {
  N: number; outCh: number; inCh: number; group: number
  hist: string; posHash: string
  counters: { n0: string; n1: string }[]
  mixer: string; apm: string | null; seen: number; logLoss: number
}

/**
 * 一层的脉冲预测器。
 *
 * 所有神经元共享同一套上下文表（表是"模型"，上下文是"地址"），这是 CM 与
 * "给每个神经元训练一个小网络"最本质的区别：参数量不随神经元数增长。
 */
export class SpikePredictor {
  readonly N: number
  readonly outCh: number
  readonly inCh: number
  /**
   * **预测单元的粒度**：每 `group` 个连续神经元共享一次预测。
   *
   * `group = 1`（默认）= 逐神经元预测。
   * `group = outCh` = **纤维（fiber）模式**：同一空间位置下的所有输出通道共享
   *   一次预测。这不是偷懒——它们的感受野盒子是**完全重合**的（盒由 (oy,ox)
   *   决定，与输出通道无关），所以输入侧的特征本来就逐位置相同；代价是丢掉了
   *   "哪个通道会发放"的逐通道历史，换来的是
   *     · 预测器开销降到 1/group（§7 里那个"每神经元 27 次运算"直接除以 group）；
   *     · 门控变成**结构化稀疏**：整块连续神经元一起算/一起跳，而不是散点。
   */
  readonly group: number
  /** 预测单元的个数 = N / group。 */
  readonly units: number
  /** 每个空间位置下的预测单元个数 = outCh / group。 */
  readonly unitsPerPos: number
  private readonly hist: Uint8Array // 每个神经元最近 8 步的发放位
  private readonly posHash: Int32Array // 逐神经元的静态身份哈希
  private readonly m: CounterModel[]
  private readonly mixer: Mixer
  private readonly apm: APM | null
  private readonly p12: Int32Array
  readonly probs: Float32Array // 上一步的输出（0..1）
  readonly cfg: PredConfig
  /** 上一步每个神经元用的 APM 上下文（训练时回放，保证表槽一致）。 */
  private readonly apmCtxOf: Int32Array
  /** 上一步每个神经元用的混合器权重集（训练时回放）。 */
  private readonly setOf: Int32Array
  /** 上一步混合器（APM 之前）的原始概率，混合器学习必须用上游概率。 */
  private readonly rawP12: Int32Array
  /** 观测计数，用于统计。 */
  seen = 0
  /** 累计对数损失（bits）。 */
  logLoss = 0

  constructor(N: number, outCh: number, inCh: number, seed: number, cfg: PredConfig = DEFAULT_PRED, group = 1) {
    if (!Number.isInteger(group) || group < 1 || N % group !== 0) throw Error('group 必须是能整除 N 的正整数')
    // 一个单元不许跨空间位置：跨了的话"这个单元的感受野盒"就没有定义了。
    // 纤维粒度（group = outCh）已经是"逐位置"这一层能做到的最大块。
    if (outCh % group !== 0) throw Error('group 必须能整除 outCh（单元不允许跨空间位置）')
    this.N = N
    this.outCh = outCh
    this.inCh = inCh
    this.group = group
    this.units = N / group
    this.unitsPerPos = outCh / group
    this.cfg = cfg
    this.hist = new Uint8Array(this.units)
    const rng: Rng = splitmix32(seed)
    this.posHash = new Int32Array(this.units)
    for (let i = 0; i < this.units; i++) this.posHash[i] = (rng() * 0x7fffffff) | 0
    this.m = [
      new CounterModel(cfg.bits), // 0: 自身历史 × 身份
      new CounterModel(cfg.bits), // 1: 自身历史 × 本尺度密度
      new CounterModel(cfg.bits), // 2: 多尺度密度 × 膜电位档（逐位置，跨通道共享）
      new CounterModel(cfg.bits), // 3: 身份 × 时间相位（周期性 / "match" 味道）
      new CounterModel(cfg.bits), // 4: 本尺度密度（逐位置，跨通道共享）
    ]
    this.mixer = new Mixer(this.m.length, cfg.mixerSets)
    this.apm = cfg.apmOn ? new APM(cfg.apmCtx, 1, 4) : null
    this.p12 = new Int32Array(this.units)
    this.probs = new Float32Array(this.units)
    this.apmCtxOf = new Int32Array(this.units)
    this.setOf = new Int32Array(this.units)
    this.rawP12 = new Int32Array(this.units)
  }

  snapshot(): PredictorState {
    return { N: this.N, outCh: this.outCh, inCh: this.inCh, group: this.group,
      hist: packIntegers(this.hist), posHash: packIntegers(this.posHash),
      counters: this.m.map(m => m.snapshot()), mixer: this.mixer.snapshot(),
      apm: this.apm ? this.apm.snapshot() : null, seen: this.seen, logLoss: this.logLoss }
  }

  restore(state: PredictorState): void {
    if (!state || state.N !== this.N || state.outCh !== this.outCh || state.inCh !== this.inCh ||
      state.group !== this.group || !Array.isArray(state.counters) || state.counters.length !== this.m.length ||
      (this.apm ? typeof state.apm !== 'string' : state.apm !== null) ||
      !Number.isSafeInteger(state.seen) || state.seen < 0 || !Number.isFinite(state.logLoss) || state.logLoss < 0)
      throw Error('invalid predictor state')
    unpackIntegers(state.hist, this.hist); unpackIntegers(state.posHash, this.posHash)
    this.m.forEach((m, i) => m.restore(state.counters[i]))
    this.mixer.restore(state.mixer)
    if (this.apm) this.apm.restore(state.apm!)
    this.seen = state.seen; this.logLoss = state.logLoss
  }

  /** 一个单元的膜电位取组内最大值：只要有一个通道快过阈，整块都值得算。 */
  private vOf(v: Float32Array, u: number): number {
    const g = this.group
    const b = u * g
    let m = v[b]
    for (let j = 1; j < g; j++) if (v[b + j] > m) m = v[b + j]
    return m
  }

  /** 一个单元的真实标签取组内 OR：漏掉整块等于漏掉块里任何一个脉冲。 */
  private yOf(spk: Uint8Array, u: number): number {
    const g = this.group
    const b = u * g
    for (let j = 0; j < g; j++) if (spk[b + j]) return 1
    return 0
  }

  /**
   * 预测一步，返回每个神经元的发放概率（0..1）。
   *
   * **前置过滤器**：`cnt1` 全 0（感受野盒内一个脉冲都没有）时，突触电流恒为 0，
   * 于是"这个神经元本步不可能发放"是一个**定理**而不是预测——`probs[i]` 被置为
   * -1 标记，既不占用预算也不花预测器的算力。把确定性的部分拿掉之后，预测器
   * 只需要对"真有可能发放"的那部分神经元排序，每神经元的摊薄成本随之下降。
   *
   * @param cnt1 感受野尺度（1×）的逐 (位置, 输入通道) 脉冲计数
   * @param cnt2 2× 尺度的计数
   * @param v    携带的膜电位（用来量化"离阈值还有多远"）
   * @param thresh 阈值
   * @param t    时间步（相位特征）
   * @param all  true = 关闭前置过滤（压缩那一路要对每个神经元都给概率）
   * @returns 本次真正算过概率的神经元个数（用于成本对账）
   */
  predict(cnt1: Int32Array, cnt2: Int32Array, v: Float32Array, thresh: number, t: number, all = false): number {
    const { unitsPerPos, group, inCh, hist, posHash, m, mixer, apm, p12, probs } = this
    const phase = t & 7
    const posCount = this.N / this.outCh
    let evaluated = 0
    for (let p = 0; p < posCount; p++) {
      let n1 = 0, n2 = 0
      const cb = p * inCh
      for (let ic = 0; ic < inCh; ic++) { n1 += cnt1[cb + ic]; n2 += cnt2[cb + ic] }
      const d1 = quantCount(n1)
      const d2 = quantCount(n2)
      const u0 = p * unitsPerPos
      if (!all && n1 === 0) {
        // 定理：盒内无脉冲 ⟹ 电流为 0 ⟹（硬重置下膜电位始终 < θ）不可能发放
        for (let s = 0; s < unitsPerPos; s++) probs[u0 + s] = -1
        continue
      }
      // 上下文 2、4 只依赖逐位置的量，同一位置下的所有单元共用一次查表
      const c2shared = hh(d1, d2, phase)
      const c4shared = hh(d1, d2)
      evaluated += unitsPerPos
      for (let s = 0; s < unitsPerPos; s++) {
        const u = u0 + s
        const h = hist[u]
        let vq = ((this.vOf(v, u) / thresh) * 16) | 0
        if (vq < 0) vq = 0
        else if (vq > 15) vq = 15
        // 上下文 2、4：默认只用输入侧的盒计数（可跨输出通道共享 → 更便宜）；
        // 打开 stateContexts 后带上膜电位档位（发不发放 ≈ f(泄漏后的电位, 输入)）。
        const c2 = this.cfg.stateContexts ? hh(vq, d1, d2) : c2shared
        const c4 = this.cfg.stateContexts ? hh(vq, d1, phase) : c4shared
        mixer.setInput(0, m[0].prob(hh(h, posHash[u])))
        mixer.setInput(1, m[1].prob(hh(h, d1)))
        mixer.setInput(2, m[2].prob(c2))
        mixer.setInput(3, m[3].prob(hh(posHash[u], phase)))
        mixer.setInput(4, m[4].prob(c4))
        const set = (d1 & 3) | ((vq & 3) << 2)
        this.setOf[u] = set
        const raw = mixer.predict(set)
        if (apm) {
          const actx = (d1 & 7) | ((vq & 1) << 3)
          this.apmCtxOf[u] = actx
          p12[u] = apm.apply(raw, actx)
        } else p12[u] = raw
        this.rawP12[u] = raw
        probs[u] = p12[u] / PSCALE
      }
    }
    // group > 1 时"每单元的开销"没变，但单元数少了 group 倍 —— 这就是省下来的。
    void group
    return evaluated
  }

  /**
   * 观察 + 学习。`cnt1/cnt2/v/thresh/t` 必须与上一次 `predict()` 完全一致，
   * 否则学到的上下文地址和预测时的不是同一个——这和算术编码器里"编解码两端
   * 模型必须逐比特同步"是同一条纪律。
   *
   * 顺序也是有讲究的（照抄 coder.ts 的 step()/learn() 分工）：
   *   1) 先按**更新前**的计数取出各上下文概率，填进混合器；
   *   2) 用预测时的 pr 更新混合器权重与 APM；
   *   3) 最后才更新计数与自身历史。
   * 反过来会让混合器"看到"已经包含了当前观测的表，等价于偷看未来。
   */
  learnWithCounts(spk: Uint8Array, cnt1: Int32Array, cnt2: Int32Array, v: Float32Array, thresh: number, t: number, train = true): void {
    const { unitsPerPos, inCh, hist, posHash, m, mixer, apm, probs } = this
    const phase = t & 7
    const posCount = this.N / this.outCh
    for (let p = 0; p < posCount; p++) {
      let n1 = 0, n2 = 0
      const cb = p * inCh
      for (let ic = 0; ic < inCh; ic++) { n1 += cnt1[cb + ic]; n2 += cnt2[cb + ic] }
      const d1 = quantCount(n1)
      const d2 = quantCount(n2)
      const c2shared = hh(d1, d2, phase)
      const c4shared = hh(d1, d2)
      const u0 = p * unitsPerPos
      for (let s = 0; s < unitsPerPos; s++) {
        const u = u0 + s
        const y = this.yOf(spk, u)
        const h = hist[u]
        if (probs[u] < 0) {
          // 前置过滤掉的单元：本步没有算过概率，权重集/APM 槽都还停留在
          // 上一个单元上，不能拿来学习；只推进自身历史。
          hist[u] = ((h << 1) | y) & 0xff
          this.logLoss += y ? 40 : 0 // 被定理排除却发了脉冲：只可能出现在无重置网络
          continue
        }
        let vq = ((this.vOf(v, u) / thresh) * 16) | 0
        if (vq < 0) vq = 0
        else if (vq > 15) vq = 15
        const c2 = this.cfg.stateContexts ? hh(vq, d1, d2) : c2shared
        const c4 = this.cfg.stateContexts ? hh(vq, d1, phase) : c4shared
        const c0 = hh(h, posHash[u])
        const c1 = hh(h, d1)
        const c3 = hh(posHash[u], phase)
        // ① 更新前的概率 —— 与 predict() 里的一模一样
        mixer.setInput(0, m[0].prob(c0))
        mixer.setInput(1, m[1].prob(c1))
        mixer.setInput(2, m[2].prob(c2))
        mixer.setInput(3, m[3].prob(c3))
        mixer.setInput(4, m[4].prob(c4))
        // ② 权重与校准：混合器用**上游**概率学（与 coder.ts 的 prOverride 同义），
        //    APM 用被插值命中的那两个槽学。
        if (train) {
          mixer.learnAt(this.setOf[u], y, this.rawP12[u])
          if (apm) {
            apm.apply(this.rawP12[u], this.apmCtxOf[u])
            apm.learn(y)
          }
        }
        // ③ 计数与历史（历史必须无条件推进：它是网络自己的观测，不是训练）
        if (train) {
          m[0].learn(c0, y)
          m[1].learn(c1, y)
          m[2].learn(c2, y)
          m[3].learn(c3, y)
          m[4].learn(c4, y)
        }
        hist[u] = ((h << 1) | y) & 0xff
        this.seen++
        const q = probs[u]
        const pp = y ? q : 1 - q
        this.logLoss += -Math.log2(pp < 1e-6 ? 1e-6 : pp > 0.999999999 ? 0.999999999 : pp)
      }
    }
  }

  /** 每步的算术量估计（乘加 + 哈希 + 取数，粗粒度，用于成本对账）。 */
  opsPerNeuron(): number {
    return 3 * 2 /* 3 个逐神经元的 hh */ + 3 /* 3 次上下文查表 */ +
      (this.m.length + 1) * 2 /* 混合器点积 + 学习 */ + (this.apm ? 6 : 0)
  }
}
