// SPDX-License-Identifier: MPL-2.0
//
// CM 内核的最小移植版（给 SNN 实验用）。
//
// 这里只搬了 UCZ 压缩器里真正影响"预测下一个脉冲"的那些部件，并且保持与
// `src/lib/coder.ts` / `src/lib/bitcm.ts` 完全相同的定点约定，于是压缩器上
// 调出来的常数（12 位概率、2047 的 stretch 上限、`err*8 >> 16` 的学习率、
// 33 槽 APM、计数老化）可以原样搬到一个完全不同的任务上：
//
//   字节流里的"下一个比特是 0 还是 1"  ⟶  脉冲流里的"下一个神经元发不发脉冲"
//
// 之所以要搬而不是"另写一个分类器"：UCZ 的整套设计前提是**零先验、在线学习**——
// 表一开始全空，权重一开始全 0，唯一的监督信号就是刚发生的那个比特。SNN 门控
// 面对的是同一个问题：没有一个"训练好的 CNN"告诉我们哪些神经元会发放，只能在
// 网络自己跑的过程中现学。
//
// 移植清单：
//   squash / STRETCH      —— coder.ts 的对数几率 ↔ 12 位概率查表（逐位相同）
//   hh()                  —— coder.ts 的 FNV 上下文哈希（含尾部折叠优化）
//   CounterModel          —— bitcm.ts 的双计数上下文模型（含 MAX_COUNT 老化）
//   Mixer                 —— coder.ts 的逐上下文权重集 logistic 混合器
//   APM                   —— coder.ts 的 33 槽自适应概率映射（SSE 校准层）
//   RangeEncoder/Decoder  —— 二值区间编码，用来**真的**把脉冲张量压一遍，
//                            而不是只算一个 -log2(p) 的理论码长

import { packIntegers, unpackIntegers } from './cm-state'

export const PBITS = 12
export const PSCALE = 1 << PBITS // 4096
export const STRETCH_MAX = 2047

// squash: 对数几率(d, 1 单位 = 1/256 nat) → 12 位概率。与 coder.ts 同一张表。
const SQ = [
  1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975,
  4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094,
]

/**
 * 对数几率 → 概率（0..4095）。
 *
 * 与 coder.ts 逐位相同，包括那个 off-by-one：`squash(0)` 是 **2047** 而不是
 * 2048（插值表在 0 附近偏了一个单位）。保留它是为了"移植"这件事可核验；
 * 差这 1/4096 对混合器输出（约 0.4999 而非 0.5）没有任何实际影响。
 */
export function squash(d: number): number {
  if (d > 2047) return 4095
  if (d < -2047) return 0
  const w = d & 127
  const i = (d >> 7) + 16
  return (SQ[i] * (128 - w) + SQ[i + 1] * w + 64) >> 7
}

/** 概率 → 对数几率（coder.ts 里的同一张反向表）。 */
export const STRETCH: Int16Array = (() => {
  const t = new Int16Array(PSCALE)
  let pi = 0
  for (let x = -STRETCH_MAX; x <= STRETCH_MAX; x++) {
    const v = squash(x)
    for (let i = pi; i <= v; i++) t[i] = x
    pi = v + 1
  }
  for (let i = pi; i < PSCALE; i++) t[i] = STRETCH_MAX
  return t
})()

export function stretch(p: number): number {
  return STRETCH[p < 0 ? 0 : p > PSCALE - 1 ? PSCALE - 1 : p | 0]
}

const FNV_P = 0x01000193
/** P^k mod 2^32，用于折叠尾部为 0 的参数（coder.ts 的同款优化）。 */
const FNV_TAIL = [1, 0x01000193, 0x26027a69, 0x3ee6b34b, 0x502c3f11, 0x46a747c3].map(v => v | 0)

/**
 * FNV 风格上下文哈希。把任意多个"弱特征"压成定长表下标——这正是 CM 里
 * "稀疏/间接上下文"的来源：特征之间不需要有意义的关系，只要联合分布有信息量。
 */
export function hh(a: number, b = 0, c = 0, d = 0, e = 0, f = 0): number {
  let h = 0x811c9dc5
  h = Math.imul(h ^ a, FNV_P)
  if (f !== 0) {
    h = Math.imul(h ^ b, FNV_P); h = Math.imul(h ^ c, FNV_P); h = Math.imul(h ^ d, FNV_P)
    h = Math.imul(h ^ e, FNV_P); h = Math.imul(h ^ f, FNV_P); return h
  }
  if (e !== 0) {
    h = Math.imul(h ^ b, FNV_P); h = Math.imul(h ^ c, FNV_P); h = Math.imul(h ^ d, FNV_P)
    h = Math.imul(h ^ e, FNV_P); return Math.imul(h, FNV_TAIL[1])
  }
  if (d !== 0) {
    h = Math.imul(h ^ b, FNV_P); h = Math.imul(h ^ c, FNV_P); h = Math.imul(h ^ d, FNV_P)
    return Math.imul(h, FNV_TAIL[2])
  }
  if (c !== 0) { h = Math.imul(h ^ b, FNV_P); h = Math.imul(h ^ c, FNV_P); return Math.imul(h, FNV_TAIL[3]) }
  if (b !== 0) { h = Math.imul(h ^ b, FNV_P); return Math.imul(h, FNV_TAIL[4]) }
  return Math.imul(h, FNV_TAIL[5])
}

/**
 * 双计数上下文模型：`(n0, n1)` 两个 Uint16，概率取拉普拉斯平滑后的比值。
 * 与 bitcm.ts 同款：计数封顶 MAX_COUNT 后对折，保证模型能跟着非平稳的
 * 脉冲流漂移（SNN 的发放率在整个序列上绝不是平稳的）。
 */
const MAX_COUNT = 1000

export class CounterModel {
  private readonly mask: number
  private readonly n0: Uint16Array
  private readonly n1: Uint16Array

  constructor(bits: number) {
    if (!Number.isInteger(bits) || bits < 1 || bits > 24) throw Error('上下文位数必须是 1..24 的整数')
    this.mask = (1 << bits) - 1
    this.n0 = new Uint16Array(1 << bits)
    this.n1 = new Uint16Array(1 << bits)
  }

  snapshot(): { n0: string; n1: string } { return { n0: packIntegers(this.n0), n1: packIntegers(this.n1) } }
  restore(state: { n0: string; n1: string }): void {
    if (!state) throw Error('invalid counter state')
    unpackIntegers(state.n0, this.n0); unpackIntegers(state.n1, this.n1)
  }

  /** 12 位概率。空上下文返回 2048（无信息），与"什么都没学到"一致。 */
  prob(ctx: number): number {
    const i = ctx & this.mask
    const a = this.n0[i], b = this.n1[i]
    if (a + b === 0) return PSCALE >> 1
    const p = (((2048 * b + 1) * PSCALE) / (2048 * (a + b) + 2)) | 0
    return p < 1 ? 1 : p > PSCALE - 1 ? PSCALE - 1 : p
  }

  learn(ctx: number, y: number): void {
    const i = ctx & this.mask
    if (y) {
      if (this.n1[i] < 65535) this.n1[i]++
    } else if (this.n0[i] < 65535) this.n0[i]++
    if (this.n0[i] + this.n1[i] >= MAX_COUNT) {
      this.n0[i] = (this.n0[i] >> 1) + 1
      this.n1[i] = (this.n1[i] >> 1) + 1
    }
  }
}

/**
 * logistic 混合器：把若干上下文模型的对数几率做加权和，再 squash 回概率。
 *
 * 与 coder.ts 完全相同的定点约定——权重是 int32、点积除以 65536 后过 squash、
 * 学习率 `err = ((y<<12) - pr) * 8`、更新 `w += (input*err + 0x8000) >> 16`。
 * 最后一路输入恒为 256（"无信息"先验），所以权重全 0 时输出正好是 p=0.5。
 */
export class Mixer {
  readonly n: number // 上下文模型数 + 1（常数项）
  readonly nsets: number // 权重集的个数（粗上下文）
  readonly inputs: Int32Array
  private readonly w: Int32Array
  private wbase = 0

  constructor(nmodels: number, nsets = 1) {
    this.n = nmodels + 1
    this.nsets = nsets
    this.inputs = new Int32Array(this.n)
    this.inputs[nmodels] = 256 // BIAS_STRETCH
    this.w = new Int32Array(this.n * nsets)
  }

  snapshot(): string { return packIntegers(this.w) }
  restore(state: string): void { unpackIntegers(state, this.w) }

  /** 读取第 i 路上下文模型的概率（12 位），内部转成对数几率。 */
  setInput(i: number, p12: number): void {
    this.inputs[i] = stretch(p12)
  }

  predict(setIdx: number): number {
    const n = this.n
    const base = (setIdx % this.nsets) * n
    this.wbase = base
    let dot = 0
    for (let i = 0; i < n; i++) dot += this.w[base + i] * this.inputs[i]
    const p = squash(Math.floor(dot / 65536))
    return p < 1 ? 1 : p > PSCALE - 1 ? PSCALE - 1 : p
  }

  learn(y: number, pr: number): void {
    const n = this.n, base = this.wbase
    const err = ((y << PBITS) - pr) * 8
    for (let i = 0; i < n; i++) this.w[base + i] += (this.inputs[i] * err + 0x8000) >> 16
  }

  /**
   * 指定权重集的学习：预测与学习之间隔着一整个神经元循环（要先跑完一层的
   * 前向、拿到真实脉冲才能学），所以 `wbase` 必须显式重设，否则会学到上一个
   * 神经元的权重集上——这正是压缩器里"编解码两端必须逐比特同步"的同一条纪律。
   */
  learnAt(setIdx: number, y: number, pr: number): void {
    this.wbase = (setIdx % this.nsets) * this.n
    this.learn(y, pr)
  }

  weightEnergy(): number {
    let s = 0
    for (let i = 0; i < this.w.length; i++) s += Math.abs(this.w[i])
    return s / this.w.length
  }
}

/**
 * APM / SSE 校准层：混合器的输出只是"各上下文的加权和"，真实条件概率对这个
 * 和并不是线性的。APM 把 stretch(pr) 量化成 32 档，对每档单独学一个校准概率，
 * 再与原始预测按比例混合。表结构与 coder.ts 一致（33 槽 × 16 位，线性插值）。
 */
const APM_SLOTS = 33
const APM_SCALE = 23

export class APM {
  private readonly t: Uint16Array
  private base = 0

  constructor(readonly nctx: number, readonly weight = 1, readonly rate = 4) {
    // 初始化为恒等映射：APM 一开始不改动上游概率（coder.ts 同样用
    // `squash((j-16)*128) * 16` 建表），它只能通过学习逐渐偏离对角线。
    this.t = new Uint16Array(nctx * APM_SLOTS)
    for (let c = 0; c < nctx; c++) {
      const base = c * APM_SLOTS
      for (let j = 0; j < APM_SLOTS; j++) this.t[base + j] = squash((j - 16) * 128) * 16
    }
  }

  snapshot(): string { return packIntegers(this.t) }
  restore(state: string): void { unpackIntegers(state, this.t) }

  apply(pr: number, ctx: number): number {
    const st = stretch(pr)
    const pos = (st + 2048) * APM_SCALE
    let q = pos >> 12
    if (q > 31) q = 31
    const base = (ctx % this.nctx) * APM_SLOTS + q
    this.base = base
    const w = pos & 4095
    const t = this.t
    const apm = (t[base] * (4096 - w) + t[base + 1] * w) >> 16 // 16 位 → 12 位
    let out = ((pr + this.weight * apm) / (1 + this.weight)) | 0
    if (out < 1) out = 1
    else if (out > PSCALE - 1) out = PSCALE - 1
    return out
  }

  /**
   * 插值表的标准 SGD（与 coder.ts 的 `learn()` 同式）：被插值命中的两个槽
   * 各自朝目标移动 1/2^rate。两端权重和恒为 1，所以不需要额外归一化。
   */
  learn(y: number): void {
    const t = this.t, base = this.base
    const target = y ? 65535 : 0
    t[base] += (target - t[base]) >> this.rate
    t[base + 1] += (target - t[base + 1]) >> this.rate
  }
}

// ===== 二值区间编码（真的把脉冲流压一遍）=====
//
// 只有"真的编出来、还能解回去"，才能说"这个预测器把脉冲张量压到了多少比特"，
// 否则那只是一个 -log2(p) 的纸面数字。

export class RangeEncoder {
  private out: number[] = []
  private low = 0
  private range = 0xffffffff
  private cache = 0
  private cacheSize = 1
  bits = 0

  /**
   * @param p12 该比特为 **1** 的概率（12 位）。区间划分用的是 0 的份额，
   *   所以这里必须先取反——这正是区间编码器最容易写反的一处。
   */
  encode(bit: number, p12: number): void {
    let p = ((PSCALE - p12) + 1) >> 1 // P(bit=0)，12 位 → 11 位
    if (p < 1) p = 1
    else if (p > 2047) p = 2047
    const bound = (this.range >>> 11) * p
    if (bit === 0) this.range = bound
    else {
      this.low += bound
      this.range -= bound
    }
    this.bits++
    while (this.range < 1 << 24) {
      this.range = (this.range << 8) >>> 0
      this.shiftLow()
    }
  }

  private shiftLow(): void {
    // 进位只在"待写链"里传播：链上第一个字节是上一个 shiftLow 留下来的
    // `cache`，后面跟着若干个被推迟的 0xFF。推迟这件事本身就是必须的——
    // 还没决定有没有进位，就不能把 0xFF 写死。
    if ((this.low >>> 0) < 0xff000000 || this.low > 0xffffffff) {
      const carry = Math.floor(this.low / 0x100000000)
      // 链上**每一个**字节都要 +carry：0xFF + 1 = 0x100，写成字节是 0x00，
      // 进位继续往前一个字节传——这正是"进位传播"的定义。只给第一个字节加，
      // 0xFF 那一段就会漏掉进位（实测：偏斜分布下几千比特后出现解码错乱）。
      let temp = this.cache + carry
      do {
        this.out.push(temp & 0xff)
        temp = 0xff + carry
      } while (--this.cacheSize !== 0)
      this.cache = (this.low >>> 24) & 0xff
    }
    this.cacheSize++
    // 关键：先左移 8 位、再截到 32 位。被移出去的那个字节已经在上面存进了
    // `cache`；如果反过来（先截取再移位），low 会一直保留到 40 位，于是
    // `(low>>32)` 不再是 0/1 的进位而是一个整字节，输出会整整差一个比特
    // （实测：解码出来的值是真值的 2 倍）。
    this.low = (this.low * 256) % 0x100000000
  }

  finish(): Uint8Array {
    for (let i = 0; i < 5; i++) this.shiftLow()
    return new Uint8Array(this.out)
  }
}

export class RangeDecoder {
  private pos = 0
  private code = 0
  private range = 0xffffffff

  constructor(private readonly src: Uint8Array) {
    if (src.length === 0) throw Error('空区间码流')
    // 第 0 字节是编码器初始 cache（恒为 0），真正的码流从第 1 字节开始。
    if (src[0] !== 0) throw Error('区间码流首字节必须为 0')
    this.pos = 1
    // `<<` 是 32 位有符号运算，必须 `>>> 0` 拉回无符号，否则 code 变负数、
    // `code < bound` 恒真、解出来全是 0。
    // `code` 是"当前区间内的偏移量"，不是"码流值 − low"。这个区别很关键：
    // 编码器的进位已经通过 cache 机制在**输出字节**里消化掉了，所以解码端
    // 根本不需要知道 low，也不可能知道（被进位改过的字节早就读进来了）。
    for (let i = 0; i < 4; i++) this.code = ((this.code << 8) | (src[this.pos++] ?? 0)) >>> 0
  }

  decode(p12: number): number {
    let p = ((PSCALE - p12) + 1) >> 1 // 与 encode 同样：P(bit=0)
    if (p < 1) p = 1
    else if (p > 2047) p = 2047
    const bound = (this.range >>> 11) * p
    let bit: number
    if (this.code < bound) {
      this.range = bound
      bit = 0
    } else {
      this.code -= bound
      this.range -= bound
      bit = 1
    }
    while (this.range < 1 << 24) {
      this.range = (this.range << 8) >>> 0
      this.code = ((this.code << 8) | (this.src[this.pos++] ?? 0)) >>> 0
    }
    return bit
  }
}

/** 理论码长 -log2(p(y))，用于与真实编码器对账。 */
export function bitCost(p12: number, y: number): number {
  const p = Math.min(1 - 1 / PSCALE, Math.max(1 / PSCALE, (p12 + 0.5) / PSCALE))
  return -Math.log2(y ? p : 1 - p)
}
