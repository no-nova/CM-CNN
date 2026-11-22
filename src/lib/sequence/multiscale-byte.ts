// SPDX-License-Identifier: MPL-2.0
// Causal structural spikes are an additive expert over the unchanged byte CM.
import { hh } from '../snn/cm'
import { SpikeByteModel, SmoothedCounter, type ByteModelState, type CounterState, type BitModifier } from './spike-byte'

const PRIOR = 24
const SLOW_MIX = 0.35
const FAST_MIX = 0.15

interface StructureState {
  fast: number[]; fastSpikes: number[]; slow: number[]; slowSpikes: number[]
  histogram: number[]; historySize: number; last: number; steps: number; spikeCount: number
}
/** Leaky neurons driven by bit rhythms and by repetitions of the historically most frequent byte. */
class StructureLif {
  readonly fast = new Int32Array(8)
  readonly fastSpikes = new Uint8Array(8)
  readonly slow = new Int32Array(4)
  readonly slowSpikes = new Uint8Array(4)
  readonly histogram = new Uint16Array(256)
  historySize = 0; last = -1; steps = 0; spikeCount = 0
  phase(): number {
    // Slow traces summarize the timing of frequently repeated bytes; fast state tracks bit rhythms.
    let signature = 0
    for (let i = 0; i < 4; i++) signature |= Math.min(3, Math.floor(this.slow[i] / 64)) << (i * 2)
    return signature
  }
  fastPhase(): number {
    let mask = 0
    for (let i = 0; i < 8; i++) if (this.fastSpikes[i]) mask |= 1 << i
    return mask
  }
  advance(byte: number): void {
    // The dominant byte is selected from the past only; no space, newline, ASCII or vocabulary rules.
    let dominant = 0
    for (let i = 1; i < 256; i++) if (this.histogram[i] > this.histogram[dominant]) dominant = i
    const event = this.historySize >= 16 && byte === dominant ? 1 : 0
    for (let i = 0; i < 8; i++) {
      let v = Math.floor(this.fast[i] * (i < 4 ? 0.5 : 0.75)) + (((byte >>> i) & 1) ? 160 : 0)
      const fired = v >= 256 ? 1 : 0; if (fired) v -= 256
      this.fast[i] = v; this.fastSpikes[i] = fired; this.spikeCount += fired
    }
    for (let i = 0; i < 4; i++) {
      let v = Math.floor(this.slow[i] * [0.75, 0.875, 0.9375, 0.96875][i]) + event * 64
      const fired = v >= 256 ? 1 : 0; if (fired) v -= 256
      this.slow[i] = v; this.slowSpikes[i] = fired; this.spikeCount += fired
    }
    this.histogram[byte]++; this.historySize++
    if (this.historySize >= 8192) {
      this.historySize = 0
      for (let i = 0; i < 256; i++) { this.histogram[i] >>>= 1; this.historySize += this.histogram[i] }
    }
    this.last = byte; this.steps++
  }
  snapshot(): StructureState { return { fast: Array.from(this.fast), fastSpikes: Array.from(this.fastSpikes), slow: Array.from(this.slow),
    slowSpikes: Array.from(this.slowSpikes), histogram: Array.from(this.histogram), historySize: this.historySize,
    last: this.last, steps: this.steps, spikeCount: this.spikeCount } }
  static restore(s: StructureState): StructureLif {
    const ints = (a: number[], n: number, max: number) => Array.isArray(a) && a.length === n && a.every(v => Number.isInteger(v) && v >= 0 && v <= max)
    if (!s || !ints(s.fast, 8, 255) || !ints(s.fastSpikes, 8, 1) || !ints(s.slow, 4, 255) || !ints(s.slowSpikes, 4, 1) ||
      !ints(s.histogram, 256, 8192) || !Number.isInteger(s.historySize) || s.historySize < 0 || s.historySize >= 8192 ||
      s.histogram.reduce((a, b) => a + b, 0) !== s.historySize || !Number.isSafeInteger(s.steps) || s.steps < 0 ||
      !Number.isSafeInteger(s.spikeCount) || s.spikeCount < 0 || s.spikeCount > 12 * s.steps ||
      !Number.isInteger(s.last) || s.last < -1 || s.last > 255 || (s.steps === 0) !== (s.last === -1)) throw Error('invalid structure state')
    const v = new StructureLif(); v.fast.set(s.fast); v.fastSpikes.set(s.fastSpikes); v.slow.set(s.slow); v.slowSpikes.set(s.slowSpikes)
    v.histogram.set(s.histogram); v.historySize = s.historySize; v.last = s.last; v.steps = s.steps; v.spikeCount = s.spikeCount; return v
  }
}
class StructuralExpert implements BitModifier {
  readonly counter = new SmoothedCounter()
  readonly fastCounter = new SmoothedCounter()
  constructor(readonly lif: StructureLif, readonly active: boolean) {}
  context(prefix: number) { return hh(51, prefix, this.lif.last + 1, (this.active ? this.lif.phase() : 0) + 1) }
  fastContext(prefix: number) { return hh(52, prefix, this.lif.last + 1, (this.active ? this.lif.fastPhase() : 0) + 1) }
  predict(prefix: number, base: number): number {
    if (this.lif.last < 0) return base
    const s = this.counter.get(this.context(prefix))
    // Smooth directly toward the intact CM forecast, so unseen buckets have exactly no effect.
    const q = (s.ones + PRIOR * base) / (s.n + PRIOR)
    const f = this.fastCounter.get(this.fastContext(prefix)), fastQ = (f.ones + PRIOR * base) / (f.n + PRIOR)
    return base + SLOW_MIX * (q - base) + FAST_MIX * (fastQ - base)
  }
  learn(prefix: number, bit: number): void {
    if (this.lif.last >= 0) { this.counter.learn(this.context(prefix), bit); this.fastCounter.learn(this.fastContext(prefix), bit) }
  }
}
export interface MultiscaleState {
  format: 'ucz-multiscale-byte-v2'; structureActive: boolean; core: ByteModelState; structure: StructureState; expert: CounterState; fastExpert: CounterState; support: number[]
}
export class MultiscaleByteModel {
  private core: SpikeByteModel
  private lif: StructureLif
  private expert: StructuralExpert
  private support = new Uint8Array(256)
  constructor(readonly structureActive = true) { this.core = new SpikeByteModel(false); this.lif = new StructureLif(); this.expert = new StructuralExpert(this.lif, structureActive) }
  get bytesSeen() { return this.core.bytesSeen }
  /** Two causal bits, measured before the next byte is consumed. */
  coarsePhase(): number {
    let fastCount = 0
    for (const sp of this.lif.fastSpikes) fastCount += sp
    return (this.lif.slow[3] >= 128 ? 1 : 0) | (fastCount >= 2 ? 2 : 0)
  }
  private compose(post?: BitModifier): BitModifier {
    if (!post) return this.expert
    return {
      predict: (prefix, base) => post.predict(prefix, this.expert.predict(prefix, base)),
      learn: (prefix, bit, base, adjusted) => {
        const fixedProbability = this.expert.predict(prefix, base)
        this.expert.learn(prefix, bit)
        post.learn(prefix, bit, fixedProbability, adjusted)
      },
    }
  }
  consume(byte: number, train: boolean, post?: BitModifier): number {
    const loss = this.core.consume(byte, train, this.compose(post))
    if (train) this.support[byte] = 1
    this.lif.advance(byte)
    return loss
  }
  sample(rng: () => number, temperature = 1, clip = true, post?: BitModifier): number {
    if (typeof clip !== 'boolean') throw Error('invalid clipping flag')
    const b = this.core.sample(rng, temperature, false, this.compose(post), clip ? this.support : undefined)
    this.lif.advance(b); return b
  }
  diagnostics() { return { steps: this.lif.steps, neurons: 12, spikes: this.lif.spikeCount,
    slowPhase: this.lif.phase(), fastSpikeMask: this.lif.fastPhase(), observedBytes: this.support.reduce((a, b) => a + b, 0) } }
  snapshot(): MultiscaleState { return { format: 'ucz-multiscale-byte-v2', structureActive: this.structureActive, core: this.core.snapshot(), structure: this.lif.snapshot(),
    expert: this.expert.counter.snapshot(), fastExpert: this.expert.fastCounter.snapshot(), support: Array.from(this.support) } }
  static restore(s: MultiscaleState): MultiscaleByteModel {
    if (!s || s.format !== 'ucz-multiscale-byte-v2' || typeof s.structureActive !== 'boolean' || !Array.isArray(s.support) || s.support.length !== 256 || s.support.some(v => v !== 0 && v !== 1)) throw Error('invalid multiscale state')
    const v = new MultiscaleByteModel(s.structureActive); v.core = SpikeByteModel.restore(s.core); v.lif = StructureLif.restore(s.structure)
    if (v.core.useSpikes || v.core.bytesSeen !== v.lif.steps) throw Error('invalid multiscale core')
    v.expert = new StructuralExpert(v.lif, s.structureActive); v.expert.counter.restore(s.expert); v.expert.fastCounter.restore(s.fastExpert); v.support.set(s.support)
    return v
  }
}
