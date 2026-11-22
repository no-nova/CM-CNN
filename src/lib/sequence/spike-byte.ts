// SPDX-License-Identifier: MPL-2.0
// Zero-corpus-start bitwise context mixer and fixed, causal recurrent LIF reservoir.
import { hh } from '../snn/cm'
import { packIntegers, unpackIntegers } from '../snn/cm-state'
import { splitmix32 } from '../snn/rand'

const TABLE_BITS = 16
const TABLE_SIZE = 1 << TABLE_BITS
const MASK = TABLE_SIZE - 1
const MAX_COUNT = 1024
const NEURONS = 64
const REC_FAN = 8
const RESERVOIR_SEED = 0x51a71e

export interface CounterState { n0: string; n1: string }
export class SmoothedCounter {
  private readonly n0 = new Uint16Array(TABLE_SIZE)
  private readonly n1 = new Uint16Array(TABLE_SIZE)
  get(ctx: number) { const i = ctx & MASK, a = this.n0[i], b = this.n1[i]; return { p: (b + 1) / (a + b + 2), n: a + b, ones: b, zeros: a } }
  learn(ctx: number, bit: number) {
    const i = ctx & MASK
    if (bit) this.n1[i]++; else this.n0[i]++
    if (this.n0[i] + this.n1[i] >= MAX_COUNT) { this.n0[i] >>>= 1; this.n1[i] >>>= 1 }
  }
  snapshot(): CounterState { return { n0: packIntegers(this.n0), n1: packIntegers(this.n1) } }
  restore(s: CounterState) { if (!s) throw Error('invalid counter'); unpackIntegers(s.n0, this.n0); unpackIntegers(s.n1, this.n1) }
}

/** One tick per completed byte. Weights are seeded fixed synthetic topology, not learned or loaded. */
class ByteLif {
  private readonly inputWeights = new Int16Array(NEURONS * 8)
  private readonly recWeights = new Int16Array(NEURONS * REC_FAN)
  private readonly recIndex = new Uint8Array(NEURONS * REC_FAN)
  readonly v = new Int32Array(NEURONS)
  readonly spikes = new Uint8Array(NEURONS)
  totalSpikes = 0
  constructor() {
    const rng = splitmix32(RESERVOIR_SEED)
    for (let i = 0; i < this.inputWeights.length; i++) this.inputWeights[i] = Math.floor(rng() * 321) - 100
    for (let i = 0; i < this.recWeights.length; i++) { this.recIndex[i] = Math.floor(rng() * NEURONS); this.recWeights[i] = Math.floor(rng() * 121) - 60 }
  }
  advance(byte: number): void {
    const next = new Uint8Array(NEURONS)
    for (let n = 0; n < NEURONS; n++) {
      let v = Math.trunc(this.v[n] * 3 / 4)
      for (let j = 0; j < 8; j++) if ((byte >>> j) & 1) v += this.inputWeights[n * 8 + j]
      for (let j = 0; j < REC_FAN; j++) if (this.spikes[this.recIndex[n * REC_FAN + j]]) v += this.recWeights[n * REC_FAN + j]
      if (v >= 512) { next[n] = 1; v = 0 }
      this.v[n] = Math.max(-1024, Math.min(1024, v))
    }
    this.spikes.set(next)
    for (const bit of next) this.totalSpikes += bit
  }
  /** Compress each group of eight spiking neurons into one activity bit. */
  signature(): number {
    let sig = 0
    for (let g = 0; g < 8; g++) { let count = 0; for (let j = 0; j < 8; j++) count += this.spikes[g * 8 + j]; if (count >= 2) sig |= 1 << g }
    return sig
  }
}

export interface ByteModelState {
  format: 'ucz-spike-byte-v1'; useSpikes: boolean; bytesSeen: number; history: number[]
  lif: { v: number[]; spikes: number[]; totalSpikes: number }; counters: CounterState[]
}
export interface BitModifier {
  predict(prefix: number, baseProbability: number): number
  learn(prefix: number, bit: number, baseProbability: number, adjustedProbability: number): void
}

export class SpikeByteModel {
  private readonly models = Array.from({ length: 5 }, () => new SmoothedCounter())
  private readonly lif = new ByteLif()
  private history: number[] = [] // most recent byte first
  bytesSeen = 0
  constructor(readonly useSpikes = true) {}
  /** Prefix encodes only bits already observed in the current byte (leading sentinel 1). */
  private contexts(prefix: number): number[] {
    const h = this.history
    return [hh(1, prefix), hh(2, prefix, h[0] + 1), hh(3, prefix, h[0] + 1, h[1] + 1),
      hh(4, prefix, h[0] + 1, h[1] + 1, h[2] + 1), hh(5, prefix, this.lif.signature() + 1)]
  }
  private probability(prefix: number): { probability: number; contexts: number[] } {
    const contexts = this.contexts(prefix)
    let p = this.models[0].get(contexts[0]).p
    for (let i = 1; i < 4; i++) if (this.history.length >= i) {
      const s = this.models[i].get(contexts[i]), weight = s.n / (s.n + 6)
      p += weight * (s.p - p)
    }
    if (this.useSpikes && this.history.length) {
      const s = this.models[4].get(contexts[4]), weight = 0.35 * s.n / (s.n + 12)
      p += weight * (s.p - p)
    }
    return { probability: Math.max(1e-6, Math.min(1 - 1e-6, p)), contexts }
  }
  private learn(contexts: number[], bit: number): void {
    this.models[0].learn(contexts[0], bit)
    for (let i = 1; i < 4; i++) if (this.history.length >= i) this.models[i].learn(contexts[i], bit)
    if (this.useSpikes && this.history.length) this.models[4].learn(contexts[4], bit)
  }
  private finish(byte: number): void {
    this.lif.advance(byte)
    this.history.unshift(byte); if (this.history.length > 3) this.history.pop()
    this.bytesSeen++
  }
  /** Score each bit before it is observed, then optionally learn it. State advances in both modes. */
  consume(byte: number, train: boolean, modifier?: BitModifier): number {
    if (!Number.isInteger(byte) || byte < 0 || byte > 255 || typeof train !== 'boolean') throw Error('invalid byte')
    let prefix = 1, loss = 0
    for (let bitPos = 7; bitPos >= 0; bitPos--) {
      const { probability: base, contexts } = this.probability(prefix), bit = (byte >>> bitPos) & 1
      const probability = modifier ? modifier.predict(prefix, base) : base
      if (!Number.isFinite(probability) || probability <= 0 || probability >= 1) throw Error('invalid modified probability')
      loss -= Math.log2(bit ? probability : 1 - probability)
      if (train) { this.learn(contexts, bit); modifier?.learn(prefix, bit, base, probability) }
      prefix = prefix * 2 + bit
    }
    this.finish(byte)
    return loss
  }
  /** Draw from the predicted bits; no probability-model training unless explicitly requested. */
  sample(rng: () => number, temperature = 1, train = false, modifier?: BitModifier, observedSupport?: Uint8Array): number {
    if (!Number.isFinite(temperature) || temperature <= 0 || temperature > 5 || typeof train !== 'boolean') throw Error('invalid sampling settings')
    if (observedSupport && (observedSupport.length !== 256 || observedSupport.some(x => x !== 0 && x !== 1))) throw Error('invalid byte support')
    let prefix = 1, byte = 0
    for (let pos = 7; pos >= 0; pos--) {
      const { probability: base, contexts } = this.probability(prefix)
      const probability = modifier ? modifier.predict(prefix, base) : base
      if (!Number.isFinite(probability) || probability <= 0 || probability >= 1) throw Error('invalid modified probability')
      let adjusted = 1 / (1 + Math.exp(-Math.log(probability / (1 - probability)) / temperature))
      // Reject only branches with zero globally observed byte support; no byte values are predefined.
      if (observedSupport) {
        let zero = false, one = false
        const knownPrefix = prefix - (1 << (7 - pos))
        for (let b = 0; b < 256; b++) if (observedSupport[b] && (b >>> (pos + 1)) === knownPrefix) {
          if ((b >>> pos) & 1) one = true; else zero = true
        }
        if (zero && !one) adjusted = 0
        else if (one && !zero) adjusted = 1
        // Empty support deliberately falls back to unconstrained sampling.
      }
      const draw = rng(); if (!Number.isFinite(draw) || draw < 0 || draw >= 1) throw Error('invalid random draw')
      const bit = draw < adjusted ? 1 : 0
      if (train) { this.learn(contexts, bit); modifier?.learn(prefix, bit, base, probability) }
      byte = byte * 2 + bit; prefix = prefix * 2 + bit
    }
    this.finish(byte); return byte
  }
  diagnostics() { return { neurons: NEURONS, ticks: this.bytesSeen, totalSpikes: this.lif.totalSpikes, lastActive: this.lif.spikes.reduce((a, b) => a + b, 0), signature: this.lif.signature() } }
  snapshot(): ByteModelState {
    return { format: 'ucz-spike-byte-v1', useSpikes: this.useSpikes, bytesSeen: this.bytesSeen, history: this.history.slice(),
      lif: { v: Array.from(this.lif.v), spikes: Array.from(this.lif.spikes), totalSpikes: this.lif.totalSpikes }, counters: this.models.map(m => m.snapshot()) }
  }
  static restore(s: ByteModelState): SpikeByteModel {
    if (!s || s.format !== 'ucz-spike-byte-v1' || typeof s.useSpikes !== 'boolean' || !Number.isSafeInteger(s.bytesSeen) || s.bytesSeen < 0 ||
      !Array.isArray(s.history) || s.history.length !== Math.min(3, s.bytesSeen) || s.history.some(x => !Number.isInteger(x) || x < 0 || x > 255) ||
      !s.lif || !Number.isSafeInteger(s.lif.totalSpikes) || s.lif.totalSpikes < 0 || s.lif.totalSpikes > s.bytesSeen * NEURONS || !Array.isArray(s.lif.v) || s.lif.v.length !== NEURONS || s.lif.v.some(v => !Number.isInteger(v) || v < -1024 || v > 1024) ||
      !Array.isArray(s.lif.spikes) || s.lif.spikes.length !== NEURONS || s.lif.spikes.some(x => x !== 0 && x !== 1) ||
      !Array.isArray(s.counters) || s.counters.length !== 5) throw Error('invalid byte-model state')
    const model = new SpikeByteModel(s.useSpikes)
    model.history = s.history.slice(); model.bytesSeen = s.bytesSeen; model.lif.v.set(s.lif.v); model.lif.spikes.set(s.lif.spikes); model.lif.totalSpikes = s.lif.totalSpikes
    model.models.forEach((m, i) => m.restore(s.counters[i])); return model
  }
}
