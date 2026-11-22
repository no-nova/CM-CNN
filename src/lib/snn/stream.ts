// SPDX-License-Identifier: MPL-2.0
// Prequential online classification. The feature-CM is separate from the frozen spike-gating CM.
import { CounterModel, PSCALE, hh } from './cm'

export interface FeatureCmState { position: { n0: string; n1: string }; temporal: { n0: string; n1: string }; previous: number[]; sequence: number }
export interface CmObservation { entropy: number; surprise: number; bits: Uint8Array; previous: Uint8Array; sequence: number }
export class FeatureContextMixer {
  private readonly position = new CounterModel(12)
  private readonly temporal = new CounterModel(13)
  private readonly previous: Uint8Array
  private sequence = 0
  constructor(readonly d: number) {
    if (!Number.isInteger(d) || d < 1) throw Error('invalid feature dimension')
    this.previous = new Uint8Array(d)
  }
  // All probabilities are from the state before this sample. No learning occurs here.
  observe(x: Float64Array): CmObservation {
    if (x.length !== this.d || !x.every(Number.isFinite)) throw Error('invalid features')
    const bits = Uint8Array.from(x, v => v > 1 / Math.sqrt(this.d) ? 1 : 0)
    const previous = this.previous.slice()
    let entropy = 0, surprise = 0
    for (let j = 0; j < this.d; j++) {
      const p1 = this.position.prob(hh(j + 1)) / PSCALE
      const p2 = this.temporal.prob(hh(j + 1, previous[j] + 1)) / PSCALE
      const p = Math.max(0.01, Math.min(0.99, (p1 + p2) / 2))
      entropy += -p * Math.log2(p) - (1 - p) * Math.log2(1 - p)
      surprise += -Math.log2(bits[j] ? p : 1 - p)
    }
    return { entropy: entropy / this.d, surprise: surprise / this.d, bits, previous, sequence: this.sequence }
  }
  learn(observation: CmObservation): void {
    if (observation.sequence !== this.sequence || observation.bits.length !== this.d || observation.previous.length !== this.d ||
        observation.previous.some((v, i) => v !== this.previous[i])) throw Error('stale CM observation')
    for (let j = 0; j < this.d; j++) {
      this.position.learn(hh(j + 1), observation.bits[j])
      this.temporal.learn(hh(j + 1, observation.previous[j] + 1), observation.bits[j])
    }
    this.previous.set(observation.bits)
    this.sequence++
  }
  snapshot(): FeatureCmState { return { position: this.position.snapshot(), temporal: this.temporal.snapshot(), previous: Array.from(this.previous), sequence: this.sequence } }
  restore(state: FeatureCmState): void {
    if (!state || !Array.isArray(state.previous) || state.previous.length !== this.d || state.previous.some(v => v !== 0 && v !== 1) ||
      !Number.isSafeInteger(state.sequence) || state.sequence < 0) throw Error('invalid feature CM state')
    this.position.restore(state.position); this.temporal.restore(state.temporal)
    this.previous.set(state.previous); this.sequence = state.sequence
  }
}

export function normalizeFeature(x: Float64Array): Float64Array {
  const copy = x.slice(); let n = 0
  for (const v of copy) n += v * v
  if (n > 0) { const factor = 1 / Math.sqrt(n); for (let j = 0; j < copy.length; j++) copy[j] *= factor }
  return copy
}

export class OnlineLinear {
  readonly weights: Float64Array
  constructor(readonly d: number, readonly classes = 10) { this.weights = new Float64Array(classes * (d + 1)) }
  probabilities(x: Float64Array): Float64Array {
    if (x.length !== this.d || !x.every(Number.isFinite)) throw Error('invalid features')
    const logits = new Float64Array(this.classes)
    for (let c = 0; c < this.classes; c++) {
      let z = this.weights[c * (this.d + 1) + this.d]
      for (let j = 0; j < this.d; j++) z += this.weights[c * (this.d + 1) + j] * x[j]
      logits[c] = z
    }
    const max = Math.max(...logits); let sum = 0
    for (let c = 0; c < this.classes; c++) { logits[c] = Math.exp(logits[c] - max); sum += logits[c] }
    for (let c = 0; c < this.classes; c++) logits[c] /= sum
    return logits
  }
  update(x: Float64Array, label: number, probs: Float64Array, rate: number): void {
    if (!Number.isInteger(label) || label < 0 || label >= this.classes || !Number.isFinite(rate) || rate < 0 || probs.length !== this.classes) throw Error('invalid online update')
    for (let c = 0; c < this.classes; c++) {
      const delta = rate * ((c === label ? 1 : 0) - probs[c]), off = c * (this.d + 1)
      for (let j = 0; j < this.d; j++) this.weights[off + j] += delta * x[j]
      this.weights[off + this.d] += delta
    }
  }
}
export const argmax = (a: Float64Array): number => { let best = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i; return best }

// Fixed before evaluation: mistakes get larger rates; predictable correct samples get smaller ones.
// H is predictive entropy, S is the observed feature's self-information (bits/feature).
export function entropyGate(correct: boolean, entropy: number, surprise: number): number {
  if (![entropy, surprise].every(Number.isFinite) || entropy < 0 || entropy > 1 || surprise < 0) throw Error('invalid CM scores')
  const novelty = Math.min(1, surprise / 2)
  return correct ? 0.05 + 0.2 * entropy * novelty : 0.5 + 0.5 * novelty
}

export interface StreamAgentState { format: 'ucz-stream-agent-v1'; d: number; mode: 'baseline' | 'entropy-gated'; rate: number; weights: number[]; cm: FeatureCmState | null }

/** One sample must be predicted and scored externally before feedback is supplied. */
export class StreamAgent {
  readonly readout: OnlineLinear
  readonly cm: FeatureContextMixer | null
  private pending: { x: Float64Array; probs: Float64Array; prediction: number; observation: CmObservation | null } | null = null
  constructor(readonly d: number, readonly mode: 'baseline' | 'entropy-gated', readonly rate = 0.3) {
    if (!Number.isInteger(d) || d < 1 || (mode !== 'baseline' && mode !== 'entropy-gated') || !Number.isFinite(rate) || rate <= 0) throw Error('invalid agent settings')
    this.readout = new OnlineLinear(d)
    this.cm = mode === 'entropy-gated' ? new FeatureContextMixer(d) : null
  }
  snapshot(): StreamAgentState {
    if (this.pending) throw Error('cannot snapshot before feedback')
    return { format: 'ucz-stream-agent-v1', d: this.d, mode: this.mode, rate: this.rate,
      weights: Array.from(this.readout.weights), cm: this.cm?.snapshot() ?? null }
  }
  static restore(state: StreamAgentState): StreamAgent {
    if (!state || state.format !== 'ucz-stream-agent-v1') throw Error('invalid agent state')
    const agent = new StreamAgent(state.d, state.mode, state.rate)
    if (!Array.isArray(state.weights) || state.weights.length !== agent.readout.weights.length || !state.weights.every(Number.isFinite) ||
      ((agent.cm === null) !== (state.cm === null))) throw Error('invalid agent state')
    agent.readout.weights.set(state.weights)
    if (agent.cm) agent.cm.restore(state.cm!)
    return agent
  }
  predict(feature: Float64Array) {
    if (this.pending) throw Error('feedback required before next prediction')
    const x = normalizeFeature(feature), probs = this.readout.probabilities(x), prediction = argmax(probs)
    const observation = this.cm?.observe(x) ?? null
    this.pending = { x, probs, prediction, observation }
    return { prediction, probabilities: probs.slice(), entropy: observation?.entropy ?? null, surprise: observation?.surprise ?? null }
  }
  feedback(label: number) {
    const p = this.pending
    if (!p) throw Error('predict before feedback')
    if (!Number.isInteger(label) || label < 0 || label >= this.readout.classes) throw Error('invalid label')
    const correct = p.prediction === label
    const gate = p.observation ? entropyGate(correct, p.observation.entropy, p.observation.surprise) : 1
    this.readout.update(p.x, label, p.probs, this.rate * gate)
    if (p.observation) this.cm!.learn(p.observation)
    this.pending = null
    return { correct, gate }
  }
}
