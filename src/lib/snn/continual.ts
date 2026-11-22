// SPDX-License-Identifier: MPL-2.0
// Bounded per-class representations: no past examples or replay buffer.
import { FeatureContextMixer, entropyGate, normalizeFeature, type FeatureCmState, type CmObservation } from './stream'

const CLASSES = 10
export const PROTOTYPE_ETA_BASE = 0.03
export const PROTOTYPE_ETA_NOVEL = 0.12
export const ANCHOR_RATE = 0.15
export const BOUNDARY_RATE = 0.12
export const BOUNDARY_MARGIN = 0.15
export const READOUT_RATE = 0.3
export const COSINE_SCALE = 5

function dot(a: Float64Array, b: Float64Array): number { let v = 0; for (let i = 0; i < a.length; i++) v += a[i] * b[i]; return v }
function vector(x: Float64Array, d: number): Float64Array {
  if (x.length !== d || !x.every(Number.isFinite)) throw Error('invalid features')
  return normalizeFeature(x)
}
function labelOk(y: number) { return Number.isInteger(y) && y >= 0 && y < CLASSES }

export function prototypeEta(surprise: number): number {
  if (!Number.isFinite(surprise) || surprise < 0) throw Error('invalid surprise')
  return PROTOTYPE_ETA_BASE + PROTOTYPE_ETA_NOVEL * Math.min(1, surprise / 2)
}

export class PrototypeBank {
  readonly means: Float64Array[]
  readonly counts: Uint32Array
  constructor(readonly d: number) {
    if (!Number.isInteger(d) || d < 1) throw Error('invalid feature dimension')
    this.means = Array.from({ length: CLASSES }, () => new Float64Array(d)); this.counts = new Uint32Array(CLASSES)
  }
  update(y: number, x: Float64Array, surprise: number): number {
    if (!labelOk(y) || x.length !== this.d || !x.every(Number.isFinite) || !x.some(v => v !== 0)) throw Error('invalid prototype input')
    const eta = prototypeEta(surprise), mu = this.means[y]
    if (this.counts[y] === 0) mu.set(x)
    else for (let j = 0; j < this.d; j++) mu[j] += eta * (x[j] - mu[j])
    if (this.counts[y] < 0xffffffff) this.counts[y]++
    return eta
  }
  scores(x: Float64Array): Float64Array {
    const scores = new Float64Array(CLASSES).fill(-Infinity)
    for (let k = 0; k < CLASSES; k++) if (this.counts[k]) scores[k] = dot(x, normalizeFeature(this.means[k]))
    return scores
  }
}
export function bestKnown(scores: Float64Array): number {
  let best = 0
  for (let k = 1; k < scores.length; k++) if (scores[k] > scores[best]) best = k
  return best // deterministic class 0 when no classes have been seen
}

/** Current-label-only row updates; boundary correction uses a competing class's prototype, not its row. */
export class AnchoredReadout {
  readonly rows: Float64Array[]
  constructor(readonly d: number) { this.rows = Array.from({ length: CLASSES }, () => new Float64Array(d)) }
  scores(x: Float64Array, bank: PrototypeBank): Float64Array {
    const out = new Float64Array(CLASSES).fill(-Infinity)
    for (let c = 0; c < CLASSES; c++) if (bank.counts[c]) out[c] = COSINE_SCALE * dot(x, this.rows[c])
    return out
  }
  update(y: number, x: Float64Array, bank: PrototypeBank, gate: number): void {
    if (!labelOk(y) || !bank.counts[y] || x.length !== this.d || !x.every(Number.isFinite) || !Number.isFinite(gate) || gate < 0 || gate > 1) throw Error('invalid anchored update')
    const row = this.rows[y], target = normalizeFeature(bank.means[y])
    // Existing scores are calculated after label feedback, but before the row update.
    const logits = this.scores(x, bank), max = Math.max(...logits)
    let sum = 0
    for (const z of logits) if (Number.isFinite(z)) sum += Math.exp(z - max)
    const py = Math.exp(logits[y] - max) / sum
    let competitor = -1, best = -Infinity
    for (let c = 0; c < CLASSES; c++) if (c !== y && bank.counts[c]) {
      const score = dot(x, normalizeFeature(bank.means[c]))
      if (score > best) { best = score; competitor = c }
    }
    const other = competitor < 0 ? null : normalizeFeature(bank.means[competitor])
    const margin = other ? dot(row, target) - dot(row, other) : Infinity
    for (let j = 0; j < this.d; j++) {
      row[j] += gate * (READOUT_RATE * (1 - py) * x[j] + ANCHOR_RATE * (target[j] - row[j]) +
        (other && margin < BOUNDARY_MARGIN ? BOUNDARY_RATE * (target[j] - other[j]) : 0))
    }
    const normalized = normalizeFeature(row); row.set(normalized)
  }
}

export type IsolatedMode = 'ncm' | 'anchored'
export interface IsolatedAgentState {
  format: 'ucz-isolated-agent-v1'; d: number; mode: IsolatedMode; means: number[][]; counts: number[];
  rows: number[][] | null; cm: FeatureCmState
}
/** Sample boundary is transactional: predict() never changes state, feedback() updates only its label. */
export class IsolatedAgent {
  readonly bank: PrototypeBank
  readonly readout: AnchoredReadout | null
  readonly cm: FeatureContextMixer
  private pending: { x: Float64Array; prediction: number; observation: CmObservation } | null = null
  constructor(readonly d: number, readonly mode: IsolatedMode) {
    if (mode !== 'ncm' && mode !== 'anchored') throw Error('invalid agent mode')
    this.bank = new PrototypeBank(d); this.readout = mode === 'anchored' ? new AnchoredReadout(d) : null
    this.cm = new FeatureContextMixer(d)
  }
  predict(feature: Float64Array) {
    if (this.pending) throw Error('feedback required before next prediction')
    const x = vector(feature, this.d)
    if (!x.some(v => v !== 0)) throw Error('zero feature vector')
    const scores = this.readout ? this.readout.scores(x, this.bank) : this.bank.scores(x)
    const prediction = bestKnown(scores), observation = this.cm.observe(x)
    this.pending = { x, prediction, observation }
    return { prediction, entropy: observation.entropy, surprise: observation.surprise }
  }
  feedback(y: number) {
    const p = this.pending
    if (!p) throw Error('predict before feedback')
    if (!labelOk(y)) throw Error('invalid label')
    const correct = p.prediction === y, gate = entropyGate(correct, p.observation.entropy, p.observation.surprise)
    const eta = this.bank.update(y, p.x, p.observation.surprise)
    if (this.readout) this.readout.update(y, p.x, this.bank, gate)
    this.cm.learn(p.observation); this.pending = null
    return { correct, gate: this.readout ? gate : null, eta }
  }
  /** Read-only evaluation: does not touch CM, prototypes, readout, or pending sample. */
  classify(feature: Float64Array): number {
    const x = vector(feature, this.d)
    return bestKnown(this.readout ? this.readout.scores(x, this.bank) : this.bank.scores(x))
  }
  snapshot(): IsolatedAgentState {
    if (this.pending) throw Error('cannot snapshot before feedback')
    return { format: 'ucz-isolated-agent-v1', d: this.d, mode: this.mode, means: this.bank.means.map(v => Array.from(v)),
      counts: Array.from(this.bank.counts), rows: this.readout ? this.readout.rows.map(v => Array.from(v)) : null, cm: this.cm.snapshot() }
  }
  static restore(s: IsolatedAgentState): IsolatedAgent {
    if (!s || s.format !== 'ucz-isolated-agent-v1') throw Error('invalid isolated agent state')
    const a = new IsolatedAgent(s.d, s.mode)
    const matrix = (m: number[][] | null) => Array.isArray(m) && m.length === CLASSES && m.every(row => Array.isArray(row) && row.length === s.d && row.every(Number.isFinite))
    if (!matrix(s.means) || !Array.isArray(s.counts) || s.counts.length !== CLASSES || s.counts.some(v => !Number.isInteger(v) || v < 0 || v > 0xffffffff) ||
      (a.readout ? !matrix(s.rows) : s.rows !== null)) throw Error('invalid isolated agent state')
    a.bank.means.forEach((v, i) => v.set(s.means[i])); a.bank.counts.set(s.counts)
    if (a.readout) a.readout.rows.forEach((v, i) => v.set(s.rows![i]))
    a.cm.restore(s.cm)
    return a
  }
}
