// SPDX-License-Identifier: MPL-2.0
// Two-bit structural residual on top of the identical fixed-phase v2 model.
import { MultiscaleByteModel, type MultiscaleState } from './multiscale-byte'
import type { BitModifier } from './spike-byte'

const PRIOR = 6
const GAIN = 2
const LIMIT = 1
const CELLS = 256 * 4

export interface CoarseState {
  format: 'ucz-coarse-byte-v3'; phaseEnabled: boolean; baseline: MultiscaleState
  n: number[]; gradient: number[]; phaseHistogram: number[]
}
export class CoarseByteModel implements BitModifier {
  private baseline = new MultiscaleByteModel(false)
  private n = new Uint32Array(CELLS)
  private gradient = new Float64Array(CELLS)
  private phaseHistogram = new Uint32Array(4)
  constructor(readonly phaseEnabled = true) {}
  get bytesSeen() { return this.baseline.bytesSeen }
  private phase() { return this.phaseEnabled ? this.baseline.coarsePhase() : 0 }
  private cell(prefix: number) {
    if (!Number.isInteger(prefix) || prefix < 1 || prefix > 255) throw Error('invalid prefix')
    return prefix * 4 + this.phase()
  }
  /** Smooth only a bounded logit residual against the fixed-phase base forecast. */
  predict(prefix: number, base: number): number {
    if (!Number.isFinite(base) || base <= 0 || base >= 1) throw Error('invalid base probability')
    const cell = this.cell(prefix), delta = Math.max(-LIMIT, Math.min(LIMIT, GAIN * this.gradient[cell] / (this.n[cell] + PRIOR)))
    if (delta === 0) return base // untouched baseline for unseen or exactly balanced residual buckets
    const z = Math.log(base / (1 - base)) + delta
    return Math.max(1e-6, Math.min(1 - 1e-6, 1 / (1 + Math.exp(-z))))
  }
  learn(prefix: number, bit: number, base: number): void {
    if ((bit !== 0 && bit !== 1) || !Number.isFinite(base) || base <= 0 || base >= 1) throw Error('invalid residual observation')
    const cell = this.cell(prefix)
    this.n[cell]++; this.gradient[cell] += bit - base
    // Bound memory and give recent observations influence on long streams.
    if (this.n[cell] >= 1024) { this.n[cell] >>>= 1; this.gradient[cell] *= 0.5 }
  }
  consume(byte: number, train: boolean): number {
    const phase = this.phase(), loss = this.baseline.consume(byte, train, this)
    if (this.phaseHistogram[phase] < 0xffffffff) this.phaseHistogram[phase]++
    return loss
  }
  sample(rng: () => number, temperature = 1, clip = true): number {
    const phase = this.phase(), b = this.baseline.sample(rng, temperature, clip, this)
    if (this.phaseHistogram[phase] < 0xffffffff) this.phaseHistogram[phase]++
    return b
  }
  diagnostics() {
    let used = 0, count = 0
    for (const n of this.n) { if (n) used++; count += n }
    return { ...this.baseline.diagnostics(), phaseEnabled: this.phaseEnabled, phaseHistogram: Array.from(this.phaseHistogram),
      residualCellsUsed: used, residualObservationsAfterAging: count, prior: PRIOR, gain: GAIN, logitLimit: LIMIT }
  }
  snapshot(): CoarseState { return { format: 'ucz-coarse-byte-v3', phaseEnabled: this.phaseEnabled, baseline: this.baseline.snapshot(),
    n: Array.from(this.n), gradient: Array.from(this.gradient), phaseHistogram: Array.from(this.phaseHistogram) } }
  static restore(s: CoarseState): CoarseByteModel {
    if (!s || s.format !== 'ucz-coarse-byte-v3' || typeof s.phaseEnabled !== 'boolean' ||
      !Array.isArray(s.n) || s.n.length !== CELLS || s.n.some(v => !Number.isInteger(v) || v < 0 || v >= 1024) ||
      !Array.isArray(s.gradient) || s.gradient.length !== CELLS || s.gradient.some((v, i) => !Number.isFinite(v) || Math.abs(v) > s.n[i] + 1e-6) ||
      !Array.isArray(s.phaseHistogram) || s.phaseHistogram.length !== 4 || s.phaseHistogram.some(v => !Number.isInteger(v) || v < 0 || v > 0xffffffff)) throw Error('invalid coarse byte state')
    const v = new CoarseByteModel(s.phaseEnabled); v.baseline = MultiscaleByteModel.restore(s.baseline)
    if (v.baseline.structureActive) throw Error('coarse baseline must use fixed phase')
    v.n.set(s.n); v.gradient.set(s.gradient); v.phaseHistogram.set(s.phaseHistogram)
    return v
  }
}
