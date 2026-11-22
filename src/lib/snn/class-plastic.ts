// SPDX-License-Identifier: MPL-2.0
// Three-factor rule: pre-synaptic input, post-synaptic WTA spikes, class feedback.
import { rebuildWeightBounds, type Layer } from './snn'
import type { Readout } from './task'

export interface ClassPlasticConfig { learningRate: number; creditGain: number; pool: number }

/** Collect eligibility traces without labels. commit() receives prediction and target afterwards.
 * A true-vs-competitor comparison supplies signed credit for each channel/spatial bin using the
 * frozen readout coefficients. Positive credit strengthens, negative credit weakens the centered
 * local correlation. No gradients pass through either SNN layer. */
export class ClassModulatedPlasticity {
  private readonly sums: Float64Array[]
  private readonly wins: Uint32Array[]
  private tie = 0
  constructor(readonly layers: Layer[], readonly config: ClassPlasticConfig) {
    if (!Number.isFinite(config.learningRate) || config.learningRate < 0 || config.learningRate > 0.5 ||
        !Number.isFinite(config.creditGain) || config.creditGain <= 0 ||
        !Number.isInteger(config.pool) || config.pool < 1 || config.pool > 32) throw Error('invalid class plasticity settings')
    const bins = config.pool ** 2
    this.sums = layers.map(l => new Float64Array(l.spec.outCh * bins * l.fanin))
    this.wins = layers.map(l => new Uint32Array(l.spec.outCh * bins))
  }
  observe = (li: number, input: Uint8Array, output: Uint8Array): void => {
    const l = this.layers[li], s = l.spec, k2 = s.k * s.k, pool = this.config.pool, bins = pool * pool
    for (let p = 0; p < l.outH * l.outW; p++) {
      const start = (this.tie + p + li) % s.outCh
      let win = -1
      for (let j = 0; j < s.outCh; j++) {
        const oc = (start + j) % s.outCh
        if (output[p * s.outCh + oc]) { win = oc; break }
      }
      if (win < 0) continue
      const oy = Math.floor(p / l.outW), ox = p % l.outW
      const gy = Math.min(pool - 1, Math.floor(oy * pool / l.outH))
      const gx = Math.min(pool - 1, Math.floor(ox * pool / l.outW))
      const unit = win * bins + gy * pool + gx
      this.wins[li][unit]++
      const y0 = oy * s.stride - s.pad, x0 = ox * s.stride - s.pad
      for (let ic = 0; ic < s.inCh; ic++) for (let ky = 0; ky < s.k; ky++) {
        const y = y0 + ky
        if (y < 0 || y >= s.inH) continue
        for (let kx = 0; kx < s.k; kx++) {
          const x = x0 + kx
          if (x >= 0 && x < s.inW)
            this.sums[li][unit * l.fanin + ic * k2 + ky * s.k + kx] += input[ic * s.inH * s.inW + y * s.inW + x]
        }
      }
    }
    this.tie++
  }
  /** Compare the true class to a competing class, with a caller-supplied reward strength.
   * A matching class or zero strength produces no synaptic update. Thresholds remain frozen. */
  commit(label: number, prediction: number, readout: Readout, strength = 1): { potentiated: number; depressed: number; kernels: number } {
    if (!Number.isInteger(label) || label < 0 || label >= readout.classes ||
      !Number.isInteger(prediction) || prediction < 0 || prediction >= readout.classes ||
      !Number.isFinite(strength) || strength < 0 || strength > 1 ||
      readout.d !== this.layers.reduce((a, l) => a + l.spec.outCh * this.config.pool ** 2, 0))
      throw Error('invalid label, prediction, or readout dimensions')
    const result = { potentiated: 0, depressed: 0, kernels: 0 }
    const bins = this.config.pool ** 2
    let offset = 0
    for (let li = 0; li < this.layers.length; li++) {
      const l = this.layers[li], s = l.spec, fanin = l.fanin, sums = this.sums[li], wins = this.wins[li]
      if (label !== prediction && strength > 0 && this.config.learningRate > 0) for (let oc = 0; oc < s.outCh; oc++) {
        const delta = new Float64Array(fanin)
        let total = 0
        for (let bin = 0; bin < bins; bin++) {
          const u = oc * bins + bin, n = wins[u]
          if (!n) continue
          total += n
          const j = offset + u
          const factor = Math.tanh(this.config.creditGain * readout.inv[j] *
            (readout.W[label * (readout.d + 1) + j] - readout.W[prediction * (readout.d + 1) + j]))
          if (factor > 0) result.potentiated++
          else if (factor < 0) result.depressed++
          let mean = 0
          for (let k = 0; k < fanin; k++) mean += sums[u * fanin + k] / n
          mean /= fanin
          for (let k = 0; k < fanin; k++) delta[k] += factor * (sums[u * fanin + k] - n * mean)
        }
        if (!total) continue
        const base = oc * fanin
        let norm2 = 0, delta2 = 0
        for (let k = 0; k < fanin; k++) { norm2 += l.W[base + k] ** 2; delta2 += delta[k] ** 2 }
        if (norm2 <= 1e-20 || delta2 <= 1e-20) continue
        // Scale by kernel RMS and winner count; then retain the original per-kernel norm.
        const rate = strength * this.config.learningRate * Math.sqrt(norm2 / fanin) / total
        let nextNorm2 = 0
        for (let k = 0; k < fanin; k++) { l.W[base + k] += rate * delta[k]; nextNorm2 += l.W[base + k] ** 2 }
        if (nextNorm2 > 1e-20) {
          const scale = Math.sqrt(norm2 / nextNorm2)
          for (let k = 0; k < fanin; k++) l.W[base + k] *= scale
        }
        result.kernels++
      }
      rebuildWeightBounds(l)
      sums.fill(0); wins.fill(0)
      offset += s.outCh * bins
    }
    return result
  }
}
