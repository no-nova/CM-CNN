// SPDX-License-Identifier: MPL-2.0
// Unsupervised sample-online competitive Hebbian learning. No labels or gradients.
import { rebuildWeightBounds, type Layer } from './snn'

export interface PlasticConfig {
  learningRate: number
  targetRate: number
  thresholdRate: number
  maxOffset: number
  /** Experimental: mean-center each receptive field and preserve each kernel's norm. */
  rule?: 'fixed-center' | 'balanced-norm'
}
export const DEFAULT_PLASTIC: PlasticConfig = {
  learningRate: 0.015, targetRate: 0.05, thresholdRate: 0.12, maxOffset: 0.4,
}

/** Accumulate pre/post correlations during one sample, apply only after its forward pass.
 * WTA selects one spiking output channel per position; cyclic tie-break avoids favoring channel 0.
 * An EMA of the winner's binary receptive field replaces a small portion of its weights.
 * Homeostasis uses per-channel firing rates, not class labels. */
export class LocalPlasticity {
  private sums: Float64Array[]
  private wins: Uint32Array[]
  private spikes: Uint32Array[]
  private stepCounts: Uint32Array
  private tie = 0
  constructor(readonly layers: Layer[], readonly config: PlasticConfig = DEFAULT_PLASTIC) {
    const c = config
    if (![c.learningRate, c.targetRate, c.thresholdRate, c.maxOffset].every(Number.isFinite) ||
      c.learningRate < 0 || c.learningRate > 1 || c.targetRate < 0 || c.targetRate > 1 ||
      c.thresholdRate < 0 || c.maxOffset < 0 || c.maxOffset >= 1 ||
      (c.rule !== undefined && c.rule !== 'fixed-center' && c.rule !== 'balanced-norm')) throw Error('invalid plasticity configuration')
    this.sums = layers.map(l => new Float64Array(l.W.length))
    this.wins = layers.map(l => new Uint32Array(l.spec.outCh))
    this.spikes = layers.map(l => new Uint32Array(l.spec.outCh))
    this.stepCounts = new Uint32Array(layers.length)
  }
  observe = (li: number, input: Uint8Array, output: Uint8Array): void => {
    const l = this.layers[li], s = l.spec, kk = s.k * s.k
    this.stepCounts[li]++
    for (let p = 0; p < l.outH * l.outW; p++) {
      const base = p * s.outCh
      for (let oc = 0; oc < s.outCh; oc++) this.spikes[li][oc] += output[base + oc]
      let win = -1
      const start = (this.tie + p + li) % s.outCh
      for (let j = 0; j < s.outCh; j++) {
        const oc = (start + j) % s.outCh
        if (output[base + oc]) { win = oc; break }
      }
      if (win < 0) continue
      this.wins[li][win]++
      const y0 = Math.floor(p / l.outW) * s.stride - s.pad
      const x0 = (p % l.outW) * s.stride - s.pad
      for (let ic = 0; ic < s.inCh; ic++) for (let ky = 0; ky < s.k; ky++) {
        const y = y0 + ky
        if (y < 0 || y >= s.inH) continue
        for (let kx = 0; kx < s.k; kx++) {
          const x = x0 + kx
          if (x >= 0 && x < s.inW)
            this.sums[li][(win * s.inCh + ic) * kk + ky * s.k + kx] += input[ic * s.inH * s.inW + y * s.inW + x]
        }
      }
    }
    this.tie++
  }
  /** Apply a bounded local update, then refresh certificate caches before any gated run. */
  commit(): void {
    for (let li = 0; li < this.layers.length; li++) {
      const l = this.layers[li], s = l.spec, fanin = l.fanin
      for (let oc = 0; oc < s.outCh; oc++) {
        const n = this.wins[li][oc]
        if (n && this.config.rule === 'balanced-norm') {
          // A fixed 0.2 center pushes the much sparser L2 inputs negative. Center
          // each winner's receptive field instead, and preserve this kernel's norm.
          const base = oc * fanin
          let mean = 0, oldNorm2 = 0
          for (let j = 0; j < fanin; j++) { mean += this.sums[li][base + j] / n; oldNorm2 += l.W[base + j] ** 2 }
          mean /= fanin
          let targetNorm2 = 0
          for (let j = 0; j < fanin; j++) targetNorm2 += (this.sums[li][base + j] / n - mean) ** 2
          if (targetNorm2 > 1e-20 && oldNorm2 > 1e-20) {
            const scale = Math.sqrt(oldNorm2 / targetNorm2)
            let newNorm2 = 0
            for (let j = 0; j < fanin; j++) {
              const i = base + j
              const target = scale * (this.sums[li][i] / n - mean)
              l.W[i] += this.config.learningRate * (target - l.W[i])
              newNorm2 += l.W[i] ** 2
            }
            if (newNorm2 > 1e-20) {
              const normScale = Math.sqrt(oldNorm2 / newNorm2)
              for (let j = 0; j < fanin; j++) l.W[base + j] *= normScale
            }
          }
        } else if (n) for (let j = 0; j < fanin; j++) {
          const i = oc * fanin + j
          // Original fixed-center rule, retained for a controlled ablation.
          const target = s.scale * 2 * (this.sums[li][i] / n - 0.2)
          l.W[i] += this.config.learningRate * (target - l.W[i])
        }
        const total = this.stepCounts[li] * l.outH * l.outW
        if (total) {
          const rate = this.spikes[li][oc] / total
          const limit = this.config.maxOffset * s.thresh
          l.thresholdOffsets[oc] = Math.max(-limit, Math.min(limit,
            l.thresholdOffsets[oc] + this.config.thresholdRate * s.thresh * (rate - this.config.targetRate)))
        }
      }
      rebuildWeightBounds(l)
      this.sums[li].fill(0); this.wins[li].fill(0); this.spikes[li].fill(0)
    }
    this.stepCounts.fill(0)
  }
}
