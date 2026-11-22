// SPDX-License-Identifier: MPL-2.0
import { rebuildWeightBounds, type Layer } from './snn'
import { splitmix32 } from './rand'

/** Shift the encoded binary event stream on the 24x24 canvas, with optional ±1 time-step jitter.
 * Clipped boundaries/collisions can remove events. The original buffer is never mutated. */
export function augmentEvents(frames: Uint8Array, T: number, H: number, W: number,
  dx: number, dy: number, jitter: number, seed: number): Uint8Array {
  if (![T, H, W].every(v => Number.isInteger(v) && v > 0) || frames.length !== T * H * W || !Number.isInteger(dx) || Math.abs(dx) > 1 ||
    !Number.isInteger(dy) || Math.abs(dy) > 1 || !Number.isFinite(jitter) || jitter < 0 || jitter > 1)
    throw Error('invalid event augmentation')
  const out = new Uint8Array(frames.length), rng = splitmix32(seed)
  for (let t = 0; t < T; t++) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (!frames[(t * H + y) * W + x]) continue
    const nx = x + dx, ny = y + dy
    if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue
    let nt = t
    if (rng() < jitter) nt += rng() < 0.5 ? -1 : 1
    if (nt < 0 || nt >= T) continue
    out[(nt * H + ny) * W + nx] = 1
  }
  return out
}

/** Threshold-only IP, algebraically identical to LocalPlasticity with weight learningRate=0. */
export class IntrinsicHomeostasis {
  private counts: Uint32Array[]
  private steps: Uint32Array
  constructor(readonly layers: Layer[], readonly target = 0.05, readonly rate = 0.12, readonly limit = 0.4) {
    if (![target, rate, limit].every(Number.isFinite) || target < 0 || target > 1 || rate < 0 || limit < 0 || limit >= 1)
      throw Error('invalid intrinsic plasticity parameters')
    this.counts = layers.map(l => new Uint32Array(l.spec.outCh))
    this.steps = new Uint32Array(layers.length)
  }
  observe = (li: number, _input: Uint8Array, output: Uint8Array): void => {
    const ch = this.layers[li].spec.outCh
    this.steps[li]++
    for (let i = 0; i < output.length; i++) this.counts[li][i % ch] += output[i]
  }
  commit(): void {
    this.layers.forEach((l, li) => {
      const total = this.steps[li] * l.outH * l.outW
      if (total) for (let c = 0; c < l.spec.outCh; c++) {
        const limit = this.limit * l.spec.thresh
        l.thresholdOffsets[c] = Math.max(-limit, Math.min(limit, l.thresholdOffsets[c] +
          this.rate * l.spec.thresh * (this.counts[li][c] / total - this.target)))
      }
      this.counts[li].fill(0)
    })
    this.steps.fill(0)
  }
}

export interface PrototypeStats { positive: number; negative: number; silentTargets: number; meanPeak: number }

/** Label-assigned L2 prototypes. The first 30 channels form ten disjoint triples; 30/31 are reserved.
 * On one sample, each assigned channel uses its peak-spike spatial position as the local winner.
 * Other assigned-class channels whose peak exceeds the mean peak receive anti-Hebbian updates.
 * Only simultaneous pre-spikes on winner post-spike steps contribute. No readout is consulted. */
export class PrototypeContrastive {
  constructor(readonly l1: Layer, readonly l2: Layer, readonly learningRate = 0.03, readonly antiScale = 0.25) {
    if (l2.spec.outCh !== 32 || l2.spec.inCh !== l1.spec.outCh || l2.spec.inH !== l1.outH ||
      l2.spec.inW !== l1.outW || !Number.isFinite(learningRate) || learningRate < 0 || learningRate > 0.5 ||
      !Number.isFinite(antiScale) || antiScale < 0 || antiScale > 1) throw Error('invalid prototype setup')
  }
  apply(label: number, pre: Uint8Array, post: Uint8Array, T: number): PrototypeStats {
    const l = this.l2, s = l.spec, ch = s.outCh, positions = l.outH * l.outW, preCh = this.l1.spec.outCh
    if (!Number.isInteger(label) || label < 0 || label >= 10 || !Number.isInteger(T) || T < 1 || T > 65535 ||
      pre.length !== T * this.l1.N || post.length !== T * l.N) throw Error('invalid prototype observation')
    const counts = new Uint16Array(positions * ch)
    for (let t = 0; t < T; t++) for (let i = 0; i < l.N; i++) counts[i] += post[t * l.N + i]
    const peaks = new Uint16Array(30), at = new Int32Array(30)
    for (let p = 0; p < positions; p++) for (let oc = 0; oc < 30; oc++) {
      const n = counts[p * ch + oc]
      if (n > peaks[oc]) { peaks[oc] = n; at[oc] = p }
    }
    let mean = 0
    for (const n of peaks) mean += n
    mean /= peaks.length
    const result: PrototypeStats = { positive: 0, negative: 0, silentTargets: 0, meanPeak: mean }
    const k = s.k, kk = k * k, fanin = l.fanin
    for (let oc = 0; oc < 30; oc++) {
      const target = Math.floor(oc / 3) === label
      if (target && !peaks[oc]) { result.silentTargets++; continue }
      if (!target && !(peaks[oc] > mean)) continue
      if (!peaks[oc]) continue
      const p = at[oc], oy = Math.floor(p / l.outW), ox = p % l.outW
      const y0 = oy * s.stride - s.pad, x0 = ox * s.stride - s.pad
      const trace = new Float64Array(fanin)
      for (let t = 0; t < T; t++) {
        if (!post[t * l.N + p * ch + oc]) continue
        for (let ic = 0; ic < s.inCh; ic++) for (let ky = 0; ky < k; ky++) {
          const y = y0 + ky
          if (y < 0 || y >= s.inH) continue
          for (let kx = 0; kx < k; kx++) {
            const x = x0 + kx
            if (x >= 0 && x < s.inW)
              trace[ic * kk + ky * k + kx] += pre[t * this.l1.N + (y * s.inW + x) * preCh + ic]
          }
        }
      }
      let avg = 0, norm = 0
      const base = oc * fanin
      for (let j = 0; j < fanin; j++) { trace[j] /= peaks[oc]; avg += trace[j]; norm += l.W[base + j] ** 2 }
      avg /= fanin
      if (norm <= 1e-20) continue
      const rate = (target ? 1 : -this.antiScale) * this.learningRate * Math.sqrt(norm / fanin)
      let next = 0
      for (let j = 0; j < fanin; j++) {
        l.W[base + j] += rate * (trace[j] - avg)
        next += l.W[base + j] ** 2
      }
      if (next > 1e-20) {
        const scale = Math.sqrt(norm / next)
        for (let j = 0; j < fanin; j++) l.W[base + j] *= scale
      }
      if (target) result.positive++; else result.negative++
    }
    rebuildWeightBounds(l)
    return result
  }
}
