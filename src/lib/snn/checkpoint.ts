// SPDX-License-Identifier: MPL-2.0
// Dense-v1 and CM-v2 inference snapshots; neither stores the plasticity trainer.
import { Net } from './experiment'
import { DEFAULT_PRED, type PredConfig, type PredictorState } from './predict'
import { rebuildWeightBounds, type LayerSpec } from './snn'
import type { Readout } from './task'

export interface DenseCheckpoint {
  format: 'ucz-snn-dense-v1'
  steps: number
  pool: number
  /** false/absent preserves the original interleaved bridge. */
  channelMajorBridge?: boolean
  layers: { spec: LayerSpec; weights: number[]; thresholdOffsets: number[] }[]
  readout: { classes: number; d: number; weights: number[]; mean: number[]; inv: number[]; l2Norm: boolean }
}

export function saveDenseCheckpoint(net: Net, readout: Readout, steps: number, pool: number, channelMajorBridge = false): DenseCheckpoint {
  return { ...(channelMajorBridge ? { channelMajorBridge: true } : {}), format: 'ucz-snn-dense-v1', steps, pool,
    layers: net.layers.map(l => ({ spec: { ...l.spec }, weights: Array.from(l.W), thresholdOffsets: Array.from(l.thresholdOffsets) })),
    readout: { classes: readout.classes, d: readout.d, weights: Array.from(readout.W),
      mean: Array.from(readout.mean), inv: Array.from(readout.inv), l2Norm: readout.l2Norm } }
}

/** Validate dimensions and finite numbers before copying. CM must be trained again separately. */
export function loadDenseCheckpoint(cp: DenseCheckpoint, pred: PredConfig = DEFAULT_PRED, seed = 777): { net: Net; readout: Readout } {
  if (!cp || cp.format !== 'ucz-snn-dense-v1' || !Number.isInteger(cp.steps) || cp.steps < 1 ||
      !Number.isInteger(cp.pool) || cp.pool < 1 || (cp.channelMajorBridge !== undefined && typeof cp.channelMajorBridge !== 'boolean') || !Array.isArray(cp.layers) || !cp.layers.length) throw Error('invalid checkpoint header')
  const validArray = (v: number[], n: number) => Array.isArray(v) && v.length === n && v.every(Number.isFinite)
  const validFloat32 = (v: number[]) => v.every(x => Math.abs(x) <= 3.4028234663852886e38)
  for (let i = 0; i < cp.layers.length; i++) {
    const layer = cp.layers[i], s = layer.spec
    if (!s || ![s.inCh, s.outCh, s.inH, s.inW, s.k, s.stride].every(x => Number.isInteger(x) && x > 0) ||
      !Number.isInteger(s.pad) || s.pad < 0 || !Number.isFinite(s.scale) || s.scale < 0 ||
      !Number.isFinite(s.decay) || s.decay < 0 || s.decay > 1 || !Number.isFinite(s.thresh) || s.thresh <= 0 ||
      !Number.isInteger(s.seed) || typeof s.hardReset !== 'boolean' ||
      !validArray(layer.weights, s.outCh * s.inCh * s.k * s.k) || !validFloat32(layer.weights) ||
      !validArray(layer.thresholdOffsets, s.outCh) || !validFloat32(layer.thresholdOffsets) ||
      layer.thresholdOffsets.some(v => v + s.thresh <= 0)) throw Error('invalid checkpoint layer')
    if (i > 0) {
      const prev = cp.layers[i - 1].spec
      const oh = Math.floor((prev.inH + 2 * prev.pad - prev.k) / prev.stride) + 1
      const ow = Math.floor((prev.inW + 2 * prev.pad - prev.k) / prev.stride) + 1
      if (s.inCh !== prev.outCh || s.inH !== oh || s.inW !== ow) throw Error('disconnected checkpoint layers')
    }
  }
  const net = new Net(cp.layers.map(l => l.spec), pred, seed)
  cp.layers.forEach((saved, i) => {
    net.layers[i].W.set(saved.weights); net.layers[i].thresholdOffsets.set(saved.thresholdOffsets)
    rebuildWeightBounds(net.layers[i])
  })
  const r = cp.readout, d = net.featureDim(cp.pool)
  if (!r || !Number.isInteger(r.classes) || r.classes < 1 || r.d !== d || typeof r.l2Norm !== 'boolean' ||
    !validArray(r.weights, r.classes * (d + 1)) || !validArray(r.mean, d) || !validArray(r.inv, d) || r.inv.some(x => x <= 0))
    throw Error('invalid checkpoint readout')
  const readout: Readout = { classes: r.classes, d, W: Float64Array.from(r.weights),
    mean: Float64Array.from(r.mean), inv: Float64Array.from(r.inv), l2Norm: r.l2Norm }
  return { net, readout }
}

function validPredConfig(config: PredConfig): boolean {
  return !!config && Number.isInteger(config.bits) && config.bits >= 1 && config.bits <= 20 &&
    Number.isInteger(config.mixerSets) && config.mixerSets >= 1 && config.mixerSets <= 1024 &&
    Number.isInteger(config.apmCtx) && config.apmCtx >= 1 && config.apmCtx <= 1024 &&
    typeof config.apmOn === 'boolean' && typeof config.stateContexts === 'boolean'
}

/** Frozen predictor state at sample boundary. This is an inference model, not a training resume file. */
export interface CmCheckpoint {
  format: 'ucz-snn-cm-v2'
  dense: DenseCheckpoint
  encoding: { type: 'poisson'; seed: number }
  cm: { group: number; budget: number; config: PredConfig; predictors: PredictorState[] }
}
export type InferenceCheckpoint = DenseCheckpoint | CmCheckpoint

export function saveCmCheckpoint(net: Net, readout: Readout, steps: number, pool: number,
  group: number, budget: number, seed: number, channelMajorBridge = false): CmCheckpoint {
  if (!Number.isInteger(group) || group < 1 || !Number.isFinite(budget) || budget <= 0 || budget > 1 ||
    !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw Error('invalid CM checkpoint settings')
  const preds = net.predsFor(group)
  if (!validPredConfig(preds[0].cfg)) throw Error('unsupported CM predictor configuration')
  if (preds.some(p => p.seen === 0)) throw Error('cannot save an untrained CM predictor')
  return { format: 'ucz-snn-cm-v2', dense: saveDenseCheckpoint(net, readout, steps, pool, channelMajorBridge),
    encoding: { type: 'poisson', seed },
    cm: { group, budget, config: { ...preds[0].cfg }, predictors: preds.map(p => p.snapshot()) } }
}

export function loadInferenceCheckpoint(cp: InferenceCheckpoint) {
  if (!cp || cp.format === 'ucz-snn-dense-v1') {
    const loaded = loadDenseCheckpoint(cp as DenseCheckpoint)
    return { ...loaded, steps: (cp as DenseCheckpoint).steps, pool: (cp as DenseCheckpoint).pool, channelMajorBridge: !!(cp as DenseCheckpoint).channelMajorBridge, cm: null, seed: null }
  }
  if (cp.format !== 'ucz-snn-cm-v2' || !cp.cm || !cp.encoding || cp.encoding.type !== 'poisson' ||
    !Number.isInteger(cp.encoding.seed) || cp.encoding.seed < 0 || cp.encoding.seed > 0xffffffff) throw Error('invalid CM checkpoint header')
  const { config, group, budget, predictors } = cp.cm
  if (!validPredConfig(config) ||
    !Number.isInteger(group) || group < 1 || !Number.isFinite(budget) || budget <= 0 || budget > 1 ||
    !Array.isArray(predictors) || !cp.dense || !Array.isArray(cp.dense.layers) || predictors.length !== cp.dense.layers.length ||
    cp.dense.layers.some(l => !l || !l.spec || l.spec.outCh % group !== 0)) throw Error('invalid CM checkpoint configuration')
  const loaded = loadDenseCheckpoint(cp.dense, config, cp.encoding.seed)
  const preds = loaded.net.predsFor(group)
  preds.forEach((p, i) => p.restore(predictors[i]))
  return { ...loaded, steps: cp.dense.steps, pool: cp.dense.pool, channelMajorBridge: !!cp.dense.channelMajorBridge, cm: { group, budget }, seed: cp.encoding.seed }
}

/** Dense logits for a single feature vector; does not modify the caller's buffer. */
export function denseLogits(readout: Readout, features: Float64Array): Float64Array {
  if (features.length !== readout.d) throw Error('invalid feature dimension')
  const x = features.slice()
  if (readout.l2Norm) {
    let norm = 0
    for (const v of x) norm += v * v
    if (norm > 0) {
      const inv = 1 / Math.sqrt(norm)
      for (let j = 0; j < x.length; j++) x[j] *= inv
    }
  }
  const scores = new Float64Array(readout.classes), d = readout.d
  for (let c = 0; c < readout.classes; c++) {
    let s = readout.W[c * (d + 1) + d]
    for (let j = 0; j < d; j++) s += readout.W[c * (d + 1) + j] * ((x[j] - readout.mean[j]) * readout.inv[j])
    scores[c] = s
  }
  return scores
}
