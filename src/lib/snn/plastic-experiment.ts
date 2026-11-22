// SPDX-License-Identifier: MPL-2.0
import { DEFAULT_EXP, Net, defaultLayers } from './experiment'
import { LocalPlasticity, DEFAULT_PLASTIC, type PlasticConfig } from './plastic'
import { l2NormalizeRows, makeDigitsDataset, readoutAccuracy, trainReadout } from './task'
import { splitmix32 } from './rand'

export interface PlasticExperimentConfig {
  nTrain: number; nTest: number; T: number; seed: number; epochs: number; pool: number
  plastic: PlasticConfig
}
export const DEFAULT_PLASTIC_EXPERIMENT: PlasticExperimentConfig = {
  nTrain: 500, nTest: 200, T: 20, seed: DEFAULT_EXP.seed, epochs: 1, pool: 3,
  plastic: DEFAULT_PLASTIC,
}

/** Same data, architecture, readout, predictor teacher and gate for both arms.
 * Plasticity sees only training samples. All weights and thresholds freeze before feature extraction.
 * The readout is supervised softmax, but its error is never propagated to either SNN layer. */
export function runPlasticDigits(cfg: PlasticExperimentConfig = DEFAULT_PLASTIC_EXPERIMENT) {
  for (const key of ['nTrain', 'nTest', 'T', 'epochs', 'pool'] as const)
    if (!Number.isInteger(cfg[key]) || cfg[key] < 1) throw Error(`${key} must be a positive integer`)
  if (!Number.isInteger(cfg.seed) || cfg.seed < 0 || cfg.seed > 0xffffffff) throw Error('seed must be a uint32')
  const H = 24, W = 24, group = 8
  const data = makeDigitsDataset({ nTrain: cfg.nTrain, nTest: cfg.nTest, T: cfg.T, H, W, encoding: 'poisson', seed: cfg.seed })
  if (!data) throw Error('digits-8x8.csv.gz unavailable')
  const specs = defaultLayers(H, W, true, { in1: 1, ch1: 16, ch2: 32, k2: 5 })
  const arms = []
  for (const plastic of [false, true]) {
    const net = new Net(specs, DEFAULT_EXP.pred, cfg.seed)
    const initial = net.layers.map(l => l.W.slice())
    const learner = new LocalPlasticity(net.layers, cfg.plastic)
    let trainingMacs = 0, trainingSteps = 0
    if (plastic) for (let epoch = 0; epoch < cfg.epochs; epoch++) for (const sm of data.train) {
      const r = net.run(sm.frames, cfg.T, { mode: 'dense', budgetFrac: 1, usePredictor: false,
        trainPredictor: false, record: false, observeSpikes: learner.observe })
      trainingMacs += r.budget.macs; trainingSteps += r.budget.neuronSteps
      learner.commit()
    }
    const weightRmsChange = net.layers.map((l, li) => {
      let sum = 0
      for (let j = 0; j < l.W.length; j++) sum += (l.W[j] - initial[li][j]) ** 2
      return Math.sqrt(sum / l.W.length)
    })
    // Predictor is trained only after the SNN is frozen, using identical dense teacher policy.
    net.ensureGroups([group])
    for (const sm of data.train) net.run(sm.frames, cfg.T, { mode: 'dense', budgetFrac: 1,
      usePredictor: true, trainPredictor: true, trainFiber: true, record: false })
    const d = net.featureDim(cfg.pool), X = new Float64Array(cfg.nTrain * d), y = new Int32Array(cfg.nTrain)
    data.train.forEach((sm, i) => {
      const r = net.run(sm.frames, cfg.T, { mode: 'dense', budgetFrac: 1, usePredictor: false,
        trainPredictor: false, record: true, pool: cfg.pool })
      X.set(r.features, i * d); y[i] = sm.label
    })
    l2NormalizeRows(X, cfg.nTrain, d)
    const readout = trainReadout(X, y, cfg.nTrain, d, data.classes, 800, 0.8, 1e-4, 12345, true)
    const labels = Int32Array.from(data.test, s => s.label)
    const ref: Uint8Array[][] = []
    const denseX = new Float64Array(cfg.nTest * d)
    let denseMacs = 0, denseSpikes = 0, denseSteps = 0
    data.test.forEach((sm, i) => {
      const r = net.run(sm.frames, cfg.T, { mode: 'dense', budgetFrac: 1, usePredictor: false,
        trainPredictor: false, record: true, pool: cfg.pool })
      denseX.set(r.features, i * d); ref.push(r.spikes)
      denseMacs += r.budget.macs; denseSpikes += r.budget.spikes; denseSteps += r.budget.neuronSteps
    })
    const denseAccuracy = readoutAccuracy(readout, denseX, labels, cfg.nTest)
    const rows = []
    for (const strategy of [
      { mode: 'event' as const, frac: 1 },
      { mode: 'cm-decay' as const, frac: 0.7 },
      { mode: 'cm-decay' as const, frac: 0.5 },
      { mode: 'random' as const, frac: 0.7 },
    ]) {
      const feats = new Float64Array(cfg.nTest * d)
      let macs = 0, ops = 0, computed = 0, steps = 0, miss = 0, extra = 0, truth = 0
      const rng = splitmix32(98)
      data.test.forEach((sm, i) => {
        const r = net.run(sm.frames, cfg.T, { mode: strategy.mode, budgetFrac: strategy.frac,
          usePredictor: strategy.mode !== 'event', trainPredictor: false, record: true, pool: cfg.pool, group, rng })
        macs += r.budget.macs; ops += r.budget.macs + r.budget.gateOps + r.budget.predOps
        computed += r.budget.computed; steps += r.budget.neuronSteps; feats.set(r.features, i * d)
        for (let li = 0; li < net.layers.length; li++) for (let j = 0; j < r.spikes[li].length; j++) {
          const a = ref[i][li][j], b = r.spikes[li][j]
          truth += a; if (a && !b) miss++; else if (!a && b) extra++
        }
      })
      rows.push({ mode: strategy.mode, budget: strategy.frac, accuracy: readoutAccuracy(readout, feats, labels, cfg.nTest),
        macRatio: macs / denseMacs, opsRatio: ops / denseMacs, candidateFraction: computed / steps,
        missedSpikes: miss, extraSpikes: extra, trueSpikes: truth })
    }
    arms.push({ plastic, denseAccuracy, denseMacs, denseSpikeRate: denseSpikes / denseSteps,
      weightRmsChange, thresholdOffsets: net.layers.map(l => Array.from(l.thresholdOffsets)),
      plasticTraining: { denseMacs: trainingMacs, neuronSteps: trainingSteps, localUpdateOps: null }, rows })
  }
  return { config: cfg, protocol: 'fixed disjoint split; poisson; frozen SNN and dense-trained supervised readout; dense-teacher CM', arms }
}
