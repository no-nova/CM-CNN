// SPDX-License-Identifier: MPL-2.0
// Separate training / validation / held-out index ranges. Selection never sees held-out results.
import { DEFAULT_EXP, Net, defaultLayers, type RunOptions } from './experiment'
import { LocalPlasticity, type PlasticConfig } from './plastic'
import { saveCmCheckpoint, saveDenseCheckpoint, type DenseCheckpoint, type CmCheckpoint } from './checkpoint'
import { l2NormalizeRows, loadDigits, makeDigitsDataset, readoutAccuracy, trainReadout, type EventSample, type Readout } from './task'
import { splitmix32 } from './rand'

export interface ValidationConfig {
  nTrain: number; nValidation: number; validationStart: number; nHoldout: number; holdoutStart: number
  steps: number; seed: number; pool: number
}
export const VALIDATION_DEFAULT: ValidationConfig = {
  nTrain: 500, nValidation: 200, validationStart: 500, nHoldout: 200, holdoutStart: 1000,
  steps: 20, seed: DEFAULT_EXP.seed, pool: 3,
}
export interface Candidate { name: string; plastic: PlasticConfig | null }
// Prespecified, identical one-pass training. A simple ablation, not a claim of optimal hyperparameters.
export const CANDIDATES: Candidate[] = [
  { name: 'random', plastic: null },
  { name: 'weights-only', plastic: { learningRate: 0.015, targetRate: 0.05, thresholdRate: 0, maxOffset: 0.4 } },
  { name: 'threshold-only', plastic: { learningRate: 0, targetRate: 0.05, thresholdRate: 0.12, maxOffset: 0.4 } },
  { name: 'combined', plastic: { learningRate: 0.015, targetRate: 0.05, thresholdRate: 0.12, maxOffset: 0.4 } },
  { name: 'gentle', plastic: { learningRate: 0.003, targetRate: 0.05, thresholdRate: 0.03, maxOffset: 0.4 } },
 ]
/** Additional ablation addressing L2 silence; the baseline is still trained identically. */
export const BALANCED_CANDIDATES: Candidate[] = [
  CANDIDATES[0],
  { name: 'balanced-weights', plastic: { learningRate: 0.015, targetRate: 0.05, thresholdRate: 0, maxOffset: 0.4, rule: 'balanced-norm' } },
  { name: 'balanced-combined', plastic: { learningRate: 0.015, targetRate: 0.05, thresholdRate: 0.12, maxOffset: 0.4, rule: 'balanced-norm' } },
]
const dense: RunOptions = { mode: 'dense', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true }

function train(candidate: Candidate, samples: EventSample[], cfg: ValidationConfig) {
  const specs = defaultLayers(24, 24, true, { in1: 1, ch1: 16, ch2: 32, k2: 5 })
  const net = new Net(specs, DEFAULT_EXP.pred, cfg.seed)
  const initial = net.layers.map(l => l.W.slice())
  let plasticMacs = 0
  if (candidate.plastic) {
    const learner = new LocalPlasticity(net.layers, candidate.plastic)
    for (const sm of samples) {
      const r = net.run(sm.frames, cfg.steps, { ...dense, record: false, observeSpikes: learner.observe })
      plasticMacs += r.budget.macs
      learner.commit()
    }
  }
  const weightRmsChange = net.layers.map((l, li) => {
    let sum = 0
    for (let j = 0; j < l.W.length; j++) sum += (l.W[j] - initial[li][j]) ** 2
    return Math.sqrt(sum / l.W.length)
  })
  const d = net.featureDim(cfg.pool), X = new Float64Array(samples.length * d)
  const labels = new Int32Array(samples.length)
  samples.forEach((sm, i) => {
    const r = net.run(sm.frames, cfg.steps, { ...dense, pool: cfg.pool })
    X.set(r.features, i * d); labels[i] = sm.label
  })
  l2NormalizeRows(X, samples.length, d)
  const readout = trainReadout(X, labels, samples.length, d, 10, 800, 0.8, 1e-4, 12345, true)
  return { net, readout, weightRmsChange, plasticMacs }
}

function channelDiagnostics(net: Net, spikes: Float64Array[], nSamples: number, steps: number) {
  return net.layers.map((l, li) => {
    const rates = Array.from(spikes[li], x => x / (nSamples * steps * l.outH * l.outW))
    let sumCos = 0, pairs = 0
    for (let a = 0; a < l.spec.outCh; a++) for (let b = a + 1; b < l.spec.outCh; b++) {
      let dot = 0, aa = 0, bb = 0
      for (let j = 0; j < l.fanin; j++) {
        const x = l.W[a * l.fanin + j], y = l.W[b * l.fanin + j]
        dot += x * y; aa += x * x; bb += y * y
      }
      if (aa && bb) { sumCos += dot / Math.sqrt(aa * bb); pairs++ }
    }
    return { rates, silentBelow001: rates.filter(r => r < 0.001).length,
      meanPairwiseCosine: pairs ? sumCos / pairs : 0,
      thresholdOffsets: Array.from(l.thresholdOffsets) }
  })
}

export function evaluate(net: Net, readout: Readout, samples: EventSample[], cfg: ValidationConfig, detailed = false) {
  const d = readout.d, X = new Float64Array(samples.length * d), labels = new Int32Array(samples.length)
  const counts = net.layers.map(l => new Float64Array(l.spec.outCh))
  const refs: Uint8Array[][] = []
  let macs = 0, spikes = 0, neuronSteps = 0
  samples.forEach((sm, i) => {
    const r = net.run(sm.frames, cfg.steps, { ...dense, pool: cfg.pool })
    X.set(r.features, i * d); labels[i] = sm.label
    macs += r.budget.macs; spikes += r.budget.spikes; neuronSteps += r.budget.neuronSteps
    if (detailed) refs.push(r.spikes)
    r.spikes.forEach((arr, li) => {
      const ch = net.layers[li].spec.outCh
      for (let j = 0; j < arr.length; j++) counts[li][j % ch] += arr[j]
    })
  })
  return { accuracy: readoutAccuracy(readout, X, labels, samples.length), macs,
    spikeRate: spikes / neuronSteps, diagnostics: channelDiagnostics(net, counts, samples.length, cfg.steps), refs, labels }
}

export function gated(net: Net, readout: Readout, samples: EventSample[], cfg: ValidationConfig,
  ref: ReturnType<typeof evaluate>, mode: 'event' | 'cm-decay' | 'random', budget: number) {
  const X = new Float64Array(samples.length * readout.d), rng = splitmix32(98)
  let macs = 0, ops = 0, miss = 0, extra = 0, trueSpikes = 0
  samples.forEach((sm, i) => {
    const r = net.run(sm.frames, cfg.steps, { mode, budgetFrac: budget, usePredictor: mode !== 'event',
      trainPredictor: false, record: true, pool: cfg.pool, group: 8, rng })
    X.set(r.features, i * readout.d); macs += r.budget.macs
    ops += r.budget.macs + r.budget.gateOps + r.budget.predOps
    r.spikes.forEach((arr, li) => {
      const expected = ref.refs[i][li]
      for (let j = 0; j < arr.length; j++) {
        const a = expected[j], b = arr[j]
        trueSpikes += a; if (a && !b) miss++; else if (!a && b) extra++
      }
    })
  })
  return { mode, budget, accuracy: readoutAccuracy(readout, X, ref.labels, samples.length),
    macRatio: macs / ref.macs, opsRatio: ops / ref.macs, miss, extra, trueSpikes }
}

export interface ValidationIO {
  log: (message: string) => void
  checkpoint: (name: string, cp: DenseCheckpoint) => void
  checkpointCM?: (name: string, cp: CmCheckpoint) => void
  results: (value: unknown) => void
}

export function runValidation(cfg: ValidationConfig, io: ValidationIO, candidates: Candidate[] = CANDIDATES) {
  if (candidates.length < 2 || candidates[0].plastic !== null || candidates.slice(1).some(c => !c.plastic))
    throw Error('candidate list must start with a random control and contain plastic candidates')
  for (const key of ['nTrain', 'nValidation', 'nHoldout', 'steps', 'pool'] as const)
    if (!Number.isInteger(cfg[key]) || cfg[key] <= 0) throw Error(`${key} must be a positive integer`)
  for (const key of ['seed', 'validationStart', 'holdoutStart'] as const)
    if (!Number.isInteger(cfg[key]) || cfg[key] < 0 || cfg[key] > 0xffffffff) throw Error(`${key} must be a uint32`)
  if (cfg.validationStart < cfg.nTrain || cfg.holdoutStart < cfg.nTrain ||
    !(cfg.validationStart + cfg.nValidation <= cfg.holdoutStart || cfg.holdoutStart + cfg.nHoldout <= cfg.validationStart))
    throw Error('training, validation, and holdout ranges must be disjoint')
  const digits = loadDigits()
  if (!digits) throw Error('digits dataset not found')
  if (cfg.validationStart + cfg.nValidation > digits.n || cfg.holdoutStart + cfg.nHoldout > digits.n)
    throw Error('evaluation range exceeds digits dataset')
  const data = makeDigitsDataset({ nTrain: cfg.nTrain, nTest: cfg.nValidation, testStart: cfg.validationStart,
    T: cfg.steps, H: 24, W: 24, encoding: 'poisson', seed: cfg.seed })
  if (!data) throw Error('digits dataset not found')
  // Do not even construct the holdout until validation selection is fixed.
  const table: { name: string; config: PlasticConfig | null; accuracy: number; spikeRate: number;
    diagnostics: ReturnType<typeof channelDiagnostics>; weightRmsChange: number[]; plasticMacs: number }[] = []
  const trained: ReturnType<typeof train>[] = []
  for (const candidate of candidates) {
    io.log(`training ${candidate.name}`)
    const model = train(candidate, data.train, cfg)
    const val = evaluate(model.net, model.readout, data.test, cfg)
    trained.push(model)
    table.push({ name: candidate.name, config: candidate.plastic, accuracy: val.accuracy, spikeRate: val.spikeRate,
      diagnostics: val.diagnostics, weightRmsChange: model.weightRmsChange, plasticMacs: model.plasticMacs })
    io.results({ phase: 'validation', config: cfg, candidates: table })
    io.log(`${candidate.name}: validation ${(100 * val.accuracy).toFixed(1)}%`)
  }
  // Strict > gives deterministic tie-breaking in the prespecified list; random is the control, not a candidate.
  let chosen = 1
  for (let i = 2; i < table.length; i++) if (table[i].accuracy > table[chosen].accuracy) chosen = i
  const selected = table[chosen].name
  io.log(`selected ${selected} from validation only; evaluating holdout`)
  const held = makeDigitsDataset({ nTrain: cfg.nTrain, nTest: cfg.nHoldout, testStart: cfg.holdoutStart,
    T: cfg.steps, H: 24, W: 24, encoding: 'poisson', seed: cfg.seed })
  if (!held) throw Error('digits dataset not found')
  const holdout = []
  for (const index of [0, chosen]) {
    const model = trained[index]
    // Dense checkpoint excludes the predictor: the teacher is trained after selection.
    io.checkpoint(table[index].name, saveDenseCheckpoint(model.net, model.readout, cfg.steps, cfg.pool))
    model.net.ensureGroups([8])
    for (const sm of data.train) model.net.run(sm.frames, cfg.steps, {
      mode: 'dense', budgetFrac: 1, usePredictor: true, trainPredictor: true, trainFiber: true, record: false,
    })
    if (io.checkpointCM) io.checkpointCM(table[index].name,
      saveCmCheckpoint(model.net, model.readout, cfg.steps, cfg.pool, 8, 0.7, cfg.seed))
    const ref = evaluate(model.net, model.readout, held.test, cfg, true)
    const rows = [gated(model.net, model.readout, held.test, cfg, ref, 'event', 1),
      gated(model.net, model.readout, held.test, cfg, ref, 'cm-decay', 0.7),
      gated(model.net, model.readout, held.test, cfg, ref, 'cm-decay', 0.5),
      gated(model.net, model.readout, held.test, cfg, ref, 'random', 0.7)]
    holdout.push({ name: table[index].name, denseAccuracy: ref.accuracy, denseMacs: ref.macs,
      spikeRate: ref.spikeRate, diagnostics: ref.diagnostics, rows })
  }
  const result = { phase: 'complete', config: cfg, candidates: table, selection: { selected,
    rule: 'highest validation dense accuracy among plastic candidates; first listed wins ties; random is control' }, holdout }
  io.results(result)
  return result
}
