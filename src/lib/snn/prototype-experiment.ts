// SPDX-License-Identifier: MPL-2.0
import { Net, defaultLayers } from './experiment'
import { DEFAULT_PRED } from './predict'
import { IntrinsicHomeostasis, PrototypeContrastive, augmentEvents } from './prototype'
import { denseLogits, loadInferenceCheckpoint, saveCmCheckpoint, type CmCheckpoint } from './checkpoint'
import { l2NormalizeRows, loadDigits, makeIndexedDigitsDataset, trainReadout, type EventSample, type Readout } from './task'
import { rebuildWeightBounds } from './snn'
import { splitmix32 } from './rand'

export interface PrototypeConfig {
  seed: number; steps: number; pool: number; learningRate: number; antiScale: number
  jitter: number; readoutL2: number; readoutEpochs: number
  trainIndices: number[]; validationIndices: number[]; testIndices: number[]
}
// Validation and test were outside earlier train and evaluation ranges; 197 images remain unused.
export const PROTOTYPE_DEFAULT: PrototypeConfig = {
  seed: 77, steps: 20, pool: 3, learningRate: 0.03, antiScale: 0.25, jitter: 0.15,
  readoutL2: 0.1, readoutEpochs: 400,
  trainIndices: [...Array.from({ length: 700 }, (_, i) => i),
    ...Array.from({ length: 280 }, (_, i) => 920 + i), ...Array.from({ length: 220 }, (_, i) => 1420 + i)],
  validationIndices: Array.from({ length: 200 }, (_, i) => 720 + i),
  testIndices: Array.from({ length: 200 }, (_, i) => 1220 + i),
}
export interface PrototypeIO {
  log: (message: string) => void
  results: (data: unknown) => void
  checkpoint: (name: string, cp: CmCheckpoint) => void
}

export function validatePrototypeConfig(cfg: PrototypeConfig): void {
  const d = loadDigits()
  if (!d) throw Error('digits dataset not found')
  const sets = [cfg.trainIndices, cfg.validationIndices, cfg.testIndices]
  if (sets.some(a => !Array.isArray(a) || !a.length || a.some(i => !Number.isInteger(i) || i < 0 || i >= d.n)))
    throw Error('invalid split indices')
  const all = sets.flat()
  if (new Set(all).size !== all.length) throw Error('train/validation/test indices must be disjoint and unique')
  if (!Number.isInteger(cfg.seed) || cfg.seed < 0 || cfg.seed > 0xffffffff ||
    !Number.isInteger(cfg.steps) || cfg.steps < 1 || cfg.steps > 65535 ||
    !Number.isInteger(cfg.pool) || cfg.pool < 1 || cfg.pool > 12 ||
    !Number.isInteger(cfg.readoutEpochs) || cfg.readoutEpochs < 1 ||
    !Number.isFinite(cfg.readoutL2) || cfg.readoutL2 < 0 ||
    !Number.isFinite(cfg.learningRate) || cfg.learningRate < 0 || cfg.learningRate > 0.5 ||
    !Number.isFinite(cfg.antiScale) || cfg.antiScale < 0 || cfg.antiScale > 1 ||
    !Number.isFinite(cfg.jitter) || cfg.jitter < 0 || cfg.jitter > 1) throw Error('invalid experiment settings')
}

function views(sm: EventSample, index: number, cfg: PrototypeConfig): EventSample[] {
  const shifted = (dx: number, dy: number, salt: number): EventSample => ({ label: sm.label,
    frames: augmentEvents(sm.frames, cfg.steps, 24, 24, dx, dy, cfg.jitter,
      (cfg.seed ^ Math.imul(index, 0x9e3779b1) ^ salt) >>> 0) })
  return [sm, shifted(-1, 1, 0x43a91), shifted(1, -1, 0x82f71)]
}

function copyNetwork(src: Net, seed: number): Net {
  const net = new Net(src.layers.map(l => ({ ...l.spec })), DEFAULT_PRED, seed)
  net.layers.forEach((l, i) => {
    l.W.set(src.layers[i].W); l.thresholdOffsets.set(src.layers[i].thresholdOffsets); rebuildWeightBounds(l)
  })
  return net
}

function features(net: Net, readout: Readout | null, samples: EventSample[], cfg: PrototypeConfig) {
  const d = net.featureDim(cfg.pool), X = new Float64Array(samples.length * d), y = new Int32Array(samples.length)
  const refs: Uint8Array[][] = []
  const activity = net.layers.map(l => new Float64Array(l.spec.outCh))
  const matrix = Array.from({ length: 10 }, () => new Float64Array(10))
  const classCounts = new Uint32Array(10)
  let macs = 0, denseOps = 0, correctPrototype = 0
  samples.forEach((sm, i) => {
    const r = net.run(sm.frames, cfg.steps, { mode: 'dense', budgetFrac: 1, usePredictor: false,
      trainPredictor: false, record: true, pool: cfg.pool, channelMajorBridge: true })
    X.set(r.features, i * d); y[i] = sm.label; refs.push(r.spikes)
    macs += r.budget.macs; denseOps += r.budget.macs + r.budget.bridgeOps
    r.spikes.forEach((arr, li) => {
      const ch = net.layers[li].spec.outCh
      for (let j = 0; j < arr.length; j++) activity[li][j % ch] += arr[j]
    })
    const last = r.spikes[1], groups = new Float64Array(10)
    for (let j = 0; j < last.length; j++) {
      const oc = j % 32
      if (oc < 30) groups[Math.floor(oc / 3)] += last[j]
    }
    let best = 0
    for (let c = 0; c < 10; c++) { matrix[sm.label][c] += groups[c]; if (groups[c] > groups[best]) best = c }
    classCounts[sm.label]++
    if (best === sm.label) correctPrototype++
  })
  let correct = 0
  const perClass = Array.from({ length: 10 }, () => ({ correct: 0, total: 0 }))
  if (readout) for (let i = 0; i < samples.length; i++) {
    const scores = denseLogits(readout, X.subarray(i * d, (i + 1) * d))
    const pred = scores.indexOf(Math.max(...scores)), label = y[i]
    perClass[label].total++
    if (pred === label) { correct++; perClass[label].correct++ }
  }
  const diagnostics = net.layers.map((l, li) => {
    const rates = Array.from(activity[li], x => x / (samples.length * cfg.steps * l.outH * l.outW))
    return { rates, silentBelow001: rates.filter(x => x < 0.001).length }
  })
  return { X, y, refs, macs, denseOps, accuracy: readout ? correct / samples.length : null, perClass,
    prototypeAccuracy: correctPrototype / samples.length, diagnostics,
    prototypeResponse: matrix.map((row, c) => Array.from(row, x => x / Math.max(1, classCounts[c]))) }
}

function gates(net: Net, readout: Readout, samples: EventSample[], cfg: PrototypeConfig, ref: ReturnType<typeof features>,
  mode: 'event' | 'cm-decay' | 'random', budget: number) {
  let macs = 0, ops = 0, miss = 0, extra = 0, truth = 0, correct = 0
  const rng = splitmix32(98)
  samples.forEach((sm, i) => {
    const r = net.run(sm.frames, cfg.steps, { mode, budgetFrac: budget, usePredictor: mode !== 'event',
      trainPredictor: false, record: true, pool: cfg.pool, group: 8, rng, channelMajorBridge: true })
    macs += r.budget.macs; ops += r.budget.macs + r.budget.gateOps + r.budget.predOps + r.budget.bridgeOps
    const scores = denseLogits(readout, r.features)
    if (scores.indexOf(Math.max(...scores)) === sm.label) correct++
    r.spikes.forEach((arr, li) => {
      const expected = ref.refs[i][li]
      for (let j = 0; j < arr.length; j++) {
        const a = expected[j], b = arr[j]
        truth += a; if (a && !b) miss++; else if (!a && b) extra++
      }
    })
  })
  return { mode, budget, accuracy: correct / samples.length, macRatio: macs / ref.macs,
    opsRatio: ops / ref.denseOps, missedSpikes: miss, extraSpikes: extra, trueSpikes: truth }
}

export function runPrototypeExperiment(cfg: PrototypeConfig, io: PrototypeIO) {
  validatePrototypeConfig(cfg)
  const get = (indices: number[]) => makeIndexedDigitsDataset({ indices, T: cfg.steps, H: 24, W: 24,
    encoding: 'poisson', seed: cfg.seed })!
  const order = cfg.trainIndices.slice(), rng = splitmix32(cfg.seed ^ 0xd1a17)
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [order[i], order[j]] = [order[j], order[i]] }
  const train = get(order), validation = get(cfg.validationIndices)
  const net = new Net(defaultLayers(24, 24, true, { in1: 1, ch1: 16, k2: 5, ch2: 32 }), DEFAULT_PRED, cfg.seed)
  const intrinsic = new IntrinsicHomeostasis(net.layers)
  const progress = (name: string, i: number, n: number) => {
    if ((i + 1) % 200 === 0 || i + 1 === n) io.log(`${name}: ${i + 1}/${n}`)
  }
  let ipMacs = 0
  io.log('intrinsic adaptation on original train images (no labels)')
  train.forEach((sm, i) => {
    const r = net.run(sm.frames, cfg.steps, { mode: 'event', budgetFrac: 1, usePredictor: false,
      trainPredictor: false, record: false, channelMajorBridge: true, observeSpikes: intrinsic.observe })
    ipMacs += r.budget.macs; intrinsic.commit(); progress('IP', i, train.length)
  })
  const baseline = copyNetwork(net, cfg.seed)
  const initial = net.layers[1].W.slice()
  const learner = new PrototypeContrastive(net.layers[0], net.layers[1], cfg.learningRate, cfg.antiScale)
  let plasticMacs = 0, positive = 0, negative = 0, silentTargets = 0
  io.log('prototype WTA / anti-Hebbian, all 3 views per training image; no readout')
  train.forEach((sm, i) => {
    for (const v of views(sm, order[i], cfg)) {
      const r = net.run(v.frames, cfg.steps, { mode: 'event', budgetFrac: 1, usePredictor: false,
        trainPredictor: false, record: true, pool: cfg.pool, channelMajorBridge: true })
      const stats = learner.apply(v.label, r.spikes[0], r.spikes[1], cfg.steps)
      positive += stats.positive; negative += stats.negative; silentTargets += stats.silentTargets
      plasticMacs += r.budget.macs
    }
    progress('prototype', i, train.length)
  })
  let change = 0
  for (let i = 0; i < initial.length; i++) change += (net.layers[1].W[i] - initial[i]) ** 2
  const plasticity = { positive, negative, silentTargets, weightRmsChangeL2: Math.sqrt(change / initial.length),
    ipMacs, plasticMacs, updateOps: null, eventViews: train.length * 3 }
  const arms: { name: string; net: Net; readout: Readout }[] = []
  const validationResults = []
  for (const [name, model] of [['intrinsic-control', baseline], ['class-prototype', net]] as const) {
    io.log(`extract 3-view train features and fit regularized readout: ${name}`)
    const dims = model.featureDim(cfg.pool), n = train.length * 3
    const X = new Float64Array(n * dims), labels = new Int32Array(n)
    train.forEach((sm, i) => {
      views(sm, order[i], cfg).forEach((v, j) => {
        const r = model.run(v.frames, cfg.steps, { mode: 'event', budgetFrac: 1, usePredictor: false,
          trainPredictor: false, record: true, pool: cfg.pool, channelMajorBridge: true })
        X.set(r.features, (i * 3 + j) * dims); labels[i * 3 + j] = sm.label
      })
      progress(`readout features ${name}`, i, train.length)
    })
    l2NormalizeRows(X, n, dims)
    // opt into penalty/n: the legacy readout applied its penalty once per sample without /n.
    const readout = trainReadout(X, labels, n, dims, 10, cfg.readoutEpochs, 0.8, cfg.readoutL2, 12345, true, true)
    const trainCorrect = [0, 0, 0]
    for (let i = 0; i < n; i++) {
      const scores = denseLogits(readout, X.subarray(i * dims, (i + 1) * dims))
      if (scores.indexOf(Math.max(...scores)) === labels[i]) trainCorrect[i % 3]++
    }
    const trainingAccuracy = { allViews: trainCorrect.reduce((a, b) => a + b, 0) / n,
      original: trainCorrect[0] / train.length, shiftA: trainCorrect[1] / train.length, shiftB: trainCorrect[2] / train.length }
    const val = features(model, readout, validation, cfg)
    arms.push({ name, net: model, readout })
    validationResults.push({ name, accuracy: val.accuracy, trainingAccuracy, prototypeAccuracy: val.prototypeAccuracy,
      diagnostics: val.diagnostics, perClass: val.perClass, prototypeResponse: val.prototypeResponse })
    io.results({ phase: 'validation', config: cfg, plasticity, validation: validationResults })
    io.log(`${name}: train all ${(100 * trainingAccuracy.allViews).toFixed(1)}%, validation dense ${(100 * val.accuracy!).toFixed(1)}%`)
  }
  // No hyperparameters are selected on the test samples.
  const test = get(cfg.testIndices), testResults = []
  for (const arm of arms) {
    io.log(`train CM teacher on original train streams: ${arm.name}`)
    arm.net.ensureGroups([8])
    let teacherMacs = 0
    train.forEach((sm, i) => {
      const r = arm.net.run(sm.frames, cfg.steps, { mode: 'event', budgetFrac: 1, usePredictor: true,
        trainPredictor: true, trainFiber: true, record: false, channelMajorBridge: true })
      teacherMacs += r.budget.macs; progress(`CM ${arm.name}`, i, train.length)
    })
    const checkpoint = saveCmCheckpoint(arm.net, arm.readout, cfg.steps, cfg.pool, 8, 0.7, cfg.seed, true)
    io.checkpoint(arm.name, checkpoint)
    const loaded = loadInferenceCheckpoint(checkpoint)
    const ref = features(loaded.net, loaded.readout, test, cfg)
    const rows = [gates(loaded.net, loaded.readout, test, cfg, ref, 'event', 1),
      gates(loaded.net, loaded.readout, test, cfg, ref, 'cm-decay', 0.7),
      gates(loaded.net, loaded.readout, test, cfg, ref, 'cm-decay', 0.5),
      gates(loaded.net, loaded.readout, test, cfg, ref, 'random', 0.7)]
    testResults.push({ name: arm.name, accuracy: ref.accuracy, prototypeAccuracy: ref.prototypeAccuracy,
      diagnostics: ref.diagnostics, perClass: ref.perClass, prototypeResponse: ref.prototypeResponse,
      denseMacs: ref.macs, denseOps: ref.denseOps, teacherMacs, rows })
    io.log(`${arm.name}: test dense ${(100 * ref.accuracy!).toFixed(1)}%, CM70 ${(100 * rows[1].accuracy).toFixed(1)}%`)
  }
  const result = { phase: 'complete', config: cfg, plasticity, validation: validationResults, test: testResults }
  io.results(result)
  return result
}
