// SPDX-License-Identifier: MPL-2.0
import { loadInferenceCheckpoint, type CmCheckpoint } from './checkpoint'
import { makeIndexedDigitsDataset } from './task'
import { STREAM_DEFAULT } from './stream-experiment'
import { splitmix32 } from './rand'
import { StreamAgent, normalizeFeature, argmax } from './stream'
import { IsolatedAgent, ANCHOR_RATE, BOUNDARY_MARGIN, BOUNDARY_RATE, COSINE_SCALE, PROTOTYPE_ETA_BASE, PROTOTYPE_ETA_NOVEL, READOUT_RATE } from './continual'

export interface ContinualItem { index: number; label: number; x: Float64Array }
export interface ContinualConfig { seeds: number[]; streamIndices: number[]; validationIndices: number[]; testIndices: number[] }
export const CONTINUAL_DEFAULT: ContinualConfig = { seeds: [77, 78, 79], streamIndices: STREAM_DEFAULT.streamIndices,
  validationIndices: STREAM_DEFAULT.validationIndices, testIndices: STREAM_DEFAULT.testIndices }
const names = ['baseline', 'entropy-gated', 'ncm', 'anchored'] as const
type Name = typeof names[number]

function evaluate(predict: (x: Float64Array) => number, items: ContinualItem[]) {
  const perClass = Array.from({ length: 10 }, () => ({ correct: 0, total: 0 }))
  for (const { x, label } of items) { const c = perClass[label]; c.total++; if (predict(x) === label) c.correct++ }
  const sum = (a: typeof perClass) => { const n = a.reduce((v, z) => v + z.total, 0); return n ? a.reduce((v, z) => v + z.correct, 0) / n : null }
  return { all: sum(perClass), old: sum(perClass.slice(0, 5)), new: sum(perClass.slice(5)), perClass }
}

export function runContinualSeed(seed: number, stream: ContinualItem[], validation: ContinualItem[], test: ContinualItem[]) {
  if (!stream.length || !validation.length || !test.length) throw Error('empty feature split')
  const d = stream[0].x.length, rng = splitmix32(seed ^ 0x817bc)
  const shuffled = (items: ContinualItem[]) => { const a = items.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]] } return a }
  const phases = [shuffled(stream.filter(s => s.label < 5)), shuffled(stream.filter(s => s.label >= 5))]
  if (phases.some(p => !p.length)) throw Error('both incremental phases need samples')
  const baseline = new StreamAgent(d, 'baseline'), gated = new StreamAgent(d, 'entropy-gated')
  const ncm = new IsolatedAgent(d, 'ncm'), anchored = new IsolatedAgent(d, 'anchored')
  const trace: object[] = [], snapshots: any[] = [], counts: Record<Name, number> = { baseline: 0, 'entropy-gated': 0, ncm: 0, anchored: 0 }
  let count = 0, beforeOld: { ncm: number[][]; anchoredMeans: number[][]; anchoredRows: number[][]; ncmCounts: number[]; anchoredCounts: number[] } | null = null
  let frozenOld: object | null = null
  const predictions: Record<Name, (x: Float64Array) => number> = {
    baseline: x => argmax(baseline.readout.probabilities(normalizeFeature(x))),
    'entropy-gated': x => argmax(gated.readout.probabilities(normalizeFeature(x))),
    ncm: x => ncm.classify(x), anchored: x => anchored.classify(x),
  }
  for (let phase = 0; phase < 2; phase++) {
    const phaseCorrect: Record<Name, number> = { baseline: 0, 'entropy-gated': 0, ncm: 0, anchored: 0 }
    for (const item of phases[phase]) {
      const b = baseline.predict(item.x), g = gated.predict(item.x), n = ncm.predict(item.x), a = anchored.predict(item.x)
      // Feature CM is cloned per method and sees the same samples, not labels or predictions.
      if (g.entropy !== n.entropy || g.surprise !== n.surprise || n.entropy !== a.entropy || n.surprise !== a.surprise) throw Error('CM contexts diverged')
      const pred: Record<Name, number> = { baseline: b.prediction, 'entropy-gated': g.prediction, ncm: n.prediction, anchored: a.prediction }
      count++
      for (const name of names) { const ok = Number(pred[name] === item.label); phaseCorrect[name] += ok; counts[name] += ok }
      const fb = { baseline: baseline.feedback(item.label), gated: gated.feedback(item.label), ncm: ncm.feedback(item.label), anchored: anchored.feedback(item.label) }
      trace.push({ step: count, phase: phase + 1, index: item.index, label: item.label, prediction: pred,
        cumulativeAccuracy: Object.fromEntries(names.map(name => [name, counts[name] / count])),
        entropy: n.entropy, surprise: n.surprise, ncmEta: fb.ncm.eta, anchoredEta: fb.anchored.eta, anchoredGate: fb.anchored.gate })
    }
    const validationResults = Object.fromEntries(names.map(name => [name, evaluate(predictions[name], validation)]))
    const testResults = Object.fromEntries(names.map(name => [name, evaluate(predictions[name], test)]))
    snapshots.push({ phase: phase + 1, n: phases[phase].length,
      prequential: Object.fromEntries(names.map(name => [name, phaseCorrect[name] / phases[phase].length])),
      cumulative: Object.fromEntries(names.map(name => [name, counts[name] / count])), validation: validationResults, test: testResults })
    if (phase === 0) beforeOld = { ncm: ncm.bank.means.slice(0, 5).map(v => Array.from(v)),
      anchoredMeans: anchored.bank.means.slice(0, 5).map(v => Array.from(v)), anchoredRows: anchored.readout!.rows.slice(0, 5).map(v => Array.from(v)), ncmCounts: Array.from(ncm.bank.counts.slice(0, 5)), anchoredCounts: Array.from(anchored.bank.counts.slice(0, 5)) }
    else {
      const equal = (old: number[][], current: Float64Array[]) => old.every((row, i) => row.every((v, j) => Object.is(v, current[i][j])))
      frozenOld = { ncmPrototypesExact: equal(beforeOld!.ncm, ncm.bank.means), anchoredPrototypesExact: equal(beforeOld!.anchoredMeans, anchored.bank.means),
        anchoredRowsExact: equal(beforeOld!.anchoredRows, anchored.readout!.rows),
        ncmCountsExact: beforeOld!.ncmCounts.every((v, i) => v === ncm.bank.counts[i]),
        anchoredCountsExact: beforeOld!.anchoredCounts.every((v, i) => v === anchored.bank.counts[i]) }
    }
  }
  return { seed, snapshots, frozenOld, trace, agents: { baseline: baseline.snapshot(), 'entropy-gated': gated.snapshot(), ncm: ncm.snapshot(), anchored: anchored.snapshot() } }
}

export function runContinualExperiment(cp: CmCheckpoint, cfg: ContinualConfig, io: { log: (s: string) => void; results: (r: unknown) => void; agents: (seed: number, a: unknown) => void }) {
  const sets = [cfg.streamIndices, cfg.validationIndices, cfg.testIndices], indices = sets.flat()
  if (sets.some(s => !s.length) || new Set(indices).size !== indices.length || !cfg.seeds.length || new Set(cfg.seeds).size !== cfg.seeds.length ||
    cfg.seeds.some(s => !Number.isInteger(s) || s < 0 || s > 0xffffffff)) throw Error('invalid seeds or overlapping splits')
  if (!cp || cp.format !== 'ucz-snn-cm-v2' || cp.encoding.type !== 'poisson') throw Error('requires a CM checkpoint')
  const loaded = loadInferenceCheckpoint(cp), first = loaded.net.layers[0].spec
  if (first.inCh !== 1 || first.inH !== first.inW) throw Error('poisson digits require one input channel and a square canvas')
  const extract = (indices: number[], name: string): ContinualItem[] => {
    const data = makeIndexedDigitsDataset({ indices, T: loaded.steps, H: first.inH, W: first.inW, encoding: 'poisson', seed: loaded.seed! })
    if (!data) throw Error('digits dataset not found')
    return data.map((sm, i) => {
      const r = loaded.net.run(sm.frames, loaded.steps, { mode: 'event', budgetFrac: 1, usePredictor: false, trainPredictor: false,
        record: true, pool: loaded.pool, channelMajorBridge: loaded.channelMajorBridge })
      if ((i + 1) % 200 === 0 || i + 1 === data.length) io.log(`${name} features ${i + 1}/${data.length}`)
      return { index: indices[i], label: sm.label, x: r.features }
    })
  }
  const stream = extract(cfg.streamIndices, 'stream'), validation = extract(cfg.validationIndices, 'validation'), test = extract(cfg.testIndices, 'test')
  const zeroFeatures = stream.filter(s => s.x.every(v => v === 0)).length
  if (zeroFeatures) throw Error(`stream contains ${zeroFeatures} zero features; isolated cosine needs nonzero features`)
  const runs: Omit<ReturnType<typeof runContinualSeed>, 'agents'>[] = []
  const parameters = { prototypeEtaBase: PROTOTYPE_ETA_BASE, prototypeEtaNovel: PROTOTYPE_ETA_NOVEL,
    readoutRate: READOUT_RATE, anchorRate: ANCHOR_RATE, boundaryRate: BOUNDARY_RATE, boundaryMargin: BOUNDARY_MARGIN, cosineScale: COSINE_SCALE }
  for (const seed of cfg.seeds) {
    const { agents, ...result } = runContinualSeed(seed, stream, validation, test)
    io.agents(seed, agents); runs.push(result)
    io.log(`seed ${seed}: finished both phases`)
    const stats = (xs: number[]) => {
      const mean = xs.reduce((a, b) => a + b, 0) / xs.length
      return { mean, min: Math.min(...xs), max: Math.max(...xs), sampleStd: xs.length > 1 ? Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (xs.length - 1)) : null }
    }
    const summary = Object.fromEntries(names.map(name => {
      const metric = (fn: (r: typeof result) => number) => stats(runs.map(fn))
      return [name, { cumulative: metric(r => r.snapshots[1].cumulative[name]),
        oldBefore: metric(r => r.snapshots[0].test[name].old), oldAfter: metric(r => r.snapshots[1].test[name].old),
        forgetting: metric(r => r.snapshots[0].test[name].old - r.snapshots[1].test[name].old),
        newAfter: metric(r => r.snapshots[1].test[name].new), allAfter: metric(r => r.snapshots[1].test[name].all) }]
    }))
    io.results({ phase: runs.length === cfg.seeds.length ? 'complete' : 'running', config: cfg, parameters, zeroFeatures, summary, runs })
  }
  return runs
}
