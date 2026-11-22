// SPDX-License-Identifier: MPL-2.0
import { loadInferenceCheckpoint, type CmCheckpoint } from './checkpoint'
import { makeIndexedDigitsDataset } from './task'
import { PROTOTYPE_DEFAULT } from './prototype-experiment'
import { splitmix32 } from './rand'
import { OnlineLinear, StreamAgent, normalizeFeature, argmax } from './stream'

export interface StreamConfig { seed: number; learningRate: number; streamIndices: number[]; validationIndices: number[]; testIndices: number[] }
export const STREAM_DEFAULT: StreamConfig = { seed: 77, learningRate: 0.3,
  streamIndices: PROTOTYPE_DEFAULT.trainIndices, validationIndices: PROTOTYPE_DEFAULT.validationIndices, testIndices: PROTOTYPE_DEFAULT.testIndices }
interface Item { index: number; label: number; x: Float64Array }

function evaluate(model: OnlineLinear, items: Item[]) {
  const perClass = Array.from({ length: 10 }, () => ({ correct: 0, total: 0 }))
  for (const { x, label } of items) { const c = perClass[label]; c.total++; if (argmax(model.probabilities(normalizeFeature(x))) === label) c.correct++ }
  const aggregate = (cs: typeof perClass) => { const n = cs.reduce((a, b) => a + b.total, 0); return n ? cs.reduce((a, b) => a + b.correct, 0) / n : null }
  return { all: aggregate(perClass), old: aggregate(perClass.slice(0, 5)), new: aggregate(perClass.slice(5)), perClass }
}

export function runStreamExperiment(cp: CmCheckpoint, cfg: StreamConfig, io: { log: (s: string) => void; results: (r: unknown) => void }) {
  const sets = [cfg.streamIndices, cfg.validationIndices, cfg.testIndices]
  if (sets.some(a => !a.length) || new Set(sets.flat()).size !== sets.flat().length ||
      !Number.isInteger(cfg.seed) || cfg.seed < 0 || cfg.seed > 0xffffffff || !Number.isFinite(cfg.learningRate) || cfg.learningRate <= 0)
    throw Error('invalid stream experiment configuration or overlapping indices')
  if (!cp || cp.format !== 'ucz-snn-cm-v2' || cp.encoding.type !== 'poisson') throw Error('requires a CM checkpoint')
  const loaded = loadInferenceCheckpoint(cp)
  const first = loaded.net.layers[0].spec
  if (first.inCh !== 1 || first.inH !== first.inW) throw Error('poisson digits require one input channel and a square canvas')
  const extract = (indices: number[], name: string): Item[] => {
    const data = makeIndexedDigitsDataset({ indices, T: loaded.steps, H: first.inH, W: first.inW, encoding: 'poisson', seed: loaded.seed! })
    if (!data) throw Error('digits dataset not found')
    return data.map((s, i) => {
      const r = loaded.net.run(s.frames, loaded.steps, { mode: 'event', budgetFrac: 1, usePredictor: false, trainPredictor: false,
        record: true, pool: loaded.pool, channelMajorBridge: loaded.channelMajorBridge })
      if ((i + 1) % 200 === 0 || i + 1 === data.length) io.log(`${name} features ${i + 1}/${data.length}`)
      return { index: indices[i], label: s.label, x: r.features }
    })
  }
  // Precompute frozen SNN features exactly once. No labels or holdout observations train the feature-CM.
  const stream = extract(cfg.streamIndices, 'stream'), validation = extract(cfg.validationIndices, 'validation'), test = extract(cfg.testIndices, 'test')
  const zeroFeatures = stream.filter(s => s.x.every(v => v === 0)).length
  if (zeroFeatures === stream.length) throw Error('all stream features are zero: check SNN recording and activity')
  const rng = splitmix32(cfg.seed ^ 0x817bc)
  const shuffled = (items: Item[]) => { const a = items.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]] } return a }
  const phases = [shuffled(stream.filter(s => s.label < 5)), shuffled(stream.filter(s => s.label >= 5))]
  if (phases.some(a => !a.length)) throw Error('both incremental phases need samples')
  const d = stream[0].x.length, baselineAgent = new StreamAgent(d, 'baseline', cfg.learningRate), gatedAgent = new StreamAgent(d, 'entropy-gated', cfg.learningRate)
  const baseline = baselineAgent.readout, gated = gatedAgent.readout
  const trace: object[] = [], snapshots: object[] = []
  let totalBase = 0, totalGated = 0, count = 0
  for (let phase = 0; phase < 2; phase++) {
    let correctBase = 0, correctGated = 0, gateSum = 0, entropySum = 0, surpriseSum = 0
    for (const item of phases[phase]) {
      // Predict both arms and score the feature with past-only CM before consuming label or training.
      const pb = baselineAgent.predict(item.x), pg = gatedAgent.predict(item.x)
      const predB = pb.prediction, predG = pg.prediction
      const cb = predB === item.label, cg = predG === item.label
      correctBase += Number(cb); correctGated += Number(cg); totalBase += Number(cb); totalGated += Number(cg); count++
      const gate = gatedAgent.feedback(item.label).gate
      baselineAgent.feedback(item.label)
      trace.push({ step: count, phase: phase + 1, index: item.index, label: item.label, baseline: predB, gated: predG,
        baselineAccuracy: totalBase / count, gatedAccuracy: totalGated / count,
        entropy: pg.entropy, surprise: pg.surprise, gate })
      gateSum += gate; entropySum += pg.entropy!; surpriseSum += pg.surprise!
    }
    const result = { phase: phase + 1, classes: phase === 0 ? '0-4' : '5-9', n: phases[phase].length,
      prequential: { baseline: correctBase / phases[phase].length, gated: correctGated / phases[phase].length },
      gateMean: gateSum / phases[phase].length, entropyMean: entropySum / phases[phase].length, surpriseMean: surpriseSum / phases[phase].length,
      validation: { baseline: evaluate(baseline, validation), gated: evaluate(gated, validation) },
      test: { baseline: evaluate(baseline, test), gated: evaluate(gated, test) } }
    snapshots.push(result); io.log(`phase ${phase + 1}: prequential baseline ${(100 * result.prequential.baseline).toFixed(1)}%, gated ${(100 * result.prequential.gated).toFixed(1)}%`)
    io.results({ phase: phase === 1 ? 'complete' : 'phase-1', config: cfg, protocol: 'all ten classes always compete; frozen SNN; no replay; prequential score before updates', zeroFeatures, snapshots, trace })
  }
  return { snapshots, trace, baselineAgent, gatedAgent, zeroFeatures }
}
