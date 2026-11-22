// SPDX-License-Identifier: MPL-2.0
import { ClassModulatedPlasticity } from './class-plastic'
import { denseLogits, loadDenseCheckpoint, loadInferenceCheckpoint, saveCmCheckpoint, type CmCheckpoint, type DenseCheckpoint } from './checkpoint'
import { DEFAULT_PRED } from './predict'
import { l2NormalizeRows, loadDigits, makeDigitsDataset, trainReadout, type EventSample } from './task'
import { evaluate, gated, type ValidationConfig } from './validation'

export interface ClassExperimentIO {
  log: (s: string) => void
  results: (data: unknown) => void
  checkpoint: (name: string, cp: CmCheckpoint) => void
}
const OPTIONS = [0.03, 0.1] // fixed before looking at validation

/** Fine-tunes the existing intrinsic-plasticity model without backpropagating into SNN.
 * The labelled train split may be replayed: this is NOT a prequential test. */
export function runClassExperiment(base: DenseCheckpoint, cfg: ValidationConfig, io: ClassExperimentIO) {
  if (!Number.isInteger(cfg.seed) || cfg.seed < 0 || cfg.seed > 0xffffffff ||
    ![cfg.nTrain, cfg.nValidation, cfg.nHoldout, cfg.steps, cfg.pool].every(n => Number.isInteger(n) && n > 0) ||
    !Number.isInteger(cfg.validationStart) || !Number.isInteger(cfg.holdoutStart) ||
    cfg.validationStart < cfg.nTrain || cfg.holdoutStart < cfg.nTrain ||
    !(cfg.validationStart + cfg.nValidation <= cfg.holdoutStart || cfg.holdoutStart + cfg.nHoldout <= cfg.validationStart))
    throw Error('invalid or overlapping experiment ranges')
  if (base.steps !== cfg.steps || base.pool !== cfg.pool) throw Error('base checkpoint steps/pool do not match config')
  const data = makeDigitsDataset({ nTrain: cfg.nTrain, nTest: cfg.nValidation, testStart: cfg.validationStart,
    H: 24, W: 24, T: cfg.steps, encoding: 'poisson', seed: cfg.seed })
  if (!data) throw Error('digits dataset not found')
  // Check the holdout range without loading its image pixels until selection.
  const n = loadDigits()?.n ?? 0
  if (cfg.holdoutStart + cfg.nHoldout > n) throw Error('holdout exceeds dataset')
  const table: { name: string; learningRate: number; accuracy: number; spikeRate: number;
    diagnostics: ReturnType<typeof evaluate>['diagnostics']; mistakes: number; trainingMacs: number;
    potentiated: number; depressed: number; kernels: number; weightRmsChange: number[] }[] = []
  const models: { net: ReturnType<typeof loadDenseCheckpoint>['net']; readout: ReturnType<typeof loadDenseCheckpoint>['readout'] }[] = []
  const add = (name: string, lr: number) => {
    const { net, readout: originalReadout } = loadDenseCheckpoint(base, DEFAULT_PRED, cfg.seed)
    const initial = net.layers.map(l => l.W.slice())
    let mistakes = 0, trainingMacs = 0, potentiated = 0, depressed = 0, kernels = 0
    if (lr > 0) {
      const learner = new ClassModulatedPlasticity(net.layers, { learningRate: lr, creditGain: 0.1, pool: cfg.pool })
      for (const sm of data.train) {
        const r = net.run(sm.frames, cfg.steps, { mode: 'dense', budgetFrac: 1, usePredictor: false,
          trainPredictor: false, record: true, pool: cfg.pool, observeSpikes: learner.observe })
        const scores = denseLogits(originalReadout, r.features)
        const predicted = scores.indexOf(Math.max(...scores))
        const correct = predicted === sm.label
        if (!correct) mistakes++
        // Even an overfitted readout supplies a small contrastive teaching signal:
        // on correct guesses compare the label against the strongest other class.
        let rival = predicted
        if (correct) {
          rival = sm.label === 0 ? 1 : 0
          for (let c = 0; c < scores.length; c++)
            if (c !== sm.label && scores[c] > scores[rival]) rival = c
        }
        const update = learner.commit(sm.label, rival, originalReadout, correct ? 0.15 : 1)
        potentiated += update.potentiated; depressed += update.depressed; kernels += update.kernels
        trainingMacs += r.budget.macs
      }
    }
    // Refit only the shallow readout after the SNN is frozen; no error is backpropagated.
    let readout = originalReadout
    if (lr > 0) {
      const d = net.featureDim(cfg.pool), X = new Float64Array(data.train.length * d), labels = new Int32Array(data.train.length)
      data.train.forEach((sm, i) => {
        const r = net.run(sm.frames, cfg.steps, { mode: 'dense', budgetFrac: 1, usePredictor: false,
          trainPredictor: false, record: true, pool: cfg.pool })
        X.set(r.features, i * d); labels[i] = sm.label
      })
      l2NormalizeRows(X, data.train.length, d)
      readout = trainReadout(X, labels, data.train.length, d, 10, 800, 0.8, 1e-4, 12345, true)
    }
    const val = evaluate(net, readout, data.test, cfg)
    const weightRmsChange = net.layers.map((l, li) => {
      let sum = 0
      for (let j = 0; j < l.W.length; j++) sum += (l.W[j] - initial[li][j]) ** 2
      return Math.sqrt(sum / l.W.length)
    })
    table.push({ name, learningRate: lr, accuracy: val.accuracy, spikeRate: val.spikeRate,
      diagnostics: val.diagnostics, mistakes, trainingMacs, potentiated, depressed, kernels, weightRmsChange })
    models.push({ net, readout })
    io.results({ phase: 'validation', config: cfg, candidates: table })
    io.log(`${name}: validation ${(val.accuracy * 100).toFixed(1)}%, errors ${mistakes}, updates ${kernels}`)
  }
  io.log('evaluating preserved intrinsic-plasticity baseline')
  add('intrinsic-baseline', 0)
  for (const lr of OPTIONS) { io.log(`training class WTA lr=${lr}`); add(`class-wta-${lr}`, lr) }
  let chosen = 1
  for (let i = 2; i < table.length; i++) if (table[i].accuracy > table[chosen].accuracy) chosen = i
  const selected = table[chosen].name
  const preferred = table[chosen].accuracy > table[0].accuracy ? selected : table[0].name
  io.log(`validation-selected plastic=${selected}; preferred=${preferred}; evaluate reused holdout`)
  const held = makeDigitsDataset({ nTrain: cfg.nTrain, nTest: cfg.nHoldout, testStart: cfg.holdoutStart,
    H: 24, W: 24, T: cfg.steps, encoding: 'poisson', seed: cfg.seed })!
  const output = []
  for (const i of [0, chosen]) {
    const { net, readout } = models[i]
    net.ensureGroups([8])
    io.log(`training CM teacher: ${table[i].name}`)
    for (const sm of data.train as EventSample[]) net.run(sm.frames, cfg.steps, {
      mode: 'dense', budgetFrac: 1, usePredictor: true, trainPredictor: true, trainFiber: true, record: false,
    })
    const cp = saveCmCheckpoint(net, readout, cfg.steps, cfg.pool, 8, 0.7, cfg.seed)
    io.checkpoint(table[i].name, cp)
    // Use the restored model for the full heldout run, proving the model file is sufficient.
    const restored = loadInferenceCheckpoint(cp)
    const ref = evaluate(restored.net, restored.readout, held.test, cfg, true)
    const rows = [gated(restored.net, restored.readout, held.test, cfg, ref, 'event', 1),
      gated(restored.net, restored.readout, held.test, cfg, ref, 'cm-decay', 0.7),
      gated(restored.net, restored.readout, held.test, cfg, ref, 'cm-decay', 0.5),
      gated(restored.net, restored.readout, held.test, cfg, ref, 'random', 0.7)]
    output.push({ name: table[i].name, denseAccuracy: ref.accuracy, spikeRate: ref.spikeRate,
      denseMacs: ref.macs, diagnostics: ref.diagnostics, rows })
    io.log(`${table[i].name}: holdout dense ${(100 * ref.accuracy).toFixed(1)}%, CM70 ${(100 * rows[1].accuracy).toFixed(1)}%`)
  }
  const result = { phase: 'complete', config: cfg, candidates: table,
    selection: { selectedPlastic: selected, preferred, rule: 'best plastic validation accuracy; retain baseline on tie or regression' }, holdout: output }
  io.results(result)
  return result
}
