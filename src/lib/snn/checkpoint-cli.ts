// SPDX-License-Identifier: MPL-2.0
import { readFileSync } from 'node:fs'
import { denseLogits, loadInferenceCheckpoint, type InferenceCheckpoint } from './checkpoint'
import { DEFAULT_EXP } from './experiment'
import { makeDigitsDataset } from './task'

const argv = process.argv.slice(2)
const get = (name: string): string | undefined => {
  const i = argv.indexOf(name)
  if (i < 0) return undefined
  if (!argv[i + 1] || argv[i + 1].startsWith('--')) throw Error(`${name} requires a value`)
  return argv[i + 1]
}
if (argv.includes('--help')) {
  console.log('npm run snn:predict -- --model <model.json> --index <digits index> [--seed n] [--mode dense|cm] [--budget 0..1]')
} else {
  const file = get('--model'), rawIndex = get('--index'), rawSeed = get('--seed'), mode = get('--mode') ?? 'dense'
  if (!file || rawIndex === undefined) throw Error('--model and --index are required')
  if (mode !== 'dense' && mode !== 'cm') throw Error('--mode must be dense or cm')
  const cp = JSON.parse(readFileSync(file, 'utf8')) as InferenceCheckpoint
  const { net, readout, steps, pool, cm, channelMajorBridge, seed: savedSeed } = loadInferenceCheckpoint(cp)
  const index = Number(rawIndex), seed = rawSeed === undefined ? (savedSeed ?? DEFAULT_EXP.seed) : Number(rawSeed)
  const rawBudget = get('--budget')
  if (mode === 'dense' && rawBudget !== undefined) throw Error('--budget is only valid with --mode cm')
  if (mode === 'cm' && !cm) throw Error('CM inference requires a CM-v2 model; this checkpoint has no trained predictor')
  const budget = rawBudget === undefined ? (cm?.budget ?? 1) : Number(rawBudget)
  if (!Number.isInteger(index) || index < 0 || !Number.isInteger(seed) || seed < 0 || seed > 0xffffffff ||
      !Number.isFinite(budget) || budget <= 0 || budget > 1)
    throw Error('invalid index, seed or budget')
  const s = net.layers[0].spec
  if (s.inCh !== 1 || s.inH !== s.inW) throw Error('this CLI supports only square, single-channel Poisson digit input')
  const data = makeDigitsDataset({ nTrain: 0, nTest: 1, testStart: index,
    H: s.inH, W: s.inW, T: steps, encoding: 'poisson', seed })
  if (!data) throw Error('digits dataset not found')
  const sample = data.test[0]
  const result = net.run(sample.frames, steps, { mode: mode === 'cm' ? 'cm-decay' : 'dense', budgetFrac: budget,
    usePredictor: mode === 'cm', trainPredictor: false, record: true, pool, group: cm?.group ?? 1, channelMajorBridge })
  const scores = Array.from(denseLogits(readout, result.features))
  const predicted = scores.indexOf(Math.max(...scores))
  // Dense geometry count without running a second dense pass; matches stepLayer's padded MAC accounting.
  let denseMacs = 0, denseBridgeOps = 0
  for (const layer of net.layers) {
    const sp = layer.spec
    for (let y = 0; y < layer.outH; y++) for (let x = 0; x < layer.outW; x++) {
      const y0 = y * sp.stride - sp.pad, x0 = x * sp.stride - sp.pad
      const kh = Math.max(0, Math.min(sp.k, sp.inH - y0) - Math.max(0, -y0))
      const kw = Math.max(0, Math.min(sp.k, sp.inW - x0) - Math.max(0, -x0))
      denseMacs += steps * sp.outCh * sp.inCh * kh * kw
    }
  }
  if (channelMajorBridge) for (const layer of net.layers.slice(0, -1)) denseBridgeOps += steps * layer.N
  const cost = result.budget
  console.log(JSON.stringify({ model: file, index, encoding: 'poisson', seed, predicted, label: sample.label,
    logits: scores, mode, channelMajorBridge, group: mode === 'cm' ? cm!.group : null, budgetFraction: mode === 'cm' ? budget : null,
    cost, denseMacs, denseBridgeOps, macRatio: cost.macs / denseMacs,
    opsRatio: (cost.macs + cost.gateOps + cost.predOps + cost.bridgeOps) / (denseMacs + denseBridgeOps),
    note: 'operation-count model, not measured speed; excludes training/readout/memory; CM state is frozen' }, null, 2))
}
