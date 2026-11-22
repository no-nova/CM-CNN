// SPDX-License-Identifier: MPL-2.0
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { createHash } from 'node:crypto'
import { MultiscaleByteModel } from './multiscale-byte'
import { CoarseByteModel } from './coarse-byte'
import { splitmix32 } from '../snn/rand'
const args = process.argv.slice(2)
function opt(flag: string, fallback: string) { const i = args.indexOf(flag); if (i < 0) return fallback; if (!args[i + 1] || args[i + 1].startsWith('--')) throw Error(`${flag} requires a value`); return args[i + 1] }
function uint(value: string, field: string, min: number, max: number): number {
  if (!/^\d+$/.test(value)) throw Error(`invalid ${field}`)
  const n = Number(value); if (!Number.isSafeInteger(n) || n < min || n > max) throw Error(`invalid ${field}`); return n
}
if (args.includes('--help')) console.log('npm run snn:sequence-coarse -- [--input path] [--train-bytes 4096] [--eval-bytes 2048] [--generate-bytes 512] [--seeds 17,23,31] [--temperature 0.8] [--out directory]')
else {
  const input = opt('--input', 'data/tinyshakespeare-prefix.txt'), raw = readFileSync(input)
  const trainN = uint(opt('--train-bytes', '4096'), 'train-bytes', 1, 100000000), evalN = uint(opt('--eval-bytes', '2048'), 'eval-bytes', 0, 100000000)
  const generateN = uint(opt('--generate-bytes', '512'), 'generate-bytes', 1, 1000000)
  const temp = Number(opt('--temperature', '0.8')), seedText = opt('--seeds', '17,23,31')
  if (!/^\d+(,\d+)*$/.test(seedText) || !Number.isFinite(temp) || temp <= 0 || temp > 5) throw Error('invalid seeds or temperature')
  const seeds = seedText.split(',').map(s => uint(s, 'seed', 0, 0xffffffff))
  if (new Set(seeds).size !== seeds.length) throw Error('duplicate seeds')
  if (raw.length < trainN + evalN) throw Error(`input has ${raw.length} bytes, need ${trainN + evalN}`)
  const train = raw.subarray(0, trainN), evaluation = raw.subarray(trainN, trainN + evalN)
  const base = new MultiscaleByteModel(false), coarse = new CoarseByteModel(true), fixed = new CoarseByteModel(false)
  const names = ['constantPhase', 'coarsePhase', 'fixedResidual'] as const
  const models = [base, coarse, fixed], totals = [0, 0, 0], sums = [0, 0, 0]
  let count = 0; const trainWindows = []
  for (let i = 0; i < train.length; i++) {
    for (let j = 0; j < models.length; j++) { const loss = models[j].consume(train[i], true); totals[j] += loss; sums[j] += loss }
    count++
    if (count === 512 || i === train.length - 1) {
      trainWindows.push({ end: i + 1, n: count, ...Object.fromEntries(names.map((k, j) => [k, sums[j] / count])) }); sums.fill(0); count = 0
    }
  }
  const baseState = base.snapshot(), state = coarse.snapshot(), fixedState = fixed.snapshot()
  const evaluationRun = (learn: boolean) => {
    const copies = [MultiscaleByteModel.restore(baseState), CoarseByteModel.restore(state), CoarseByteModel.restore(fixedState)]
    const totals = [0, 0, 0], block = [0, 0, 0], windows = []; let n = 0
    for (let i = 0; i < evaluation.length; i++) {
      for (let j = 0; j < copies.length; j++) { const loss = copies[j].consume(evaluation[i], learn); totals[j] += loss; block[j] += loss }
      n++
      if (n === 512 || i === evaluation.length - 1) {
        windows.push({ end: i + 1, n, ...Object.fromEntries(names.map((k, j) => [k, block[j] / n])) }); n = 0; block.fill(0)
      }
    }
    return { n: evaluation.length, bitsPerByte: Object.fromEntries(names.map((k, j) => [k, evaluation.length ? totals[j] / evaluation.length : null])),
      windows, coarseDiagnosticsAfter: copies[1].diagnostics() }
  }
  const out = opt('--out', 'experiments/snn-sequence-coarse'); mkdirSync(out, { recursive: true })
  const frozen = evaluationRun(false), online = evaluationRun(true)
  writeFileSync(join(out, 'model.json'), JSON.stringify(state))
  writeFileSync(join(out, 'constant-phase-model.json'), JSON.stringify(baseState))
  writeFileSync(join(out, 'fixed-residual-model.json'), JSON.stringify(fixedState))
  const support = new Set(train)
  const generation = seeds.flatMap(seed => [false, true].map(clip => {
    const model = CoarseByteModel.restore(state), rng = splitmix32(seed), bytes = Buffer.alloc(generateN)
    for (let i = 0; i < bytes.length; i++) bytes[i] = model.sample(rng, temp, clip)
    const variant = clip ? 'observed-support' : 'unclipped', bin = `generated-${seed}-${variant}.bin`, text = `generated-${seed}-${variant}.txt`
    const decoded = bytes.toString('utf8'); writeFileSync(join(out, bin), bytes); writeFileSync(join(out, text), decoded, 'utf8')
    let validUtf8 = true; try { new TextDecoder('utf-8', { fatal: true }).decode(bytes) } catch { validUtf8 = false }
    return { seed, variant, bytes: generateN, binaryFile: bin, previewFile: text, validUtf8,
      unseenBytes: bytes.reduce((n, b) => n + Number(!support.has(b)), 0),
      replacementCharacters: Array.from(decoded).filter(c => c === '\ufffd').length }
  }))
  const hash = (b: Buffer) => createHash('sha256').update(b).digest('hex')
  const result = { input, inputSha256: hash(raw), trainN, evalN, trainSha256: hash(train), evaluationSha256: hash(evaluation),
    parameters: { stateBits: 2, slowThreshold: 128, fastSpikeThreshold: 2, residualPrior: 6, residualGain: 2, maxLogitShift: 1,
      contexts: 'current bit prefix + coarse phase only; no prevByte in residual key', temperature: temp },
    trainPrequential: { bitsPerByte: Object.fromEntries(names.map((k, j) => [k, totals[j] / trainN])), windows: trainWindows },
    holdoutFrozen: frozen, holdoutOnline: online, diagnostics: coarse.diagnostics(), generation,
    notes: ['All three share the unchanged v2 fixed-phase baseline; fixedResidual is an extra matched residual-capacity control.',
      'Training and online holdout score every bit before its update; frozen holdout still advances input context.',
      'Sampling starts after the training prefix, not the holdout, and never learns its own generated bytes.'] }
  writeFileSync(join(out, 'results.json'), JSON.stringify(result, null, 2)); console.log(JSON.stringify(result, null, 2))
}
