// SPDX-License-Identifier: MPL-2.0
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { createHash } from 'node:crypto'
import { SpikeByteModel } from './spike-byte'
import { MultiscaleByteModel } from './multiscale-byte'
import { splitmix32 } from '../snn/rand'
const args = process.argv.slice(2)
function opt(flag: string, fallback: string) { const i = args.indexOf(flag); if (i < 0) return fallback; if (!args[i + 1] || args[i + 1].startsWith('--')) throw Error(`${flag} requires a value`); return args[i + 1] }
function uint(value: string, field: string, min: number, max: number): number {
  if (!/^\d+$/.test(value)) throw Error(`invalid ${field}`)
  const n = Number(value); if (!Number.isSafeInteger(n) || n < min || n > max) throw Error(`invalid ${field}`); return n
}
if (args.includes('--help')) console.log('npm run snn:sequence -- [--input path] [--train-bytes 4096] [--eval-bytes 2048] [--generate-bytes 512] [--seeds 17,23,31] [--temperature 0.8] [--out directory]')
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
  const spike = new MultiscaleByteModel(), control = new SpikeByteModel(false), legacy = new SpikeByteModel(true), constant = new MultiscaleByteModel(false)
  const totals = [0, 0, 0, 0], block = [0, 0, 0, 0]
  let count = 0
  const windows = []
  for (let i = 0; i < train.length; i++) {
    const losses = [spike.consume(train[i], true), control.consume(train[i], true), legacy.consume(train[i], true), constant.consume(train[i], true)]
    for (let j = 0; j < losses.length; j++) { totals[j] += losses[j]; block[j] += losses[j] }
    count++
    if (count === 512 || i === train.length - 1) {
      windows.push({ end: i + 1, n: count, multiscale: block[0] / count, cmOnly: block[1] / count, legacySpike: block[2] / count, constantPhase: block[3] / count })
      block.fill(0); count = 0
    }
  }
  const state = spike.snapshot(), controlState = control.snapshot(), legacyState = legacy.snapshot(), constantState = constant.snapshot()
  const evalMode = (model: { consume: (byte: number, train: boolean) => number }, learn: boolean) => {
    let loss = 0; for (const b of evaluation) loss += model.consume(b, learn)
    return evaluation.length ? loss / evaluation.length : null
  }
  const holdout = { n: evalN, multiscaleFrozen: evalMode(MultiscaleByteModel.restore(state), false),
    multiscaleOnline: evalMode(MultiscaleByteModel.restore(state), true), cmOnlyFrozen: evalMode(SpikeByteModel.restore(controlState), false),
    cmOnlyOnline: evalMode(SpikeByteModel.restore(controlState), true), legacyFrozen: evalMode(SpikeByteModel.restore(legacyState), false),
    legacyOnline: evalMode(SpikeByteModel.restore(legacyState), true),
    constantPhaseFrozen: evalMode(MultiscaleByteModel.restore(constantState), false),
    constantPhaseOnline: evalMode(MultiscaleByteModel.restore(constantState), true) }
  const out = opt('--out', 'experiments/snn-sequence-v2'); mkdirSync(out, { recursive: true })
  writeFileSync(join(out, 'model.json'), JSON.stringify(state))
  writeFileSync(join(out, 'cm-only-model.json'), JSON.stringify(controlState))
  writeFileSync(join(out, 'constant-phase-model.json'), JSON.stringify(constantState))
  const support = new Set(train), trigrams = new Set<string>()
  for (let i = 2; i < train.length; i++) trigrams.add(`${train[i-2]},${train[i-1]},${train[i]}`)
  const generation = seeds.flatMap(seed => [false, true].map(clip => {
    const model = MultiscaleByteModel.restore(state), rng = splitmix32(seed), bytes = Buffer.alloc(generateN)
    for (let i = 0; i < bytes.length; i++) bytes[i] = model.sample(rng, temp, clip)
    const variant = clip ? 'observed-support' : 'unclipped', bin = `generated-${seed}-${variant}.bin`, text = `generated-${seed}-${variant}.txt`
    const decoded = bytes.toString('utf8'); writeFileSync(join(out, bin), bytes); writeFileSync(join(out, text), decoded, 'utf8')
    let validUtf8 = true; try { new TextDecoder('utf-8', { fatal: true }).decode(bytes) } catch { validUtf8 = false }
    let unseenTrigrams = 0; for (let i = 2; i < bytes.length; i++) if (!trigrams.has(`${bytes[i-2]},${bytes[i-1]},${bytes[i]}`)) unseenTrigrams++
    return { seed, variant, temperature: temp, bytes: generateN, binaryFile: bin, previewFile: text, validUtf8,
      printableAsciiFraction: bytes.reduce((n, b) => n + Number((b >= 32 && b <= 126) || b === 9 || b === 10 || b === 13), 0) / bytes.length,
      unseenBytes: bytes.reduce((n, b) => n + Number(!support.has(b)), 0),
      unseenTrigramFraction: generateN >= 3 ? unseenTrigrams / (generateN - 2) : null,
      replacementCharacters: Array.from(decoded).filter(c => c === '\ufffd').length }
  }))
  const result = { input, inputSha256: createHash('sha256').update(raw).digest('hex'), trainN, evalN,
    trainSha256: createHash('sha256').update(train).digest('hex'), evaluationSha256: createHash('sha256').update(evaluation).digest('hex'),
    onlineFromByteZero: { multiscale: totals[0] / trainN, cmOnly: totals[1] / trainN, legacySpike: totals[2] / trainN, constantPhase: totals[3] / trainN, uniform: 8, windows },
    holdoutBitsPerByte: holdout, structure: spike.diagnostics(), generation,
    notes: ['Identical original CM contexts, short-stream state-derived structural experts; no pretraining.',
      'Holdout is teacher-forced; frozen means tables frozen while byte/spike context advances.',
      'Samples start after training prefix, never learn from generated bytes.',
      'Observed support clips only globally unseen byte values, with no encoding/character rules; paired runs consume identical random draws.'] }
  writeFileSync(join(out, 'results.json'), JSON.stringify(result, null, 2))
  console.log(JSON.stringify(result, null, 2))
}
