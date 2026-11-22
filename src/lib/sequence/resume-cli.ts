// SPDX-License-Identifier: MPL-2.0
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { SpikeByteModel, type ByteModelState } from './spike-byte'
import { splitmix32 } from '../snn/rand'
import { MultiscaleByteModel, type MultiscaleState } from './multiscale-byte'
import { CoarseByteModel, type CoarseState } from './coarse-byte'
const args = process.argv.slice(2)
function opt(flag: string, fallback: string) { const i = args.indexOf(flag); if (i < 0) return fallback; if (!args[i + 1] || args[i + 1].startsWith('--')) throw Error(`${flag} requires a value`); return args[i + 1] }
function uint(s: string, name: string, min: number, max: number) { if (!/^\d+$/.test(s)) throw Error(`invalid ${name}`); const x = Number(s); if (!Number.isSafeInteger(x) || x < min || x > max) throw Error(`invalid ${name}`); return x }
if (args.includes('--help')) console.log('npm run snn:sequence-resume -- --model model.json [--input new-bytes] [--generate-bytes 512] [--seed 17] [--temperature 0.8] [--support observed|none] [--out directory]')
else {
  const path = opt('--model', ''), input = opt('--input', ''), out = opt('--out', 'experiments/snn-sequence-resume')
  if (!path) throw Error('--model is required')
  const saved = JSON.parse(readFileSync(path, 'utf8')) as ByteModelState | MultiscaleState | CoarseState
  const v3 = saved.format === 'ucz-coarse-byte-v3', v2 = saved.format === 'ucz-multiscale-byte-v2'
  const model = v3 ? CoarseByteModel.restore(saved as CoarseState) : v2 ? MultiscaleByteModel.restore(saved as MultiscaleState) : SpikeByteModel.restore(saved as ByteModelState)
  const support = opt('--support', (v2 || v3) ? 'observed' : 'none')
  if ((support !== 'observed' && support !== 'none') || (!(v2 || v3) && support === 'observed')) throw Error('observed support requires a v2 or v3 checkpoint')
  const n = uint(opt('--generate-bytes', '512'), 'generate-bytes', 1, 1000000), seed = uint(opt('--seed', '17'), 'seed', 0, 0xffffffff)
  const temperature = Number(opt('--temperature', '0.8'))
  if (!Number.isFinite(temperature) || temperature <= 0 || temperature > 5) throw Error('invalid temperature')
  const data = input ? readFileSync(input) : Buffer.alloc(0)
  let loss = 0
  for (const b of data) loss += model.consume(b, true)
  mkdirSync(out, { recursive: true })
  writeFileSync(join(out, 'model-after-input.json'), JSON.stringify(model.snapshot()))
  const rng = splitmix32(seed), generated = Buffer.alloc(n)
  for (let i = 0; i < n; i++) generated[i] = model instanceof MultiscaleByteModel || model instanceof CoarseByteModel ? model.sample(rng, temperature, support === 'observed') : model.sample(rng, temperature, false)
  writeFileSync(join(out, 'generated.bin'), generated)
  writeFileSync(join(out, 'generated.txt'), generated.toString('utf8'))
  writeFileSync(join(out, 'model-after-generation.json'), JSON.stringify(model.snapshot()))
  let validUtf8 = true; try { new TextDecoder('utf-8', { fatal: true }).decode(generated) } catch { validUtf8 = false }
  const result = { sourceModel: path, input: input || null, learnedBytes: data.length, onlineInputBitsPerByte: data.length ? loss / data.length : null,
    generationBytes: n, seed, temperature, support, validUtf8, generatedBytesUsedAsTraining: false,
    note: 'Generated context advances but counters remain frozen. Each invocation starts a fresh sampling RNG at the supplied seed.' }
  writeFileSync(join(out, 'results.json'), JSON.stringify(result, null, 2)); console.log(JSON.stringify(result, null, 2))
}
