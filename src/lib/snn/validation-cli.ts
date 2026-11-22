// SPDX-License-Identifier: MPL-2.0
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { BALANCED_CANDIDATES, CANDIDATES, runValidation, VALIDATION_DEFAULT } from './validation'

const argv = process.argv.slice(2)
const get = (name: string) => {
  const i = argv.indexOf(name)
  if (i < 0) return undefined
  if (!argv[i + 1] || argv[i + 1].startsWith('--')) throw Error(`${name} requires a value`)
  return argv[i + 1]
}
const num = (name: string, fallback: number) => {
  const raw = get(name)
  if (raw === undefined) return fallback
  const n = Number(raw)
  if (!Number.isInteger(n) || n < 0) throw Error(`${name} requires a non-negative integer`)
  return n
}
if (argv.includes('--help')) {
  console.log('npm run snn:validation -- [--quick] [--balanced] [--seed n] [--out directory]')
} else {
  const quick = argv.includes('--quick')
  const cfg = { ...VALIDATION_DEFAULT, ...(quick ? {
    nTrain: 12, nValidation: 6, validationStart: 12, nHoldout: 6, holdoutStart: 20, steps: 3,
  } : {}), seed: num('--seed', VALIDATION_DEFAULT.seed) }
  const out = get('--out') ?? 'experiments/snn-validation'
  mkdirSync(out, { recursive: true })
  runValidation(cfg, {
    log: message => console.log(`[${new Date().toISOString()}] ${message}`),
    checkpoint: (name, cp) => writeFileSync(join(out, `${name}.dense.json`), JSON.stringify(cp)),
    checkpointCM: (name, cp) => writeFileSync(join(out, `${name}.cm.json`), JSON.stringify(cp)),
    results: result => writeFileSync(join(out, 'results.json'), JSON.stringify(result, null, 2)),
  }, argv.includes('--balanced') ? BALANCED_CANDIDATES : CANDIDATES)
  console.log(`results: ${join(out, 'results.json')}`)
}
