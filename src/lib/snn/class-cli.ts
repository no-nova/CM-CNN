// SPDX-License-Identifier: MPL-2.0
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { runClassExperiment } from './class-experiment'
import type { DenseCheckpoint } from './checkpoint'
import { VALIDATION_DEFAULT } from './validation'
const argv = process.argv.slice(2)
const get = (name: string) => {
  const i = argv.indexOf(name)
  if (i < 0) return undefined
  if (!argv[i + 1] || argv[i + 1].startsWith('--')) throw Error(`${name} requires a value`)
  return argv[i + 1]
}
if (argv.includes('--help')) console.log('npm run snn:class -- [--quick] [--base dense.json] [--seed n] [--out directory]')
else {
  const quick = argv.includes('--quick')
  if (quick) console.warn('quick is a smoke test only: the default base was trained on these images, so accuracy is not an evaluation')
  const seedRaw = get('--seed'), seed = seedRaw === undefined ? 77 : Number(seedRaw)
  const cfg = { ...VALIDATION_DEFAULT, ...(quick ? { nTrain: 12, nValidation: 6, validationStart: 12, nHoldout: 6, holdoutStart: 20 } : {}), seed }
  const basePath = get('--base') ?? 'experiments/snn-validation-seed77/threshold-only.dense.json'
  const base = JSON.parse(readFileSync(basePath, 'utf8')) as DenseCheckpoint
  const out = get('--out') ?? 'experiments/snn-class-wta'
  mkdirSync(out, { recursive: true })
  runClassExperiment(base, cfg, {
    log: s => console.log(`[${new Date().toISOString()}] ${s}`),
    results: value => writeFileSync(join(out, 'results.json'), JSON.stringify({ basePath, ...value as object }, null, 2)),
    checkpoint: (name, cp) => writeFileSync(join(out, `${name}.cm.json`), JSON.stringify(cp)),
  })
}
