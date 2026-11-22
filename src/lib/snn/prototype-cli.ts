// SPDX-License-Identifier: MPL-2.0
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { PROTOTYPE_DEFAULT, runPrototypeExperiment } from './prototype-experiment'
const argv = process.argv.slice(2)
const get = (name: string) => {
  const i = argv.indexOf(name)
  if (i < 0) return undefined
  if (!argv[i + 1] || argv[i + 1].startsWith('--')) throw Error(`${name} requires a value`)
  return argv[i + 1]
}
if (argv.includes('--help')) console.log('npm run snn:prototype -- [--quick] [--seed n] [--out directory]')
else {
  const quick = argv.includes('--quick')
  const seedArg = get('--seed'), seed = seedArg === undefined ? PROTOTYPE_DEFAULT.seed : Number(seedArg)
  const cfg = { ...PROTOTYPE_DEFAULT, ...(quick ? { trainIndices: PROTOTYPE_DEFAULT.trainIndices.slice(0, 30),
    validationIndices: Array.from({ length: 10 }, (_, i) => 1700 + i),
    testIndices: Array.from({ length: 10 }, (_, i) => 1710 + i),
    steps: 4, readoutEpochs: 30 } : {}), seed }
  const out = get('--out') ?? 'experiments/snn-prototype'
  mkdirSync(out, { recursive: true })
  runPrototypeExperiment(cfg, { log: s => console.log(`[${new Date().toISOString()}] ${s}`),
    results: value => writeFileSync(join(out, 'results.json'), JSON.stringify(value, null, 2)),
    checkpoint: (name, cp) => writeFileSync(join(out, `${name}.cm.json`), JSON.stringify(cp)) })
}
