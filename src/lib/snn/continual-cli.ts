// SPDX-License-Identifier: MPL-2.0
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { createHash } from 'node:crypto'
import { CONTINUAL_DEFAULT, runContinualExperiment } from './continual-experiment'
import type { CmCheckpoint } from './checkpoint'
const args = process.argv.slice(2)
function opt(flag: string, fallback: string) { const i = args.indexOf(flag); if (i < 0) return fallback; if (!args[i + 1] || args[i + 1].startsWith('--')) throw Error(`${flag} requires a value`); return args[i + 1] }
if (args.includes('--help')) console.log('npm run snn:continual -- [--model path] [--seeds 77,78,79] [--out directory] [--quick]')
else {
  const model = opt('--model', 'experiments/snn-prototype/intrinsic-control.cm.json'), bytes = readFileSync(model)
  const sha256 = createHash('sha256').update(bytes).digest('hex'), cp = JSON.parse(bytes.toString('utf8')) as CmCheckpoint
  const seedsText = opt('--seeds', '77,78,79')
  if (!/^\d+(,\d+)*$/.test(seedsText)) throw Error('seeds must be comma-separated unsigned integers')
  const seeds = seedsText.split(',').map(Number)
  const config = { ...CONTINUAL_DEFAULT, seeds, ...(args.includes('--quick') ? {
    streamIndices: CONTINUAL_DEFAULT.streamIndices.slice(0, 60), validationIndices: Array.from({ length: 10 }, (_, i) => 1700 + i),
    testIndices: Array.from({ length: 10 }, (_, i) => 1710 + i) } : {}) }
  const out = opt('--out', 'experiments/snn-continual'); mkdirSync(out, { recursive: true })
  runContinualExperiment(cp, config, { log: s => console.log(`[${new Date().toISOString()}] ${s}`),
    results: r => writeFileSync(join(out, 'results.json'), JSON.stringify({ ...(r as object), model, modelSha256: sha256 }, null, 2)),
    agents: (seed, agents) => writeFileSync(join(out, `agents-seed-${seed}.json`), JSON.stringify({ model, modelSha256: sha256, agents })) })
}
