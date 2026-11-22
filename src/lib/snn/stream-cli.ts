// SPDX-License-Identifier: MPL-2.0
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { createHash } from 'node:crypto'
import { STREAM_DEFAULT, runStreamExperiment } from './stream-experiment'
import type { CmCheckpoint } from './checkpoint'
const args = process.argv.slice(2)
function option(name: string, fallback: string) { const i = args.indexOf(name); if (i < 0) return fallback; if (!args[i + 1] || args[i + 1].startsWith('--')) throw Error(`${name} requires a value`); return args[i + 1] }
if (args.includes('--help')) console.log('npm run snn:stream -- [--model path] [--seed n] [--out directory] [--quick]')
else {
  const model = option('--model', 'experiments/snn-prototype/intrinsic-control.cm.json')
  const bytes = readFileSync(model)
  const sha256 = createHash('sha256').update(bytes).digest('hex')
  const cp = JSON.parse(bytes.toString('utf8')) as CmCheckpoint
  const quick = args.includes('--quick')
  const config = { ...STREAM_DEFAULT, seed: Number(option('--seed', '77')), ...(quick ? {
    streamIndices: STREAM_DEFAULT.streamIndices.slice(0, 60), validationIndices: Array.from({ length: 10 }, (_, i) => 1700 + i),
    testIndices: Array.from({ length: 10 }, (_, i) => 1710 + i) } : {}) }
  const out = option('--out', 'experiments/snn-stream'); mkdirSync(out, { recursive: true })
  const result = runStreamExperiment(cp, config, { log: s => console.log(`[${new Date().toISOString()}] ${s}`),
    results: r => writeFileSync(join(out, 'results.json'), JSON.stringify({ ...(r as object), model, modelSha256: sha256 }, null, 2)) })
  for (const [name, agent] of [['baseline', result.baselineAgent], ['entropy-gated', result.gatedAgent]] as const)
    writeFileSync(join(out, `${name}.agent.json`), JSON.stringify({ model, modelSha256: sha256, agent: agent.snapshot() }))
}
