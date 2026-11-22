// SPDX-License-Identifier: MPL-2.0
// Read-only diagnostic: does restricting the final classifier to old candidates reproduce phase-1 old accuracy?
import { readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { createHash } from 'node:crypto'
import { loadInferenceCheckpoint, type CmCheckpoint } from './checkpoint'
import { makeIndexedDigitsDataset } from './task'
import { IsolatedAgent, bestKnown, type IsolatedAgentState } from './continual'
import { normalizeFeature } from './stream'
const args = process.argv.slice(2)
function opt(flag: string, fallback: string) { const i = args.indexOf(flag); if (i < 0) return fallback; if (!args[i + 1] || args[i + 1].startsWith('--')) throw Error(`${flag} requires a value`); return args[i + 1] }
if (args.includes('--help')) console.log('npm run snn:continual-check -- [--out experiments/snn-continual]')
else {
  const dir = opt('--out', 'experiments/snn-continual'), result = JSON.parse(readFileSync(join(dir, 'results.json'), 'utf8'))
  if (result.phase !== 'complete') throw Error('requires complete results')
  const bytes = readFileSync(result.model), hash = createHash('sha256').update(bytes).digest('hex')
  if (hash !== result.modelSha256) throw Error('feature checkpoint hash mismatch')
  const loaded = loadInferenceCheckpoint(JSON.parse(bytes.toString('utf8')) as CmCheckpoint), first = loaded.net.layers[0].spec
  if (!loaded.cm || loaded.seed === null || first.inCh !== 1 || first.inH !== first.inW) throw Error('invalid feature model')
  const samples = makeIndexedDigitsDataset({ indices: result.config.testIndices, T: loaded.steps, H: first.inH, W: first.inW, encoding: 'poisson', seed: loaded.seed })!
  const old = samples.filter(s => s.label < 5).map(s => {
    const r = loaded.net.run(s.frames, loaded.steps, { mode: 'event', budgetFrac: 1, usePredictor: false, trainPredictor: false,
      record: true, pool: loaded.pool, channelMajorBridge: loaded.channelMajorBridge })
    return { label: s.label, x: normalizeFeature(r.features) }
  })
  const runs = result.runs.map((run: any) => {
    const saved = JSON.parse(readFileSync(join(dir, `agents-seed-${run.seed}.json`), 'utf8')) as { modelSha256: string; agents: Record<string, IsolatedAgentState> }
    if (saved.modelSha256 !== hash) throw Error('agent checkpoint hash mismatch')
    const methods = Object.fromEntries(['ncm', 'anchored'].map(name => {
      const agent = IsolatedAgent.restore(saved.agents[name]); let correct = 0, allCorrect = 0, displaced = 0
      for (const item of old) {
        const scores = agent.readout ? agent.readout.scores(item.x, agent.bank) : agent.bank.scores(item.x)
        const allPred = bestKnown(scores)
        for (let c = 5; c < 10; c++) scores[c] = -Infinity
        const oldPred = bestKnown(scores)
        correct += Number(oldPred === item.label); allCorrect += Number(allPred === item.label)
        if (oldPred === item.label && allPred >= 5) displaced++
      }
      const before = run.snapshots[0].test[name].old, full = run.snapshots[1].test[name].old
      if (correct / old.length !== before || allCorrect / old.length !== full) throw Error('diagnostic does not reproduce recorded test accuracy')
      return [name, { oldCount: old.length, phase1AllClassesCorrect: before * old.length, phase2AllClassesCorrect: allCorrect,
        phase2OldCandidatesOnlyCorrect: correct, previouslyCorrectDisplacedToNew: displaced }]
    }))
    return { seed: run.seed, methods }
  })
  const output = { note: 'Diagnostic only: old-only candidate restriction uses task identity and is not the reported class-incremental metric.', runs }
  writeFileSync(join(dir, 'diagnostics.json'), JSON.stringify(output, null, 2))
  console.log(JSON.stringify(output, null, 2))
}
