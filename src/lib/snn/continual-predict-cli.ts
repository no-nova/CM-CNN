// SPDX-License-Identifier: MPL-2.0
import { readFileSync, writeFileSync, renameSync, unlinkSync } from 'node:fs'
import { createHash, randomBytes } from 'node:crypto'
import { loadInferenceCheckpoint, type CmCheckpoint } from './checkpoint'
import { makeIndexedDigitsDataset } from './task'
import { IsolatedAgent, type IsolatedAgentState } from './continual'
const args = process.argv.slice(2)
function required(flag: string): string { const i = args.indexOf(flag); if (i < 0 || !args[i + 1] || args[i + 1].startsWith('--')) throw Error(`${flag} requires a value`); return args[i + 1] }
if (args.includes('--help')) console.log('npm run snn:continual-predict -- --state agents.json --method ncm|anchored --index n [--label n --update]')
else {
  const path = required('--state'), method = required('--method'), index = Number(required('--index'))
  if ((method !== 'ncm' && method !== 'anchored') || !Number.isInteger(index) || index < 0) throw Error('invalid method or index')
  const update = args.includes('--update'), label = args.includes('--label') ? Number(required('--label')) : null
  if ((label !== null) !== update || (label !== null && (!Number.isInteger(label) || label < 0 || label > 9))) throw Error('--label 0..9 and --update must be supplied together')
  const saved = JSON.parse(readFileSync(path, 'utf8')) as { model: string; modelSha256: string; agents: Record<string, IsolatedAgentState> }
  if (!saved || typeof saved.model !== 'string' || !/^[a-f0-9]{64}$/.test(saved.modelSha256) || !saved.agents) throw Error('invalid agent file')
  const bytes = readFileSync(saved.model), hash = createHash('sha256').update(bytes).digest('hex')
  if (hash !== saved.modelSha256) throw Error('feature checkpoint hash mismatch')
  const cp = JSON.parse(bytes.toString('utf8')) as CmCheckpoint, loaded = loadInferenceCheckpoint(cp)
  if (!loaded.cm || loaded.seed === null) throw Error('expected CM feature checkpoint')
  const agent = IsolatedAgent.restore(saved.agents[method]), first = loaded.net.layers[0].spec
  if (agent.mode !== method || first.inCh !== 1 || first.inH !== first.inW || agent.d !== loaded.net.featureDim(loaded.pool)) throw Error('agent and feature model mismatch')
  const sample = makeIndexedDigitsDataset({ indices: [index], T: loaded.steps, H: first.inH, W: first.inW, encoding: 'poisson', seed: loaded.seed })![0]
  const r = loaded.net.run(sample.frames, loaded.steps, { mode: 'event', budgetFrac: 1, usePredictor: false, trainPredictor: false,
    record: true, pool: loaded.pool, channelMajorBridge: loaded.channelMajorBridge })
  const prediction = agent.predict(r.features), feedback = update ? agent.feedback(label!) : null
  if (update) {
    saved.agents[method] = agent.snapshot()
    const temp = `${path}.${process.pid}.${randomBytes(6).toString('hex')}.tmp`
    try { writeFileSync(temp, JSON.stringify(saved), { flag: 'wx' }); renameSync(temp, path) }
    finally { try { unlinkSync(temp) } catch { /* renamed or never created */ } }
  }
  console.log(JSON.stringify({ index, method, ...prediction, updated: update, feedback,
    note: 'Prediction precedes optional feedback; no label inferred from dataset; frozen event-mode SNN.' }, null, 2))
}
