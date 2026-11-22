// SPDX-License-Identifier: MPL-2.0
import { readFileSync, writeFileSync, renameSync, unlinkSync } from 'node:fs'
import { createHash, randomBytes } from 'node:crypto'
import { loadInferenceCheckpoint, type CmCheckpoint } from './checkpoint'
import { makeIndexedDigitsDataset } from './task'
import { StreamAgent, type StreamAgentState } from './stream'
const args = process.argv.slice(2)
function required(flag: string): string { const i = args.indexOf(flag); if (i < 0 || !args[i + 1] || args[i + 1].startsWith('--')) throw Error(`${flag} requires a value`); return args[i + 1] }
if (args.includes('--help')) console.log('npm run snn:stream-predict -- --state path --index n [--label n --update]')
else {
  const statePath = required('--state'), index = Number(required('--index'))
  if (!Number.isInteger(index) || index < 0) throw Error('invalid sample index')
  const update = args.includes('--update'), labelArg = args.includes('--label') ? Number(required('--label')) : null
  if ((labelArg !== null) !== update || (labelArg !== null && (!Number.isInteger(labelArg) || labelArg < 0 || labelArg > 9)))
    throw Error('--label 0..9 and --update must be supplied together')
  const saved = JSON.parse(readFileSync(statePath, 'utf8')) as { model: string; modelSha256: string; agent: StreamAgentState }
  if (!saved || typeof saved.model !== 'string' || !/^[a-f0-9]{64}$/.test(saved.modelSha256)) throw Error('invalid agent file')
  const bytes = readFileSync(saved.model), hash = createHash('sha256').update(bytes).digest('hex')
  if (hash !== saved.modelSha256) throw Error('feature checkpoint hash mismatch')
  const cp = JSON.parse(bytes.toString('utf8')) as CmCheckpoint, loaded = loadInferenceCheckpoint(cp)
  if (!loaded.cm || loaded.seed === null) throw Error('expected CM feature checkpoint')
  const agent = StreamAgent.restore(saved.agent), first = loaded.net.layers[0].spec
  if (first.inCh !== 1 || first.inH !== first.inW) throw Error('poisson digits require one input channel and a square canvas')
  if (agent.d !== loaded.net.featureDim(loaded.pool)) throw Error('agent and feature dimensions differ')
  const sample = makeIndexedDigitsDataset({ indices: [index], T: loaded.steps, H: first.inH, W: first.inW, encoding: 'poisson', seed: loaded.seed })![0]
  const r = loaded.net.run(sample.frames, loaded.steps, { mode: 'event', budgetFrac: 1, usePredictor: false,
    trainPredictor: false, record: true, pool: loaded.pool, channelMajorBridge: loaded.channelMajorBridge })
  const prediction = agent.predict(r.features)
  const feedback = update ? agent.feedback(labelArg!) : null
  if (update) {
    const temp = `${statePath}.${process.pid}.${randomBytes(6).toString('hex')}.tmp`
    try { writeFileSync(temp, JSON.stringify({ ...saved, agent: agent.snapshot() }), { flag: 'wx' }); renameSync(temp, statePath) }
    finally { try { unlinkSync(temp) } catch { /* renamed or never created */ } }
  }
  console.log(JSON.stringify({ index, prediction: prediction.prediction, probabilities: Array.from(prediction.probabilities),
    entropy: prediction.entropy, surprise: prediction.surprise, updated: update, feedback,
    note: 'Prediction precedes optional feedback. Event-mode frozen SNN; supplied label is not inferred from the dataset.' }, null, 2))
}
