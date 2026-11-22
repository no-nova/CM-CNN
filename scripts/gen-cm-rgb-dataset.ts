import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { buildRgbBenchmark, encodeRgbSet, encodeRgbStream, readDigits } from '../src/lib/vision/cm-rgb-dataset.ts'

const root = new URL('..', import.meta.url).pathname
const out = join(root, 'experiments/cm-cnn-rgb-v4/dataset')
mkdirSync(out, { recursive: true })
const digits = readDigits(join(root, 'data/digits-8x8.csv.gz'))
const b = buildRgbBenchmark(digits)
writeFileSync(join(out, 'train.bin'), encodeRgbSet(b.train))
writeFileSync(join(out, 'test.bin'), encodeRgbSet(b.test))
writeFileSync(join(out, 'dev.bin'), encodeRgbSet(b.dev))
writeFileSync(join(out, 'stream.bin'), encodeRgbStream(b.stream))
let mn = 1, mx = 0
for (const v of b.test.pixels) { mn = Math.min(mn, v); mx = Math.max(mx, v) }
console.log(JSON.stringify({ out, train: b.train.count, test: b.test.count, dev: b.dev.count, seqs: b.stream.seqs, min: mn, max: mx, bytes: readFileSync(join(out, 'test.bin')).length }))
