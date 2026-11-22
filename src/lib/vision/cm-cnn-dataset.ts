// SPDX-License-Identifier: MPL-2.0
//
// CM-CNN-Accel benchmark generator.
//
// The source corpus is the 8x8 UCI-style digit set shipped in
// data/digits-8x8.csv.gz (1797 rows: 64 pixel values 0..16, then the label).
// Every 8x8 digit is bicubically upscaled to a random 18..24 px square,
// translated to a random position on a 32x32 canvas, composited over
// structured distractor strokes and mild sensor noise.  The same generator
// also emits a 6-frame saccade stream (360 sequences x 6 frames) in which the
// digit slides by (+-1, +-1) px per frame over a fixed background.
//
// Everything is seeded; the generated binary files are the single source of
// truth for the native trainer/benchmark.  Nothing here imports from the
// codec; this module is standalone.

import { gunzipSync } from 'node:zlib'

export const IMG_SIZE = 32
export const FRAMES = 6
export const TRAIN_COUNT = 1437
export const TEST_COUNT = 360
export const SOURCE_COUNT = 1797

export interface DigitSample {
  pixels: Float32Array // 64 values in [0,1]
  label: number
}

export interface ImageSet {
  images: Float32Array // count * 32 * 32, row-major, [0,1]
  labels: Uint8Array
  count: number
  h: number
  w: number
}

export interface StreamSet {
  frames: Float32Array // seqs * FRAMES * 32 * 32
  labels: Uint8Array
  seqs: number
  framesPerSeq: number
  h: number
  w: number
}

/** Deterministic 32-bit PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Parse the gzipped 8x8 digit CSV.  Values are divided by 16. */
export function parseDigitsGz(gz: Buffer): DigitSample[] {
  const text = gunzipSync(gz).toString('utf8')
  const out: DigitSample[] = []
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue
    const parts = line.split(',')
    if (parts.length < 65) throw new Error(`bad csv row: ${parts.length} cols`)
    const pixels = new Float32Array(64)
    for (let i = 0; i < 64; i++) pixels[i] = Number(parts[i]) / 16
    const label = Number(parts[64])
    if (!Number.isInteger(label) || label < 0 || label > 9) throw new Error(`bad label ${label}`)
    out.push({ pixels, label })
  }
  if (out.length !== SOURCE_COUNT) throw new Error(`expected ${SOURCE_COUNT} rows, got ${out.length}`)
  return out
}

/** Catmull-Rom bicubic sample with clamped edges. */
function cubic(p0: number, p1: number, p2: number, p3: number, t: number): number {
  const a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3
  const b = p0 - 2.5 * p1 + 2 * p2 - 0.5 * p3
  const c = -0.5 * p0 + 0.5 * p2
  return ((a * t + b) * t + c) * t + p1
}

/** Bicubic upscale of an 8x8 grayscale image to size x size. */
export function bicubicUpscale(src: Float32Array, size: number): Float32Array {
  const dst = new Float32Array(size * size)
  const s = 8 / size
  const clamp = (v: number) => Math.max(0, Math.min(7, v))
  // Two-pass separable Catmull-Rom: 8x8 -> size x 8 -> size x size.
  const tmp = new Float32Array(size * 8)
  for (let y = 0; y < size; y++) {
    const sy = (y + 0.5) * s - 0.5
    const y0 = Math.floor(sy)
    const ty = sy - y0
    const r0 = clamp(y0 - 1)
    const r1 = clamp(y0)
    const r2 = clamp(y0 + 1)
    const r3 = clamp(y0 + 2)
    for (let x = 0; x < 8; x++) {
      tmp[y * 8 + x] = cubic(src[r0 * 8 + x], src[r1 * 8 + x], src[r2 * 8 + x], src[r3 * 8 + x], ty)
    }
  }
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const sx = (x + 0.5) * s - 0.5
      const x0 = Math.floor(sx)
      const tx = sx - x0
      const c0 = clamp(x0 - 1)
      const c1 = clamp(x0)
      const c2 = clamp(x0 + 1)
      const c3 = clamp(x0 + 2)
      dst[y * size + x] = cubic(tmp[y * 8 + c0], tmp[y * 8 + c1], tmp[y * 8 + c2], tmp[y * 8 + c3], tx)
    }
  }
  return dst
}

export interface ComposeOptions {
  /** 0..1 strength of distractor strokes. */
  clutter?: number
  /** uniform sensor-noise amplitude. */
  noise?: number
  /** force a specific scale / position (used by the saccade stream). */
  scale?: number
  dx?: number
  dy?: number
  /** background to reuse (stream frames share one background). */
  background?: Float32Array
}

export interface ComposedSample {
  image: Float32Array
  scale: number
  dx: number
  dy: number
  background: Float32Array
}

/** Structured background: gradient + low-frequency blobs + line strokes. */
export function makeBackground(rng: () => number, clutter = 0.22): Float32Array {
  const bg = new Float32Array(IMG_SIZE * IMG_SIZE)
  const gx = (rng() - 0.5) * 0.10
  const gy = (rng() - 0.5) * 0.10
  const base = 0.02 + rng() * 0.03
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      bg[y * IMG_SIZE + x] = base + gx * (x / 31) + gy * (y / 31)
    }
  }
  // Two low-frequency blobs.
  for (let b = 0; b < 2; b++) {
    const cx = rng() * IMG_SIZE
    const cy = rng() * IMG_SIZE
    const rad = 3 + rng() * 6
    const amp = 0.03 + rng() * 0.05
    for (let y = 0; y < IMG_SIZE; y++) {
      for (let x = 0; x < IMG_SIZE; x++) {
        const d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
        bg[y * IMG_SIZE + x] += amp * Math.exp(-d2 / (2 * rad * rad))
      }
    }
  }
  // Structured distractor strokes: 2..4 anti-aliased line segments.
  const strokes = 2 + Math.floor(rng() * 3)
  for (let s = 0; s < strokes; s++) {
    const x0 = rng() * 31
    const y0 = rng() * 31
    const ang = rng() * Math.PI * 2
    const len = 5 + rng() * 14
    const x1 = x0 + Math.cos(ang) * len
    const y1 = y0 + Math.sin(ang) * len
    const amp = (0.10 + rng() * 0.18) * clutter
    const width = 0.8 + rng() * 1.1
    const vx = x1 - x0
    const vy = y1 - y0
    const vv = vx * vx + vy * vy + 1e-6
    for (let y = 0; y < IMG_SIZE; y++) {
      for (let x = 0; x < IMG_SIZE; x++) {
        const t = Math.max(0, Math.min(1, ((x - x0) * vx + (y - y0) * vy) / vv))
        const px = x0 + t * vx
        const py = y0 + t * vy
        const d = Math.hypot(x - px, y - py)
        if (d < width) bg[y * IMG_SIZE + x] += amp * (1 - d / width)
      }
    }
  }
  // A few isolated dots / corners.
  const dots = 2 + Math.floor(rng() * 3)
  for (let d = 0; d < dots; d++) {
    const cx = Math.floor(rng() * IMG_SIZE)
    const cy = Math.floor(rng() * IMG_SIZE)
    const amp = (0.08 + rng() * 0.12) * clutter
    bg[cy * IMG_SIZE + cx] += amp
  }
  return bg
}

/**
 * Compose one sample: digit (bicubic 18..24 px) over a clutter background.
 * Digit ink is composited with a max-blend so it stays legible over strokes.
 */
export function composeSample(digit: Float32Array, rng: () => number, opts: ComposeOptions = {}): ComposedSample {
  const scale = opts.scale ?? 18 + Math.floor(rng() * 7)
  const dx = opts.dx ?? Math.floor(rng() * (IMG_SIZE - scale + 1))
  const dy = opts.dy ?? Math.floor(rng() * (IMG_SIZE - scale + 1))
  const bg = opts.background ? opts.background.slice() : makeBackground(rng, opts.clutter ?? 0.22)
  const up = bicubicUpscale(digit, scale)
  const image = bg
  for (let y = 0; y < scale; y++) {
    for (let x = 0; x < scale; x++) {
      const v = up[y * scale + x] * 0.95
      const yy = dy + y
      const xx = dx + x
      if (yy < 0 || yy >= IMG_SIZE || xx < 0 || xx >= IMG_SIZE) continue
      const idx = yy * IMG_SIZE + xx
      image[idx] = Math.max(image[idx], v)
    }
  }
  const noise = opts.noise ?? 0.018
  if (noise > 0) {
    for (let i = 0; i < image.length; i++) {
      // sum of two uniforms -> cheap, deterministic, roughly Gaussian noise
      const n = (rng() + rng() - 1) * noise
      image[i] = Math.max(0, Math.min(1, image[i] + n))
    }
  }
  return { image, scale, dx, dy, background: bg }
}

/** Deterministically shuffle indices (Fisher-Yates, seeded). */
export function shuffledIndices(n: number, seed: number): number[] {
  const rng = mulberry32(seed)
  const idx = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    const t = idx[i]
    idx[i] = idx[j]
    idx[j] = t
  }
  return idx
}

export interface BuiltBenchmark {
  train: ImageSet
  test: ImageSet
  stream: StreamSet
  stats: {
    trainClassCounts: number[]
    testClassCounts: number[]
    trainScaleMean: number
    trainObjectFractionMean: number
    trainPixelMean: number
    trainPixelStd: number
    testPixelMean: number
    testPixelStd: number
  }
}

/**
 * Build the full benchmark from parsed digits.
 *
 * Determinism contract: the whole function is a pure function of (digits,
 * seeds).  The runner persists the resulting arrays to .bin files so the
 * native side never re-interprets the randomness.
 */
export function buildBenchmark(digits: DigitSample[], opts: { splitSeed?: number; sampleSeed?: number; streamSeed?: number } = {}): BuiltBenchmark {
  if (digits.length !== SOURCE_COUNT) throw new Error(`expected ${SOURCE_COUNT} digits`)
  const order = shuffledIndices(digits.length, opts.splitSeed ?? 0x5eed1234)
  const trainIdx = order.slice(0, TRAIN_COUNT)
  const testIdx = order.slice(TRAIN_COUNT, TRAIN_COUNT + TEST_COUNT)
  if (trainIdx.length !== TRAIN_COUNT || testIdx.length !== TEST_COUNT) throw new Error('bad split')

  const sampleRng = mulberry32(opts.sampleSeed ?? 0xc0ffee42)
  const trainImages = new Float32Array(TRAIN_COUNT * IMG_SIZE * IMG_SIZE)
  const trainLabels = new Uint8Array(TRAIN_COUNT)
  let scaleSum = 0
  let fracSum = 0
  for (let i = 0; i < TRAIN_COUNT; i++) {
    const d = digits[trainIdx[i]]
    const s = composeSample(d.pixels, sampleRng)
    trainImages.set(s.image, i * IMG_SIZE * IMG_SIZE)
    trainLabels[i] = d.label
    scaleSum += s.scale
    fracSum += (s.scale * s.scale) / (IMG_SIZE * IMG_SIZE)
  }

  const testImages = new Float32Array(TEST_COUNT * IMG_SIZE * IMG_SIZE)
  const testLabels = new Uint8Array(TEST_COUNT)
  for (let i = 0; i < TEST_COUNT; i++) {
    const d = digits[testIdx[i]]
    const s = composeSample(d.pixels, sampleRng)
    testImages.set(s.image, i * IMG_SIZE * IMG_SIZE)
    testLabels[i] = d.label
  }

  // Saccade stream: 6 frames per test object, shared background, +-(1,1) px
  // per frame.  Sampling parameters are independent of the static test image
  // so the stream is not a trivial repeat of the static frames.
  const streamRng = mulberry32(opts.streamSeed ?? 0x51ace5)
  const streamFrames = new Float32Array(TEST_COUNT * FRAMES * IMG_SIZE * IMG_SIZE)
  const streamLabels = new Uint8Array(TEST_COUNT)
  for (let s = 0; s < TEST_COUNT; s++) {
    const d = digits[testIdx[s]]
    const scale = 18 + Math.floor(streamRng() * 5) // 18..22, room to move
    const dir = Math.floor(streamRng() * 4)
    const sx = dir & 1 ? 1 : -1
    const sy = dir & 2 ? 1 : -1
    const margin = 5
    const dx0 = margin + Math.floor(streamRng() * (IMG_SIZE - scale - 2 * margin + 1))
    const dy0 = margin + Math.floor(streamRng() * (IMG_SIZE - scale - 2 * margin + 1))
    const bg = makeBackground(streamRng, 0.22)
    streamLabels[s] = d.label
    for (let f = 0; f < FRAMES; f++) {
      const composed = composeSample(d.pixels, streamRng, {
        scale,
        dx: dx0 + sx * f,
        dy: dy0 + sy * f,
        background: bg,
        noise: 0.014,
      })
      streamFrames.set(composed.image, (s * FRAMES + f) * IMG_SIZE * IMG_SIZE)
    }
  }

  const stats = (arr: Float32Array) => {
    let sum = 0
    let sum2 = 0
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i]
      sum2 += arr[i] * arr[i]
    }
    const mean = sum / arr.length
    return { mean, std: Math.sqrt(Math.max(0, sum2 / arr.length - mean * mean)) }
  }
  const tr = stats(trainImages)
  const te = stats(testImages)
  const trainClassCounts = Array(10).fill(0)
  const testClassCounts = Array(10).fill(0)
  for (const l of trainLabels) trainClassCounts[l]++
  for (const l of testLabels) testClassCounts[l]++

  return {
    train: { images: trainImages, labels: trainLabels, count: TRAIN_COUNT, h: IMG_SIZE, w: IMG_SIZE },
    test: { images: testImages, labels: testLabels, count: TEST_COUNT, h: IMG_SIZE, w: IMG_SIZE },
    stream: { frames: streamFrames, labels: streamLabels, seqs: TEST_COUNT, framesPerSeq: FRAMES, h: IMG_SIZE, w: IMG_SIZE },
    stats: {
      trainClassCounts,
      testClassCounts,
      trainScaleMean: scaleSum / TRAIN_COUNT,
      trainObjectFractionMean: fracSum / TRAIN_COUNT,
      trainPixelMean: tr.mean,
      trainPixelStd: tr.std,
      testPixelMean: te.mean,
      testPixelStd: te.std,
    },
  }
}

/**
 * Build a small held-out "dev" image set for gate hyper-parameter tuning.
 * It uses the same digits and composition process but an independent seed, so
 * it never reuses a training/test *image*.  Tuning thresholds on dev keeps the
 * reported test accuracy honest.
 */
export function buildDevSet(digits: DigitSample[], count = 360, seed = 0x0de0b6b3): ImageSet {
  const rng = mulberry32(seed)
  const images = new Float32Array(count * IMG_SIZE * IMG_SIZE)
  const labels = new Uint8Array(count)
  for (let i = 0; i < count; i++) {
    const d = digits[Math.floor(rng() * digits.length)]
    const s = composeSample(d.pixels, rng)
    images.set(s.image, i * IMG_SIZE * IMG_SIZE)
    labels[i] = d.label
  }
  return { images, labels, count, h: IMG_SIZE, w: IMG_SIZE }
}

// ---------------------------------------------------------------------------
// Binary persistence
// ---------------------------------------------------------------------------

const IMG_MAGIC = 'CMIMG001'
const VID_MAGIC = 'CMVID001'

export function encodeImageSet(set: ImageSet): Buffer {
  const head = Buffer.alloc(32)
  head.write(IMG_MAGIC, 0, 'ascii')
  head.writeUInt32LE(set.count, 8)
  head.writeUInt32LE(set.h, 12)
  head.writeUInt32LE(set.w, 16)
  head.writeUInt32LE(1, 20)
  head.writeUInt32LE(0, 24)
  head.writeUInt32LE(0, 28)
  const labels = Buffer.from(set.labels)
  const pad = (4 - (labels.length % 4)) % 4
  const data = Buffer.from(set.images.buffer, set.images.byteOffset, set.images.byteLength)
  return Buffer.concat([head, labels, Buffer.alloc(pad), data])
}

export function decodeImageSet(buf: Buffer): ImageSet {
  if (buf.toString('ascii', 0, 8) !== IMG_MAGIC) throw new Error('bad image magic')
  const count = buf.readUInt32LE(8)
  const h = buf.readUInt32LE(12)
  const w = buf.readUInt32LE(16)
  const ch = buf.readUInt32LE(20)
  if (ch !== 1) throw new Error(`unsupported channels ${ch}`)
  const labels = new Uint8Array(buf.subarray(32, 32 + count))
  const off = 32 + count + ((4 - (count % 4)) % 4)
  const need = count * h * w * 4
  if (buf.length !== off + need) throw new Error(`image length mismatch ${buf.length} vs ${off + need}`)
  const images = new Float32Array(buf.buffer.slice(buf.byteOffset + off, buf.byteOffset + off + need))
  return { images, labels, count, h, w }
}

export function encodeStreamSet(set: StreamSet): Buffer {
  const head = Buffer.alloc(32)
  head.write(VID_MAGIC, 0, 'ascii')
  head.writeUInt32LE(set.seqs, 8)
  head.writeUInt32LE(set.framesPerSeq, 12)
  head.writeUInt32LE(set.h, 16)
  head.writeUInt32LE(set.w, 20)
  head.writeUInt32LE(1, 24)
  head.writeUInt32LE(0, 28)
  const labels = Buffer.from(set.labels)
  const pad = (4 - (labels.length % 4)) % 4
  const data = Buffer.from(set.frames.buffer, set.frames.byteOffset, set.frames.byteLength)
  return Buffer.concat([head, labels, Buffer.alloc(pad), data])
}

export function decodeStreamSet(buf: Buffer): StreamSet {
  if (buf.toString('ascii', 0, 8) !== VID_MAGIC) throw new Error('bad stream magic')
  const seqs = buf.readUInt32LE(8)
  const framesPerSeq = buf.readUInt32LE(12)
  const h = buf.readUInt32LE(16)
  const w = buf.readUInt32LE(20)
  const labels = new Uint8Array(buf.subarray(32, 32 + seqs))
  const off = 32 + seqs + ((4 - (seqs % 4)) % 4)
  const need = seqs * framesPerSeq * h * w * 4
  if (buf.length !== off + need) throw new Error(`stream length mismatch ${buf.length} vs ${off + need}`)
  const frames = new Float32Array(buf.buffer.slice(buf.byteOffset + off, buf.byteOffset + off + need))
  return { frames, labels, seqs, framesPerSeq, h, w }
}

/** ASCII preview used by tests and by the heatmap documentation. */
export function asciiImage(pixels: Float32Array | ArrayLike<number>, h = IMG_SIZE, w = IMG_SIZE): string {
  const ramp = ' .:-=+*#%@'
  const lines: string[] = []
  for (let y = 0; y < h; y++) {
    let line = ''
    for (let x = 0; x < w; x++) {
      const v = Math.max(0, Math.min(1, pixels[y * w + x] as number))
      line += ramp[Math.floor(v * (ramp.length - 1) + 0.5)]
    }
    lines.push(line)
  }
  return lines.join('\n')
}
