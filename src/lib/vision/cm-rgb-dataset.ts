// SPDX-License-Identifier: MPL-2.0
//
// CM-CNN-RGB-v4 benchmark generator.
//
// 32x32x3 full-canvas procedural scenes: a coloured digit over one of five
// natural RGB texture families (sky/clouds, foliage, water, brick, wood).
// Every pixel is non-zero; there is no blank background to exploit.  The
// digit split is byte-identical to the Phase-A benchmark, so the test digits
// are the same, but the RGB scenes and stream are generated independently.

import { readFileSync } from 'node:fs'
import {
  bicubicUpscale,
  mulberry32,
  parseDigitsGz,
  shuffledIndices,
  type DigitSample,
} from './cm-cnn-dataset.ts'

export const IMG = 32
export const CH = 3
export const FRAMES = 6
export const TRAIN_COUNT = 1437
export const TEST_COUNT = 360
export const SOURCE_COUNT = 1797

export interface RgbSet {
  pixels: Float32Array // count * 32 * 32 * 3
  labels: Uint8Array
  count: number
  h: number
  w: number
  c: number
}

export interface RgbStream {
  pixels: Float32Array // seqs * frames * 32 * 32 * 3
  labels: Uint8Array
  seqs: number
  framesPerSeq: number
  h: number
  w: number
  c: number
}

type Rng = () => number

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v)
const lerp = (a: number, b: number, t: number) => a + (b - a) * t

/** Smooth value noise with a deterministic lattice. */
function valueNoise(rng: Rng, cells: number): (u: number, v: number) => number {
  const g = new Float32Array(cells * cells)
  for (let i = 0; i < g.length; i++) g[i] = rng()
  const at = (x: number, y: number) => g[((y % cells) + cells) % cells * cells + (((x % cells) + cells) % cells)]
  return (u, v) => {
    const x = u * cells, y = v * cells
    const x0 = Math.floor(x), y0 = Math.floor(y)
    const tx = x - x0, ty = y - y0
    const sx = tx * tx * (3 - 2 * tx), sy = ty * ty * (3 - 2 * ty)
    const a = lerp(at(x0, y0), at(x0 + 1, y0), sx)
    const b = lerp(at(x0, y0 + 1), at(x0 + 1, y0 + 1), sx)
    return lerp(a, b, sy)
  }
}

type Texture = (u: number, v: number, rng: Rng) => [number, number, number]

/** Five full-canvas texture families, all with non-zero RGB. */
function makeTexture(kind: number, rng: Rng): Texture {
  if (kind === 0) {
    // sky + clouds
    const n = valueNoise(rng, 6)
    return (u, v) => {
      const cloud = n(u * 1.3, v * 1.7 + 0.2)
      const c = 0.35 + 0.55 * cloud
      return [lerp(0.30, 0.92, c) * 0.62 + 0.10, lerp(0.42, 0.96, c) * 0.72 + 0.12, lerp(0.62, 1.0, c) * 0.92 + 0.06]
    }
  }
  if (kind === 1) {
    // foliage / grass
    const n = valueNoise(rng, 9), m = valueNoise(rng, 17)
    return (u, v) => {
      const leaf = 0.55 * n(u, v) + 0.45 * m(u * 1.7, v * 1.3)
      return [0.10 + 0.28 * leaf, 0.22 + 0.52 * leaf, 0.08 + 0.20 * leaf]
    }
  }
  if (kind === 2) {
    // water waves
    const n = valueNoise(rng, 5)
    return (u, v) => {
      const wave = 0.5 + 0.5 * Math.sin((u * 8 + v * 5 + n(u, v) * 3) * Math.PI)
      return [0.08 + 0.16 * wave, 0.26 + 0.34 * wave, 0.45 + 0.45 * wave]
    }
  }
  if (kind === 3) {
    // brick / grid
    const bw = 0.10 + rng() * 0.06, bh = 0.045 + rng() * 0.03
    const mortar = rng() * 0.5 + 0.5
    return (u, v) => {
      const row = Math.floor(v / bh)
      const uu = (u + (row & 1) * bw * 0.5) / bw
      const fx = uu - Math.floor(uu)
      const fy = v / bh - row
      const edge = Math.min(fx, 1 - fx, fy, 1 - fy)
      if (edge < 0.06) {
        const g = 0.55 + 0.25 * mortar
        return [g, g * 0.97, g * 0.90]
      }
      const tint = 0.5 + 0.5 * n2(u, v, mortar)
      return [0.48 + 0.28 * tint, 0.20 + 0.16 * tint, 0.12 + 0.10 * tint]
    }
  }
  // wood grain
  const n = valueNoise(rng, 7)
  return (u, v) => {
    const rings = 0.5 + 0.5 * Math.sin((v * 14 + n(u, v) * 5 + u * 2) * Math.PI)
    const g = 0.45 + 0.35 * rings
    return [g, g * 0.58, g * 0.28]
  }
}

// small helper used by the brick closure (keeps it deterministic and seeded)
function n2(u: number, v: number, s: number): number {
  const x = Math.sin(u * 91.7 + v * 47.3 + s * 13.1) * 43758.5453
  return x - Math.floor(x)
}

function rgbAt(tex: Texture, u: number, v: number, rng: Rng): [number, number, number] {
  const [r, g, b] = tex(u, v, rng)
  const grain = (rng() - 0.5) * 0.015
  return [clamp01(r + grain), clamp01(g + grain), clamp01(b + grain)]
}

/** HSV -> RGB, all channels in [0,1]. */
function hsv(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6) % 6
  const f = h * 6 - Math.floor(h * 6)
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s)
  switch (i) {
    case 0: return [v, t, p]
    case 1: return [q, v, p]
    case 2: return [p, v, t]
    case 3: return [p, q, v]
    case 4: return [t, p, v]
    default: return [v, p, q]
  }
}

export interface RgbSceneOpts {
  scale?: number
  dx?: number
  dy?: number
  textureKind?: number
  textureSeed?: number
  hue?: number
  noise?: number
}

/** Compose one 32x32 RGB scene with a coloured digit over a full texture. */
export function composeRgb(digit: Float32Array, rng: Rng, opts: RgbSceneOpts = {}): { image: Float32Array; scale: number; dx: number; dy: number } {
  const scale = opts.scale ?? 18 + Math.floor(rng() * 7)
  const dx = opts.dx ?? Math.floor(rng() * (IMG - scale + 1))
  const dy = opts.dy ?? Math.floor(rng() * (IMG - scale + 1))
  const kind = opts.textureKind ?? Math.floor(rng() * 5)
  const tex = makeTexture(kind, mulberry32(opts.textureSeed ?? Math.floor(rng() * 0xffffffff)))
  const hue = opts.hue ?? rng()
  const fg = hsv(hue, 0.72 + 0.22 * rng(), 0.82 + 0.16 * rng())
  const accent = hsv((hue + 0.5) % 1, 0.65, 0.85)
  const up = bicubicUpscale(digit, scale)
  const img = new Float32Array(IMG * IMG * 3)
  for (let y = 0; y < IMG; y++) {
    for (let x = 0; x < IMG; x++) {
      let [r, g, b] = rgbAt(tex, x / IMG, y / IMG, rng)
      // low-frequency lighting so the texture is not spatially stationary
      const light = 0.78 + 0.30 * (0.5 + 0.5 * Math.sin((x * 0.21 + y * 0.13) + kind))
      r *= light; g *= light; b *= light
      // Mute the texture so the coloured foreground object carries the class signal.
      r = 0.45 * r + 0.26; g = 0.45 * g + 0.26; b = 0.45 * b + 0.26
      const sx = x - dx, sy = y - dy
      if (sx >= 0 && sy >= 0 && sx < scale && sy < scale) {
        const a = clamp01(up[sy * scale + sx] * 1.25)
        const edge = Math.min(sx, sy, scale - 1 - sx, scale - 1 - sy)
        const rim = edge < 1.5 ? 0.55 : 0
        const fr = fg[0] * (1 - rim) + accent[0] * rim
        const fgG = fg[1] * (1 - rim) + accent[1] * rim
        const fb = fg[2] * (1 - rim) + accent[2] * rim
        r = lerp(r, fr, a)
        g = lerp(g, fgG, a)
        b = lerp(b, fb, a)
      }
      const nz = (rng() - 0.5) * (opts.noise ?? 0.012)
      const o = (y * IMG + x) * 3
      img[o] = clamp01(r + nz)
      img[o + 1] = clamp01(g + nz)
      img[o + 2] = clamp01(b + nz)
    }
  }
  // 100% non-zero guarantee (no blank background anywhere)
  for (let i = 0; i < img.length; i++) if (img[i] < 0.005) img[i] = 0.005
  return { image: img, scale, dx, dy }
}

export interface BuiltRgb {
  train: RgbSet
  test: RgbSet
  dev: RgbSet
  stream: RgbStream
}

/** Build the deterministic RGB benchmark (same digit split as Phase A). */
export function buildRgbBenchmark(digits: DigitSample[]): BuiltRgb {
  const order = shuffledIndices(digits.length, 0x5eed1234)
  const trainIdx = order.slice(0, TRAIN_COUNT)
  const testIdx = order.slice(TRAIN_COUNT, TRAIN_COUNT + TEST_COUNT)
  const rng = mulberry32(0x22b16e5)
  const train = { pixels: new Float32Array(TRAIN_COUNT * IMG * IMG * CH), labels: new Uint8Array(TRAIN_COUNT), count: TRAIN_COUNT, h: IMG, w: IMG, c: CH }
  for (let i = 0; i < TRAIN_COUNT; i++) {
    const d = digits[trainIdx[i]]
    const s = composeRgb(d.pixels, rng)
    train.pixels.set(s.image, i * IMG * IMG * CH)
    train.labels[i] = d.label
  }
  const test = { pixels: new Float32Array(TEST_COUNT * IMG * IMG * CH), labels: new Uint8Array(TEST_COUNT), count: TEST_COUNT, h: IMG, w: IMG, c: CH }
  for (let i = 0; i < TEST_COUNT; i++) {
    const d = digits[testIdx[i]]
    const s = composeRgb(d.pixels, rng)
    test.pixels.set(s.image, i * IMG * IMG * CH)
    test.labels[i] = d.label
  }
  const devRng = mulberry32(0x0de0b6b4)
  const dev = { pixels: new Float32Array(TEST_COUNT * IMG * IMG * CH), labels: new Uint8Array(TEST_COUNT), count: TEST_COUNT, h: IMG, w: IMG, c: CH }
  for (let i = 0; i < TEST_COUNT; i++) {
    const d = digits[Math.floor(devRng() * digits.length)]
    const s = composeRgb(d.pixels, devRng)
    dev.pixels.set(s.image, i * IMG * IMG * CH)
    dev.labels[i] = d.label
  }
  const srng = mulberry32(0x7c0a57)
  const stream = { pixels: new Float32Array(TEST_COUNT * FRAMES * IMG * IMG * CH), labels: new Uint8Array(TEST_COUNT), seqs: TEST_COUNT, framesPerSeq: FRAMES, h: IMG, w: IMG, c: CH }
  for (let s = 0; s < TEST_COUNT; s++) {
    const d = digits[testIdx[s]]
    const scale = 18 + Math.floor(srng() * 5)
    const margin = 4
    const dx0 = margin + Math.floor(srng() * (IMG - scale - 2 * margin + 1))
    const dy0 = margin + Math.floor(srng() * (IMG - scale - 2 * margin + 1))
    const dir = Math.floor(srng() * 4)
    const sx = dir & 1 ? 1 : -1, sy = dir & 2 ? 1 : -1
    const kind = Math.floor(srng() * 5)
    const seed = Math.floor(srng() * 0xffffffff)
    const hue = srng()
    const tex = makeTexture(kind, mulberry32(seed))
    stream.labels[s] = d.label
    for (let f = 0; f < FRAMES; f++) {
      // Reuse the same texture function closure for all frames.
      const frameRng = mulberry32(0xabc000 + s * 37 + f)
      const img = new Float32Array(IMG * IMG * CH)
      const up = bicubicUpscale(d.pixels, scale)
      const fg = hsv(hue, 0.86, 0.94)
      const dx = dx0 + sx * f, dy = dy0 + sy * f
      for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
        let [r, g, b] = rgbAt(tex, x / IMG, y / IMG, frameRng)
        const light = 0.8 + 0.25 * (0.5 + 0.5 * Math.sin((x * 0.2 + y * 0.15) + kind))
        r *= light; g *= light; b *= light
        const sx2 = x - dx, sy2 = y - dy
        if (sx2 >= 0 && sy2 >= 0 && sx2 < scale && sy2 < scale) {
          const a = Math.pow(clamp01(up[sy2 * scale + sx2]), 0.75)
          r = lerp(r, fg[0], a * 0.96); g = lerp(g, fg[1], a * 0.96); b = lerp(b, fg[2], a * 0.96)
        }
        const nz = (frameRng() - 0.5) * 0.012
        const o = (y * IMG + x) * 3
        img[o] = clamp01(r + nz); img[o + 1] = clamp01(g + nz); img[o + 2] = clamp01(b + nz)
      }
      for (let i = 0; i < img.length; i++) if (img[i] < 0.005) img[i] = 0.005
      stream.pixels.set(img, (s * FRAMES + f) * IMG * IMG * CH)
    }
  }
  return { train, test, dev, stream }
}

// ---------------------------------------------------------------------------
// Binary persistence (CMRGB001 / CMRGBV01)
// ---------------------------------------------------------------------------

export function encodeRgbSet(set: RgbSet): Buffer {
  const head = Buffer.alloc(32)
  head.write('CMRGB001', 0, 'ascii')
  head.writeUInt32LE(set.count, 8)
  head.writeUInt32LE(set.h, 12)
  head.writeUInt32LE(set.w, 16)
  head.writeUInt32LE(set.c, 20)
  const labels = Buffer.from(set.labels)
  const pad = (4 - (labels.length % 4)) % 4
  return Buffer.concat([head, labels, Buffer.alloc(pad), Buffer.from(set.pixels.buffer, set.pixels.byteOffset, set.pixels.byteLength)])
}

export function decodeRgbSet(buf: Buffer): RgbSet {
  if (buf.subarray(0, 8).toString('ascii') !== 'CMRGB001') throw new Error('bad RGB set magic')
  const count = buf.readUInt32LE(8), h = buf.readUInt32LE(12), w = buf.readUInt32LE(16), c = buf.readUInt32LE(20)
  const labels = new Uint8Array(buf.subarray(32, 32 + count))
  const off = 32 + count + ((4 - (count % 4)) % 4)
  const pixels = new Float32Array(count * h * w * c)
  Buffer.from(pixels.buffer).set(buf.subarray(off, off + pixels.byteLength))
  return { pixels, labels, count, h, w, c }
}

export function encodeRgbStream(s: RgbStream): Buffer {
  const head = Buffer.alloc(32)
  head.write('CMRGBV01', 0, 'ascii')
  head.writeUInt32LE(s.seqs, 8)
  head.writeUInt32LE(s.framesPerSeq, 12)
  head.writeUInt32LE(s.h, 16)
  head.writeUInt32LE(s.w, 20)
  head.writeUInt32LE(s.c, 24)
  const labels = Buffer.from(s.labels)
  const pad = (4 - (labels.length % 4)) % 4
  return Buffer.concat([head, labels, Buffer.alloc(pad), Buffer.from(s.pixels.buffer, s.pixels.byteOffset, s.pixels.byteLength)])
}

export function decodeRgbStream(buf: Buffer): RgbStream {
  if (buf.subarray(0, 8).toString('ascii') !== 'CMRGBV01') throw new Error('bad RGB stream magic')
  const seqs = buf.readUInt32LE(8), framesPerSeq = buf.readUInt32LE(12), h = buf.readUInt32LE(16), w = buf.readUInt32LE(20), c = buf.readUInt32LE(24)
  const labels = new Uint8Array(buf.subarray(32, 32 + seqs))
  const off = 32 + seqs + ((4 - (seqs % 4)) % 4)
  const pixels = new Float32Array(seqs * framesPerSeq * h * w * c)
  Buffer.from(pixels.buffer).set(buf.subarray(off, off + pixels.byteLength))
  return { pixels, labels, seqs, framesPerSeq, h, w, c }
}

export function readDigits(path: string): DigitSample[] {
  return parseDigitsGz(readFileSync(path))
}
