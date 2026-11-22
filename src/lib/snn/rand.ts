// SPDX-License-Identifier: MPL-2.0
// 确定性伪随机数：整个实验必须逐比特可复现，否则"A/B 两种门控的差几个脉冲"
// 这种结论没法复核。splitmix32 + Box-Muller，无外部依赖。

export type Rng = () => number

/** splitmix32：状态是一个 uint32，返回值是 [0,1) 的 32 位精度双精度。 */
export function splitmix32(seed: number): Rng {
  let a = seed >>> 0
  return () => {
    a = (a + 0x9e3779b9) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 16), 0x21f0aaad)
    t = Math.imul(t ^ (t >>> 15), 0x735a2d97)
    return ((t ^ (t >>> 15)) >>> 0) / 4294967296
  }
}

/** 标准正态（Box-Muller，成对生成，丢弃第二个）。 */
export function makeGauss(rng: Rng): () => number {
  let spare = 0
  let hasSpare = false
  return () => {
    if (hasSpare) {
      hasSpare = false
      return spare
    }
    let u = 0, v = 0, s = 0
    do {
      u = rng() * 2 - 1
      v = rng() * 2 - 1
      s = u * u + v * v
    } while (s >= 1 || s === 0)
    const m = Math.sqrt((-2 * Math.log(s)) / s)
    spare = v * m
    hasSpare = true
    return u * m
  }
}
