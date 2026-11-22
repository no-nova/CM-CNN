// SPDX-License-Identifier: MPL-2.0
//
// CM-SNN 实验的不变量测试。
//
// 这个文件只测**应该永远成立**的命题，不测"这次跑出来是多少"——后者属于
// 实验报告（docs/SNN_CM.md）与结果回执（experiments/snn-results.json）。
// 之所以区分开：跑分会被机器和随机种子影响，而不变量不会；把两者混在一起
// 会让 CI 在换机器时莫名其妙地红。
//
// 用 `node tests/run-snn.mjs` 跑（esbuild 打包后交给 node，与 tests/run.mjs 同套路）。

import { strict as assert } from 'node:assert'
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { spawnSync } from 'node:child_process'
import {
  RangeDecoder, RangeEncoder, squash, stretch, hh, PSCALE, bitCost,
} from '../src/lib/snn/cm'
import {
  boxCountScratch, boxCountsFromII, buildIntegral, buildLayer, lifSequential, psnScan,
  psnToeplitz, resetLayer, selectThresh, selectTopB, stepLayer, outSize, TOPB_BINS,
  type GateMode, type LayerSpec, type StepOptions, type StepStats,
} from '../src/lib/snn/snn'
import { SpikePredictor, quantCount } from '../src/lib/snn/predict'
import { Net, runExperiment, runTheoremCheck, defaultLayers, DEFAULT_EXP } from '../src/lib/snn/experiment'
import {
  l2NormalizeRows, makeMovingBarDataset, readoutAccuracy, trainReadout,
} from '../src/lib/snn/task'
import { splitmix32 } from '../src/lib/snn/rand'
import { LocalPlasticity } from '../src/lib/snn/plastic'
import { runPlasticDigits } from '../src/lib/snn/plastic-experiment'
import { denseLogits, loadDenseCheckpoint, loadInferenceCheckpoint, saveCmCheckpoint, saveDenseCheckpoint, type DenseCheckpoint } from '../src/lib/snn/checkpoint'
import { ClassModulatedPlasticity } from '../src/lib/snn/class-plastic'
import { runClassExperiment } from '../src/lib/snn/class-experiment'
import { runValidation, VALIDATION_DEFAULT } from '../src/lib/snn/validation'
import { loadDigits, makeDigitsDataset, makeIndexedDigitsDataset } from '../src/lib/snn/task'
import { augmentEvents, IntrinsicHomeostasis, PrototypeContrastive } from '../src/lib/snn/prototype'
import { PROTOTYPE_DEFAULT, runPrototypeExperiment, validatePrototypeConfig } from '../src/lib/snn/prototype-experiment'
import { FeatureContextMixer, StreamAgent, entropyGate } from '../src/lib/snn/stream'
import { runStreamExperiment, STREAM_DEFAULT } from '../src/lib/snn/stream-experiment'
import { IsolatedAgent, prototypeEta } from '../src/lib/snn/continual'
import { runContinualSeed, runContinualExperiment, CONTINUAL_DEFAULT } from '../src/lib/snn/continual-experiment'
import { SpikeByteModel } from '../src/lib/sequence/spike-byte'
import { MultiscaleByteModel } from '../src/lib/sequence/multiscale-byte'
import { CoarseByteModel } from '../src/lib/sequence/coarse-byte'

const TINY = {
  T: 6, H: 12, W: 12, trainPerClass: 2, testPerClass: 2, budgets: [0.5, 0.2],
  seed: 0x1234, bound: 'perCh' as const, pred: {
    bits: 12, mixerSets: 4, apmOn: true, apmCtx: 4, stateContexts: false,
  },
  shape: { in1: 1, ch1: 4, k2: 5, ch2: 8 },
}

// ===== 1. CM 内核：与 UCZ 压缩器同一套定点约定 =====

export function testCmPrimitives(): void {
  // squash(0) 必须落在 0.5 附近（±1/4096）。UCZ 的插值表在 0 附近差一个单位
  // （2047 而非 2048），移植时原样保留——差这 1/4096 不影响任何结论。
  const half = squash(0)
  assert.ok(Math.abs(half - (PSCALE >> 1)) <= 1, `squash(0) 应≈0.5，实测 ${half}`)
  assert.ok(Math.abs(squash(-1) + squash(1) - (PSCALE - 1)) <= 2, 'squash 应近似关于 0 对称')
  for (const p of [1, 2, 17, 512, 2047, 2048, 3000, 4094, 4095]) {
    const back = squash(stretch(p))
    assert.ok(Math.abs(back - p) <= 8, `stretch/squash 往返失配: p=${p} -> ${back}`)
  }
  assert.equal(squash(100000), PSCALE - 1)
  assert.equal(squash(-100000), 0)
  // 单调性：概率模型的查表一旦不单调，编码器会写出负数区间
  let prev = -1
  for (let d = -2047; d <= 2047; d += 7) {
    const v = squash(d)
    assert.ok(v >= prev, `squash 非单调于 d=${d}`)
    prev = v
  }
  // hh 的尾部折叠：显式传 0 与省略参数必须逐位相同（coder.ts 的同款优化）
  assert.equal(hh(7), hh(7, 0, 0, 0, 0, 0))
  assert.equal(hh(7, 9), hh(7, 9, 0, 0, 0, 0))
  assert.equal(hh(7, 9, 11), hh(7, 9, 11, 0, 0, 0))
  assert.notEqual(hh(1, 2), hh(2, 1))
  // 理论码长
  assert.ok(Math.abs(bitCost(PSCALE >> 1, 1) - 1) < 0.01, 'p=0.5 时应为 1 比特')
  assert.ok(bitCost(4000, 1) < 0.05, 'p→1 时码长应趋于 0')
  assert.ok(bitCost(1, 1) > 11, 'p→0 时码长应大于 11 比特')
}

export function testRangeCoderRoundTrip(): void {
  const rng = splitmix32(20240607)
  // 覆盖"极端偏斜"这一最容易被写反的情形：区间划分用的是 0 的份额。
  for (const p1 of [0.5, 0.25, 0.1, 0.02, 0.002]) {
    const p12 = Math.round(p1 * PSCALE)
    const bits: number[] = []
    const enc = new RangeEncoder()
    let ideal = 0
    for (let i = 0; i < 4000; i++) {
      const b = rng() < p1 ? 1 : 0
      bits.push(b)
      enc.encode(b, p12)
      ideal += bitCost(p12, b)
    }
    const out = enc.finish()
    const dec = new RangeDecoder(out)
    for (let i = 0; i < bits.length; i++) {
      assert.equal(dec.decode(p12), bits[i], `p1=${p1} 第 ${i} 位解码不一致`)
    }
    // 真编码长度必须贴近理论码长（< 1.5%），这是"编码器没写错"的硬闸门。
    // 先扣掉 finish() 的 5 字节固定收尾（那是终止码，不是预测误差）。
    const overhead = (out.length * 8 - 5 * 8 - ideal) / ideal
    assert.ok(overhead < 0.015, `p1=${p1} 编码开销 ${(overhead * 100).toFixed(2)}% 过大`)
    assert.ok(out.length * 8 > ideal * 0.9, '编码长度不该低于熵下界')
  }
  // 空流之外的健壮性
  assert.throws(() => new RangeDecoder(new Uint8Array([1, 2, 3, 4, 5])), /首字节/)
}

// ===== 2. 多尺度盒滤波：门控的"小型 CNN 卷积部分" =====

export function testBoxCounts(): void {
  const H = 8, W = 8, inCh = 2
  const S = new Uint8Array(inCh * H * W)
  // 在通道 0 的 (3,4) 与通道 1 的 (3,5) 各放一个脉冲
  S[0 * H * W + 3 * W + 4] = 1
  S[1 * H * W + 3 * W + 5] = 1
  const scratch = boxCountScratch(inCh, H, W)
  const out = new Int32Array(H * W * inCh)
  buildIntegral(S, inCh, H, W, scratch)
  boxCountsFromII(scratch, inCh, H, W, 3, 1, 1, H, W, out)
  // 盒以 (oy,ox) 为中心、半宽 1 ⇒ 3×3；(3,4) 处的盒应同时看到两个脉冲
  assert.equal(out[(3 * W + 4) * inCh + 0], 1, '通道 0 盒内应有 1 个脉冲')
  assert.equal(out[(3 * W + 4) * inCh + 1], 1, '通道 1 的脉冲也应落进同一盒')
  assert.equal(out[(0 * W + 0) * inCh + 0], 0, '远处盒内应无脉冲')
  // 盒和必须与暴力统计逐点一致（含边界裁剪）。
  // 覆盖偶数核与大核：`--k2 10` 这类配置会让输出图比输入大 1（k 为偶数时
  // pad = k>>1 不是完美对称），这里就是防止那种配置悄悄算错。
  for (const k of [1, 2, 3, 5, 6, 10]) {
    for (const pad of [0, 1, 2, 3, 5]) {
      const oh = outSize(H, k, 1, pad), ow = outSize(W, k, 1, pad)
      const o2 = new Int32Array(oh * ow * inCh)
      boxCountsFromII(scratch, inCh, H, W, k, 1, pad, oh, ow, o2)
      for (let oy = 0; oy < oh; oy++) {
        for (let ox = 0; ox < ow; ox++) {
          for (let ic = 0; ic < inCh; ic++) {
            let brute = 0
            for (let ky = 0; ky < k; ky++) {
              for (let kx = 0; kx < k; kx++) {
                const y = oy - pad + ky, x = ox - pad + kx
                if (y >= 0 && y < H && x >= 0 && x < W) brute += S[ic * H * W + y * W + x]
              }
            }
            assert.equal(o2[(oy * ow + ox) * inCh + ic], brute,
              `积分图与暴力统计不符 k=${k} pad=${pad} @(${oy},${ox},${ic})`)
          }
        }
      }
    }
  }
}

// ===== 3. 门控：可证明无损的那一条 =====

function singleLayerSpec(hardReset: boolean): LayerSpec {
  return { inCh: 2, outCh: 3, inH: 8, inW: 8, k: 3, stride: 1, pad: 1, decay: 0.9, thresh: 0.35, scale: 0.5, seed: 4242, hardReset }
}

/** 造一串随机输入脉冲流，逐模式跑一层，返回脉冲张量。 */
function runStream(spec: LayerSpec, frames: Uint8Array, T: number, mode: GateMode, budget = 1, offsets?: number[]): Uint8Array {
  const L = buildLayer(spec)
  if (offsets) L.thresholdOffsets.set(offsets)
  resetLayer(L)
  const scratch = boxCountScratch(spec.inCh, spec.inH, spec.inW)
  const counts = new Int32Array(L.outH * L.outW * spec.inCh)
  const out = new Uint8Array(T * L.N)
  const st: StepStats = { computed: 0, skipped: 0, spikes: 0, macs: 0, gateOps: 0, runs: 0, runLen: 0, tiles: 0 }
  const opts: StepOptions = {
    mode, bound: 'perCh', budget: Math.max(1, Math.round(budget * L.N)),
    select: new Uint8Array(L.N), take: new Uint8Array(L.N), hist: new Int32Array(TOPB_BINS),
  }
  for (let t = 0; t < T; t++) {
    const input = frames.subarray(t * spec.inCh * spec.inH * spec.inW, (t + 1) * spec.inCh * spec.inH * spec.inW)
    buildIntegral(input, spec.inCh, spec.inH, spec.inW, scratch)
    boxCountsFromII(scratch, spec.inCh, spec.inH, spec.inW, spec.k, spec.stride, spec.pad, L.outH, L.outW, counts)
    stepLayer(L, input, counts, opts, L.spk, st)
    out.set(L.spk, t * L.N)
  }
  return out
}

export function testEventGatingIsLossless(): void {
  // 定理：硬重置 ⟹ 膜电位恒 < θ ⟹ 感受野盒内无脉冲时电流为 0 ⟹ 不可能发放。
  // 所以"零输入跳过"对 LIF 是逐位无损的，与稠密逐比特相同。
  const rng = splitmix32(555)
  for (const hardReset of [true, false]) {
    for (const density of [0.02, 0.1, 0.3]) {
      const spec = singleLayerSpec(hardReset)
      const T = 12
      const frames = new Uint8Array(T * spec.inCh * spec.inH * spec.inW)
      for (let i = 0; i < frames.length; i++) frames[i] = rng() < density ? 1 : 0
      const dense = runStream(spec, frames, T, 'dense')
      const event = runStream(spec, frames, T, 'event')
      assert.deepEqual(event, dense, `event 门控必须逐位无损 (hardReset=${hardReset}, density=${density})`)
    }
  }
}

export function testCertificateNeverMissesWithoutReset(): void {
  // 无重置 ⟹ 携带状态恒为真值上界 ⟹ 证书判定"不会发放"时一定不会发放。
  const rng = splitmix32(9001)
  for (const density of [0.02, 0.1, 0.3]) {
    const spec = singleLayerSpec(false)
    const T = 16
    const frames = new Uint8Array(T * spec.inCh * spec.inH * spec.inW)
    for (let i = 0; i < frames.length; i++) frames[i] = rng() < density ? 1 : 0
    const dense = runStream(spec, frames, T, 'dense')
    const cert = runStream(spec, frames, T, 'cert')
    let missed = 0
    for (let i = 0; i < dense.length; i++) if (dense[i] && !cert[i]) missed++
    assert.equal(missed, 0, `无重置网络的上界证书漏了 ${missed} 个脉冲 (density=${density})`)
  }
}

export function testAdaptiveThresholdGateSafety(): void {
  const rng = splitmix32(105)
  const offsets = [-0.1, 0.04, 0.12]
  const T = 12, spec = singleLayerSpec(false)
  const frames = new Uint8Array(T * spec.inCh * spec.inH * spec.inW)
  for (let i = 0; i < frames.length; i++) frames[i] = rng() < 0.14 ? 1 : 0
  const dense = runStream(spec, frames, T, 'dense', 1, offsets)
  const event = runStream(spec, frames, T, 'event', 1, offsets)
  const cert = runStream(spec, frames, T, 'cert', 1, offsets)
  assert.deepEqual(event, dense, '自适应阈值下 event 应逐位无损')
  for (let i = 0; i < dense.length; i++) assert.ok(!dense[i] || cert[i], '无重置证书不应漏脉冲')
}

export function testHardResetBreaksMonotonicity(): void {
  // 反向命题：硬重置时同样的证书**会**漏脉冲（这正是 C2 节要展示的事实）。
  // 这条测试是"我们的理论解释是对的"的证据，而不是缺陷。
  const rows = runTheoremCheck({ ...TINY } as never, makeMovingBarDataset({
    perClass: 1, testPerClass: 2, T: TINY.T, H: TINY.H, W: TINY.W, seed: 7,
  }).test)
  const nr = rows.noReset.reduce((a, x) => a + x.miss, 0)
  const hr = rows.hardReset.reduce((a, x) => a + x.miss, 0)
  assert.equal(nr, 0, '无重置时证书不应漏脉冲')
  assert.ok(hr > 0, '硬重置时证书应当漏脉冲（否则说明"重置破坏单调性"的解释不成立）')
}

// ===== 4. 时间折叠（可能性 2）=====

export function testTimeFolding(): void {
  const T = 12, N = 5
  const rng = splitmix32(31337)
  const I = new Float32Array(T * N)
  for (let i = 0; i < I.length; i++) I[i] = rng() * 1.2 - 0.4
  const seq = lifSequential(I, T, N, 0.9, 0.8, false)
  const toe = psnToeplitz(I, T, N, 0.9, 0.8)
  const scan = psnScan(I, T, N, 0.9, 0.8)
  const maxErr = (a: Float32Array, b: Float32Array) => {
    let m = 0
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
    return m
  }
  // 无重置时膜电位对输入是线性的 ⟹ T 步递归可以折成一次带因果掩码的矩阵乘
  assert.ok(maxErr(seq.v, toe.v) < 1e-5, `Toeplitz 与逐步递归不符: ${maxErr(seq.v, toe.v)}`)
  assert.ok(maxErr(seq.v, scan.v) < 1e-5, `结合扫描与逐步递归不符: ${maxErr(seq.v, scan.v)}`)
  assert.deepEqual(Array.from(toe.s), Array.from(seq.s), 'Toeplitz 脉冲串应与逐步递归一致')
  assert.deepEqual(Array.from(scan.s), Array.from(seq.s), '扫描脉冲串应与逐步递归一致')
  // 深度：折叠后是 O(1)/O(log T)，逐步是 O(T)
  assert.equal(toe.depth, 1)
  assert.ok(scan.depth < T)
  assert.equal(seq.depth, T)
  // 带重置的 LIF 折不了：与无重置的脉冲串必须有实质差异
  const lif = lifSequential(I, T, N, 0.9, 0.8, true)
  let diff = 0
  for (let i = 0; i < lif.s.length; i++) if (lif.s[i] !== seq.s[i]) diff++
  assert.ok(diff > 0, '带重置与无重置的脉冲串必须有差异（否则"重置让时间折不了"无从谈起）')
}

// ===== 5. 预测器与选点 =====

export function testPredictorLearnsOnline(): void {
  // 零先验：第 0 步只能给 p=0.5；喂一段有规律的序列后，对数损失必须下降。
  const N = 64, outCh = 8, inCh = 1
  const pred = new SpikePredictor(N, outCh, inCh, 99, { bits: 12, mixerSets: 4, apmOn: true, apmCtx: 4, stateContexts: false })
  const cnt1 = new Int32Array((N / outCh) * inCh)
  const cnt2 = new Int32Array((N / outCh) * inCh)
  const v = new Float32Array(N)
  const spk = new Uint8Array(N)
  const rng = splitmix32(4)
  const early: number[] = []
  const late: number[] = []
  const T = 400
  for (let t = 0; t < T; t++) {
    // 规律：神经元 i 在 t % (i % 5 + 2) === 0 时发放
    for (let i = 0; i < N; i++) spk[i] = (t % ((i % 5) + 2)) === 0 ? 1 : 0
    // 让盒计数与发放相关，给特征一点信号
    for (let p = 0; p < N / outCh; p++) {
      const c = spk[p * outCh] ? 3 : 0
      cnt1[p] = c
      cnt2[p] = c + (rng() < 0.5 ? 1 : 0)
    }
    const before = pred.logLoss
    // all=true：关掉"盒内无脉冲 ⟹ 不可能发放"的前置过滤，让概率一路都真的算
    // （这条测试只考察预测器本身能不能学，过滤是另一码事）。
    pred.predict(cnt1, cnt2, v, 1, t, true)
    pred.learnWithCounts(spk, cnt1, cnt2, v, 1, t, true)
    const step = pred.logLoss - before
    if (t < 64) early.push(step)
    if (t > T - 64) late.push(step)
  }
  const avg = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length / N
  const e = avg(early), l = avg(late)
  assert.ok(l < e - 0.05, `在线学习没有生效：早期 ${e.toFixed(4)} → 晚期 ${l.toFixed(4)} 比特/神经元步`)
  assert.ok(l < 0.9, `晚期对数损失应明显好过 1 比特，实测 ${l.toFixed(4)}`)
}

export function testSelectTopBRespectsBudgetAndFilter(): void {
  const n = 100
  const p = new Float32Array(n)
  const out = new Uint8Array(n)
  const hist = new Int32Array(64)
  for (let i = 0; i < n; i++) p[i] = i / n
  // 预算正好：高分段的 30 个
  assert.equal(selectTopB(p, n, 30, out, hist), 30)
  let cnt = 0
  for (let i = 0; i < n; i++) if (out[i]) cnt++
  assert.equal(cnt, 30)
  // 分桶并列的神经元按索引顺序取舍（概率量化到 32 个桶），所以顺序断言只能
  // 在**桶的粒度**上做：落选者的桶不得高于入选者的桶。
  const bucketOf = (x: number) => Math.min(31, Math.max(0, (x * 32) | 0))
  for (let i = 0; i < n; i++) {
    if (out[i]) continue
    for (let j = 0; j < n; j++) {
      if (!out[j]) continue
      assert.ok(bucketOf(p[j]) >= bucketOf(p[i]),
        `选了第 ${j} 个(${p[j]})却没选桶更高的第 ${i} 个(${p[i]})`)
    }
  }
  // 前置过滤器标记（p < 0）不参与排序、不占预算
  for (let i = 0; i < 50; i++) p[i] = -1
  assert.equal(selectTopB(p, n, 30, out, hist), 30)
  for (let i = 0; i < 50; i++) assert.equal(out[i], 0, '被过滤掉的神经元不应入选')
  // 预算超过合格数时全选合格者
  assert.equal(selectTopB(p, n, 999, out, hist), 50)
  // 计数分档
  assert.equal(quantCount(0), 0)
  assert.equal(quantCount(1000), 7)
}

// ===== 6. 端到端：小样本跑通全部口径 =====

export function testEndToEndTiny(): void {
  const res = runExperiment(TINY)
  const byMode = (m: string) => res.runs.filter(x => x.mode === m)
  assert.ok(res.runs.length >= 8, '各档门控都应跑出来')
  // 稠密自比：0 漏 0 多，比值为 1
  const dense = res.runs[0]
  assert.equal(dense.mode, 'dense')
  assert.equal(dense.missRate, 0)
  assert.equal(dense.extraRate, 0)
  assert.ok(Math.abs(dense.macRatio - 1) < 1e-9)
  // 事件驱动：逐位无损，且乘加一定少于稠密
  const ev = byMode('event')[0]
  assert.equal(ev.missRate, 0, 'event 门控必须无损')
  assert.equal(ev.extraRate, 0, 'event 门控必须无损')
  assert.ok(ev.macs < dense.macs, 'event 门控应比稠密省乘加')
  // 预测器门控：预算越小，候选越少、漏得越多（单调性）。
  // 只比"逐神经元 + 有教师"的那一组：自训练用的是另一个 Net，纤维块是另一种
  // 粒度，混进单调性检查里都不是同一个变量。
  const cm = byMode('cm-decay').filter(x => !x.label.includes('自训练') && !x.group)
  assert.ok(cm.length >= 2)
  for (let i = 1; i < cm.length; i++) {
    assert.ok(cm[i].candidateFrac < cm[i - 1].candidateFrac, '预算收紧时候选比例必须下降')
    assert.ok(cm[i].missRate > cm[i - 1].missRate, '预算收紧时漏脉冲率必须上升')
    assert.ok(cm[i].macs < cm[i - 1].macs, '预算收紧时乘加必须下降')
  }
  // 同等候选比例下，预测器排序必须显著优于随机（这是整个想法的核心断言）
  const rnd = byMode('random')[0]
  const cmSame = cm.find(x => Math.abs(x.budgetFrac - rnd.budgetFrac) < 1e-9)!
  assert.ok(Math.abs(cmSame.candidateFrac - rnd.candidateFrac) < 0.05, '两组候选比例应接近才可比')
  assert.ok(cmSame.missRate < rnd.missRate - 0.05,
    `CM 排序应显著优于随机：${cmSame.missRate.toFixed(3)} vs ${rnd.missRate.toFixed(3)}`)
  // 压缩：CM 必须同时优于"原样存"和"逐神经元自适应码"
  for (const c of res.coding) {
    assert.ok(c.cmIdealBits < c.rawBits, 'CM 应优于原样存')
    assert.ok(c.cmIdealBits < c.perNeuronBits, 'CM 应优于逐神经元自适应码')
    assert.ok(Math.abs(c.coderOverhead) < 0.05, `区间编码开销异常: ${(c.coderOverhead * 100).toFixed(2)}%`)
    // 校准：分桶后的平均预测应贴近实测发放率（只检查样本量足够的桶）
    for (const b of c.calibration) {
      // 硬检查：前置过滤标记的 -1 绝不能漏进统计表（否则平均预测会是负数）
      assert.ok(b.pred >= 0 && b.pred <= 1, '分桶平均预测越界（前置过滤的 -1 漏进来了）')
      // 软检查：只在样本量足够时才谈校准。样本太少时上下文刚建立、计数还没
      // 收敛，过预测是**正常的**，不是 bug——报告里那张表就是用来显示这一点的。
      if (b.n < 5000) continue
      assert.ok(Math.abs(b.pred - b.obs) < 0.15, `校准偏差过大: 预测 ${b.pred} 实测 ${b.obs} (n=${b.n})`)
    }
  }
  // 熵下界：信息论给出的无损门控下界必须落在 (0, 候选比例) 之间
  const ent = res.coding[res.coding.length - 1].entropyFloor
  assert.ok(ent > 0 && ent < 1, `熵下界应在 0..1 比特之间，实测 ${ent}`)
  // 时间折叠
  assert.ok(Math.abs(res.timeFold.psnSeqVsToeplitzMaxErr) < 1e-4)
  assert.ok(Math.abs(res.timeFold.psnSeqVsScanMaxErr) < 1e-4)
  assert.equal(res.timeFold.toeplitzDepth, 1)
  // 定理
  assert.equal(res.theorem.noReset.reduce((a, x) => a + x.miss, 0), 0)
  assert.ok(res.theorem.hardReset.reduce((a, x) => a + x.miss, 0) > 0)
}

/**
 * fanin 越大，预测器的固定开销越摊得薄——这是 §7 的判据，这里把它锁成不变量。
 *
 * 只锁**预测器**那一项（它才是 ∝ 1/fanin 的）：盒滤波 + top-B 选取里含一个与
 * fanin 无关的常数地板（每步 `2×TOPB_BINS` 次直方图清零/阈值搜索），小 fanin 时
 * 这个地板很显眼，但它是"选取"的成本，不是"预测"的成本。
 *
 * 断言取方向和量级，不锁具体数字：数字随配置漂移，而"开销 ∝ 1/fanin"是结构性的。
 */
export function testWidthBreakEven(): void {
  const overheadAt = (ch1: number): { fanin: number; pred: number; gate: number } => {
    const res = runExperiment({ ...TINY, budgets: [0.2], shape: { in1: 1, ch1, k2: 5, ch2: 8 } })
    const dense = res.runs[0]
    const run = res.runs.find(x => x.mode === 'cm-decay' && Math.abs(x.budgetFrac - 0.2) < 1e-9)!
    const li = 1 // L2：fanin 由 shape 决定；L1 的 fanin 恒为 in1×25
    return {
      fanin: res.layers[li].fanin,
      pred: run.perLayer[li].predOps / dense.perLayer[li].macs,
      gate: (run.perLayer[li].gateOps) / dense.perLayer[li].macs,
    }
  }
  const narrow = overheadAt(2) // fanin = 2×5×5 = 50
  const wide = overheadAt(8) // fanin = 8×5×5 = 200
  assert.equal(wide.fanin / narrow.fanin, 4, '两档的 fanin 应当正好差 4 倍')
  assert.ok(wide.pred < narrow.pred / 2.5,
    `fanin ×4 时预测器开销应降到 1/2.5 以下：${(narrow.pred * 100).toFixed(1)}% → ${(wide.pred * 100).toFixed(1)}%`)
  // 选取/盒滤波是地板：会降（常数被摊薄），但绝不会像预测器那样按比例降
  assert.ok(wide.gate > narrow.gate * 0.3, '盒滤波+选取不该跟着 fanin 同比例消失（它是地板）')
}

/**
 * 结构化稀疏（§8）：门控粒度换成"纤维块"之后，被算的神经元必须形成更长、
 * 更少的连续段——这正是"GPU/SIMD 上访存不碎片化"的可测代理量。
 *
 * 这是**方向性**命题：段长上限就是块大小 C，所以只要分组真的生效，段长就
 * 必然变长，与机器、数据、随机种子无关。
 */
export function testFiberBlocksAreLessFragmented(): void {
  const H = TINY.H, W = TINY.W, T = TINY.T
  const specs = defaultLayers(H, W, true, TINY.shape)
  const ds = makeMovingBarDataset({ perClass: 2, testPerClass: 2, T, H, W, seed: TINY.seed })
  const runLenOf = (group: number | 'fiber'): { runLen: number; segsPerStep: number } => {
    const net = new Net(specs, TINY.pred, TINY.seed)
    // 教师：稠密跑一遍训练集，顺便把纤维预测器也训起来
    for (const sm of ds.train) {
      net.run(sm.frames, T, {
        mode: 'dense', budgetFrac: 1, usePredictor: true, trainPredictor: true,
        trainFiber: true, record: false,
      })
    }
    let computed = 0, runs = 0, steps = 0
    for (const sm of ds.test) {
      const r = net.run(sm.frames, T, {
        mode: 'cm-decay', budgetFrac: 0.3, usePredictor: true, trainPredictor: false,
        record: true, group,
      })
      computed += r.budget.computed; runs += r.budget.runs; steps += r.budget.neuronSteps
    }
    return {
      runLen: computed / Math.max(1, runs),
      segsPerStep: runs / Math.max(1, steps),
    }
  }
  const perNeuron = runLenOf(1)
  const fiber = runLenOf('fiber')
  assert.ok(fiber.runLen > perNeuron.runLen * 2,
    `纤维块的平均段长应远大于逐神经元：${perNeuron.runLen.toFixed(2)} → ${fiber.runLen.toFixed(2)}`)
  assert.ok(perNeuron.runLen >= 1 - 1e-9, '平均段长不可能小于 1（runs 计数器坏了）')
  assert.ok(fiber.segsPerStep < perNeuron.segsPerStep / 2,
    `纤维块每步的连续段数应远少于逐神经元：${perNeuron.segsPerStep.toFixed(3)} → ${fiber.segsPerStep.toFixed(3)}`)
}

/**
 * 自适应阈值（§9）：`cm-thresh` 没有预算、没有排序，所以它的逐时间步候选比例
 * 必须**跟着输入密度走**；而固定 top-B 必须是一条直线。
 *
 * 这是定义层面的命题：`cm-thresh` 的判据里根本没有"选前 B 个"这一步，
 * 常数预算那条直线不是巧合，是 top-B 的定义。
 */
export function testAdaptiveBudgetFollowsTheFrame(): void {
  const H = 16, W = 16, T = 10
  // 前 5 步稀疏（每 8 行亮 1 行），后 5 步密集（每 2 行亮 1 行）——密度差 4 倍。
  // 注意**不能**把前半段留成全黑：黑帧里"盒内无脉冲"的前置过滤会把候选集抽空，
  // 那时候 top-B 想给 30% 也给不出来，测到的就不是"固定预算"而是"过滤"了。
  const frames = new Uint8Array(T * H * W)
  for (let t = 0; t < T; t++) {
    const off = t * H * W
    const every = t < (T >> 1) ? 8 : 2
    for (let y = 0; y < H; y++) {
      if (y % every !== 0) continue
      for (let x = 0; x < W; x++) frames[off + y * W + x] = 1
    }
  }
  const specs = defaultLayers(H, W, true, { in1: 1, ch1: 4, k2: 5, ch2: 8 })
  const net = new Net(specs, DEFAULT_EXP.pred, 7)
  const candOf = (mode: GateMode, tau?: number): number[] => {
    // 先让预测器在同一条流上学几遍（在线学习，没有外部教师）
    for (let e = 0; e < 3; e++) {
      net.run(frames, T, {
        mode: 'dense', budgetFrac: 1, usePredictor: true, trainPredictor: true, record: false,
      })
    }
    const r = net.run(frames, T, {
      mode, budgetFrac: mode === 'cm-thresh' ? 1 : 0.3,
      usePredictor: true, trainPredictor: false, record: true, tau,
    })
    return Array.from(r.perStepCand)
  }
  const fixed = candOf('cm-decay')
  const adapt = candOf('cm-thresh', 0.05)
  const mean = (a: number[], from: number, to: number) => {
    let s = 0
    for (let i = from; i < to; i++) s += a[i]
    return s / (to - from)
  }
  const half = T >> 1
  const fixedSwing = Math.abs(mean(fixed, 0, half) - mean(fixed, half, T))
  const adaptSwing = mean(adapt, half, T) - mean(adapt, 0, half)
  assert.ok(fixedSwing < 0.05,
    `固定预算的逐时间步候选比例必须基本是常数：实测两段差 ${(fixedSwing * 100).toFixed(2)} 个百分点`)
  assert.ok(adaptSwing > 0.05,
    `自适应阈值必须在有输入的那半段多算：安静段 ${(mean(adapt, 0, half) * 100).toFixed(1)}% → `
    + `亮段 ${(mean(adapt, half, T) * 100).toFixed(1)}%`)
}

/**
 * 读出层 L2 归一（§11 合体实验的第三条杠杆）。
 *
 * 归一之后读的是**方向**不是**模长**，所以把一个样本的整体发放数乘 k 倍，
 * 归一结果必须完全不变——这正是"漏掉 35% 的脉冲也不该影响分类"的数学根据。
 * 这是一个恒等式，不是跑分。
 */
export function testReadoutL2Normalization(): void {
  const n = 4, d = 5
  const X = new Float64Array(n * d)
  const rng = splitmix32(4242)
  for (let i = 0; i < X.length; i++) X[i] = rng() * 3
  l2NormalizeRows(X, n, d)
  for (let i = 0; i < n; i++) {
    let s = 0
    for (let j = 0; j < d; j++) s += X[i * d + j] * X[i * d + j]
    assert.ok(Math.abs(s - 1) < 1e-12, `第 ${i} 行归一后必须是单位向量：‖x‖² = ${s}`)
  }
  // 尺度不变性：整行 ×k 之后归一，结果与原来逐位相同
  const Y = new Float64Array(n * d)
  for (let i = 0; i < Y.length; i++) Y[i] = X[i]
  for (let i = 0; i < n; i++) {
    const k = 0.01 + i * 7.3 // 模拟"门控漏掉了不同比例的脉冲"
    for (let j = 0; j < d; j++) Y[i * d + j] *= k
  }
  l2NormalizeRows(Y, n, d)
  for (let i = 0; i < Y.length; i++) {
    assert.ok(Math.abs(Y[i] - X[i]) < 1e-12,
      `整行缩放后归一必须与原向量逐位相同：${Y[i]} vs ${X[i]}`)
  }
  // 全零行不能被除出 NaN
  const Z = new Float64Array([0, 0, 0, 1, 0, 0])
  l2NormalizeRows(Z, 2, 3)
  for (const v of Z) assert.ok(Number.isFinite(v), '全零行归一后不能出现 NaN')
  assert.deepEqual(Array.from(Z), [0, 0, 0, 1, 0, 0], '全零行保持原样')
}

export function testDeterminism(): void {
  // 同一配置跑两次必须逐位相同，否则"A/B 差几个脉冲"这种结论没法复核
  const a = runExperiment(TINY)
  const b = runExperiment(TINY)
  assert.deepEqual(
    a.runs.map(r => [r.macs, r.gateOps, r.predOps, r.missRate, r.accuracy]),
    b.runs.map(r => [r.macs, r.gateOps, r.predOps, r.missRate, r.accuracy]),
    '同一配置的两次运行必须逐位一致',
  )
}

export function testTaskAndReadout(): void {
  const data = makeMovingBarDataset({ perClass: 2, testPerClass: 2, T: 8, H: 12, W: 12, seed: 3 })
  assert.equal(data.train.length, 8)
  assert.equal(data.test.length, 8)
  assert.equal(data.classes, 4)
  // 事件流必须是 0/1，且密度在合理范围
  let ones = 0
  for (const v of data.train[0].frames) { assert.ok(v === 0 || v === 1); ones += v }
  assert.ok(ones > 0 && ones < data.train[0].frames.length, '事件流不能全 0 或全 1')
  // 读出层：可分的构造数据上必须能过随机猜测
  const net = new Net(defaultLayers(12, 12), TINY.pred, 1)
  const dims = net.featureDim()
  const X = new Float64Array(8 * dims)
  const y = new Int32Array(8)
  data.train.forEach((s, i) => {
    const r = net.run(s.frames, 8, { mode: 'dense', budgetFrac: 1, usePredictor: false, trainPredictor: false, record: true })
    X.set(r.features, i * dims)
    y[i] = s.label
  })
  const ro = trainReadout(X, y, 8, dims, 4, 200, 0.8)
  assert.ok(readoutAccuracy(ro, X, y, 8) >= 0.75, '读出层应在构造数据上过随机猜测')
}

export function testLocalPlasticity(): void {
  const spec: LayerSpec = { inCh: 1, outCh: 2, inH: 3, inW: 3, k: 3, stride: 1,
    pad: 1, decay: 0.8, thresh: 0.5, scale: 0.3, seed: 2, hardReset: true }
  const run = () => {
    const net = new Net([spec], TINY.pred, 9)
    const l = net.layers[0], old = l.W.slice()
    const learner = new LocalPlasticity(net.layers, { learningRate: 0.2, targetRate: 0.05, thresholdRate: 0.1, maxOffset: 0.4 })
    const input = new Uint8Array(9).fill(1), output = new Uint8Array(l.N).fill(1)
    learner.observe(0, input, output)
    assert.deepEqual(l.W, old, '统计期间不得提前改权重')
    learner.commit()
    assert.notDeepEqual(l.W, old, '局部可塑性应改变权重')
    assert.ok(l.thresholdOffsets.every(v => v > 0 && v <= 0.2), '高发放率应升高阈值并受限')
    // 上界必须从新权重重建；每通道最终前缀为该通道所有正权重之和。
    for (let oc = 0; oc < spec.outCh; oc++) {
      let sum = 0
      for (let j = 0; j < 9; j++) sum += Math.max(0, l.W[oc * 9 + j])
      assert.ok(Math.abs(l.topPerCh[oc * 10 + 9] - sum) < 1e-6)
    }
    const frozen = l.W.slice(); learner.commit(); assert.deepEqual(l.W, frozen, '空提交不得改变权重')
    return Array.from(l.W)
  }
  assert.deepEqual(run(), run(), '局部学习必须可复现')
}

export function testDigitsSplitDisjointOnWrap(): void {
  const common = { T: 3, H: 8, W: 8, encoding: 'poisson' as const, seed: 11 }
  const big = makeDigitsDataset({ ...common, nTrain: 1000, nTest: 1 })!
  const reference = makeDigitsDataset({ ...common, nTrain: 1001, nTest: 1 })!
  assert.deepEqual(big.test[0], reference.train[1000], '环绕时测试集应退回未见过的区间')
  const legacy = makeDigitsDataset({ ...common, nTrain: 500, nTest: 1 })!
  const long = makeDigitsDataset({ ...common, nTrain: 1478, nTest: 1 })!
  assert.deepEqual(legacy.test[0], long.train[1477], '历史基准的测试样本必须保持不变')
}

export function testPlasticExperimentProtocol(): void {
  const result = runPlasticDigits({ nTrain: 6, nTest: 3, T: 2, pool: 2, epochs: 1, seed: 77,
    plastic: { learningRate: 0.01, targetRate: 0.05, thresholdRate: 0.05, maxOffset: 0.4 } })
  assert.equal(result.arms.length, 2)
  assert.deepEqual(result.arms[0].weightRmsChange, [0, 0])
  assert.ok(result.arms[1].weightRmsChange.every(x => x > 0), '两层均须学到局部更新')
  for (const arm of result.arms) {
    const event = arm.rows[0]
    assert.equal(event.missedSpikes, 0)
    assert.equal(event.extraSpikes, 0)
    assert.equal(event.accuracy, arm.denseAccuracy)
    assert.ok(arm.rows.every(r => Number.isFinite(r.opsRatio) && r.opsRatio >= 0))
  }
}

export function testBalancedPlasticityNorm(): void {
  const spec: LayerSpec = { inCh: 1, outCh: 2, inH: 3, inW: 3, k: 3, stride: 1,
    pad: 1, decay: 0.8, thresh: 0.5, scale: 0.3, seed: 2, hardReset: true }
  const l = buildLayer(spec), initial = l.W.slice()
  const norm = (weights: Float32Array, oc: number) => {
    let sq = 0; for (let j = 0; j < l.fanin; j++) sq += weights[oc * l.fanin + j] ** 2
    return Math.sqrt(sq)
  }
  const learner = new LocalPlasticity([l], { learningRate: 0.2, targetRate: 0.05, thresholdRate: 0,
    maxOffset: 0.4, rule: 'balanced-norm' })
  const input = new Uint8Array(9); input[4] = 1
  const out = new Uint8Array(l.N); out[4 * spec.outCh] = 1
  learner.observe(0, input, out); learner.commit()
  assert.notDeepEqual(l.W, initial)
  for (let oc = 0; oc < spec.outCh; oc++) assert.ok(Math.abs(norm(l.W, oc) - norm(initial, oc)) < 1e-6,
    '平衡规则必须保留各核的范数')
  assert.ok(l.thresholdOffsets.every(v => v === 0), '阈值消融不应改变阈值')
}

export function testValidationAndCheckpoint(): void {
  const cfg = { ...VALIDATION_DEFAULT, nTrain: 6, nValidation: 3, validationStart: 6,
    nHoldout: 3, holdoutStart: 9, steps: 2, pool: 2, seed: 77 }
  const checkpoints = new Map<string, DenseCheckpoint>()
  const cmCheckpoints = new Map<string, Parameters<typeof loadInferenceCheckpoint>[0]>()
  const phases: string[] = []
  const result = runValidation(cfg, { log: () => {}, checkpoint: (name, cp) => checkpoints.set(name, cp),
    checkpointCM: (name, cp) => cmCheckpoints.set(name, cp),
    results: value => phases.push((value as { phase: string }).phase) })
  assert.deepEqual(phases, ['validation', 'validation', 'validation', 'validation', 'validation', 'complete'])
  assert.equal(result.candidates.length, 5)
  assert.equal(result.holdout.length, 2)
  for (const arm of result.holdout) {
    assert.equal(arm.rows[0].miss, 0); assert.equal(arm.rows[0].extra, 0)
    const cp = JSON.parse(JSON.stringify(checkpoints.get(arm.name))) as DenseCheckpoint
    const { net, readout } = loadDenseCheckpoint(cp)
    const held = makeDigitsDataset({ nTrain: cfg.nTrain, nTest: cfg.nHoldout, testStart: cfg.holdoutStart,
      T: cfg.steps, H: 24, W: 24, encoding: 'poisson', seed: cfg.seed })!
    const X = new Float64Array(cfg.nHoldout * readout.d)
    const labels = Int32Array.from(held.test, x => x.label)
    held.test.forEach((sm, i) => X.set(net.run(sm.frames, cp.steps, { mode: 'dense', budgetFrac: 1,
      usePredictor: false, trainPredictor: false, record: true, pool: cp.pool }).features, i * readout.d))
    let correct = 0
    for (let i = 0; i < cfg.nHoldout; i++) {
      const scores = denseLogits(readout, X.subarray(i * readout.d, (i + 1) * readout.d))
      if (scores.indexOf(Math.max(...scores)) === labels[i]) correct++
    }
    assert.equal(correct / cfg.nHoldout, arm.denseAccuracy, '独立 logits 推断应等于实验结果')
    assert.equal(readoutAccuracy(readout, X, labels, cfg.nHoldout), arm.denseAccuracy, '重载模型预测应相同')
    const cm = loadInferenceCheckpoint(cmCheckpoints.get(arm.name)!)
    let cmCorrect = 0, cmOps = 0
    held.test.forEach(sm => {
      const r = cm.net.run(sm.frames, cfg.steps, { mode: 'cm-decay', budgetFrac: 0.7, usePredictor: true,
        trainPredictor: false, record: true, pool: cfg.pool, group: 8 })
      const scores = denseLogits(cm.readout, r.features)
      if (scores.indexOf(Math.max(...scores)) === sm.label) cmCorrect++
      cmOps += r.budget.macs + r.budget.gateOps + r.budget.predOps
    })
    assert.equal(cmCorrect / cfg.nHoldout, arm.rows[1].accuracy, 'CM 文件应重现评估准确率')
    assert.equal(cmOps / arm.denseMacs, arm.rows[1].opsRatio, 'CM 文件应重现门控算术账本')
    cp.layers[0].weights.pop()
    assert.throws(() => loadDenseCheckpoint(cp), /invalid checkpoint layer/)
  }
  assert.throws(() => runValidation({ ...cfg, holdoutStart: 8 }, { log: () => {}, checkpoint: () => {}, results: () => {} }), /disjoint/)
  assert.throws(() => makeDigitsDataset({ nTrain: 6, nTest: 3, testStart: 5, T: 2, H: 8, W: 8, encoding: 'poisson' }), /训练区之外/)
}

export function testClassModulation(): void {
  const spec: LayerSpec = { inCh: 1, outCh: 2, inH: 3, inW: 3, k: 3, stride: 1,
    pad: 1, decay: 0.8, thresh: 0.5, scale: 0.3, seed: 2, hardReset: true }
  const make = (label: number, prediction: number) => {
    const l = buildLayer(spec), before = l.W.slice(), offsets = l.thresholdOffsets.slice()
    const r = { classes: 2, d: 2, W: new Float64Array([1, -1, 0, -1, 1, 0]),
      mean: new Float64Array(2), inv: new Float64Array([1, 1]), l2Norm: false }
    const learner = new ClassModulatedPlasticity([l], { learningRate: 0.2, creditGain: 1, pool: 1 })
    const input = new Uint8Array(9); input[4] = 1
    const out = new Uint8Array(l.N); out[4 * 2] = 1; out[5 * 2 + 1] = 1
    learner.observe(0, input, out)
    assert.deepEqual(l.W, before, '预测之前仅记录局部痕迹')
    const counts = learner.commit(label, prediction, r)
    assert.deepEqual(l.thresholdOffsets, offsets, '继承的稳态阈值不能被修改')
    return { l, before, counts }
  }
  const positive = make(0, 1), negative = make(1, 0), correct = make(0, 0)
  assert.ok(positive.counts.potentiated > 0 && positive.counts.depressed > 0, '必须同时有正负调制')
  assert.notDeepEqual(positive.l.W, positive.before)
  assert.deepEqual(correct.l.W, correct.before, '预测正确时不能误更新')
  // Reversing the label/guess reverses the local update direction.
  let dot = 0
  for (let i = 0; i < positive.l.W.length; i++)
    dot += (positive.l.W[i] - positive.before[i]) * (negative.l.W[i] - negative.before[i])
  assert.ok(dot < 0, '类调制符号应反转更新方向')
  for (const run of [positive, negative]) for (let oc = 0; oc < spec.outCh; oc++) {
    let old = 0, now = 0
    for (let k = 0; k < 9; k++) { old += run.before[oc * 9 + k] ** 2; now += run.l.W[oc * 9 + k] ** 2 }
    assert.ok(Math.abs(old - now) < 1e-6, '单核范数保持')
  }
}

export function testCmCheckpointRoundTrip(): void {
  const shape = { in1: 1, ch1: 4, k2: 3, ch2: 8 }
  const net = new Net(defaultLayers(8, 8, true, shape), TINY.pred, 77)
  const ds = makeMovingBarDataset({ perClass: 2, testPerClass: 1, T: 4, H: 8, W: 8, seed: 7 })
  net.ensureGroups([2])
  const d = net.featureDim(2), X = new Float64Array(ds.train.length * d), y = new Int32Array(ds.train.length)
  ds.train.forEach((sample, i) => {
    const r = net.run(sample.frames, 4, { mode: 'dense', budgetFrac: 1, usePredictor: true,
      trainPredictor: true, trainFiber: true, record: true, pool: 2 })
    X.set(r.features, i * d); y[i] = sample.label
  })
  const ro = trainReadout(X, y, ds.train.length, d, 4, 20)
  const cp = JSON.parse(JSON.stringify(saveCmCheckpoint(net, ro, 4, 2, 2, 0.7, 77)))
  const loaded = loadInferenceCheckpoint(cp)
  assert.equal(loaded.seed, 77); assert.deepEqual(loaded.cm, { group: 2, budget: 0.7 })
  const run = (n: Net, trainPredictor: boolean) => {
    const probs: number[][] = []
    const r = n.run(ds.test[0].frames, 4, { mode: 'cm-decay', budgetFrac: 0.7, usePredictor: true,
      trainPredictor, record: true, pool: 2, group: 2, onStep: (_li, _t, p) => probs.push(Array.from(p)) })
    return { probs, spikes: r.spikes.map(x => Array.from(x)), features: Array.from(r.features), budget: r.budget }
  }
  assert.deepEqual(run(loaded.net, false), run(net, false), '重载前后的概率、脉冲和门控账本应一致')
  assert.deepEqual(run(loaded.net, true), run(net, true), '恢复后继续在线更新应一致')
  assert.deepEqual(run(loaded.net, false), run(net, false), '继续学习之后也应一致')
  const dir = mkdtempSync(join(tmpdir(), 'snn-cm-checkpoint-'))
  try {
    const modelPath = join(dir, 'model.json'), densePath = join(dir, 'dense.json')
    writeFileSync(modelPath, JSON.stringify(cp)); writeFileSync(densePath, JSON.stringify(saveDenseCheckpoint(net, ro, 4, 2)))
    const invoke = (path: string) => spawnSync(process.execPath, ['scripts/run-snn-predict.mjs', '--model', path,
      '--index', '3', '--mode', 'cm'], { cwd: process.cwd(), encoding: 'utf8' })
    const child = invoke(modelPath)
    assert.equal(child.status, 0, child.stderr)
    const output = JSON.parse(child.stdout)
    assert.equal(output.mode, 'cm'); assert.equal(output.seed, 77); assert.equal(output.group, 2)
    const digits = makeDigitsDataset({ nTrain: 0, nTest: 1, testStart: 3, T: 4, H: 8, W: 8, encoding: 'poisson', seed: 77 })!
    const restored = loadInferenceCheckpoint(cp)
    const direct = restored.net.run(digits.test[0].frames, 4, { mode: 'cm-decay', budgetFrac: 0.7,
      usePredictor: true, trainPredictor: false, record: true, pool: 2, group: 2 })
    assert.deepEqual(output.logits, Array.from(denseLogits(restored.readout, direct.features)))
    assert.deepEqual(output.cost, direct.budget)
    const dense = restored.net.run(digits.test[0].frames, 4, { mode: 'dense', budgetFrac: 1,
      usePredictor: false, trainPredictor: false, record: true, pool: 2 })
    assert.equal(output.denseMacs, dense.budget.macs, '解析几何账本应等于稠密参考')
    const wrong = invoke(densePath)
    assert.notEqual(wrong.status, 0); assert.match(wrong.stderr, /no trained predictor/)
  } finally { rmSync(dir, { recursive: true, force: true }) }
  cp.cm.predictors[0].hist = 'invalid'
  assert.throws(() => loadInferenceCheckpoint(cp), /invalid CM/)
}

export function testClassExperimentPipeline(): void {
  const cfg = { ...VALIDATION_DEFAULT, nTrain: 6, nValidation: 3, validationStart: 6,
    nHoldout: 3, holdoutStart: 9, steps: 2, pool: 2, seed: 77 }
  const net = new Net(defaultLayers(24, 24, true, { in1: 1, ch1: 8, ch2: 8, k2: 3 }), TINY.pred, 77)
  const ds = makeDigitsDataset({ nTrain: cfg.nTrain, nTest: cfg.nValidation, testStart: cfg.validationStart,
    T: cfg.steps, H: 24, W: 24, encoding: 'poisson', seed: cfg.seed })!
  const d = net.featureDim(cfg.pool), X = new Float64Array(ds.train.length * d), y = new Int32Array(ds.train.length)
  ds.train.forEach((sm, i) => {
    const r = net.run(sm.frames, cfg.steps, { mode: 'dense', budgetFrac: 1, usePredictor: false,
      trainPredictor: false, record: true, pool: cfg.pool })
    X.set(r.features, i * d); y[i] = sm.label
  })
  const base = saveDenseCheckpoint(net, trainReadout(X, y, ds.train.length, d, 10, 20, 0.8, 1e-4, 123, true), cfg.steps, cfg.pool)
  const saved = new Map<string, unknown>(), phases: string[] = []
  const result = runClassExperiment(base, cfg, { log: () => {}, checkpoint: (name, cp) => saved.set(name, cp),
    results: r => phases.push((r as { phase: string }).phase) })
  assert.equal(result.candidates.length, 3); assert.deepEqual(phases, ['validation', 'validation', 'validation', 'complete'])
  assert.ok(result.candidates[1].kernels > 0 && result.candidates[1].weightRmsChange.every(x => x > 0))
  const baselineOffsets = base.layers.map(l => l.thresholdOffsets)
  for (const arm of result.holdout) {
    const cp = saved.get(arm.name) as Parameters<typeof loadInferenceCheckpoint>[0]
    const loaded = loadInferenceCheckpoint(cp)
    assert.deepEqual(loaded.net.layers.map(l => Array.from(l.thresholdOffsets)), baselineOffsets)
    assert.equal(arm.rows[0].miss, 0); assert.equal(arm.rows[0].extra, 0)
    assert.ok(Number.isFinite(arm.rows[1].opsRatio))
  }
  assert.throws(() => runClassExperiment(base, { ...cfg, holdoutStart: 7 },
    { log: () => {}, checkpoint: () => {}, results: () => {} }), /overlapping/)
}

export function testLayerBridgeAndIndexedData(): void {
  const specs = defaultLayers(8, 8, true, { in1: 1, ch1: 4, ch2: 8, k2: 3 })
  const sample = makeMovingBarDataset({ perClass: 1, testPerClass: 1, T: 4, H: 8, W: 8, seed: 3 }).train[0]
  const net = new Net(specs, TINY.pred, 77)
  let previous = new Uint8Array(0), steps = 0
  const opts = { mode: 'dense' as const, budgetFrac: 1, usePredictor: false, trainPredictor: false,
    record: true, channelMajorBridge: true, observeSpikes: (li: number, input: Uint8Array, output: Uint8Array) => {
      if (li === 0) previous = output.slice()
      else {
        const pos = net.layers[0].outH * net.layers[0].outW, ch = specs[0].outCh
        for (let p = 0; p < pos; p++) for (let c = 0; c < ch; c++)
          assert.equal(input[c * pos + p], previous[p * ch + c])
        steps++
      }
    } }
  const dense = net.run(sample.frames, 4, opts)
  const event = net.run(sample.frames, 4, { ...opts, mode: 'event' })
  assert.equal(steps, 8)
  assert.deepEqual(dense.spikes, event.spikes, '正确布局下 event 仍逐脉冲无损')
  assert.equal(dense.budget.bridgeOps, 4 * net.layers[0].N)
  const common = { T: 3, H: 8, W: 8, encoding: 'poisson' as const, seed: 77 }
  const indexed = makeIndexedDigitsDataset({ ...common, indices: [0, 1477] })!
  const legacy = makeDigitsDataset({ ...common, nTrain: 500, nTest: 1 })!
  assert.deepEqual(indexed, [legacy.train[0], legacy.test[0]])
  assert.throws(() => makeIndexedDigitsDataset({ ...common, indices: [1, 1] }), /唯一/)
  validatePrototypeConfig(PROTOTYPE_DEFAULT)
  const digits = loadDigits()!
  const pixels = (i: number) => Buffer.from(digits.px.subarray(i * 64, i * 64 + 64)).toString('base64')
  const originals = new Set(PROTOTYPE_DEFAULT.trainIndices.map(pixels))
  assert.ok(PROTOTYPE_DEFAULT.validationIndices.every(i => !originals.has(pixels(i))), '验证原图不能与训练原图完全相同')
  assert.ok(PROTOTYPE_DEFAULT.testIndices.every(i => !originals.has(pixels(i))), '测试原图不能与训练原图完全相同')
  assert.throws(() => validatePrototypeConfig({ ...PROTOTYPE_DEFAULT, testIndices: [0] }), /disjoint/)
}

export function testEventAugmentation(): void {
  const T = 3, H = 4, W = 4, input = new Uint8Array(T * H * W)
  input[(1 * H + 1) * W + 1] = 1
  assert.deepEqual(augmentEvents(input, T, H, W, 0, 0, 0, 9), input)
  const shifted = augmentEvents(input, T, H, W, 1, -1, 0, 9)
  assert.equal(shifted[(1 * H + 0) * W + 2], 1)
  assert.equal(shifted.reduce((a, b) => a + b, 0), 1)
  const jitter = augmentEvents(input, T, H, W, 1, 0, 1, 77)
  assert.deepEqual(jitter, augmentEvents(input, T, H, W, 1, 0, 1, 77))
  assert.equal(jitter[(1 * H + 1) * W + 2], 0)
  assert.equal(jitter[(0 * H + 1) * W + 2] + jitter[(2 * H + 1) * W + 2], 1)
  assert.equal(input[(1 * H + 1) * W + 1], 1, '增强不得改变原输入')
}

export function testPrototypeLocalUpdates(): void {
  const one: LayerSpec = { inCh: 1, outCh: 2, inH: 3, inW: 3, k: 1, stride: 1, pad: 0,
    decay: 0.8, thresh: 0.5, scale: 0.3, seed: 2, hardReset: true }
  const two: LayerSpec = { ...one, inCh: 2, outCh: 32, seed: 3 }
  const l1 = buildLayer(one), l2 = buildLayer(two); l2.W.fill(0.2)
  const before = l2.W.slice(), pre = new Uint8Array(2 * l1.N), post = new Uint8Array(2 * l2.N)
  for (let t = 0; t < 2; t++) pre[t * l1.N + 0 * 2 + 0] = 1
  post[0 * l2.N + 0 * 32 + 3] = 1 // class 1 target
  post[0 * l2.N + 0 * 32 + 0] = 1; post[1 * l2.N + 0 * 32 + 0] = 1 // other class > mean
  post[0 * l2.N + 0 * 32 + 30] = 1 // reserved channel remains untouched
  const learner = new PrototypeContrastive(l1, l2, 0.2, 0.5)
  const stats = learner.apply(1, pre, post, 2)
  assert.equal(stats.positive, 1); assert.equal(stats.negative, 1); assert.equal(stats.silentTargets, 2)
  assert.ok(l2.W[3 * 2] > before[3 * 2] && l2.W[3 * 2 + 1] < before[3 * 2 + 1], '真类赢家朝脉冲方向增强')
  assert.ok(l2.W[0] < before[0] && l2.W[1] > before[1], '其他强响应通道反向更新')
  assert.deepEqual(l2.W.slice(60), before.slice(60), '两个保留通道不更新')
  assert.deepEqual(l2.W.slice(2, 6), before.slice(2, 6), '非赢家不更新')
  const frozen = l2.W.slice()
  learner.apply(1, pre, new Uint8Array(post.length), 2)
  assert.deepEqual(l2.W, frozen, '无后脉冲时不得更新')
}

export function testThresholdAndRegularizationScaling(): void {
  const spec = singleLayerSpec(true), a = buildLayer(spec), b = buildLayer(spec)
  const slow = new LocalPlasticity([a], { learningRate: 0, targetRate: 0.05, thresholdRate: 0.12, maxOffset: 0.4 })
  const fast = new IntrinsicHomeostasis([b])
  const input = new Uint8Array(spec.inCh * spec.inH * spec.inW), out = new Uint8Array(a.N)
  for (let i = 0; i < out.length; i++) out[i] = i % 7 === 0 ? 1 : 0
  for (let i = 0; i < 5; i++) { slow.observe(0, input, out); fast.observe(0, input, out) }
  slow.commit(); fast.commit(); assert.deepEqual(a.thresholdOffsets, b.thresholdOffsets)
  const X = new Float64Array([1, 0, 0, 1, 0.9, 0.1, 0.1, 0.9]), y = new Int32Array([0, 1, 0, 1])
  const dup = new Float64Array([...X, ...X]), yd = new Int32Array([...y, ...y])
  const small = trainReadout(X.slice(), y, 4, 2, 2, 100, 0.3, 0.1, 9, false, true)
  const large = trainReadout(dup, yd, 8, 2, 2, 100, 0.3, 0.1, 9, false, true)
  for (let i = 0; i < small.W.length; i++) assert.ok(Math.abs(small.W[i] - large.W[i]) < 0.01, '正则不能随样本数翻倍')
}

export function testPrototypePipeline(): void {
  const cfg = { ...PROTOTYPE_DEFAULT, trainIndices: PROTOTYPE_DEFAULT.trainIndices.slice(0, 8),
    validationIndices: [1700, 1701, 1702], testIndices: [1710, 1711, 1712],
    steps: 2, readoutEpochs: 3 }
  const checkpoints = new Map<string, Parameters<typeof loadInferenceCheckpoint>[0]>()
  const result = runPrototypeExperiment(cfg, { log: () => {}, results: () => {}, checkpoint: (n, cp) => checkpoints.set(n, cp) })
  assert.equal(result.phase, 'complete'); assert.equal(result.plasticity.eventViews, 24)
  assert.ok(result.plasticity.positive > 0); assert.ok(result.plasticity.weightRmsChangeL2 > 0)
  for (const arm of result.test) {
    const cp = loadInferenceCheckpoint(checkpoints.get(arm.name)!)
    assert.equal(cp.channelMajorBridge, true)
    assert.equal(arm.rows[0].missedSpikes, 0); assert.equal(arm.rows[0].extraSpikes, 0)
    const samples = makeIndexedDigitsDataset({ indices: cfg.testIndices, T: cfg.steps, H: 24, W: 24,
      encoding: 'poisson', seed: cfg.seed })!
    let ok = 0
    for (const sm of samples) {
      const r = cp.net.run(sm.frames, cfg.steps, { mode: 'cm-decay', budgetFrac: 0.7,
        usePredictor: true, trainPredictor: false, record: true, pool: cfg.pool, group: 8, channelMajorBridge: cp.channelMajorBridge })
      const score = denseLogits(cp.readout, r.features)
      if (score.indexOf(Math.max(...score)) === sm.label) ok++
    }
    assert.equal(ok / samples.length, arm.rows[1].accuracy, '恢复的门控模型必须复现评估')
    const dir = mkdtempSync(join(tmpdir(), 'snn-bridge-cli-'))
    try {
      const path = join(dir, 'model.json')
      writeFileSync(path, JSON.stringify(checkpoints.get(arm.name)))
      const child = spawnSync(process.execPath, ['scripts/run-snn-predict.mjs', '--model', path, '--index',
        String(cfg.testIndices[0]), '--mode', 'cm'], { cwd: process.cwd(), encoding: 'utf8' })
      assert.equal(child.status, 0, child.stderr)
      const output = JSON.parse(child.stdout)
      const loaded = loadInferenceCheckpoint(checkpoints.get(arm.name)!)
      const expected = loaded.net.run(samples[0].frames, cfg.steps, { mode: 'cm-decay', budgetFrac: 0.7,
        usePredictor: true, trainPredictor: false, record: true, pool: cfg.pool, group: 8, channelMajorBridge: true })
      assert.equal(output.channelMajorBridge, true)
      assert.deepEqual(output.logits, Array.from(denseLogits(loaded.readout, expected.features)))
      assert.deepEqual(output.cost, expected.budget)
      assert.equal(output.denseBridgeOps, cfg.steps * loaded.net.layers[0].N)
    } finally { rmSync(dir, { recursive: true, force: true }) }
  }
}


export function testStreamPrequential(): void {
  const cm = new FeatureContextMixer(4), x = Float64Array.of(1, 0, 0, 0)
  const a = cm.observe(x), repeated = cm.observe(x)
  assert.equal(a.entropy, 1); assert.equal(a.surprise, 1)
  assert.deepEqual(a, repeated, 'observe must not update the CM')
  cm.learn(a)
  assert.throws(() => cm.learn(a), /stale/)
  assert.ok(cm.observe(x).entropy < a.entropy)
  assert.ok(entropyGate(false, .5, 2) > entropyGate(true, .5, 2))
  const agent = new StreamAgent(4, 'entropy-gated')
  assert.throws(() => agent.feedback(0), /predict/)
  const before = agent.readout.weights.slice(), prediction = agent.predict(x)
  assert.equal(prediction.prediction, 0)
  assert.deepEqual(agent.readout.weights, before)
  assert.throws(() => agent.predict(x), /feedback/)
  assert.throws(() => agent.feedback(10), /label/)
  assert.deepEqual(agent.readout.weights, before)
  const feedback = agent.feedback(1)
  assert.equal(feedback.correct, false); assert.ok(feedback.gate > 0)
  assert.notDeepEqual(agent.readout.weights, before)
  assert.throws(() => agent.feedback(1), /predict/)
  const resumed = StreamAgent.restore(JSON.parse(JSON.stringify(agent.snapshot())))
  assert.deepEqual(resumed.predict(x), agent.predict(x))
  assert.throws(() => resumed.snapshot(), /feedback/)
  assert.deepEqual(resumed.feedback(0), agent.feedback(0))
  assert.deepEqual(resumed.snapshot(), agent.snapshot())
}

export function testStreamPipeline(): void {
  // Self-contained tiny frozen checkpoint, with no dependency on experiment artifacts.
  const net = new Net(defaultLayers(8, 8, true, { in1: 1, ch1: 4, k2: 3, ch2: 8 }), TINY.pred, 77)
  net.ensureGroups([2])
  const sample = makeIndexedDigitsDataset({ indices: [0], T: 2, H: 8, W: 8, encoding: 'poisson', seed: 77 })![0]
  net.run(sample.frames, 2, { mode: 'dense', budgetFrac: 1, usePredictor: true,
    trainPredictor: true, trainFiber: true, record: false })
  const d = net.featureDim(2)
  const ro = { classes: 10, d, W: new Float64Array(10 * (d + 1)), mean: new Float64Array(d), inv: new Float64Array(d).fill(1), l2Norm: true }
  const cp = saveCmCheckpoint(net, ro, 2, 2, 2, 0.7, 77, true)
  const cfg = { ...STREAM_DEFAULT, streamIndices: STREAM_DEFAULT.streamIndices.slice(0, 20),
    validationIndices: [1700, 1701], testIndices: [1710, 1711] }
  const reports: any[] = []
  const r = runStreamExperiment(cp, cfg, { log: () => {}, results: v => reports.push(v) })
  assert.equal(r.trace.length, 20); assert.equal(reports.length, 2); assert.equal(reports[1].phase, 'complete')
  assert.ok(r.zeroFeatures < 20, 'frozen SNN must provide recorded nonzero features')
  const trace = r.trace as { phase: number; label: number; index: number; baseline: number; gated: number; baselineAccuracy: number; gatedAccuracy: number }[]
  assert.ok(trace.filter(t => t.phase === 1).every(t => t.label < 5))
  assert.ok(trace.filter(t => t.phase === 2).every(t => t.label >= 5))
  assert.equal(new Set(trace.map(t => t.index)).size, 20)
  let b = 0, g = 0
  trace.forEach((t, i) => { b += Number(t.label === t.baseline); g += Number(t.label === t.gated)
    assert.equal(t.baselineAccuracy, b / (i + 1)); assert.equal(t.gatedAccuracy, g / (i + 1)) })
  assert.throws(() => runStreamExperiment(cp, { ...cfg, testIndices: cfg.validationIndices }, { log: () => {}, results: () => {} }), /overlapping/)
  const dir = mkdtempSync(join(tmpdir(), 'snn-stream-cli-'))
  try {
    const model = join(dir, 'feature.json'), state = join(dir, 'agent.json'), bytes = JSON.stringify(cp)
    writeFileSync(model, bytes)
    writeFileSync(state, JSON.stringify({ model, modelSha256: createHash('sha256').update(bytes).digest('hex'), agent: r.gatedAgent.snapshot() }))
    const run = (...args: string[]) => spawnSync(process.execPath, ['scripts/run-snn-stream-predict.mjs', '--state', state, '--index', '1700', ...args], { cwd: process.cwd(), encoding: 'utf8' })
    const before = readFileSync(state, 'utf8'), plain = run()
    assert.equal(plain.status, 0, plain.stderr); assert.equal(readFileSync(state, 'utf8'), before)
    const update = run('--label', '3', '--update')
    assert.equal(update.status, 0, update.stderr)
    assert.equal(JSON.parse(update.stdout).prediction, JSON.parse(plain.stdout).prediction)
    assert.notEqual(readFileSync(state, 'utf8'), before)
    const isolated = new IsolatedAgent(d, 'anchored'), probe = new Float64Array(d).fill(1)
    isolated.predict(probe); isolated.feedback(0)
    const isolatedState = join(dir, 'isolated.json')
    writeFileSync(isolatedState, JSON.stringify({ model, modelSha256: createHash('sha256').update(bytes).digest('hex'), agents: { anchored: isolated.snapshot() } }))
    const runIsolated = (...extra: string[]) => spawnSync(process.execPath, ['scripts/run-snn-continual-predict.mjs', '--state', isolatedState,
      '--method', 'anchored', '--index', '1700', ...extra], { cwd: process.cwd(), encoding: 'utf8' })
    const stateBefore = readFileSync(isolatedState, 'utf8'), pre = runIsolated()
    assert.equal(pre.status, 0, pre.stderr); assert.equal(readFileSync(isolatedState, 'utf8'), stateBefore)
    const trained = runIsolated('--label', '1', '--update')
    assert.equal(trained.status, 0, trained.stderr)
    assert.equal(JSON.parse(pre.stdout).prediction, JSON.parse(trained.stdout).prediction)
    const trainedState = JSON.parse(readFileSync(isolatedState, 'utf8')).agents.anchored
    assert.deepEqual(trainedState.rows[0], isolated.snapshot().rows![0])
    assert.deepEqual(trainedState.means[0], isolated.snapshot().means[0])
    writeFileSync(model, bytes + ' ')
    assert.notEqual(run().status, 0, 'mismatched model hash must be rejected')
    assert.notEqual(runIsolated().status, 0, 'isolated agent also rejects a mismatched model')
  } finally { rmSync(dir, { recursive: true, force: true }) }
}


export function testIsolatedContinualAgents(): void {
  assert.ok(prototypeEta(2) > prototypeEta(0)); assert.equal(prototypeEta(100), prototypeEta(2))
  for (const mode of ['ncm', 'anchored'] as const) {
    const agent = new IsolatedAgent(10, mode), x0 = new Float64Array(10), x1 = new Float64Array(10)
    x0[0] = 1; x1[1] = 1
    assert.throws(() => agent.feedback(0), /predict/)
    assert.equal(agent.predict(x0).prediction, 0)
    assert.throws(() => agent.snapshot(), /feedback/)
    agent.feedback(0)
    const old = agent.snapshot(), old0 = old.means[0], oldRow = old.rows?.[0]
    const pred = agent.predict(x1)
    assert.equal(pred.prediction, 0, 'new classes cannot be predicted before feedback')
    assert.throws(() => agent.feedback(42), /label/)
    agent.feedback(1)
    const after = agent.snapshot()
    assert.deepEqual(after.means[0], old0, 'another label cannot change the old prototype')
    assert.equal(after.counts[0], old.counts[0])
    if (after.rows) assert.deepEqual(after.rows[0], oldRow, 'another label cannot change an old readout row')
    assert.equal(agent.classify(x0), 0); assert.equal(agent.classify(x1), 1)
    const snapshot = agent.snapshot()
    agent.classify(x0)
    assert.deepEqual(agent.snapshot(), snapshot, 'evaluation is read-only')
    const restored = IsolatedAgent.restore(JSON.parse(JSON.stringify(snapshot)))
    assert.deepEqual(agent.predict(x0), restored.predict(x0))
    assert.deepEqual(agent.feedback(0), restored.feedback(0))
    assert.deepEqual(agent.snapshot(), restored.snapshot())
  }
}

export function testContinualSynthetic(): void {
  const dataset = Array.from({ length: 40 }, (_, i) => {
    const x = new Float64Array(10), label = i % 10; x[label] = 1
    return { index: i, label, x }
  })
  const holdout = dataset.slice(0, 10)
  const run = runContinualSeed(77, dataset, holdout, holdout)
  assert.equal(run.trace.length, 40)
  assert.ok(Object.values(run.frozenOld!).every(Boolean))
  assert.equal(run.snapshots[1].test.ncm.old, 1)
  assert.equal(run.snapshots[1].test.anchored.old, 1)
  const again = runContinualSeed(77, dataset, holdout, holdout)
  assert.deepEqual(run.trace, again.trace)
  const trace = run.trace as any[]
  assert.ok(trace.filter(t => t.phase === 1).every(t => t.label < 5))
  assert.ok(trace.filter(t => t.phase === 2).every(t => t.label >= 5))
  const sums: Record<string, number> = { baseline: 0, 'entropy-gated': 0, ncm: 0, anchored: 0 }
  trace.forEach((t, i) => { for (const name of Object.keys(sums)) { sums[name] += Number(t.prediction[name] === t.label)
    assert.equal(t.cumulativeAccuracy[name], sums[name] / (i + 1)) } })
  assert.throws(() => runContinualExperiment({} as any, { ...CONTINUAL_DEFAULT, streamIndices: [0], validationIndices: [0], testIndices: [1] },
    { log: () => {}, results: () => {}, agents: () => {} }), /overlapping/)
}


export function testSpikeByteModel(): void {
  const fresh = new SpikeByteModel(), empty = fresh.snapshot()
  assert.equal(fresh.consume(0x41, false), 8, 'empty tables predict the first byte at one bit per bit')
  assert.deepEqual(fresh.snapshot().counters, empty.counters, 'frozen scoring cannot learn')
  assert.equal(fresh.bytesSeen, 1)
  const model = new SpikeByteModel(), text = Buffer.from('to be or not to be, that is the question. '.repeat(80))
  const early = text.subarray(0, 200).reduce((sum, x) => sum + model.consume(x, true), 0) / 200
  const later = text.subarray(200).reduce((sum, x) => sum + model.consume(x, true), 0) / (text.length - 200)
  assert.ok(later < early, 'online prequential loss should drop on a repeated stream')
  const snapshot = model.snapshot(), restored = SpikeByteModel.restore(JSON.parse(JSON.stringify(snapshot)))
  assert.deepEqual(restored.snapshot(), snapshot)
  const noSpikes = JSON.parse(JSON.stringify(snapshot)), allSpikes = JSON.parse(JSON.stringify(snapshot))
  noSpikes.lif.spikes.fill(0); allSpikes.lif.spikes.fill(1)
  assert.notEqual(SpikeByteModel.restore(noSpikes).consume(116, false), SpikeByteModel.restore(allSpikes).consume(116, false),
    'spike context must actually affect the mixed prediction')
  const r1 = splitmix32(123), r2 = splitmix32(123), a = [], b = []
  for (let i = 0; i < 64; i++) { a.push(model.sample(r1, 0.8)); b.push(restored.sample(r2, 0.8)) }
  assert.deepEqual(a, b, 'sampling must reproduce from the same snapshot and seed')
  assert.deepEqual(model.snapshot(), restored.snapshot())
  assert.deepEqual(model.snapshot().counters, snapshot.counters, 'generated samples must not train frozen counters')
  assert.equal(model.bytesSeen, snapshot.bytesSeen + 64)
  assert.ok(model.diagnostics().totalSpikes > 0, 'the LIF reservoir must actually spike')
  assert.throws(() => SpikeByteModel.restore({ ...snapshot, history: [300] }), /invalid/)
  assert.throws(() => model.consume(256, true), /invalid/)
  assert.throws(() => model.sample(splitmix32(1), 0), /invalid/)
  const learning = SpikeByteModel.restore(snapshot)
  learning.sample(splitmix32(1), 1, true)
  assert.notDeepEqual(learning.snapshot().counters, snapshot.counters, 'explicit self-training is distinguishable')
}

export function testSequenceCli(): void {
  const dir = mkdtempSync(join(tmpdir(), 'ucz-sequence-'))
  try {
    const input = join(dir, 'input.txt'), out = join(dir, 'run'), resume = join(dir, 'resume')
    writeFileSync(input, 'abc def; byte stream, example.\n'.repeat(50))
    const cmd = (script: string, ...args: string[]) => spawnSync(process.execPath, [script, ...args], { cwd: process.cwd(), encoding: 'utf8' })
    const first = cmd('scripts/run-snn-sequence.mjs', '--input', input, '--train-bytes', '512', '--eval-bytes', '128',
      '--generate-bytes', '32', '--seeds', '17', '--out', out)
    assert.equal(first.status, 0, first.stderr)
    const results = JSON.parse(readFileSync(join(out, 'results.json'), 'utf8'))
    assert.equal(results.trainN, 512); assert.equal(results.holdoutBitsPerByte.n, 128)
    const model = join(out, 'model.json'), modelBefore = readFileSync(model, 'utf8')
    const next = cmd('scripts/run-snn-sequence-resume.mjs', '--model', model, '--seed', '17', '--generate-bytes', '32', '--out', resume)
    assert.equal(next.status, 0, next.stderr)
    assert.equal(readFileSync(model, 'utf8'), modelBefore)
    assert.deepEqual(readFileSync(join(out, 'generated-17-observed-support.bin')), readFileSync(join(resume, 'generated.bin')))
    const inputState = JSON.parse(readFileSync(join(resume, 'model-after-input.json'), 'utf8'))
    const outputState = JSON.parse(readFileSync(join(resume, 'model-after-generation.json'), 'utf8'))
    assert.deepEqual(inputState.core.counters, outputState.core.counters)
    assert.deepEqual(inputState.expert, outputState.expert)
    assert.deepEqual(inputState.fastExpert, outputState.fastExpert)
    assert.deepEqual(inputState.support, outputState.support)
    assert.equal(outputState.core.bytesSeen, 544)
    const noClip = cmd('scripts/run-snn-sequence-resume.mjs', '--model', model, '--seed', '17', '--support', 'none', '--generate-bytes', '32', '--out', resume)
    assert.equal(noClip.status, 0, noClip.stderr)
    assert.deepEqual(readFileSync(join(out, 'generated-17-unclipped.bin')), readFileSync(join(resume, 'generated.bin')))
    const more = join(dir, 'more.bin'); writeFileSync(more, Buffer.from([255]))
    const continued = cmd('scripts/run-snn-sequence-resume.mjs', '--model', model, '--input', more, '--generate-bytes', '32', '--out', resume)
    assert.equal(continued.status, 0, continued.stderr)
    assert.equal(JSON.parse(readFileSync(join(resume, 'model-after-input.json'), 'utf8')).support[255], 1)
    const oldModel = join(dir, 'v1.json'); writeFileSync(oldModel, JSON.stringify(new SpikeByteModel().snapshot()))
    const oldResume = cmd('scripts/run-snn-sequence-resume.mjs', '--model', oldModel, '--generate-bytes', '32', '--out', resume)
    assert.equal(oldResume.status, 0, oldResume.stderr)
    assert.equal(JSON.parse(readFileSync(join(resume, 'results.json'), 'utf8')).support, 'none')
    const invalid = cmd('scripts/run-snn-sequence.mjs', '--input', input, '--train-bytes', '99999', '--out', out)
    assert.notEqual(invalid.status, 0)
  } finally { rmSync(dir, { recursive: true, force: true }) }
}


export function testMultiscaleAndSupport(): void {
  const multi = new MultiscaleByteModel(), baseline = new SpikeByteModel(false)
  const input = Buffer.from('the rhythm of the word and the rhythm of the space.\n'.repeat(50))
  let multiLoss = 0, baselineLoss = 0
  for (const b of input) { multiLoss += multi.consume(b, true); baselineLoss += baseline.consume(b, true) }
  assert.ok(Number.isFinite(multiLoss) && Number.isFinite(baselineLoss))
  const state = multi.snapshot(), cm = baseline.snapshot()
  assert.deepEqual(state.core.counters, cm.counters, 'all original CM tables must remain identical')
  assert.deepEqual(state.core.history, cm.history)
  assert.ok(state.structure.spikeCount > 0)
  const resumed = MultiscaleByteModel.restore(JSON.parse(JSON.stringify(state)))
  assert.deepEqual(resumed.snapshot(), state)
  const frozen = resumed.snapshot(), rng = splitmix32(17), known = new Set(input)
  for (let i = 0; i < 512; i++) assert.ok(known.has(resumed.sample(rng, 0.8, true)), 'support clipping must rule out all unseen bytes')
  const after = resumed.snapshot()
  assert.deepEqual(after.core.counters, frozen.core.counters)
  assert.deepEqual(after.expert, frozen.expert); assert.deepEqual(after.fastExpert, frozen.fastExpert)
  assert.deepEqual(after.support, frozen.support, 'sampling must not train the support')
  assert.notDeepEqual(after.structure, frozen.structure, 'generated context must advance')
  const unsupported = new SpikeByteModel(false), singleton = new Uint8Array(256), r = splitmix32(100)
  for (let i = 0; i < 256; i++) {
    singleton.fill(0); singleton[i] = 1
    assert.equal(unsupported.sample(r, 0.8, false, undefined, singleton), i, 'all byte values including non-ASCII must be supported')
  }
  const coldA = new MultiscaleByteModel(), coldB = new MultiscaleByteModel(), r1 = splitmix32(5), r2 = splitmix32(5)
  assert.equal(coldA.sample(r1, 0.8, true), coldB.sample(r2, 0.8, false), 'empty support must fall back without a hardcoded prior')
  const a = MultiscaleByteModel.restore(JSON.parse(JSON.stringify(state))), b = MultiscaleByteModel.restore(JSON.parse(JSON.stringify(state)))
  const ra = splitmix32(37), rb = splitmix32(37)
  for (let i = 0; i < 100; i++) assert.equal(a.sample(ra, 0.8, true), b.sample(rb, 0.8, true))
  assert.deepEqual(a.snapshot(), b.snapshot())
  const changed = JSON.parse(JSON.stringify(state)); changed.structure.steps++
  assert.throws(() => MultiscaleByteModel.restore(changed), /invalid/)
}


export function testCoarseResidual(): void {
  const baseline = new MultiscaleByteModel(false), coarse = new CoarseByteModel(), fixed = new CoarseByteModel(false)
  const input = Buffer.from('words and boundaries, lines and bits.\n'.repeat(70))
  assert.equal(coarse.predict(1, 0.37), 0.37, 'a new bucket must not alter the baseline')
  for (const byte of input) { baseline.consume(byte, true); coarse.consume(byte, true); fixed.consume(byte, true) }
  const before = coarse.snapshot()
  assert.deepEqual(before.baseline, baseline.snapshot(), 'every baseline table, support bit, and state must match exactly')
  assert.deepEqual(fixed.snapshot().baseline, baseline.snapshot())
  assert.equal(before.phaseHistogram.reduce((a, b) => a + b, 0), input.length)
  assert.ok(before.phaseHistogram.filter(v => v > 0).length > 1, 'coarse phase should not be a constant on this stream')
  const restored = CoarseByteModel.restore(JSON.parse(JSON.stringify(before)))
  assert.deepEqual(restored.snapshot(), before)
  const old = restored.snapshot(), random = splitmix32(77), allowed = new Set(input)
  for (let i = 0; i < 200; i++) assert.ok(allowed.has(restored.sample(random, 0.8, true)))
  const after = restored.snapshot()
  assert.deepEqual(after.n, old.n); assert.deepEqual(after.gradient, old.gradient)
  assert.deepEqual(after.baseline.core.counters, old.baseline.core.counters)
  assert.deepEqual(after.baseline.expert, old.baseline.expert)
  assert.deepEqual(after.baseline.fastExpert, old.baseline.fastExpert)
  assert.deepEqual(after.baseline.support, old.baseline.support)
  assert.equal(after.baseline.core.bytesSeen, old.baseline.core.bytesSeen + 200)
  const a = CoarseByteModel.restore(JSON.parse(JSON.stringify(before))), b = CoarseByteModel.restore(JSON.parse(JSON.stringify(before)))
  const r1 = splitmix32(99), r2 = splitmix32(99)
  for (let i = 0; i < 80; i++) assert.equal(a.sample(r1, .8, true), b.sample(r2, .8, true))
  assert.deepEqual(a.snapshot(), b.snapshot())
  // Two models with identical coarse phase and residual tables must ignore the exact previous byte in the residual key.
  const alternate = JSON.parse(JSON.stringify(before)); alternate.baseline.core.history[0] ^= 1
  const c = CoarseByteModel.restore(alternate)
  for (let prefix = 1; prefix < 128; prefix++) assert.equal(c.predict(prefix, .4), coarse.predict(prefix, .4))
  const probe = JSON.parse(JSON.stringify(before))
  probe.baseline.structure.fastSpikes.fill(0); probe.baseline.structure.slow[3] = 0
  probe.n[4] = 10; probe.gradient[4] = 3; probe.n[5] = 10; probe.gradient[5] = -3
  const p0 = CoarseByteModel.restore(probe).predict(1, .5)
  probe.baseline.structure.slow[3] = 128
  const p1 = CoarseByteModel.restore(probe).predict(1, .5)
  assert.ok(p0 > .5 && p1 < .5, 'coarse neural state should actually affect the residual forecast')
  assert.throws(() => coarse.predict(1, 0), /invalid/)
  const broken = JSON.parse(JSON.stringify(before)); broken.baseline.structureActive = true
  assert.throws(() => CoarseByteModel.restore(broken), /fixed phase/)
}

export function testCoarseCli(): void {
  const dir = mkdtempSync(join(tmpdir(), 'ucz-coarse-'))
  try {
    const input = join(dir, 'data.bin'), out = join(dir, 'experiment'), resume = join(dir, 'resume')
    writeFileSync(input, Buffer.from('first observed, then learned. '.repeat(50)))
    const run = (script: string, ...args: string[]) => spawnSync(process.execPath, [script, ...args], { cwd: process.cwd(), encoding: 'utf8' })
    const result = run('scripts/run-snn-sequence-coarse.mjs', '--input', input, '--train-bytes', '512', '--eval-bytes', '128', '--generate-bytes', '32', '--seeds', '17', '--out', out)
    assert.equal(result.status, 0, result.stderr)
    const report = JSON.parse(readFileSync(join(out, 'results.json'), 'utf8'))
    assert.equal(report.trainN, 512); assert.equal(report.holdoutFrozen.n, 128)
    assert.equal(report.generation.find((v: any) => v.variant === 'observed-support').unseenBytes, 0)
    const state = join(out, 'model.json'), before = readFileSync(state, 'utf8')
    const continued = run('scripts/run-snn-sequence-resume.mjs', '--model', state, '--seed', '17', '--generate-bytes', '32', '--out', resume)
    assert.equal(continued.status, 0, continued.stderr); assert.equal(readFileSync(state, 'utf8'), before)
    assert.deepEqual(readFileSync(join(out, 'generated-17-observed-support.bin')), readFileSync(join(resume, 'generated.bin')))
    const resumed = JSON.parse(readFileSync(join(resume, 'model-after-generation.json'), 'utf8'))
    const original = JSON.parse(before)
    assert.deepEqual(resumed.n, original.n); assert.deepEqual(resumed.gradient, original.gradient)
    assert.equal(resumed.baseline.core.bytesSeen, 544)
  } finally { rmSync(dir, { recursive: true, force: true }) }
}

export function runSnnTests(): void {
  const cases: [string, () => void][] = [
    ['粗粒度脉冲残差与共同底座', testCoarseResidual],
    ['粗粒度序列 CLI 与恢复', testCoarseCli],
    ['多尺度字节模型与零先验支持集', testMultiscaleAndSupport],
    ['字节脉冲语言模型因果性与恢复', testSpikeByteModel],
    ['序列生成 CLI', testSequenceCli],
    ['类原型和读出行严格隔离', testIsolatedContinualAgents],
    ['多方法类增量合成实验', testContinualSynthetic],
    ['流式先预测后反馈', testStreamPrequential],
    ['类增量实验流程', testStreamPipeline],
    ['层间布局与索引数据', testLayerBridgeAndIndexedData],
    ['事件位移与时间抖动', testEventAugmentation],
    ['类别原型局部正负更新', testPrototypeLocalUpdates],
    ['阈值和正则化缩放', testThresholdAndRegularizationScaling],
    ['原型实验端到端', testPrototypePipeline],
    ['类别调制训练与 CM 模型端到端', testClassExperimentPipeline],
    ['类别调制正负局部更新', testClassModulation],
    ['CM 状态序列化往返', testCmCheckpointRoundTrip],
    ['平衡局部更新保持范数', testBalancedPlasticityNorm],
    ['独立验证和模型重载', testValidationAndCheckpoint],
    ['可塑性实验与无损对照', testPlasticExperimentProtocol],
    ['自适应阈值的门控安全', testAdaptiveThresholdGateSafety],
    ['局部可塑性与上界刷新', testLocalPlasticity],
    ['数字数据切分不重叠', testDigitsSplitDisjointOnWrap],
    ['CM 定点内核', testCmPrimitives],
    ['区间编码器往返', testRangeCoderRoundTrip],
    ['多尺度盒滤波', testBoxCounts],
    ['event 门控逐位无损', testEventGatingIsLossless],
    ['无重置上界证书不漏脉冲', testCertificateNeverMissesWithoutReset],
    ['硬重置破坏单调性', testHardResetBreaksMonotonicity],
    ['时间折叠', testTimeFolding],
    ['预测器在线学习', testPredictorLearnsOnline],
    ['top-B 选点与前置过滤', testSelectTopBRespectsBudgetAndFilter],
    ['端到端（小样本）', testEndToEndTiny],
    ['fanin 越大越划算', testWidthBreakEven],
    ['纤维块的段长更长（结构化稀疏）', testFiberBlocksAreLessFragmented],
    ['自适应预算跟着画面走', testAdaptiveBudgetFollowsTheFrame],
    ['读出层 L2 归一的尺度不变性', testReadoutL2Normalization],
    ['可复现性', testDeterminism],
    ['任务与读出层', testTaskAndReadout],
  ]
  let failed = 0
  for (const [name, fn] of cases) {
    const t0 = Date.now()
    try {
      fn()
      console.log(`  ✓ ${name} (${Date.now() - t0} ms)`)
    } catch (e) {
      failed++
      console.log(`  ✗ ${name} (${Date.now() - t0} ms)\n    ${(e as Error).message}`)
    }
  }
  if (failed) {
    console.log(`\n${failed} 项失败`)
    process.exitCode = 1
  } else {
    console.log(`\n全部 ${cases.length} 项通过`)
  }
}

// 直接 `node tests/run-snn.mjs` 时执行；被别的测试文件 import 时只导出用例。
if (require.main === module) runSnnTests()
void DEFAULT_EXP
