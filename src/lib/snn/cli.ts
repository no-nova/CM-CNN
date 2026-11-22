// SPDX-License-Identifier: MPL-2.0
// `npm run snn` 的入口：跑一遍完整实验，打印报告，可选把原始结果写成 JSON。
//
// 用法：
//   npm run snn                    默认规模（L2 fanin = 4×5×5 = 100）
//   npm run snn -- --quick         小样本（给测试/演示用，几秒跑完）
//   npm run snn -- --json out.json 同时把结构化结果写到文件
//   npm run snn -- --ch1 16        L2 fanin = 16×5×5 = 400（真实视觉网络的量级）
//   npm run snn -- --ch1 16 --k2 6 L2 fanin = 16×6×6 = 576
//
//   npm run snn -- --digits                        §11 合体实验（真实手写数字 × 七档配置）
//   npm run snn -- --digits --enc poisson          只跑率编码那一半（省一半时间）
//
// `--ch1` 是 L1 的输出通道数（= L2 的输入通道数），`--k2` 是 L2 的核边长。
// 两者决定 L2 的 fanin = ch1 × k2²；权重尺度会按 1/√fanin 自动归一，所以
// fanin 是唯一变量（否则换 fanin 会同时改掉发放率，A/B 就没法比了）。

import { writeFileSync } from 'node:fs'
import { DEFAULT_PLASTIC_EXPERIMENT, runPlasticDigits } from './plastic-experiment'
import {
  DEFAULT_EXP, DIGITS_VARIANTS, runDigitsMatrix, runExperiment, type ExpConfig,
} from './experiment'
import { formatDigitsMatrixReport, formatReport } from './report'

const QUICK: Partial<ExpConfig> = {
  T: 12, H: 16, W: 16, trainPerClass: 4, testPerClass: 4, budgets: [0.3, 0.2, 0.1],
}

function numFlag(argv: string[], name: string): number | null {
  const i = argv.indexOf(name)
  if (i < 0) return null
  const v = Number(argv[i + 1])
  if (!Number.isInteger(v) || v < 1) throw Error(`${name} 需要正整数，收到 "${argv[i + 1]}"`)
  return v
}

function realFlag(argv: string[], name: string, fallback: number, max: number, inclusiveMax = true): number {
  const i = argv.indexOf(name)
  if (i < 0) return fallback
  const raw = argv[i + 1]
  const value = raw === undefined || raw.startsWith('--') ? NaN : Number(raw)
  if (!Number.isFinite(value) || value < 0 || (inclusiveMax ? value > max : value >= max))
    throw Error(`${name} must be between 0 and ${max}${inclusiveMax ? '' : ' (exclusive)'}`)
  return value
}

export function main(argv: string[] = process.argv.slice(2)): number {
  const quick = argv.includes('--quick')
  const jsonAt = argv.indexOf('--json')
  const jsonPath = jsonAt >= 0 ? argv[jsonAt + 1] : null
  const cfg: Partial<ExpConfig> = quick ? QUICK : {}
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log('       npm run snn -- --plastic [--quick] [--ntrain n] [--ntest n] [--epochs n] [--steps n] [--seed n] [--json file]')
    console.log('       可塑性参数: --plastic-lr x --target-rate x --threshold-lr x --max-offset x')
    console.log('用法: npm run snn -- [--quick] [--json <file>] [--ch1 <n>] [--k2 <n>] [--ch2 <n>]')
    console.log('       npm run snn -- --digits [--enc poisson|saccade] [--ntrain n] [--ntest n] [--json <file>]')
    return 0
  }
  if (argv.includes('--plastic')) {
    const config = { ...DEFAULT_PLASTIC_EXPERIMENT,
      ...(quick ? { nTrain: 30, nTest: 20, T: 6 } : {}),
      nTrain: numFlag(argv, '--ntrain') ?? (quick ? 30 : 500),
      nTest: numFlag(argv, '--ntest') ?? (quick ? 20 : 200),
      T: numFlag(argv, '--steps') ?? (quick ? 6 : 20),
      epochs: numFlag(argv, '--epochs') ?? 1,
      seed: realFlag(argv, '--seed', DEFAULT_PLASTIC_EXPERIMENT.seed, 0xffffffff),
      plastic: {
        learningRate: realFlag(argv, '--plastic-lr', DEFAULT_PLASTIC_EXPERIMENT.plastic.learningRate, 1),
        targetRate: realFlag(argv, '--target-rate', DEFAULT_PLASTIC_EXPERIMENT.plastic.targetRate, 1),
        thresholdRate: realFlag(argv, '--threshold-lr', DEFAULT_PLASTIC_EXPERIMENT.plastic.thresholdRate, 100),
        maxOffset: realFlag(argv, '--max-offset', DEFAULT_PLASTIC_EXPERIMENT.plastic.maxOffset, 1, false),
      },
    }
    if (!Number.isInteger(config.seed)) throw Error('--seed must be an integer')
    const t = performance.now()
    const result = runPlasticDigits(config)
    console.log(JSON.stringify(result, null, 2))
    console.log(`[总耗时 ${((performance.now() - t) / 1000).toFixed(1)} s]`)
    if (jsonPath) { writeFileSync(jsonPath, JSON.stringify(result, null, 2)); console.log(`[JSON 已写入 ${jsonPath}]`) }
    return 0
  }
  // ---- §11 合体实验：只跑真实手写数字那一段 ----
  // 它比主干还贵（宽通道 + 纤维块 × 七档 × 两路编码），所以单独一个入口，
  // 也方便两路编码分成两个进程并行跑。
  if (argv.includes('--digits')) {
    const encAt = argv.indexOf('--enc')
    const enc = encAt >= 0 ? argv[encAt + 1] : null
    const encodings: ('poisson' | 'saccade')[] =
      enc === 'poisson' ? ['poisson'] : enc === 'saccade' ? ['saccade'] : ['poisson', 'saccade']
    const nTrain = numFlag(argv, '--ntrain') ?? 500
    const nTest = numFlag(argv, '--ntest') ?? 200
    const t0 = performance.now()
    const matrix = runDigitsMatrix(DEFAULT_EXP, DIGITS_VARIANTS, encodings, nTrain, nTest, 3)
    const txt = formatDigitsMatrixReport(matrix, nTrain, nTest)
    console.log(txt)
    console.log(`[总耗时 ${((performance.now() - t0) / 1000).toFixed(1)} s]`)
    const jsonPath2 = jsonAt >= 0 ? argv[jsonAt + 1] : null
    if (jsonPath2) {
      writeFileSync(jsonPath2, JSON.stringify({ encodings, nTrain, nTest, pool: 3, matrix }, null, 2))
      console.log(`[JSON 已写入 ${jsonPath2}]`)
    }
    return 0
  }

  const ch1 = numFlag(argv, '--ch1')
  const k2 = numFlag(argv, '--k2')
  const in1 = numFlag(argv, '--in1')
  const ch2 = numFlag(argv, '--ch2')
  if (in1 !== null) cfg.shape = { ...DEFAULT_EXP.shape, in1 }
  if (ch2 !== null) cfg.shape = { ...(cfg.shape ?? DEFAULT_EXP.shape), ch2 }
  if (ch1 !== null || k2 !== null) {
    const base = cfg.shape ?? DEFAULT_EXP.shape
    cfg.shape = { ...base, ch1: ch1 ?? base.ch1, k2: k2 ?? base.k2 }
    const sh = cfg.shape
    console.log(`[L2 fanin = ${sh.ch1}×${sh.k2}×${sh.k2} = ${sh.ch1 * sh.k2 * sh.k2}]`)
  }
  const res = runExperiment(cfg)
  console.log(formatReport(res))
  console.log(`[总耗时 ${(res.ms / 1000).toFixed(1)} s]`)
  if (jsonPath) {
    writeFileSync(jsonPath, JSON.stringify(res, null, 2))
    console.log(`[JSON 已写入 ${jsonPath}]`)
  }
  return 0
}

if (import.meta.url === `file://${process.argv[1]}`) process.exitCode = main()
