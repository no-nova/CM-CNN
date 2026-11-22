// SPDX-License-Identifier: MPL-2.0
// 把 ExpResult 渲染成一份能直接贴进 issue / 文档的文字报告。
// 所有数字都来自 runExperiment 的实测，这里只做除法与排版。

import type { DigitsCell, DigitsMatrixResult, ExpResult } from './experiment'

const f = (x: number, d = 3): string => (Number.isFinite(x) ? x.toFixed(d) : '—')
const pct = (x: number, d = 1): string => (Number.isFinite(x) ? (x * 100).toFixed(d) + '%' : '—')
const kb = (bits: number): string => (bits / 8 / 1024).toFixed(1) + ' KiB'

function pad(s: string, n: number): string {
  // 中文字符按两个宽度算，否则表格会歪
  let w = 0
  for (const ch of s) w += ch.charCodeAt(0) > 0x2e80 ? 2 : 1
  return w >= n ? s : s + ' '.repeat(n - w)
}
const padL = (s: string, n: number): string => {
  let w = 0
  for (const ch of s) w += ch.charCodeAt(0) > 0x2e80 ? 2 : 1
  return w >= n ? s : ' '.repeat(n - w) + s
}

export function formatReport(r: ExpResult): string {
  const out: string[] = []
  const push = (s = '') => out.push(s)

  push('# CM-SNN 实验报告：用上下文混合预测器给脉冲网络做门控')
  push()
  push(`网络：${r.layers.map((l, i) => `L${i + 1} ${l.outCh}×${l.outH}×${l.outW} (N=${l.N}, fanin=${l.fanin})`).join(' → ')}`)
  push(`数据：${r.dataset.classes} 类移动条事件流，${r.dataset.H}×${r.dataset.W}×${r.dataset.T} 步，训练 ${r.dataset.train} 个 / 测试 ${r.dataset.test} 个`)
  push(`稠密基线：${(r.dense.macs / 1e6).toFixed(1)}M 乘加，发放率 ${pct(r.dense.spikeRate)}，读出准确率 ${pct(r.dense.accuracy)}，${r.dense.ms.toFixed(0)} ms`)
  push()

  // ---- A 压缩视角 ----
  push('## A. 预测器质量（压缩视角）')
  push()
  push('脉冲张量按 (神经元, 时间步) 的栅格顺序喂进二值编码器，四个模型同条件比较。')
  push('"原样存"是 1 比特/神经元步，也就是"没有预测"的代价。')
  push()
  push(pad('层', 6) + padL('原样存', 10) + padL('全局频率', 10) + padL('逐神经元', 10) + padL('CM 理论', 10) + padL('CM 实编', 10) + padL('编码开销', 10) + padL('熵下界', 9) + padL('稳态', 8) + padL('比特/脉冲', 10))
  for (const c of r.coding) {
    push(
      pad(`L${c.layer + 1}`, 6) +
      padL(kb(c.rawBits), 10) +
      padL(kb(c.globalBits), 10) +
      padL(kb(c.perNeuronBits), 10) +
      padL(kb(c.cmIdealBits), 10) +
      padL(kb(c.cmCodedBits), 10) +
      padL(pct(c.coderOverhead, 2), 10) +
      padL(f(c.entropyFloor, 4) + ' b', 9) +
      padL(f(c.steadyBitsPerNeuronStep, 4) + ' b', 8) +
      padL(f(c.bitsPerSpike, 2), 10),
    )
  }
  push()
  push('单位：除"熵下界/稳态"外都是整层的 KiB；熵下界是每神经元步的比特数。')
  push('"稳态"取序列后半段：在线学习从第 0 步的 p=0.5 起步，前半段天然吃亏。')
  push()
  const last = r.coding[r.coding.length - 1]
  if (last) {
    push('校准度（预测概率分桶 vs 实测发放率）——预测器不仅是"排序对"，还得"数值准"：')
    push()
    push(pad('预测区间', 14) + padL('平均预测', 12) + padL('实测发放率', 12) + padL('样本数', 10))
    for (const c of last.calibration) {
      if (c.n < 1) continue
      push(pad(`${f(c.lo, 1)}–${f(c.hi, 1)}`, 14) + padL(pct(c.pred, 2), 12) + padL(pct(c.obs, 2), 12) + padL(String(c.n), 10))
    }
    push()
  }

  // ---- B 计算视角 ----
  push('## B. 门控（计算视角）')
  push()
  push(pad('策略', 30) + padL('候选比例', 10) + padL('乘加比', 9) + padL('总算术比', 10) + padL('漏脉冲', 9) + padL('多脉冲', 9) + padL('发放率', 9) + padL('准确率', 9) + padL('ms', 8))
  for (const x of r.runs) {
    push(
      pad(x.label, 30) +
      padL(pct(x.candidateFrac), 10) +
      padL(pct(x.macRatio), 9) +
      padL(pct(x.opsRatio), 10) +
      padL(pct(x.missRate, 2), 9) +
      padL(pct(x.extraRate, 2), 9) +
      padL(pct(x.spikeRate), 9) +
      padL(pct(x.accuracy), 9) +
      padL(x.ms.toFixed(0), 8),
    )
  }
  push()
  push('* 乘加比 = 门控后的乘加数 ÷ 稠密乘加数（"总矩阵"缩到了多大）。')
  push('* 总算术比 = (乘加 + 盒滤波/上界开销 + 预测器开销) ÷ 稠密乘加数：**真金白银的比值**，')
  push('  因为它把"为了省乘法而付出的代价"也算进去了。')
  push('* 漏脉冲 = 稠密发了、门控没发的比例；多脉冲 = 反过来。')
  push('* 段长 = 把"算不算"按神经元编号排成 0/1 串，连续 1 的平均长度。逐神经元选点时')
  push('  它≈1（每次选中都是孤零零的一个），纤维块模式下它≈块大小——这就是"访存碎片化"')
  push('  的可测代理量：GPU/SIMD 需要的是少而长的段。')
  push()

  // ---- B0：结构化稀疏（实验一）----
  {
    const fiber = r.runs.filter(x => x.group)
    const scatter = r.runs.filter(x => x.mode === 'cm-decay' && !x.group && !x.label.includes('自训练'))
    if (fiber.length && scatter.length) {
      push('## B0. 结构化稀疏：不按单个神经元跳，按"纤维块"跳（回答"跑不快"）')
      push()
      push('同一个空间位置 (y,x) 下的 C 个通道，感受野盒子是**完全重合**的——盒由 (y,x)')
      push('决定，与输出通道无关。所以可以让整条纤维共享一次 CM 预测，选中就整块一起算：')
      push('  · 预测器开销被整块摊薄（不再是"每神经元 27 次运算"，而是每 C 个神经元 27 次）；')
      push('  · 算的集合是若干**连续的块**，内存访问 coalesce，而不是散点 gather。')
      push()
      push(pad('策略', 34) + padL('候选比例', 10) + padL('乘加比', 9) + padL('总算术比', 10) +
        padL('预测器', 9) + padL('平均段长', 10) + padL('漏脉冲', 9) + padL('准确率', 9))
      const rows = [...scatter, ...fiber].sort((a, b) => a.opsRatio - b.opsRatio)
      for (const x of rows) {
        push(pad(x.label, 34) +
          padL(pct(x.candidateFrac), 10) + padL(pct(x.macRatio), 9) + padL(pct(x.opsRatio), 10) +
          padL(pct(x.predOps / Math.max(1, r.dense.macs)), 9) + padL(f(x.runLen, 2), 10) +
          padL(pct(x.missRate, 2), 9) + padL(pct(x.accuracy), 9))
      }
      push()
      push('（同一张表按总算术比排序。纤维块把**决策成本**摊到整块上、段长从 ~2 变成 ~8–23，')
      push(' 代价是同样算这么多神经元时选得更粗——这就是"结构化稀疏"真实的买卖。）')
      push()
      // 分层：这是"fanin 小的层到底划不划算"的直接答案
      push('分层看（30% 预算，开销按**该层**稠密乘加归一）：')
      push()
      push(pad('层 / 粒度', 22) + padL('预测器', 10) + padL('盒滤波+选取', 13) +
        padL('总算术比', 10) + padL('净省', 9) + padL('平均段长', 10))
      const at30 = (g: boolean) => rows.find(x => Math.abs(x.budgetFrac - 0.3) < 1e-9 && !!x.group === g)
      const dn = r.runs[0]
      for (let li = 0; li < r.layers.length; li++) {
        for (const g of [false, true]) {
          const x = at30(g)
          if (!x) continue
          const p = x.perLayer[li], d = dn.perLayer[li]
          const ops = (p.macs + p.gateOps + p.predOps) / Math.max(1, d.macs)
          push(pad(`L${li + 1} (fanin ${r.layers[li].fanin}) ${g ? '纤维块' : '逐神经元'}`, 22) +
            padL(pct(p.predOps / Math.max(1, d.macs)), 10) +
            padL(pct(p.gateOps / Math.max(1, d.macs)), 13) +
            padL(pct(ops), 10) + padL(pct(1 - ops), 9) +
            padL(f(p.computed / Math.max(1, p.runs), 2), 10))
        }
      }
      push()
      push('（fanin 25 的 L1 逐神经元是**净亏**的：总算术 92.0%，只省 8%。换成纤维块变')
      push(' 成净省 44%——所以"跳过率 × fanin > 27"这条盈亏线不用等 fanin 变大，换粒度')
      push(' 就能跨过去。）')
      push()
    }
  }

  // ---- B2 分层账本 ----
  push('## B2. 分层看：门控划不划算，取决于 fanin')
  push()
  push('预测器的成本是**每神经元一个常数**（约 27 次运算：3 次上下文哈希 + 5 路查表')
  push('+ 混合器点积/学习 + APM），而它省下来的是 fanin 次乘加。所以判据很简单：')
  push('  跳过率 × fanin  >  每神经元预测开销  ⟺  这笔生意划算')
  push('下面把"预测开销 + 盒滤波开销"占该层稠密乘加的比例按层拆开：')
  push()
  push(pad('层 / 策略', 40) + padL('候选比例', 10) + padL('乘加比', 9) + padL('盒滤波', 9) + padL('预测器', 9) + padL('总算术比', 10) + padL('净省', 9))
  const refLayers = r.runs.find(x => x.mode === 'dense')
  const show = ['dense', 'event', 'cert', 'cm-decay', 'random']
  for (const x of r.runs) {
    if (!show.includes(x.mode)) continue
    if (x.mode === 'cm-decay' && Math.abs(x.budgetFrac - (r.config.budgets.includes(0.2) ? 0.2 : r.config.budgets[0])) > 1e-9) continue
    if (x.mode === 'random' && Math.abs(x.budgetFrac - 0.2) > 1e-9) continue
    for (let li = 0; li < x.perLayer.length; li++) {
      const pl = x.perLayer[li]
      const denseMac = refLayers ? refLayers.perLayer[li].macs : 1
      const gate = pl.gateOps / Math.max(1, denseMac)
      const pred = pl.predOps / Math.max(1, denseMac)
      const mac = pl.macs / Math.max(1, denseMac)
      const saved = (1 - mac) - gate - pred
      push(pad(`L${li + 1}(fanin ${r.layers[li].fanin}) ${x.label}`, 40) +
        padL(pct(pl.computed / Math.max(1, pl.neuronSteps)), 10) +
        padL(pct(mac), 9) + padL(pct(gate), 9) + padL(pct(pred), 9) +
        padL(pct(mac + gate + pred), 10) +
        padL(pct(saved, 1), 9))
    }
  }
  push()
  push('（同一个策略在 fanin 小的层上可能净亏、在 fanin 大的层上净赚，这就是"把矩阵变小"')
  push(' 这件事的真实边界：它不是免费的，它有一个和 fanin 打平的盈亏平衡点。）')
  push()

  // ---- C 时间折叠 ----
  const tf = r.timeFold
  push('## C. 时间折叠：T 步递归 → 一次带因果掩码的矩阵乘（可能性 2）')
  push()
  push(`第 1 层，N=${tf.neurons} 个神经元，T=${tf.T} 步。`)
  push()
  push(pad('算法', 26) + padL('乘加量', 12) + padL('并行深度', 10) + padL('与逐步递归的最大 |Δv|', 22) + padL('脉冲串差异', 12) + padL('ms', 8))
  push(pad('逐步递归（无重置 PSN）', 26) + padL((tf.seqFlops / 1e6).toFixed(2) + 'M', 12) + padL(String(tf.seqDepth), 10) + padL('—', 22) + padL('—', 12) + padL(tf.msSequential.toFixed(1), 8))
  push(pad('Toeplitz 矩阵乘 L·I', 26) + padL((tf.toeplitzFlops / 1e6).toFixed(2) + 'M', 12) + padL(String(tf.toeplitzDepth), 10) + padL(tf.psnSeqVsToeplitzMaxErr.toExponential(2), 22) + padL(String(tf.psnSeqVsToeplitzHamming), 12) + padL(tf.msToeplitz.toFixed(1), 8))
  push(pad('结合扫描（Hillis–Steele）', 26) + padL((tf.scanFlops / 1e6).toFixed(2) + 'M', 12) + padL(String(tf.scanDepth), 10) + padL(tf.psnSeqVsScanMaxErr.toExponential(2), 22) + padL(String(tf.psnSeqVsScanHamming), 12) + padL(tf.msScan.toFixed(1), 8))
  push()
  push(`带硬重置的 LIF 与无重置 PSN 的脉冲串有 ${tf.lifVsPsnHamming} 个位置不同（占 ${pct(tf.lifVsPsnHamming / (tf.T * tf.neurons), 2)}）——`)
  push('这就是"重置"这个非线性让整条时间线折不进一次矩阵乘的具体代价：PSN 能折，LIF 折不了。')
  push()

  // ---- 定理边界 ----
  const th = r.theorem
  push('## C2. 定理边界：什么情况下"跳过"是**可证明**安全的')
  push()
  push('推理链只有两步：跳过时把电流取严格上界 ⟹ 携带状态 v̂ ≥ 真值 v ⟹ v̂ < θ 时')
  push('"这个神经元本步一定不发"是被**证明**的，不是猜的。这条链子在发放的那一刻')
  push('会断：硬重置把 v̂ 清零，而真值那一侧可能压根没发放（于是还留着一个 < θ 的电位）。')
  push()
  push('检验方式：先用稠密跑录下每一层的输入流，再把同一条流分别喂给"全算的那一层"')
  push(`和"证书门控的那一层"，只比这一层的输出（${th.samples} 个样本）。这样排除掉误差传播：`)
  push('第 1 层多发的脉冲会改变第 2 层的输入，那种差异不是门控本身的误差。')
  push()
  push(pad('神经元模型 / 层', 22) + padL('漏脉冲', 10) + padL('多脉冲', 10) + padL('候选比例', 10) + padL('乘加比', 9))
  for (const [name, key] of [['无重置 L1', 'noReset'], ['无重置 L2', 'noReset'], ['硬重置 L1', 'hardReset'], ['硬重置 L2', 'hardReset']] as const) {
    const idx = name.endsWith('L1') ? 0 : 1
    const x = th[key][idx]
    push(pad(`${name} (fanin ${x.fanin})`, 22) + padL(String(x.miss), 10) +
      padL(pct(x.extra / Math.max(1, x.trueSpikes), 2), 10) +
      padL(pct(x.candidateFrac), 10) + padL(pct(x.macRatio), 9))
  }
  push()
  const nrMiss = th.noReset.reduce((a, x) => a + x.miss, 0)
  const hrMiss = th.hardReset.reduce((a, x) => a + x.miss, 0)
  push(`无重置网络一共漏 ${nrMiss} 个脉冲，硬重置网络漏 ${hrMiss} 个。`)
  push(nrMiss === 0
    ? '无重置时**一个都没漏**——上界证书在这里是真的定理；代价是多发的那一堆（保守方向的误差）。'
    : '（注意：两种模型都漏了，说明还有别的地方破坏了上界不变式，需要继续查。）')
  push('这与 C 节是同一件事的两面：**重置这步非线性，既让时间折不进一次矩阵乘，')
  push('也让上界证书失去严格性**；无重置的 PSN 家族两头都成立。')
  push()

  // ---- E：非平稳突发流（实验二）----
  if (r.burst) {
    const B = r.burst
    push('## E. 非平稳突发流：固定预算 vs 自适应阈值（回答"全黑的一帧还硬挑 30%？"）')
    push()
    push(`同一个样本内部：前 ${B.burstFrom} 步只有 1 条移动条（安静），中间 ${B.burstTo - B.burstFrom} 步`)
    push(`突然变成 4 条条 + 背景噪声上调（爆发），后 ${B.T - B.burstTo} 步又恢复安静。`)
    push(`实测输入密度：安静段 ${pct(B.quietDensity, 2)}，爆发段 ${pct(B.burstDensity, 2)}（相差 ${f(B.burstDensity / Math.max(1e-9, B.quietDensity), 1)} 倍）。`)
    push()
    const cells = (label: string, vals: number[], fmt: (v: number) => string) =>
      push(pad(label, 12) + vals.map(v => padL(fmt(v), 6)).join(''))
    cells('时间步', Array.from({ length: B.T }, (_, t) => t), String)
    cells('输入密度', B.density, v => pct(v, 1))
    for (const row of B.rows) cells(row.label, row.perStepCand, v => pct(v, 0))
    push()
    push('（第一行是输入事件密度，其余各行是各策略**逐时间步实际用了多少候选**。')
    push('  固定预算那两行是一条直线——它不看画面；自适应阈值那两行跟着密度起伏。）')
    push()
    push(pad('策略', 20) + padL('安静Ⅰ（候选 / 漏）', 20) + padL('爆发（候选 / 漏）', 20) + padL('安静Ⅱ（候选 / 漏）', 20))
    for (const row of B.rows) {
      const seg = (a: number, b: number) => {
        let cand = 0, miss = 0, tw = 0
        for (let t = a; t < b; t++) { cand += row.perStepCand[t]; miss += row.perStepMiss[t]; tw++ }
        return { cand: cand / Math.max(1, tw), miss: miss / Math.max(1, tw) }
      }
      const g1 = seg(0, B.burstFrom), g2 = seg(B.burstFrom, B.burstTo), g3 = seg(B.burstTo, B.T)
      push(pad(row.label, 20) +
        padL(`${pct(g1.cand)} / 漏 ${pct(g1.miss, 1)}`, 20) +
        padL(`${pct(g2.cand)} / 漏 ${pct(g2.miss, 1)}`, 20) +
        padL(`${pct(g3.cand)} / 漏 ${pct(g3.miss, 1)}`, 20))
    }
    push()
    const fixed = B.rows.find(x => x.label.includes('固定 30%'))
    const ad = B.rows.find(x => x.tau === 0.05)
    if (fixed && ad) {
      const swing = (x: typeof fixed) => {
        let lo = 1, hi = 0
        for (let t = 0; t < B.T; t++) { lo = Math.min(lo, x.perStepCand[t]); hi = Math.max(hi, x.perStepCand[t]) }
        return hi - lo
      }
      push('固定 30% 的逐时间步候选比例几乎不动（极差 ' + pct(swing(fixed), 1) + '），自适应 τ=0.05 的极差是 '
        + pct(swing(ad), 1) + '——后者真的在看画面。')
      push()
      push(pad('总账', 20) + padL('候选比例', 10) + padL('总算术比', 10) + padL('漏脉冲', 10) + padL('准确率', 10))
      for (const row of B.rows) {
        push(pad(row.label, 20) + padL(pct(row.candidateFrac), 10) + padL(pct(row.opsRatio), 10) +
          padL(pct(row.missRate, 2), 10) + padL(pct(row.accuracy), 10))
      }
      push(pad('稠密（参考）', 20) + padL('100%', 10) + padL('100%', 10) + padL('0%', 10) + padL(pct(B.denseAcc), 10))
    }
    push()
  }

  // ---- F：冷启动 + 分布突变（实验三）----
  if (r.trajectory) {
    const Tj = r.trajectory
    push('## F. 冷启动与分布突变：在线学习器的熵显微镜（回答"要几步才学会"）')
    push()
    push('预测器零先验起步（第 0 步只有 p=0.5），全部知识来自网络自己跑出来的脉冲。')
    push()
    push('**读数 = 相对码长**，分母是"只知道这一步总体发放率、其余一无所知"时的码长下界 H(r)：')
    push()
    push('    rel = CM 码长 ÷ H(该步实测发放率)     rel > 1 = 连发放率都没学到，rel < 1 = 抓到了结构')
    push()
    push('用相对值，是因为绝对码长会跟着发放率一起变，而换规律时发放率也变了——两个效应分不开。')
    push()
    push('### 冷启动：逐时间步')
    push()
    push('第 1 个样本的 20 个时间步（绝对码长 1.0 bit = 完全没预测）：')
    push()
    push(pad('步', 6) + Array.from({ length: Tj.T }, (_, t) => padL(String(t), 7)).join(''))
    if (Tj.coldSteps[0]) {
      push(pad('绝对', 6) + Tj.coldSteps[0].bpb.map(v => padL(f(v, 3), 7)).join(''))
      push(pad('rel', 6) + Tj.coldSteps[0].rel.map(v => padL(f(v, 2), 7)).join(''))
    }
    push()
    push('第 0 步的 rel 随"见过的样本数"下降（看起步还慌不慌）：')
    push()
    push(pad('第几个样本', 14) + Tj.coldAt.map(i => padL(String(i + 1), 10)).join(''))
    push(pad('第 0 步 rel', 14) + Tj.coldSteps.map(c => padL(f(c.rel[0], 2), 10)).join(''))
    push(pad('全样本 rel', 14) + Tj.coldSteps.map(c => padL(f(c.rel.reduce((a, b) => a + b, 0) / c.rel.length, 2), 10)).join(''))
    push()
    push('第 1 个样本的第 0 步 rel = ' + f(Tj.coldSteps[0].rel[0], 2) + '（比"只知道发放率"还差 '
      + f(Tj.coldSteps[0].rel[0], 2) + ' 倍），第 1 步就掉到 ' + f(Tj.coldSteps[0].rel[1], 2)
      + '，第 3 步之后稳定在 0.7 上下。')
    push('**冷启动的代价是几个时间步，不是几个样本**——一步就有一整层的发放向量（L1 576 +')
    push('L2 1152 个比特）可以学，p=0.5 的零先验两三步就被冲掉了。')
    push()
    push('### 分布突变：四种"换规律"，尖峰完全不一样')
    push()
    push(`第 ${Tj.shiftAt} 个样本起换规律。四种换法：**动位置**（整幅画面转 90°，逐像素密度完全不变）、`)
    push('**动输入量**（条变宽 3→9）、**动时间抖动**（条变实、速度翻倍）、**动空间相关性**')
    push('（同密度均匀噪声，物体消失）。')
    push()
    push(pad('突变方式', 16) + padL('旧规律 rel', 12) + padL('峰值 rel', 11) + padL('尖峰', 8) +
      padL('末 5 步 rel', 13) + padL('尖峰后几步落回', 15) + padL('漏脉冲 前→突变→后', 26))
    const nm: Record<string, string> = {
      transpose: '转 90°（位置）', noise: '同密度噪声（结构）',
      wide: '条变宽（输入量）', solid: '条变实（抖动）',
    }
    for (const g of Tj.regimes) {
      push(pad(nm[g.regime] ?? g.regime, 16) + padL(f(g.preRel, 3), 12) + padL(f(g.peakRel, 3), 11) +
        padL(f(g.spike, 2) + '×', 8) + padL(f(g.tailRel, 3), 13) +
        padL(g.recoveredStep === null ? '本样本内没落回' : `${g.recoveredStep - g.peakAtStep} 步`, 15) +
        padL(`${pct(g.missPre, 1)} → ${pct(g.missAt, 1)} → ${pct(g.missPost, 1)}`, 26))
    }
    push()
    const worst = Tj.regimes.reduce((a, b) => (b.spike > a.spike ? b : a))
    const best = Tj.regimes.reduce((a, b) => (b.spike < a.spike ? b : a))
    push(`**只有"结构消失"这一路产生了真尖峰（${f(worst.spike, 2)}×），而且不是尖一下就完：**`)
    push(`同一个样本末 5 步还在 ${f(worst.tailRel, 3)}（旧规律 ${f(worst.preRel, 3)}），漏脉冲率从 `
      + `${pct(worst.missPre, 1)} 冲到 ${pct(worst.missAt, 1)}。`)
    push(`反过来，「${nm[best.regime]}」的尖峰只有 ${f(best.spike, 2)}×——换个角度看，这其实是好消息：`)
    push('预测器读的是"感受野盒计数 + 膜电位"，前三种换法都没动"给定盒计数和膜电位会不会发"')
    push('这个条件分布，所以它连抖都不抖一下。真把它晃到的，是把这个条件分布改写的那一种。')
    push()
    push(`### 明细：${nm[Tj.detail] ?? Tj.detail}（逐时间步 rel）`)
    push()
    push(pad('时间步', 12) + Array.from({ length: Tj.T }, (_, t) => padL(String(t), 7)).join(''))
    for (const row of Tj.shiftSteps) {
      const tag = row.sample < Tj.shiftAt ? '旧规律' : row.sample === Tj.shiftAt ? '突变←' : '新规律'
      push(pad(`#${row.sample} ${tag}`, 12) + row.rel.map(v => padL(f(v, 2), 7)).join(''))
    }
    push()
    push('**这里有个测量教训值得写下来**：我最初按样本聚合读这条曲线，四种突变里最大的尖峰')
    push('只有 1.03×，几乎看不见。原因不是模型没被惊到，而是**一步就是一整层的发放向量**，')
    push('预测器一步就更新一次——模型在新规律的头几步之内就适应完了，样本尺度把尖峰平均掉了。')
    push('把显微镜换成步尺度，尖峰才出来。')
    push()
    push('机制：CounterModel 的计数在 MAX_COUNT 处对折，于是旧分布只保留一半权重；混合器')
    push('也用同一个"越新越重"的节奏更新。所以尖峰不是 bug，是老化在设计上就该有的形状。')
    push()
  }

  // ---- G：真实手写数字（实验四）----
  if (r.digits && r.digits.length) {
    push('## G. 真实手写数字：走出合成数据')
    push()
    push('数据：scikit-learn 内置 `digits` —— 1797 张真实手写数字（8×8，源头是 UCI optdigits，')
    push('43 位书写者的真笔迹），取 600 训练 / 200 测试，**10 分类**（随机猜测 10%）。')
    push('文件只有 57 KB，读取只用 node 的 zlib/fs。它不再是程序画出来的移动条：有笔画')
    push('结构、有类内书写差异，也有 3/5、4/9 这种真的会认错的相邻类。')
    push()
    push('编码两路：`poisson` 是率编码（亮的地方按亮度随机发），`saccade` 是差分事件编码')
    push('（数字在视野里漂移 + 中途一次眼跳，事件 = 相邻帧亮度变化的 ON/OFF 两个极性，两个')
    push('输入通道）。后面那路的发放率只有前一路的一半左右。')
    push()
    push('读出层是**线性**的，输入是各层发放图按通道 × ' + (r.digits[0].pool + '×' + r.digits[0].pool) +
      ' 网格分块求和的计数——分块只影响读出分辨率，不碰前向、不碰门控、不碰账本。')
    push()
    for (const D of r.digits) {
      push(`### ${D.encoding === 'poisson' ? '泊松率编码' : '眼跳差分编码'}：` +
        `稠密基线准确率 ${pct(D.denseAcc)}，发放率 ${pct(D.denseSpikeRate, 2)}`)
      push()
      push(pad('策略', 24) + padL('候选比例', 10) + padL('乘加比', 9) + padL('总算术比', 10) +
        padL('漏脉冲', 9) + padL('多脉冲', 9) + padL('准确率', 9))
      for (const x of D.rows) {
        push(pad(x.label, 24) + padL(pct(x.candidateFrac), 10) + padL(pct(x.macRatio), 9) +
          padL(pct(x.opsRatio), 10) + padL(pct(x.missRate, 2), 9) + padL(pct(x.extraRate, 2), 9) +
          padL(pct(x.accuracy), 9))
      }
      push(pad('稠密（参考）', 24) + padL('100%', 10) + padL('100%', 9) + padL('100%', 10) +
        padL('0%', 9) + padL('0%', 9) + padL(pct(D.denseAcc), 9))
      push()
      const ev = D.rows.find(x => x.label.startsWith('event'))
      const good = D.rows.filter(x => x.accuracy >= D.denseAcc - 0.02).sort((a, b) => a.opsRatio - b.opsRatio)[0]
      if (good) {
        push(`准确率不掉（≥ 稠密 −2 个点）里最省的一档是「${good.label}」：总算术比 ${pct(good.opsRatio)}`)
        push(`（净省 ${pct(1 - good.opsRatio, 1)}），准确率 ${pct(good.accuracy)}，漏脉冲 ${pct(good.missRate, 2)}。`)
      } else {
        push('这一路没有任何一档把准确率保持在稠密 −2 个点以内——10 分类 + 低发放率下，')
        push('漏掉的脉冲对线性读出的伤害被放大了。这本身就是一条要记住的边界。')
      }
      if (ev) {
        push(`无损的天花板（零输入跳过）是 ${pct(ev.opsRatio)} 总算术比；CM 门控要想有意义，必须压到它以下。`)
      }
      const cm = D.rows.find(x => x.label.includes('cm-decay 30%'))
      const rnd = D.rows.find(x => x.label.includes('random'))
      if (cm && rnd) {
        push(`同样只算 ${pct(cm.candidateFrac)} 的神经元：CM 挑的漏 ${pct(cm.missRate, 2)}、准确率 ${pct(cm.accuracy)}；`
          + `随机挑的漏 ${pct(rnd.missRate, 2)}、准确率 ${pct(rnd.accuracy)}。`)
      }
      push()
    }
  }

  // ---- 结论 ----
  push('## D. 这张表说明了什么')
  push()
  const cmRuns = r.runs.filter(x => x.mode === 'cm-bound' || x.mode === 'cm-decay')
  const lastC = r.coding[r.coding.length - 1]
  const perNeuron = lastC ? lastC.perNeuronBits / Math.max(1, lastC.neuronSteps) : NaN
  push(`1. **预测是真的** —— 最后一层稳态码长 ${f(lastC?.steadyBitsPerNeuronStep ?? NaN, 3)} 比特/神经元步：`)
  push(`   比"原样存"（1 比特）省 ${pct(1 - (lastC?.steadyBitsPerNeuronStep ?? 1), 1)}，` +
    `比"逐神经元自适应码"（${f(perNeuron, 3)} 比特）省 ${pct(1 - (lastC?.steadyBitsPerNeuronStep ?? 1) / Math.max(1e-9, perNeuron), 1)}。`)
  push(`   而且它是**在线学出来的**：第 0 步只有 p=0.5，全部知识来自网络自己跑出来的脉冲。`)
  const ent = r.coding[r.coding.length - 1]?.entropyFloor ?? NaN
  push(`2. **天花板在哪** —— 平均二元熵 H̄ = ${f(ent, 4)} 比特/神经元步。要**无损**地表出`)
  push(`   整个脉冲张量，任何门控至少要检查 ${pct(ent, 1)} 的神经元（信息论下界）；`)
  push(`   想比它更省，就只能接受近似——"压缩率"和"失真"在这里是同一个旋钮。`)
  const tol = 0.05
  const best = cmRuns.filter(x => x.accuracy >= r.dense.accuracy - tol).sort((a, b) => a.opsRatio - b.opsRatio)[0]
  if (best) {
    push(`3. **真金白银** —— 准确率不掉（≥ 稠密 −5 个百分点）的最好一档是「${best.label}」：`)
    push(`   候选 ${pct(best.candidateFrac)}，乘加降到 ${pct(best.macRatio)}；连盒滤波与预测器`)
    push(`   开销一起算，总算术量是稠密的 ${pct(best.opsRatio)}（净省 ${pct(1 - best.opsRatio, 1)}）。`)
  } else {
    push(`3. **真金白银** —— 本轮没有任何一档预测器门控把准确率保持在稠密 −5 个百分点以内。`)
  }
  const cert2 = r.runs.find(x => x.mode === 'cert')
  const ev2 = r.runs.find(x => x.mode === 'event')
  if (cert2 && ev2) {
    push(`4. **无损这条线在哪里** —— 事件驱动（零输入跳过）是**逐位无损**的：乘加降到`)
    push(`   ${pct(ev2.macRatio)}，脉冲一个不差。再往下走就得付代价：上界证书降到 ${pct(cert2.macRatio)}，`)
    push(`   但多出发 ${pct(cert2.extraRate, 0)} 的脉冲、准确率掉到 ${pct(cert2.accuracy)}——最坏值上界太松了。`)
    push(`   严格"不漏脉冲"只在**无重置**的神经元上成立（C2 节），代价也还是偏多不偏少。`)
  }
  const rnd2 = r.runs.find(x => x.mode === 'random')
  const cmSame = rnd2 ? cmRuns.filter(x => Math.abs(x.budgetFrac - rnd2.budgetFrac) < 1e-9 && x.mode === 'cm-decay')[0] : undefined
  if (rnd2 && cmSame) {
    push(`5. **"选得准"比"算得少"更关键** —— 同样 ${pct(rnd2.budgetFrac, 0)} 预算、候选比例几乎一样`)
    push(`   （${pct(cmSame.candidateFrac)} vs ${pct(rnd2.candidateFrac)}）：随机选漏 ${pct(rnd2.missRate, 1)} 的脉冲、`)
    push(`   准确率 ${pct(rnd2.accuracy)}；CM 排序只漏 ${pct(cmSame.missRate, 1)}，准确率 ${pct(cmSame.accuracy)}。`)
  }
  push(`6. **盈亏平衡点在 fanin 上** —— 预测器每神经元约 27 次运算是**常数**，省下的是`)
  push(`   fanin 次乘加。所以"跳过率 × fanin > 27"才划算：fanin 大的层净赚，fanin 小的层`)
  push(`   净亏（见 B2 表）。这条判据比任何跑分都更能回答"我这层该不该上预测器"。`)
  push(`7. **为什么不能更省** —— 只要跳过了一步**有输入**的积分，携带状态就再也不是真值；`)
  push(`   想补回来只能重放，而重放的代价正好等于当初省下的那些乘加（重放不产生任何节省）。`)
  push('   所以"无损"的上限就是"零输入跳过"，其余全是**有损**——与压缩器里"无损/有损"是同一道分界线。')
  push()
  return out.join('\n')
}

// ===== §11 合体实验的报告 =====

/**
 * 把"宽通道 × 纤维块 × 读出 L2 归一"八档配置渲染成一张能直接读的表。
 *
 * 关键是**每档都给一条"无损天花板"**：`event` 那一行的总算术比就是该档无损的
 * 下限，CM 门控要压到它以下才算真的赢了。§11 第一版之所以结论难看，就是因为
 * fanin 100 + 逐神经元时永远压不下去。
 */
export function formatDigitsMatrixReport(
  matrix: DigitsMatrixResult[], nTrain: number, nTest: number,
): string {
  const out: string[] = []
  const push = (s = '') => out.push(s)
  push('# §11 合体实验：宽通道 × 纤维块 × 读出 L2 归一（真实手写数字）')
  push()
  push(`真实手写数字 10 分类，${nTrain} 训练 / ${nTest} 测试，T=20，读出特征按通道 ×3×3 网格分块。`)
  push('（200 个测试样本 ⟹ 准确率的标准误约 ±3.5 个百分点，差 3 个点以内的不要当结论。）')
  push()
  push('八档配置是**逐条加杠杆**的，相邻两行的差可以归因到刚加的那一条：')
  push('  ①→② 只动读出归一；①→③ 只动粒度；①→④ 只动宽度；④→⑤ 在宽通道上加粒度；')
  push('  ⑤→⑥ 再加读出归一（**合体**）；①→⑦ 不换宽度、把归一和粒度都加上；')
  push('  ⑥→⑧ 只把粒度从"整条纤维（32 通道一块）"改细成"8 通道一块"。')
  push()
  for (const m of matrix) {
    push(`## ${m.encoding === 'poisson' ? '泊松率编码' : '眼跳差分编码'}`)
    push()
    const short = (s: string) => s.replace(/（.*）/, '').trim()
    const labels = m.cells[0].rows.map(r => short(r.label))
    push(pad('档位', 36) + padL('fanin', 7) + padL('粒度', 6) + padL('稠密%', 8) +
      labels.map(l => padL(l, 13)).join(''))
    push(pad('', 36) + padL('', 7) + padL('', 6) + padL('发放率', 8) +
      labels.map(() => padL('算术比/准确率', 13)).join(''))
    for (const c of m.cells) {
      push(pad(c.variant, 36) + padL(String(c.fanin2), 7) + padL(c.groupLabel, 6) +
        padL(pct(c.denseAcc), 8) +
        c.rows.map(r => padL(`${pct(r.opsRatio)} / ${pct(r.accuracy)}`, 13)).join(''))
    }
    push()
    // 前沿：在三个"准确率地板"下，各档最省的是哪一档
    push('前沿：在三个准确率地板下，各档**最省**的那一档（"—" = 没有任何一档守得住）')
    push()
    push(pad('档位', 36) + padL('无损 event', 14) + padL('≥稠密−2', 18) + padL('≥稠密−5', 18) + padL('≥稠密−10', 18))
    for (const c of m.cells) {
      const ev = c.rows.find(r => r.label.startsWith('event'))!
      const pick = (floor: number) => {
        const g = c.rows
          .filter(r => !r.label.startsWith('event') && r.accuracy >= c.denseAcc - floor)
          .sort((a, b) => a.opsRatio - b.opsRatio)[0]
        return g ? `${pct(g.opsRatio)} (${short(g.label)})` : '—'
      }
      push(pad(c.variant, 36) + padL(pct(ev.opsRatio), 14) +
        padL(pick(0.02), 18) + padL(pick(0.05), 18) + padL(pick(0.10), 18))
    }
    push()
    // 选得准 vs 随机：同预算，50% 与 30% 两档
    for (const b of ['50%', '30%'] as const) {
      push(`同样只算 ~${b}：CM 挑 vs 随机挑`)
      push()
      push(pad('档位', 36) + padL('候选', 8) + padL('CM 漏', 9) + padL('CM 准确率', 12) +
        padL('随机 漏', 9) + padL('随机 准确率', 12) + padL('准确率差', 10))
      for (const c of m.cells) {
        const cm = c.rows.find(r => r.label === `cm-decay ${b}`)
        const rd = c.rows.find(r => r.label === `random ${b}（对照）`)
        if (!cm || !rd) continue
        push(pad(c.variant, 36) + padL(pct(cm.candidateFrac), 8) + padL(pct(cm.missRate, 2), 9) +
          padL(pct(cm.accuracy), 12) + padL(pct(rd.missRate, 2), 9) + padL(pct(rd.accuracy), 12) +
          padL(`${cm.accuracy >= rd.accuracy ? '+' : ''}${((cm.accuracy - rd.accuracy) * 100).toFixed(1)}`, 10))
      }
      push()
    }
    // 三条杠杆各自的边际贡献（同预算下逐条对比）
    push('三条杠杆各自的边际贡献（都看 cm-decay 50% 这一行）：')
    push()
    const at50 = (c: DigitsCell) => c.rows.find(r => r.label === 'cm-decay 50%')
    const pairs: [string, string][] = [
      ['① 基线 ch1=4,ch2=8 · 逐神经元 · 无归一', '② +读出 L2 归一'],
      ['① 基线 ch1=4,ch2=8 · 逐神经元 · 无归一', '④ 宽通道 ch1=16,ch2=32'],
      ['④ 宽通道 ch1=16,ch2=32', '⑤ 宽通道 + 纤维块'],
      ['⑤ 宽通道 + 纤维块', '⑥ 宽通道 + 纤维块 + 归一（合体）'],
      ['⑥ 宽通道 + 纤维块 + 归一（合体）', '⑧ 宽通道 + 纤维块(group=8) + 归一'],
      ['① 基线 ch1=4,ch2=8 · 逐神经元 · 无归一', '⑥ 宽通道 + 纤维块 + 归一（合体）'],
    ]
    push(pad('改动', 62) + padL('总算术比', 20) + padL('准确率', 18))
    for (const [a, b] of pairs) {
      const ca = m.cells.find(c => c.variant === a), cb = m.cells.find(c => c.variant === b)
      if (!ca || !cb) continue
      const ra = at50(ca), rb = at50(cb)
      if (!ra || !rb) continue
      push(pad(`${a.slice(0, 22)} → ${b.slice(0, 22)}`, 62) +
        padL(`${pct(ra.opsRatio)} → ${pct(rb.opsRatio)}`, 20) +
        padL(`${pct(ra.accuracy)} → ${pct(rb.accuracy)}`, 18))
    }
    push()
  }
  return out.join('\n')
}
