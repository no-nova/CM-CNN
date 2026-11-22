= Resolution of open questions

The table indexes the questions against the measurement that answers each, and the subsections below give the underlying numbers.

#figure(table(
  columns: 4,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*ID*], [*Question*], [*Resolution in this revision*], [*Where*],
  [Q1], [Equation (2)], [left side is now $‖Delta y‖_2^2$; a test asserts the source text], [§3.3],
  [Q2], [4B full-model PPL], [not run end to end (needs a GGUF SSM runtime); the FFN-stream square law ($R^2$ = 0.9948 / 0.9895) carries the argument], [§20.1],
  [Q3], [why the norm of the block activation orders blocks], [weight-block norms are near uniform (CV 0.05–0.09); activation energy carries the variance (Spearman 0.976)], [§17],
  [Q4], [leak-free calibration], [permutation rank correlation 0.94 across seeds, active-set Jaccard 0.94–1.00 across corpora], [§16],
  [Q5], [one evaluation set for every ordering claim], [all orderings re-measured on 16$times$512 leak-free windows; $rho$ and lane sweeps on both models], [§15],
  [Q6], [hardware width], [thread sweep 1–32 and an ISA-width ablation (scalar / AVX2 / AVX-512) both reproduce the ordering], [§20.4],
  [Q7], [dispatch cost], [threshold recomputation costs 256.3 ms/token; caching brings the overhead to 0.75 ms], [§14, §20.4],
  [Q8], [prompt-level statistics], [64 prompts, prompt as the sampling unit; agreement 0.637 [0.531, 0.726], first divergence 18.7, step KL 0.021], [§20.2],
), caption: [Question-to-evidence index. Every row points at the section that carries the measurement, not at a table number, so the index survives renumbering.])
