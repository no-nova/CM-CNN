== Calibration corpus and budget (Q11)

#figure(table(
  columns: 2,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Calibration pair*], [*Jaccard of kept blocks*],
  [wiki vs zh], [0.940],
  [code vs wiki], [0.940],
  [code vs zh], [1.000],
  [code vs math], [1.000],
  [math vs wiki], [0.940],
  [math vs zh], [1.000],
), caption: [Runtime overlap of the canonical active set (mean block energy above the calibrated threshold) between calibration corpora, averaged over 24 layers.])

#figure(table(
  columns: 3,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*$N_(cal)$ tokens*], [*$Delta$PPL*], [*95% CI (window bootstrap)*],
  [128], [+0.625], [[+0.395, +0.896]],
  [512], [+0.483], [[+0.211, +0.734]],
  [2048], [+0.483], [[+0.211, +0.734]],
  [8192], [+0.483], [[+0.211, +0.734]],
), caption: [Calibration-size sweep, Qwen3.5-0.8B, $tau$=0.20. The threshold is a quantile of the calibration energies, so 512 tokens already pin it down; more calibration data does not move the budget.])

Across the four calibration corpora the $Delta$PPL at $tau$=0.20 is +0.482 $plus.minus$ 0.306 (mean $plus.minus$ standard deviation), and the kept-block sets agree on 0.94–1.00 of their union. The tail of the distribution is what the budget actually selects, and that tail is corpus independent; the ranking inside the active region is not, which is why the per-domain numbers differ by a few tenths of a perplexity point while the block sets do not.

Raising the calibration budget moves the 0.8B result from N=128: +0.625, N=512: +0.483, N=2048: +0.483, N=8192: +0.483 (0.6B: +0.631, +0.285, +0.439, +0.496), i.e. 512 tokens are enough to fix the quantile on the 0.8B model and the 0.6B curve keeps moving at the same budget. Since the threshold is a per-layer quantile of the calibration energies, the budget matters only insofar as it changes the estimated quantile, and a model with a heavier activation tail needs more tokens to stabilise it.
