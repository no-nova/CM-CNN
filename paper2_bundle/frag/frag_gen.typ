== Prompt-level generation statistics (Q9)

#figure(table(
  columns: 6,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Protocol*], [*Token agreement*], [*First divergence*], [*Median*], [*Step KL*], [*Top-1 prob*],
  [Greedy, 64 prompts $times$ 32 tok], [0.637 [0.531, 0.726]], [18.69 [15.19, 21.78]], [22.5], [0.0205], [0.9722],
  [$T$=0.7, seed 0 (16 prompts)], [--], [--], [--], [0.0835], [0.355],
  [$T$=0.7, seed 1 (16 prompts)], [--], [--], [--], [0.0943], [0.314],
  [$T$=0.7, seed 2 (16 prompts)], [--], [--], [--], [0.0828], [0.293],
), caption: [Prompt is the sampling unit; intervals are 1000-sample paired bootstraps over prompts. Sampling rows draw one trajectory from the dense model at temperature 0.7 and score every step of it under both models, so their KL is a trajectory-averaged divergence, not a per-token deviation.])

The greedy agreement of 0.637 is dominated by the 32-token budget: 11 of 64 prompts diverge within the first three tokens and then stay apart, which pulls the mean below the median 0.875. The per-step KL of 0.021 with a top-1 probability match of 0.972 says the sparse model rarely changes which token is most likely; sampling raises the divergence to 0.08–0.09 because it exposes the whole distribution rather than its argmax.
