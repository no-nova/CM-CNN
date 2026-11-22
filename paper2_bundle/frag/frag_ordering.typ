= One evaluation set for every ordering claim

*Qwen3.5-0.8B*, dense PPL 21.0925.

#figure(
  table(
    columns: 5,
    align: (col, row) => if col == 0 { left } else if col >= 4 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ordering*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.30*],
    [*Gini*],
    [*E keep*],
    [random], [+1.591], [+3.419], [0.460], [0.9367],
    [native], [+1.412], [+2.470], [0.516], [0.9468],
    [actmag], [+1.278], [+2.383], [0.562], [0.9577],
    [wnorm], [+1.774], [+3.329], [0.493], [0.9433],
    [impsort], [+0.977], [+2.372], [0.569], [0.9592],
    [ours], [+0.963], [+1.870], [0.563], [0.9589],
  ),
  caption: [Ordering strategies on Qwen3.5-0.8B, calibration rows 0–120, evaluation rows 300–420, 16 × 512 tokens.],
)

#figure(
  table(
    columns: 3,
    align: (col, row) => if col == 0 { left } else if col >= 2 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ρ*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.40*],
    [1], [+1.057], [+3.759],
    [2], [+0.888], [+4.100],
    [4], [+0.963], [+3.964],
    [8], [+1.399], [+3.906],
    [full], [+1.285], [+5.393],
  ),
  caption: [Window radius ρ on Qwen3.5-0.8B: how many neighbouring blocks a neuron may be placed in.],
)

#figure(
  table(
    columns: 3,
    align: (col, row) => if col == 0 { left } else if col >= 2 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*lane*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.40*],
    [16], [+0.886], [+3.877],
    [32], [+0.963], [+3.964],
    [64], [+1.030], [+4.131],
  ),
  caption: [Lane width on Qwen3.5-0.8B. 32 is one warp on the GPU and two ZMM registers on the CPU.],
)

#figure(
  table(
    columns: 4,
    align: (col, row) => if col == 0 { left } else if col >= 3 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ordering / gate*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.30*],
    [*skip*],
    [native/topk], [+2.208], [+3.820], [0.196],
    [native/abs], [+1.412], [+2.470], [0.200],
    [perm/topk], [+1.553], [+3.687], [0.196],
    [perm/abs], [+0.963], [+1.870], [0.200],
  ),
  caption: [Ordering × gate factorial on Qwen3.5-0.8B at matched block budget.],
)

*Qwen3-0.6B*, dense PPL 24.4455.

#figure(
  table(
    columns: 5,
    align: (col, row) => if col == 0 { left } else if col >= 4 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ordering*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.30*],
    [*Gini*],
    [*E keep*],
    [random], [+2.591], [+6.606], [0.592], [0.9625],
    [native], [+1.745], [+4.707], [0.648], [0.9701],
    [actmag], [+0.925], [+2.206], [0.694], [0.9772],
    [wnorm], [+1.399], [+3.625], [0.631], [0.9697],
    [impsort], [+1.161], [+2.824], [0.699], [0.9786],
    [ours], [+1.468], [+3.277], [0.691], [0.9781],
  ),
  caption: [Ordering strategies on Qwen3-0.6B, calibration rows 0–120, evaluation rows 300–420, 16 × 512 tokens.],
)

#figure(
  table(
    columns: 3,
    align: (col, row) => if col == 0 { left } else if col >= 2 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ρ*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.40*],
    [1], [+1.378], [+6.362],
    [2], [+1.210], [+6.196],
    [4], [+1.468], [+6.134],
    [8], [+1.219], [+5.117],
    [full], [+2.170], [+10.264],
  ),
  caption: [Window radius ρ on Qwen3-0.6B: how many neighbouring blocks a neuron may be placed in.],
)

#figure(
  table(
    columns: 3,
    align: (col, row) => if col == 0 { left } else if col >= 2 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*lane*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.40*],
    [16], [+1.317], [+6.379],
    [32], [+1.468], [+6.134],
    [64], [+1.001], [+5.192],
  ),
  caption: [Lane width on Qwen3-0.6B. 32 is one warp on the GPU and two ZMM registers on the CPU.],
)

#figure(
  table(
    columns: 4,
    align: (col, row) => if col == 0 { left } else if col >= 3 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ordering / gate*],
    [*ΔPPL τ0.20*],
    [*ΔPPL τ0.30*],
    [*skip*],
    [native/topk], [+2.428], [+5.905], [0.198],
    [native/abs], [+1.745], [+4.707], [0.200],
    [perm/topk], [+1.876], [+4.340], [0.198],
    [perm/abs], [+1.468], [+3.277], [0.200],
  ),
  caption: [Ordering × gate factorial on Qwen3-0.6B at matched block budget.],
)

Two effects are visible in these tables. First, the absolute ΔPPL of every
strategy is several times larger here than in the 8-window tables of Section 7; the
difference is calibration proximity, not implementation, and the leak-free numbers
are the ones to quote. Second, the margin between the co-activation permutation and
the simpler activation-magnitude sort is small at τ = 0.20 and larger at τ = 0.30:
at low budgets the tail of near-dead channels dominates, and any ordering that puts
that tail last works; at higher budgets the within-block co-movement of the surviving
channels decides, which is what the output Gram matrix measures.
