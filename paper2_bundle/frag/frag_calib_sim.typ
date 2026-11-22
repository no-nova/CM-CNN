= Calibration hygiene and permutation similarity

*qwen3.5-0.8b*: calibration rows [0, 120], evaluation rows [300, 420], calibration windows found inside the evaluation stream: 0.

#figure(
  table(
    columns: 5,
    align: (col, row) => if col == 0 { left } else if col >= 4 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*cal \ eval*],
    [*code*],
    [*math*],
    [*wiki*],
    [*zh*],
    [code], [+0.168], [+0.087], [+0.403], [+0.010],
    [math], [+0.124], [+0.166], [+0.705], [+0.019],
    [wiki], [+0.224], [+0.206], [+0.963], [+0.036],
    [zh], [+0.034], [+0.068], [-0.018], [+0.017],
  ),
  caption: [Cross-domain calibration on qwen3.5-0.8b, ΔPPL at τ = 0.20. Rows are the calibration corpus, columns the evaluation corpus.],
)

#figure(
  table(
    columns: 3,
    align: (col, row) => if col == 0 { left } else if col >= 2 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*pair*],
    [*rank Spearman*],
    [*block Jaccard*],
    [seed0_vs_seed1], [0.945], [0.073],
    [seed0_vs_seed2], [0.945], [0.073],
    [seed1_vs_seed2], [0.942], [0.071],
    [wiki_vs_zh], [0.461], [0.043],
    [code_vs_wiki], [0.525], [0.044],
    [code_vs_zh], [0.451], [0.042],
    [code_vs_math], [0.499], [0.043],
    [math_vs_wiki], [0.582], [0.046],
    [math_vs_zh], [0.426], [0.042],
  ),
  caption: [Permutation similarity across domains on qwen3.5-0.8b. The block-identity metric reported earlier was zero because two independent clusterings never share all 32 channels of a block; these two metrics are the meaningful ones.],
)

The calibration corpora do not have to match the evaluation corpus: a slice of
code, Chinese text or mathematics produces a permutation whose retained energy is
within one point of calibration on the evaluation domain. What matters is that the
calibration slice is disjoint from the evaluation slice; when it is not, as in the
earlier 8-window runs, the reported ΔPPL shrinks by a factor of two to four.
