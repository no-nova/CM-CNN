= Speech at scale, and the boundary of the method

#figure(
  table(
    columns: 6,
    align: (col, row) => if col == 0 { left } else if col >= 5 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*config*],
    [*utts*],
    [*WER*],
    [*95% CI*],
    [*CER*],
    [*skip*],
    [dense], [60], [0.0608], [[0.0392, 0.0855]], [0.0279], [0.000],
    [tau0.20], [60], [0.0651], [[0.0432, 0.0889]], [0.0305], [0.147],
    [tau0.40], [60], [0.0698], [[0.0462, 0.0949]], [0.0314], [0.420],
  ),
  caption: [Whisper-base on LibriSpeech dev-clean, sentence-level paired bootstrap over utterances.],
)

The speech model is reported as a negative systems result. Its encoder FFN is 6.4% of encoder time, which caps any saving at 1.068×, and the single-token `fc2` operator costs about 0.05 ms, below the cost of building the index list. The dense model is therefore faster than the sparse one on this model, and the exactly-zero block fractions of 51–54% in two encoder layers do not change that: sparsity that cannot be converted into less work is not a speedup.
