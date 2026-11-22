= Cache-line census

#figure(
  table(
    columns: 7,
    align: (col, row) => if col == 0 { left } else if col >= 6 { right } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*τ*],
    [*blocks*],
    [*row lines*],
    [*block lines*],
    [*row GB/s*],
    [*block GB/s*],
    [*block ms*],
    [0.00], [288], [207360], [207360], [[—]], [24.87], [0.534],
    [0.20], [230], [166400], [165600], [13.02], [43.84], [0.242],
    [0.40], [173], [125440], [124560], [11.99], [33.07], [0.241],
    [0.60], [115], [84480], [82800], [8.18], [26.27], [0.202],
    [0.80], [58], [43520], [41760], [4.30], [16.32], [0.164],
  ),
  caption: [Distinct 64-byte lines touched by one `ffn_down` call on `blk.16` of the 4B checkpoint, counted exactly from the block map.],
)

The census and the timing together locate the gain. At τ = 0 the two layouts touch the same number of lines, and the block-major kernel is faster only because consecutive rows of a block are contiguous and the dequantiser can read 64 bytes at a time instead of 18. At τ ≥ 0.20 the row-major stream keeps paying for lines that hold one useful block and two dead ones, while the block-major stream drops whole lines, which is where the 3.8–5.7× at the single-layer level comes from.
