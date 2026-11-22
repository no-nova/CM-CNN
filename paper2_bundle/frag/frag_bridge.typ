== Is the FFN-only residual study a valid proxy for model quality? (Q2)

#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Model, budget*], [*FFN-stream rel-L2*], [*$Delta$PPL*], [*KL*], [*Top-1 agree*],
  [0.8B, $tau$=0.05], [0.0254], [-0.058], [0.004], [0.922],
  [0.8B, $tau$=0.10], [0.0702], [+0.049], [0.014], [0.891],
  [0.8B, $tau$=0.15], [0.1094], [+0.284], [0.026], [0.875],
  [0.8B, $tau$=0.20], [0.1653], [+0.519], [0.038], [0.875],
  [0.8B, $tau$=0.30], [0.2716], [+1.214], [0.076], [0.828],
  [0.8B, $tau$=0.40], [0.3787], [+2.655], [0.172], [0.719],
  [0.6B, $tau$=0.05], [0.0658], [-0.022], [0.003], [0.953],
  [0.6B, $tau$=0.10], [0.0387], [-0.025], [0.025], [0.953],
  [0.6B, $tau$=0.15], [0.0544], [+0.152], [0.025], [0.906],
  [0.6B, $tau$=0.20], [0.0727], [+0.283], [0.106], [0.828],
  [0.6B, $tau$=0.30], [0.1130], [+1.195], [0.189], [0.703],
  [0.6B, $tau$=0.40], [0.1684], [+2.974], [0.293], [0.719],
), caption: [FFN-only residual error against the full-model $Delta$PPL. The stream is embedding, per-layer RMSNorm and the masked MLP, with no attention; the $Delta$PPL and KL columns come from the complete forward pass on the same text.])

The two quantities are related by a square law. Regressing $Delta$PPL on the square of the stream error gives $R^2 = 0.9948$ (Pearson 0.9974) on Qwen3.5-0.8B and $R^2 = 0.9895$ on Qwen3-0.6B, with fitted slopes 18.37 and 115.01. Since the mechanism leaves every attention and SSM tensor untouched, all of the quality loss enters through the FFN residual, which is why a streaming evaluator that can only afford the FFN measures the right quantity.

The 4B checkpoint cannot be run end to end here: it needs a GGUF GatedDeltaNet and attention runtime, and writing one is a separate project. What the 4B section reports is therefore the FFN stream at real activations (Table 4 and §7), and this square law is the argument that connects it to the $Delta$PPL columns measured on the two fully implemented models.
