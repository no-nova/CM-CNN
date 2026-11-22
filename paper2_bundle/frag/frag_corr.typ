== Energy versus output error across architectures (Q4)

#figure(table(
  columns: 6,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Model*], [*FFN type*], [*CV(block weight norm)*], [*CV(block activation)*], [*Pearson*], [*Spearman*],
  [Qwen3.5-0.8B], [SwiGLU gate/up/down, 3584], [0.050], [1.456], [0.837], [0.976],
  [Qwen3-0.6B], [SwiGLU gate/up/down, 3072], [0.091], [1.567], [0.808], [0.975],
  [Qwen3.5-4B-IQ4_NL], [IQ4_NL SwiGLU, 9216], [--], [--], [--], [E_cap 0.934],
  [whisper-base], [GELU ff1/ff2, 512->2048->512], [0.042], [2.033], [0.898], [0.951],
  [SmolVLM2-256M / vision], [ViT GELU, 768->3072], [0.034], [0.959], [0.887], [0.968],
  [SmolVLM2-256M / text], [LM SwiGLU, 576->1536], [0.015], [1.800], [0.902], [0.990],
), caption: [Correlation between a 32-lane block's activation energy and the L2 norm of its contribution to the layer output, per model. Qwen3.5-0.8B and Qwen3-0.6B are 0.976 and 0.975 from Table 8; the two vision/audio rows are computed here with the same statistic; the SmolVLM2 rows are measured with hooks on `fc2` (vision, 1024 patch tokens from a 512px input) and on `down_proj` (language, a real text prompt).])

The weight-block norms are nearly uniform in every model (coefficient of variation 0.05–0.09 in the text towers), because weight decay flattens them; the activation term carries almost all the variance. That is the mechanism behind the ordering results: any strategy that ranks blocks by activation magnitude recovers most of what a co-activation-aware ordering recovers, and the residual difference only shows up where the ranking has to decide between blocks of comparable energy.
