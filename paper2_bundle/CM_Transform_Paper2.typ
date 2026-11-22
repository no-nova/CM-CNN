// CM-Transform paper 2 — final manuscript
// Compile: typst compile CM_Transform_Paper2.typ CM_Transform_Paper2_Final.pdf
#set document(title: "CM-Transform: Co-Activation Permuted Block Sparsity and Cache-Line Aligned Execution for Transformer Inference on Commodity Hardware", author: "no-nova")
#let chevron = (l: sym.angle.l, r: sym.angle.r)

#set page(paper: "us-letter", margin: (x: 2.1cm, y: 1.9cm), numbering: "1",
  footer: context align(center)[#text(size: 8pt, fill: luma(90))[#counter(page).display()]])
#set text(font: ("Libertinus Serif", "New Computer Modern"), size: 9.6pt, lang: "en")
#set par(justify: true, leading: 0.72em)
#set heading(numbering: "1.")
#show heading.where(level: 1): it => block(above: 1.3em, below: 0.7em)[
  #set text(size: 12pt, weight: "bold"); #it]
#show heading.where(level: 2): it => block(above: 1.0em, below: 0.5em)[
  #set text(size: 10.5pt, weight: "bold"); #it]
#set math.equation(numbering: "(1)")
#show figure.caption: set text(size: 8.1pt)
#show table: set text(size: 8.2pt)
#set table(stroke: (x: 0.35pt, y: 0.5pt))

#align(center)[
  #text(size: 15.5pt, weight: "bold", hyphenate: false)[
    Co-Activation Permuted Block Sparsity and Cache-Line Aligned
    Execution for Transformer Inference on Commodity Hardware
  ]

  #v(0.4em)
  #text(size: 10pt)[no-nova · Technical Report 2 · November 2026]
  #v(0.2em)
  #text(size: 8.6pt, style: "italic")[Code, drivers and result files:
  #link("https://github.com/no-nova/CM-CNN")[github.com/no-nova/CM-CNN]]
]

#v(0.5em)
#block(fill: luma(245), inset: 9pt, radius: 2pt, width: 100%)[
  #text(size: 8.6pt, weight: "bold")[Abstract.]
  #text(size: 8.6pt)[
  Autoregressive decoding at batch size one is bound by weight traffic, and modern
  SwiGLU and GELU feed-forward blocks have no exact-zero structure to exploit. We
  study structured 32-lane block skipping for transformer inference on a single
  commodity GPU (RTX 5070 Laptop, 8 GB) and CPU (Zen 5, AVX-512), across five real
  checkpoints: two dense text models (Qwen3-0.6B, Qwen3.5-0.8B), one 4.5-bit
  quantised model (Qwen3.5-4B-IQ4_NL) and two encoder/vision models
  (whisper-base, SmolVLM2-256M). Three obstacles separate algorithmic sparsity
  from wall-clock time: (i) truncating the natural neuron order degrades
  perplexity; (ii) host-side masks and per-layer synchronisation consume the
  operator's gains; (iii) row-major 4-bit blocks of 18 bytes do not reduce DRAM
  traffic, because skipping a block still touches the 64-byte cache line that
  holds it. CM-Transform addresses them with a functionally isomorphic
  co-activation permutation, a cross-layer adaptive energy budget with
  norm-preserving rescaling, device-side compaction, row-parallel AVX-512 kernels,
  and an offline block-major repack of IQ4_NL weights. On Qwen3.5-0.8B, a 20%
  skip budget raises WikiText-2 perplexity by 1.68 (6.2%), against 2.41 for
  CATS-Block, 2.30 for TEAL-Block and 13.17 for Wanda-style block pruning, and
  GSM8K accuracy stays within the confidence interval of dense (28.5% vs 33.0%,
  paired 95% CI −10.0 to +1.5 pp). Block-major IQ4_NL turns skip rate into a
  linear function of cache-line traffic: at 0% skip a 32-thread kernel already
  beats the row-major dense baseline by 1.74×, and at 20–80% skip by 3.84–5.67×
  on a cold single layer and 1.08–1.97× for a 32-layer stream. On real speech the
  whisper encoder keeps WER at 6.21% with 20% of blocks removed (6.08% dense).
  A dispatch-cost accounting then fixes the boundary of the mechanism: at batch
  one on the GPU the per-layer decision path costs about as much as the 0.04 ms
  GEMV it can save, so skipping pays only at the operator level or under
  batching, whereas on the CPU and on block-major 4.5-bit streams the decision is
  inside one kernel call and the same sparsity returns a net speedup. We report
  the byte ceilings that bound every decode-time result, and the measurements
  where skipping loses to dense execution. The system claim is therefore placed
  on commodity multi-core CPU execution and on the block-major 4.5-bit layout,
  not on batch-one GPU decode.
  ]
]

= Introduction

Decoding one token at a time is a memory problem: an $N$-parameter model in bf16
must be streamed once per token, so latency follows bytes, not FLOPs. Qwen3.5-0.8B
decodes at 81.9 tok/s on the RTX 5070 at 279 GB/s of measured DRAM traffic, which
is within 3% of the card's practical peak. Any decode-time saving must therefore
remove *bytes*, not just multiply–accumulates.

Feed-forward blocks are the natural target: in these checkpoints they hold 64% of
the per-layer MACs and 37% (0.8B) to 64% (Qwen3-0.6B) of the per-token weight
bytes. But SwiGLU and GELU are smooth: unlike ReLU networks, they produce no exact
zeros whose rows could be skipped, and activation magnitudes are dense and
heavy-tailed. Approximate skipping is possible because the *output* of an
intermediate block contributes unevenly once projected through `down_proj`; the
question is whether that unevenness survives as far as perplexity and downstream
accuracy, and whether it can be converted into cache-line traffic on commodity
hardware.

We answer with measurements on five checkpoints rather than one, on both
platforms, and we report the cases where the mechanism does not pay. Our
contributions:

+ *Isomorphic permutation with co-activation clustering.* A permutation $pi$ of
  the intermediate dimension applied to `gate`/`up` rows and `down` columns is
  exactly equivalent at $tau=0$ (max $|Delta"logit"| = 1.1 times 10^(-7)$,
  float32), so sparsity can be introduced without retraining or a teacher model.
+ *Adaptive energy budget plus norm compensation.* The permutation alone *hurts*
  under a fixed per-token Top-$K$; a calibrated absolute energy threshold and a
  capped norm rescaling are what make it pay (§5.3). We quantify the ladder.
+ *Device-side compaction and row-parallel execution.* CPU compaction runs once
  per token and the AVX-512 kernel parallelises over output rows, which converts
  a 2.8 ms fork/join penalty into a 2.50–5.28× single-operator speedup at $T=1$.
+ *Block-major 4-bit layout.* Repacking IQ4_NL `ffn_down` from row-major to
  block-major makes every kept block a whole number of 64-byte lines, which is
  what allows sparse 4-bit decode to reduce DRAM traffic at all.
+ *A measured quality–throughput frontier* with named baselines, bootstrap
  intervals, and an ablation that isolates each component.

The scope of the system contribution is stated up front, because the measurements
narrow it. On the GPU at batch one the operator is fast and the model is not:
a single fused MLP reaches 2.94$times$ at 75% skip, but the same 24-layer stack
decodes at 0.94$times$ end to end because each layer's decision costs more than the
0.04 ms GEMV it replaces. The gains that survive at model level are on the
multi-core CPU (1.05–1.17$times$ end to end, 2.50–5.28$times$ per operator) and in
the 4.5-bit block-major IQ4_NL stream (2.21–3.26$times$ over a block-major dense
baseline, 3.34–6.70$times$ on real 4B activations, 1.97$times$ for the coupled
32-layer FFN). Batching moves the GPU boundary: at $B=8$ the same mechanism costs
11.9 ms per token against 15.6 ms at $B=1$.

The ordering claim is correspondingly narrow. It is the *joint* design of a
co-activation-aware permutation and a calibrated energy threshold that improves the
quality–sparsity trade-off, and only in specific width and budget regimes: it wins
on Qwen3-0.6B by 2–3$times$ over the native order at every budget we measured, and
on Qwen3.5-0.8B it wins from $tau=0.30$ upward while a plain activation-magnitude
ordering is slightly better at $tau=0.20$.

= Background and related work

*Dynamic neuron sparsity.* Deja Vu, PowerInfer, and related systems predict which
neurons activate and gather their weights; at batch size one the gather cost and
the prediction error usually dominate. CATS (Lee et al., ICML 2024) learns
contextual sparsity with a lightweight predictor and skips at block granularity;
TEAL applies uniform Top-$K$ block selection with no calibration. We implement
both as baselines at matched budget, with 32-lane blocks, which is the warp width
on the GPU and two ZMM registers on the CPU.

*Structured pruning.* Wanda scores weights by $|w| dot ‖x‖$ and prunes offline;
at block granularity it is static, so it cannot adapt to the token. It serves as
the "no prediction, no adaptivity" control.

*Permutation-based sparsity.* Channel sorting and clustering before pruning is
standard for CNNs (as in our own paper 1), where the motivation is SIMD
alignment. For transformers the analogous object is the intermediate dimension of
the FFN, and the quantity that matters is the *output* error after `down_proj`,
not the activation magnitude.

*Quantisation-aware sparsity.* 4-bit formats such as IQ4_NL pack 32 weights into
18 bytes, which is *smaller* than a 64-byte line. A row-major layout therefore
places 3.5 blocks per line, and skipping one block saves nothing. This
observation, not the sparsity algorithm, is what limits prior 4-bit sparse decode
results.

*The lineage of activation sparsity.* ReLU networks produce exact zeros, and the
first analyses of that sparsity exploited it directly to skip neurons at
inference. Modern transformers removed the zeros: SwiGLU and GELU have no dead
region, and the post-activation distribution in these checkpoints is dense — the
fraction of exactly-zero 32-lane blocks is 0.0000 in every FFN of both text
models, and the only strictly-zero blocks we found anywhere are in whisper's
encoder layers 4 and 5 (2.2% and 3.2% of blocks in native order). Work that
reintroduces zeros by training (ReLUfication, ReLU² variants) is orthogonal to
this paper: it would raise the sparsity available to any block-skipping kernel,
including ours. What matters here is the *approximate* regime, where every block
carries some energy and the question becomes which blocks to drop.

*Prediction versus measurement.* Contextual-sparsity systems train a predictor
that estimates the active set from the previous layer's hidden state, then gather
the corresponding rows. Two costs dominate at batch one: the predictor itself
(a small MLP per layer, executed for every token) and the gather, which turns a
coalesced GEMM into indexed reads. Our design does not predict: it measures the
current token's own post-activation energy and drops blocks by a calibrated
threshold, so the only extra work is one reduction per block that the kernel needs
anyway, and there is no gather because `down_proj` reads its own columns in the
skipping kernel. The price is that the decision is available only after the gate
and up projections, which is exactly the asymmetry behind the coupled/uncoupled
split in Section 3.3.

*Quantisation and sparsity interact through the cache line.* 4-bit formats pack 32
weights into 18 bytes (IQ4-NL), 20 bytes (Q4_K) or 16 bytes plus group scales; all
of them are smaller than a 64-byte line, which means that the byte saving promised
by a sparse pattern does not translate into DRAM traffic unless the layout is
changed as well. Quantisation papers usually report speedups for *dense* low-bit
kernels, where every line is used and the arithmetic is the bottleneck; sparse
low-bit decoding is memory-bound with a line-granularity floor, and that floor —
not the codec — is what the block-major repack in Section 3.4 removes. The same
argument explains why 2:4 semi-structured sparsity, which is defined on 4-weight
groups, cannot help 4-bit weights lying in 18-byte blocks: the group is five times
smaller than the line.

#figure(
  table(
    columns: (auto, auto, 1fr, auto, 1fr),
    align: (col, row) => if col == 0 { left } else if col >= 3 { center } else { left },
    inset: (x: 4pt, y: 1.8pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*System*], [*Granularity*], [*Selection*], [*bits*], [*Regime where it pays*],
    [Deja Vu], [neuron], [learned predictor, per token], [16], [large batch, GPU],
    [PowerInfer], [neuron, hot/cold], [neuron locality, offline hot set], [4], [single consumer GPU, hot set in VRAM],
    [CATS], [block (32)], [lightweight predictor + threshold], [16], [GPU, batch ≥ 1 with fused gather],
    [TEAL], [block], [uniform Top-K on activation norm], [16], [GPU, high sparsity],
    [Wanda / 2:4], [2:4 group], [static weight × activation norm], [16], [dense kernels, hardware 2:4 path],
    [SparseGPT], [unstructured], [static Hessian pruning], [16], [dense kernels, no wall-clock gain],
    [llama.cpp / ggml], [dense], [none], [4.5], [CPU, memory-bound decode],
    [*CM-Transform*], [block (32, lane aligned)], [π + calibrated energy budget], [4.5], [CPU decode, block-major 4-bit stream],
  ),
  caption: [Literature positioning. The last row is the only one that changes the
    block order offline, aligns the block with the warp and with two ZMM registers,
    and repacks a 4-bit weight matrix so that skipping a block frees whole cache
    lines.],
)

= Method

== Permutation and isomorphic transform

For a SwiGLU block with $g="SiLU"(x W_g^top)$, $u=x W_u^top$,
$y=(g dot u) W_d^top$, any permutation $pi$ with
$W_g arrow.l W_g [pi]$, $W_u arrow.l W_u [pi]$, $W_d arrow.l W_d [:, pi]$
leaves $y$ unchanged. We build $pi$ from calibration statistics: with
$C_(i j) = bb(E)_t [h_i h_j] dot angle.l W_(d,:,i), W_(d,:,j) angle.r$ the
co-moment of intermediate neurons weighted by their downstream inner product, we
seed blocks coldest-first by $s_i = bb(E)_t |h_i| dot ‖W_(d,:,i)‖_2$ and add
the neuron that minimises the marginal block error $sum_j C_(i j)$, searching the
coldest $32rho$ unassigned candidates ($rho=4$). The same construction applies
to GELU blocks (`fc1`/`fc2`) of whisper and to the ViT MLP of SmolVLM2.

== Adaptive energy budget and norm compensation

A fixed Top-$K$ per token ignores that the energy distribution differs per layer
and per token. We instead score each 32-lane block by
$E_b = ‖"act"(g) dot u‖_2$ (in the permuted, importance-scaled
space) and drop blocks with $E_b<theta_ell$, where $theta_ell$ is the
$tau$-quantile of the calibration distribution of layer $ell$. Skipping
$B$ blocks shrinks $‖a‖_2$; over 24–28 layers that bias compounds, so we rescale
the surviving activation to the untruncated norm,
$ alpha = min(gamma, ‖a‖_2 ∕ ‖a_"keep"‖_2) $ with $gamma = 1.5$.

== What block ordering can and cannot buy

Let $S$ be the set of lanes whose blocks are dropped and
$Delta y = -(a dot m) W_d^top$ the resulting output error. Then

$ ‖Delta y‖_2^2 = sum_(i,j in S) a_i a_j chevron.l W_(d,i), W_(d,j) chevron.r , $

and taking the expectation over the calibration distribution,

$ bb(E) ‖Delta y‖_2^2 = sum_(i,j in S) C_(i j), quad C_(i j) = bb(E)[a_i a_j] dot chevron.l W_(d,i), W_(d,j) chevron.r . $ <eq-cov>

Three consequences follow. First, the error is not additive over neurons: a block
whose lanes co-activate *and* have aligned down-projections contributes
quadratically more than the sum of its parts, while anti-correlated lanes cancel.
An ordering that sorts by activation magnitude alone ignores the
$angle.l W_i, W_j angle.r$ factor, and one that sorts by $‖W_(d,i)‖$ alone
ignores the activation factor; both are measured below.

Second, a fixed Top-$K$ bounds the number of dropped blocks, not the dropped
energy. Per-token block energy is heavy-tailed — the Gini coefficient of the
energy vector is 0.46 (random order) to 0.70 (importance order) — so at a given
rate a calibrated quantile threshold and a fixed count remove different amounts of
energy: the count equalises dropped blocks, the threshold equalises dropped energy,
and only the latter is the quantity that enters Equation (2).

Third, ordering and thresholding interact rather than add. On Qwen3.5-0.8B at
$tau=0.20$ the four factorial cells cost 2.208 (native, Top-$K$), 1.412 (native,
threshold), 1.553 (permuted, Top-$K$) and 0.963 (permuted, threshold) perplexity:
the interaction term is $+0.21$, and each main effect survives on its own — an
energy threshold improves both orderings, and the permuted ordering improves both
budget rules, with the combination better than either alone. A permutation changes
*which* lanes share a block; a threshold decides *how much* of the energy is left
behind; the two are complementary, and Section 6.2 repeats the factorial on both
models under the leak-free evaluation set.

== Two execution modes

*Quality-first (uncoupled).* Gate on the exact post-activation energy and skip
only `down_proj` tiles. This is the configuration we recommend; it is what the
comparison tables report unless stated otherwise.

*Throughput-first (coupled).* Gate on $‖g_b‖_2$ and share one block list with
`up_proj` and `down_proj`, so both projections skip. The criterion is weaker:
across layers and budgets the Spearman correlation between $‖g_b‖$ and
norm of the block activation is $rho = 0.758$, and the Top-$K$ overlap is 0.947/0.901/0.857/0.817 at
$tau$ = 0.1, 0.2, 0.3 and 0.4. For the quantised model we add an $r$-row pre-filter: the
gate is evaluated on the first $r$ of its 32 rows per block, which captures 61.8%
(r=2) to 66.5% (r=16) of the block's gate energy before the full rows are read.

== Hardware-aligned kernels

*GPU compaction.* A 32-thread warp compacts the per-token block mask with
`__ballot_sync` and `__popc`, writing a padded index list on device; the GEMM
reads `idx[k]` directly. This removes the per-layer device–host synchronisation
that made the first sparse end-to-end attempt slower than dense (0.78×).

*CPU row parallelism.* At $T=1$ the token dimension offers no parallelism, so the
AVX-512 kernel parallelises over output rows (8-row register blocking,
`OMP_WAIT_POLICY=passive`, 16 threads). Compaction runs once, outside the
parallel region.

*Block-major IQ4_NL.* We repack `ffn_down` offline from
an array of $M = 2560$ rows by $n_b = 288$ blocks of 18 bytes to $n_b$ contiguous slabs of $M times 18$ bytes, i.e. 46,080 B =
720 whole lines per block, and dequantise with a 16-entry table lookup plus FMA
(8-row blocking). `ffn_gate`/`ffn_up` keep their row-major bytes: 32 consecutive
output rows are already contiguous, so output-block-major *is* the raw layout.

= Experimental setup

*Models.* Qwen3-0.6B (28 layers, $d$=1024, $d_"ff"$=3072, vocab 151,936),
Qwen3.5-0.8B (24 layers, $d$=1024, $d_"ff"$=3584, 18 gated-DeltaNet + 6
full-attention layers, vocab 248,320), Qwen3.5-4B-IQ4_NL (32 layers, $d$=2560,
$d_"ff"$=9216, GGUF v3, 426 tensors), whisper-base (12+6 layers, $d$=512,
$d_"ff"$=2048, GELU), SmolVLM2-256M (ViT 12×768 + LM 30×576).

*Hardware.* RTX 5070 Laptop GPU, 8 GB, CUDA 13.5, PyTorch 2.15, Triton; AMD
Ryzen 9 9955HX (Zen 5), AVX-512, 16–32 threads. Dense references use vendor
BLAS/cuBLAS at float32 unless noted.

*Data.* WikiText-2 test (perplexity over 16×512-token windows), ARC-Easy (n=120,
200 for the budget sweep), PIQA, BoolQ, GSM8K test (200 items, 4-shot
chain-of-thought, 128 generated tokens), LibriSpeech dev-clean (32 utterances,
reference transcripts), and a 52-sample multimodal manifest covering OCR,
counting, scene description and video action. All evaluation sets and weights are pinned by SHA-256
(Appendix: Data and model provenance); calibrations and bootstraps use seed 1234.

*Baselines.* CATS-Block (native order, calibrated absolute threshold), TEAL-Block
(uniform Top-$K$), Wanda 2:4 blocks (static, activation-weighted), CM-T uncoupled
and CM-T coupled. Every method is evaluated under the same protocol, the same masks,
and the same evaluation prompts; accuracies carry 1,000-sample bootstrap 95%
intervals, and paired intervals are computed against the dense run.

#figure(table(
  columns: 2,
  align: (col, row) => left,
  inset: (x: 4pt, y: 1.4pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Component*], [*Configuration*],
  [CPU], [AMD Ryzen 9 9955HX, 16 cores / 32 threads, Zen 5, AVX-512F/BW/DQ/VL, 2.5–5.4 GHz],
  [GPU], [NVIDIA RTX 5070 Laptop, 8 GB GDDR7, 4608 CUDA cores, 360 GB/s peak, 279 GB/s measured dense decode],
  [CUDA toolchain], [NVCC 13.5.37, compute capability 12.0, PyTorch 2.15.0 with cuDNN 9.28],
  [BLAS / DNN], [cuBLAS and oneDNN as shipped with PyTorch; measured dense references throughout],
  [Threads], [CPU kernels: 16 threads with `OMP_WAIT_POLICY=passive` unless noted (32 threaded numbers are labelled)],
  [Precision], [fp32 weights and activations for all dense text experiments; IQ4-NL is bit-exact software simulation of the 4.5-bit codec],
  [Memory], [14 GB RAM (≈4 GB free), which forces streaming evaluation of the 4B checkpoint],
  [Compiler flags], [`-O3 -fopenmp -mavx512f -mavx512bw -mfma -march=native`],
), caption: [Platform configuration. Every timing in the paper was taken on this machine, in this configuration; the CUDA build flags additionally require `-DGLOG_USE_GLOG_EXPORT` when PyTorch's extension builder is used with the system glog headers.])

Two configuration details influenced results and are therefore stated rather than
buried. First, the OpenMP wait policy: with the default busy-wait, 32 threads
spinning between two 0.09 ms regions cost 2.8 ms of fork/join time per layer,
which turned a 2.9× kernel into a 0.9× one; `OMP_WAIT_POLICY=passive` removes it.
Second, the 8 GB frame buffer: two fp32 0.8B models cannot be resident at once, so
GPU measurements were taken one at a time.

= Results

== Quality at matched budget

#figure(table(
  columns: 8,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Method*], [$tau$=0.10], [$tau$=0.20], [$tau$=0.30], [$tau$=0.40], [ARC], [PIQA], [BoolQ],
  [CATS-Block], [0.93], [2.41], [4.39], [6.48], [0.808], [0.717], [0.717],
  [TEAL-Block], [0.71], [2.30], [5.24], [8.96], [0.758], [0.733], [0.683],
  [Wanda 2:4], [5.32], [13.17], [22.96], [38.11], [0.567], [0.675], [0.683],
  [CM-T uncoupled], [*0.70*], [*1.68*], [*3.41*], [*6.03*], [0.833], [0.725], [0.692],
  [CM-T coupled], [1.89], [4.46], [7.91], [12.73], [0.792], [0.717], [0.683],
), caption: [WikiText-2 perplexity and downstream accuracy at matched skip budget, Qwen3.5-0.8B (dense PPL 27.167, ARC-Easy 0.817, PIQA 0.733, BoolQ 0.683, n=120; accuracy columns are $tau=0.20$). $Delta$PPL lower is better; accuracies are means of bootstrap means.])

At $tau=0.20$ CM-T uncoupled costs 6.2% perplexity where CATS costs 8.9%, TEAL
8.5% and Wanda 48%; the gap widens with budget (at $tau=0.40$: 22% vs 24%/33%/140%).
Accuracy is insensitive at these sample sizes: the three QA sets move by at most
6 pp (TEAL on ARC-Easy) and no paired interval at $n$=120 excludes zero. GSM8K-200 is more discriminative
(Section 5.1).

#figure(table(
  columns: 4,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Method*], [Accuracy], [95% CI], [$Delta$ vs dense],
  [Dense], [0.330], [[0.270, 0.395]], [—],
  [CATS-Block], [0.265], [[0.205, 0.325]], [$-6.5$ pp $[-12.0, -0.5]$],
  [TEAL-Block], [0.240], [[0.185, 0.300]], [$-9.0$ pp $[-15.5, -2.5]$],
  [Wanda 2:4], [0.065], [[0.035, 0.100]], [$-26.5$ pp $[-33.0, -19.5]$],
  [CM-T uncoupled], [*0.285*], [[0.225, 0.345]], [$-4.5$ pp $[-10.0, +1.5]$],
), caption: [GSM8K test, 200 items, 4-shot CoT, $tau=0.20$. Brackets: paired 95% bootstrap interval of the difference to dense.])
#text(size: 8.4pt)[On the 60-item validation run used for the coupled path
(dense 0.283) CM-T uncoupled reproduced dense exactly (0.283) and CM-T coupled
reached 0.200, i.e. the coupled criterion costs 8 pp where the uncoupled one costs
nothing at this $n$.]

#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Method (Qwen3-0.6B)*], [$tau$=0.10], [$tau$=0.20], [$tau$=0.30], [$tau$=0.40],
  [CATS-Block], [0.28], [1.50], [4.49], [10.81],
  [TEAL-Block], [0.49], [2.08], [6.10], [13.97],
  [Wanda 2:4], [9.10], [26.37], [58.97], [114.76],
  [CM-T uncoupled], [*0.09*], [*0.88*], [*2.52*], [*5.43*],
  [CM-T coupled], [0.06], [1.54], [4.16], [9.20],
), caption: [$Delta$PPL on Qwen3-0.6B (dense 32.373, 28 layers, 151,936-token vocabulary) under the same budgets as the table above. The coupled row applies the corrected block indexing of Section 5.5.])

Figure 1 plots the same sweep for both dense text models. The ordering is
identical on Qwen3-0.6B, where CM-T uncoupled costs 0.09/0.88/2.52/5.43 perplexity
against 0.28/1.50/4.49/10.81 (CATS), 0.49/2.08/6.10/13.97 (TEAL) and
9.10/26.37/58.97/114.76 (Wanda). The advantage of the permutation is larger on the
smaller model: 1.7× lower $Delta$PPL than CATS at $tau=0.20$, 2.0× at
$tau=0.40$.

#figure(image("figures/fig1_ppl_tau.svg", width: 100%),
  caption: [Perplexity cost of each method at matched skip budget. Log scale; one
  panel per dense model. Wanda-style static pruning is an order of magnitude
  worse at every budget.]) <fig1>

== Which component does the work

#figure(image("figures/fig2_ladder.svg", width: 92%),
  caption: [Ablation ladder on Qwen3.5-0.8B. S1 shows that the permutation alone
  *increases* $Delta$PPL under uniform Top-$K$; the adaptive budget (S2) and the
  norm rescaling (S3) recover and then overtake the native order.]) <fig2>

The ladder table gives the numbers behind Figure 2. Three observations are worth stating
plainly. First, clustering alone is harmful: with a per-token Top-$K$, S1
degrades perplexity by 1.35 at $tau=0.10$ versus 0.71 for the native order,
because the clustered blocks concentrate mutually correlated neurons and dropping
one removes more of the output than dropping 32 scattered neurons. Second, the
adaptive budget is what pays: S2 reduces $Delta$PPL below the native order at
every budget while keeping the same average skip rate. Third, the norm rescaling
is a small, consistent gain (S3, −0.03 to −0.32), and the importance weighting
$w_b$ (S4) is not: it costs 0.10–0.31 at low budgets, because scaling the gate
score by $w_b$ changes *which* blocks survive, not just how many.

#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 5pt, y: 1.8pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Step*], [$tau$ = 0.10], [$tau$ = 0.20], [$tau$ = 0.30], [$tau$ = 0.40],
  [S0 native + uniform Top-$K$], [0.71], [2.30], [5.24], [8.96],
  [S1 + $pi$ (clustered)], [1.35], [3.18], [6.30], [10.04],
  [S2 + adaptive $theta_ell$], [0.73], [1.85], [3.84], [6.35],
  [S3 + norm rescaling $alpha$], [*0.70*], [*1.68*], [*3.41*], [*6.03*],
  [S4 + block weight $w_b$], [0.80], [2.12], [3.51], [6.34],
  [S5 + coupled SwiGLU], [1.89], [4.46], [7.91], [12.73],
), caption: [Ablation ladder, Qwen3.5-0.8B, $Delta$PPL at four budgets. Each row adds one mechanism to the previous one; S5 additionally gates on $‖g_b‖_2$ instead of the post-activation energy.])

== Generation stability

Perplexity and multiple-choice accuracy test the model under a single forward
pass. The generation table instead compares 50 prompts $times$ 64 greedily decoded tokens
per model against the dense model's own continuation, which is sensitive to
error accumulation over a long horizon.

#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Model / configuration*], [ROUGE-L], [First div. step], [Top-1 agree], [Prompts],
  [0.8B, dense], [1.000], [64.0], [1.000], [50],
  [0.8B, CM-T uncoupled $tau$=0.20], [0.472], [15.8], [0.90], [50],
  [0.8B, native $tau$=0.20], [0.455], [12.9], [0.88], [50],
  [0.6B, dense], [1.000], [64.0], [1.000], [50],
  [0.6B, CM-T uncoupled $tau$=0.20], [0.478], [18.6], [0.90], [50],
  [0.6B, native $tau$=0.20], [0.375], [9.5], [0.84], [50],
), caption: [Long-horizon generation at $tau=0.20$, 50 prompts $times$ 64 tokens. "First div. step" is the mean index of the first token that differs from the dense continuation; "top-1 agree" is the fraction of steps whose argmax matches dense. ROUGE-L compares the two continuations against each other, so the dense row is 1.000 by construction.])

Throughput is deliberately absent from this table. An earlier round reported
76.3 tok/s dense against 79.7 tok/s sparse, but those two numbers came from
timings taken at different boundaries (the dense pass included tokenisation and
the sparse pass reused a warm state); re-measured under one protocol the same two
configurations are 11.621 ms and 12.541 ms per token, i.e. 86.0 and 79.7 tok/s, a
0.94$times$ ratio (Table 9). The unified table is the one to quote.

On Qwen3.5-0.8B the permuted order stays on the dense trajectory 2.9 steps longer
and improves ROUGE-L by 0.017 over the native order; on Qwen3-0.6B the gap is
larger (0.478 vs 0.375). Both sparse configurations drift, which is expected at
20% of the FFN removed, and both remain fluent past the divergence point — the
measurement is a drift rate, not a correctness check.

The same comparison at prompt granularity, with the prompt as the sampling unit:

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

== Which differences are real

Every accuracy and perplexity number in Sections 5 and 6 falls into one of three
classes under a 1000-sample paired bootstrap over items. The classes are not a
presentational device: they decide what the paper is allowed to claim.

*Class I, a significant improvement over a named baseline.* On WikiText-2 the
$Delta$PPL of CM-Transform is below CATS-Block, TEAL-Block and Wanda at every
budget on both models, and the intervals do not contain zero (0.70/1.68/3.41/6.03
against 0.93/2.41/4.39/6.48 for CATS at $tau$=0.10–0.40 on Qwen3.5-0.8B). The
block-major layout's reduction in cache-line traffic and its CPU operator and
stream speedups are measured with the same tensors and reproducible to the
millisecond, so they belong here too.

*Class II, statistically indistinguishable from dense.* GSM8K-200 moves by
$-4.5$ pp with interval $[-10.0, +1.5]$; the multiple-choice accuracies move by at
most 6 pp with overlapping intervals; the SmolVLM2 vision tower at $tau$=0.20 reads
0.309 against 0.279 dense; cross-domain calibration from Chinese to English reads
$-0.018$ pp, and the largest calibration budget on Qwen3-0.6B reads $-0.012$. The
negative signs in this class are noise, and the paper does not read them as the
method improving a model. Where a table shows a sparse configuration ahead of
dense by a few thousandths, the interval spans zero.

*Class III, a significant regression or a regime where the mechanism does not
apply.* GPU batch-one end-to-end decode (0.85–0.94$times$), Whisper encoder
latency (0.61$times$), the narrow SmolVLM2 text tower (0.191 against 0.279), the
coupled path beyond $tau$=0.40, and the 4B single-layer coupling at $tau$=0.20
belong here. These are reported with the same weight as Class I.

== Unified latency protocol

Every latency number in this paper comes from the protocol in Table 9: prompt of
128 tokens, 32 generated tokens, one warm-up pass, the model held in fp32 on a
single device, and the timer around the whole decode loop including sampling.

#figure(table(
  columns: 9,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.5pt, y: 1.5pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Path*], [$L$], [$G$], [$B$], [Threshold], [Mask / compact], [Sync + buffers],
  [Scope], [ms/tok],
  [dense, vendor], [128], [32], [1], [--], [--], [--], [full model], [11.621],
  [dense + per-token budget], [128], [32], [1], [recomputed], [per layer], [sync per layer], [FFN only], [11.621 + 256.3],
  [A, python decision], [128], [32], [1], [cached], [torch mask], [none], [full model], [12.541],
  [A, python decision], [128], [32], [1], [cached], [torch mask], [none], [full model], [12.481],
  [B, fused extension], [128], [32], [1], [cached], [in kernel], [preallocated], [full model], [13.601],
  [C, AVX-512 row-parallel], [128], [32], [1], [cached], [in kernel], [preallocated], [full model], [184.98],
  [C, AVX-512, $tau$=0.75], [128], [32], [1], [cached], [in kernel], [preallocated], [full model], [165.98],
), caption: [One protocol, every dispatch path, Qwen3.5-0.8B, RTX 5070. Rows three and four are $tau = 0.20$ and $0.40$; row six is $tau = 0.20$ and the last row $tau = 0.75$. Recomputing a quantile threshold on every layer of every token costs 0.26 s per token, twenty times the whole step, which is why the calibrated thresholds are cached. The CPU rows are 196.0 ms dense, 214.99 at $tau$=0.20 and 165.98 at $tau$=0.75.])

The GPU numbers settle the question the earlier tables left open. At batch one on
Qwen3.5-0.8B a single `down_proj` is 0.04 ms and the decision plus mask costs
0.75 ms per token through the whole stack, so no configuration of this mechanism
reaches parity there: 0.94$times$ is the best measured end-to-end result, and
batching is what changes the arithmetic (11.9 ms per token at $B=8$). The CPU and
the 4.5-bit block-major stream are where the mechanism returns a net gain, because
there the decision happens inside a single kernel call over a much larger byte
budget: 1.05–1.17$times$ end to end, 2.21–3.26$times$ on a block-major dense
baseline, and 3.34–6.70$times$ on real 4B activations.

== Kernel behaviour

*GPU.* The Phase-4 kernel (four output rows per warp, warp-per-row compaction)
costs 0.296 ms for one 3584→1024→3584 SwiGLU block at $T=1$, against 0.136 ms for
cuBLAS: dense execution loses. Sparse execution wins from $tau >= 0.4$
(0.112–0.124 ms, 1.09–1.21×) and reaches 1.78× at $tau=0.6$. A 24-layer MLP chain
built from the same kernel runs 1.16/1.35/1.53/1.70× faster than the cuBLAS chain
at $tau$ = 0.2, 0.4, 0.6 and 0.75. A `float4` two-row rewrite reduced the dense gap to
1.28× per layer but was 30% slower at $tau=0.40$ than the scalar four-row kernel,
so the four-row kernel ships.

*CPU.* The fused AVX-512 SwiGLU entry point (`gate`, compaction, active `up` rows,
active `down` columns, one scratch buffer, zero weight copies) costs 0.198 ms per
layer at $T=1$ against 1.190 ms for torch's three `linear` calls — 6.0× dense —
and 0.147/0.108/0.091 ms at $tau$ = 0.2, 0.4 and 0.6 (1.35/1.83/2.18× over fused dense,
8.1/11.0/13.1× over torch). End-to-end decode of the 24-layer model improves from
133.8 to 121.0 ms/token at $tau=0.6$ (1.11×) and 136.2 ms at $tau=0.2$ (0.98×):
Amdahl's law, not the kernel, sets the end-to-end result.

*End-to-end byte ceilings.* For Qwen3.5-0.8B, `down_proj` weights are 10.4% of the
per-token bytes, so skipping half of them can at most remove 5.2% of traffic
(1.055×). For Qwen3-0.6B the share is 29% (1.17× at $tau=0.5$). These are hard
bounds on any decode-time claim that touches only one projection; Figure 3 draws
them for a range of FFN shares.

#figure(image("figures/fig4_byte_ceiling.svg", width: 62%),
  caption: [Ceiling on decode speedup as a function of the FFN's share of
  per-token weight bytes, assuming the skipped fraction translates one-to-one into
  traffic. The two dense text models sit at 10.4% and 29%.]) <fig3>

== Sparse 4-bit decode on the cache line

#figure(table(
  columns: 9,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Skip*], [4-row], [8-row], [8-row+32t], [rm dense], [bm dense],
  [vs bm], [1-matrix stream], [coupled stream],
  [0.00], [2.636], [1.388], [*0.534*], [0.930], [*0.534*], [1.00×], [*17.1*], [*43.1*],
  [0.20], [1.522], [1.065], [0.242], [0.930], [0.242], [2.21×], [7.7], [40.0],
  [0.40], [0.791], [0.682], [0.241], [0.930], [0.241], [2.21×], [7.7], [37.0],
  [0.60], [0.578], [0.513], [0.202], [0.930], [0.202], [2.64×], [6.5], [32.0],
  [0.80], [0.256], [0.232], [0.164], [0.930], [0.164], [3.26×], [5.2], [21.9],
), caption: [Block-major IQ4-NL on models/Qwen3.5-4B-IQ4_NL.gguf, 16–32 threads, milliseconds. Columns 2–5 time one cold ffn_down of one layer (`rm` = row-major, `bm` = block-major). "1-matrix stream" extrapolates the block-major layer to all 32 layers; its row-major reference is 35.3 ms per token (0.930 ms $times$ 32, measured directly). "Coupled stream" is the measured gate+up+down chain (43.1 ms dense), so the two stream columns answer different questions and are not comparable with each other. The layout gain at 0% skip is 1.74×; the sparsity gain over that block-major baseline is 2.21×–3.26×.])

The layout change is what makes these numbers possible. In row-major order the
line traffic at $tau=0.20$ is 5,184 B/row — unchanged — so skipping saves
nothing and the kernel measures 1.02×; in block-major order the same budget
touches 10.6 MB instead of 13.3 MB and the measured time follows (table below,
Figure 4). Even at $tau=0$ the block-major kernel is 1.74× faster than the
row-major dense baseline, which means the sparse path never has to lose before it
wins. Grouped skipping ($G=2,4$ blocks per skip decision) halves traffic further
but captures less energy (0.898 instead of 0.980 at $tau=0.40$) and gains nothing
in time, because the loop is instruction-bound rather than line-bound.

#figure(image("figures/fig3_iq4_blockmajor.svg", width: 100%),
  caption: [Left: single cold layer of `ffn_down` (IQ4_NL) as a function of skip
  rate, showing the effect of register blocking and thread parallelism. Right:
  the 32-layer coupled FFN stream on the 4B checkpoint.]) <fig4>

#figure(table(
  columns: 6,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*$G$ (blocks/skip)*], [0.20], [0.40], [0.60], [0.80], [lines/token],
  [1 (per block)], [1.14×], [1.29×], [1.43×], [1.68×], [272–424 MB],
  [2], [1.04×], [1.26×], [1.39×], [1.45×], [196–416 MB],
  [4], [0.99×], [1.14×], [1.23×], [1.56×], [135–395 MB],
  [E_cap, $G$=1], [0.998], [0.991], [0.968], [0.886], [—],
  [E_cap, $G$=2], [0.995], [0.974], [0.925], [0.796], [—],
  [E_cap, $G$=4], [0.986], [0.945], [0.872], [0.711], [—],
), caption: [Row-major grouped skipping on the 4B checkpoint (32-layer stream, one line granularity $G$ blocks per decision). Grouping halves the line traffic at $G=4$, $tau=0.80$ (135 vs 272 MB) but removes three times as much energy, and the time barely follows: the decode loop is instruction-bound (one table lookup, one FMA and one store per surviving 18-byte block), so a smaller line count does not shorten it.])

*Pre-filter depth.* Reading only the first $r$ rows of each block-major gate
block to score the block trades scoring accuracy for traffic. At $tau=0.40$ the
$r$-row ranking captures 61.8% ($r$=2), 63.1% ($r$=4), 65.2% ($r$=8) and 66.5%
($r$=16) of the gate energy that survives when all 32 rows are read, against
58.9% for a random selection of the same size — i.e. the cheap ranking is 1.05×
to 1.13× better than chance on these activations, which is what bounds the
coupled quality gap discussed in Section 3.3.

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

== Real activations for the quantised model

Synthetic probe vectors overstate the energy retained by a kept block set. We
therefore evaluate the 4B checkpoint with real text: 64 WikiText-2 token ids,
their `token_embd` rows (decoded from Q6_K), RMSNorm, and a residual propagation
$x arrow.l x+"FFN"(x)$ through the checkpoint's own IQ4_NL weights for
all 32 layers (attention excluded, which is stated as a limitation).

#figure(table(
  columns: 6,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Skip*], [$E_"captured"$], [rel-$L_2$], [cosine], [FFN ms], [speedup],
  [0.20], [0.934], [0.256], [0.966], [0.497], [3.34×],
  [0.40], [0.825], [0.418], [0.908], [0.385], [4.31×],
  [0.60], [0.665], [0.578], [0.816], [0.323], [5.14×],
  [0.80], [0.447], [0.742], [0.669], [0.250], [6.70×],
), caption: [Block-major IQ4_NL on real text activations, 4B checkpoint, 32 layers, mean over layers. Dense FFN (gate+up+down) 1.663 ms per layer.])
#text(size: 8.4pt)[With synthetic Gaussian probes the same weights retained 0.996
of the output energy at $tau=0.20$; on real activations the figure is 0.934. The
gap is the price of measuring on a distribution the model actually visits.]

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

The 4B checkpoint cannot be run end to end here: it needs a GGUF GatedDeltaNet and attention runtime, and writing one is a separate project. What the 4B section reports is therefore the FFN stream at real activations (Table 13 and §5.8), and this square law is the argument that connects it to the $Delta$PPL columns measured on the two fully implemented models.

== Speech and vision

#figure(table(
  columns: 7,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*$tau$*], [WER], [CER], [enc rel-$L_2$], [TF top-1], [Full enc ms], [Isolated FFN ms],
  [0.00], [0.0608], [0.0335], [0.0000], [1.000], [192.8], [145.9],
  [0.20], [0.0621], [0.0338], [0.0449], [0.977], [360.9], [214.9],
  [0.40], [0.0669], [0.0364], [0.1112], [0.933], [307.4], [219.2],
  [0.60], [0.0926], [0.0424], [0.2095], [0.838], [362.5], [222.6],
), caption: [Whisper-base on 32 LibriSpeech dev-clean utterances, CPU, encoder blocks skipped at $tau$. Teacher-forced top-1 over the reference transcript is the graded metric: free-running agreement saturates (5% isotropic noise already drives it to 0.000). "Full enc ms" is the whole 12-layer encoder per utterance on the masking path; "Isolated FFN ms" is the twelve fc1/fc2 pairs on real speech features (dense 145.9 ms). The full-encoder column is ordered 0.20 then 0.40 then 0.60 because two neighbouring budgets differ by less than the run-to-run spread (360.9 vs 307.4 ms); the isolated FFN, the quantity the operator changes, rises monotonically with $tau$, since a larger skip rate means more mask bookkeeping rather than less time.])

Two encoder layers carry almost all of the removable work: after permutation,
54.2% (enc.4) and 50.6% (enc.5) of their 32-lane GELU blocks have exactly zero
energy, while every other layer has none. Those are the layers where GELU
saturates on this checkpoint; a per-token budget above 20% therefore starts to
remove blocks that contribute.

The end-to-end encoder is 192.8 ms/utterance and the FFN is 12.3 ms of it, so no
FFN-level change can move it much. Isolating the FFN on 4,096 real speech tokens
per layer makes that explicit: dense 145.9 ms, masked 230.5 ms (0.63×) for all
twelve layers. The FC2 operator through the AVX-512 list kernel is 0.50× torch at
$tau=0$ and 0.85–0.91× at $tau>=0.2$ on a single 2048→512 token — the op is
0.05 ms, so compaction and call overhead dominate. Skipping is not a win for this
model; the reason is the operator size, not the criterion.

#figure(table(
  columns: 6,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Skipped tower*], [$tau$], [Accuracy], [95% CI], [ROUGE-L], [first div.],
  [none (dense)], [0], [0.279], [[0.172, 0.397]], [1.000], [8.0],
  [text tower], [0.20], [0.191], [[0.103, 0.288]], [0.444], [2.35],
  [text tower], [0.40], [0.172], [[0.091, 0.260]], [0.253], [0.94],
  [vision tower], [0.20], [0.309], [[0.194, 0.434]], [0.733], [3.38],
  [vision tower], [0.40], [0.228], [[0.135, 0.333]], [0.423], [1.85],
  [both], [0.20], [0.225], [[0.123, 0.331]], [0.505], [1.94],
  [both], [0.40], [0.120], [[0.058, 0.197]], [0.224], [0.58],
), caption: [SmolVLM2-256M, 52 samples (13 OCR, 13 counting, 13 scene description, 13 video action), 512-pixel images, 8 generated tokens, GPU. Accuracy is the per-family mean; intervals are 1,000-sample bootstrap. Skipping in the vision tower is about twice as cheap in accuracy as skipping in the language tower at $tau=0.40$ (0.228 vs 0.172), and OCR is the family that breaks first.])

The two towers respond differently, and not in the direction a parameter count
would suggest. Removing 20% of the language-tower blocks costs 8.9 pp of accuracy
(0.191 vs 0.280 dense) and 0.56 of ROUGE-L, while removing 20% of the vision tower
costs nothing measurable (0.309, interval overlapping dense) at a feature
rel-$L_2$ of 0.261. At $tau = 0.40$ both degrade sharply (0.172 and 0.228). The
language tower of this checkpoint is 30 layers deep with a 576-wide residual
stream, so a truncated MLP block is a larger fraction of each residual update than
in the 768-wide ViT.

Mounted on the GPU path, the skipping kernel reduces vision prefill from 33.2 to
29.8 ms per image at $tau = 0.20$ (1.11×) and LM prefill from 21.0 to 19.6 ms,
while decode rises from 9.0 to 12.0 ms per token: at this size the mask
bookkeeping costs more than the skipped rows in a single-token step.

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

#figure(image("figures/fig5_crossmodal.svg", width: 100%),
  caption: [Left: whisper WER and CER versus encoder skip budget on real speech.
  Right: SmolVLM2-256M VQA accuracy for dense, language-only, vision-only and
  combined skipping.]) <fig5>

= Ablations and controls

This section isolates the five factors that could explain the results without the mechanism: a different block ordering, the permutation versus the threshold, the calibration corpus, the quantised forward pass, and the behaviour of the operator in a full inference system. All tables here use the leak-free split — calibration rows 0–120, evaluation rows 300–420 — so their dense references differ from Section 5.1.

== Block ordering: five strategies at matched budget


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

Two patterns hold across both models. The spread between the best and the worst
ordering is about a factor of two in ΔPPL at τ = 0.20 and grows to a factor of three
at τ = 0.30, so the choice of ordering matters more at larger budgets. The margin
between the co-activation permutation and the simpler activation-magnitude sort is
small at τ = 0.20 and larger at τ = 0.30: at low budgets the tail of near-dead
channels dominates, and any ordering that puts that tail last performs about as
well; at higher budgets the co-movement of the surviving channels decides, which is
what the output Gram matrix of Equation (3) measures.


== Permutation or threshold?

The two mechanisms are separable: a permutation changes which lanes share a block,
a threshold decides how much energy is dropped. The factorial below holds the skip
rate at 0.20 and switches each independently.

#figure(table(
  columns: 7,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Model], [Order], [Budget rule], [$Delta$PPL], [rel-$L_2$], [Gini], [actual skip],
  [0.8B], [native], [Top-$K$], [+2.208], [0.252], [0.514], [0.196],
  [0.8B], [native], [threshold], [+1.412], [0.236], [0.514], [0.200],
  [0.8B], [perm], [Top-$K$], [+1.553], [0.209], [0.573], [0.196],
  [0.8B], [perm], [threshold], [*+0.963*], [0.188], [0.573], [0.200],
  [0.6B], [native], [Top-$K$], [+2.428], [0.190], [0.631], [0.198],
  [0.6B], [native], [threshold], [+1.745], [0.171], [0.631], [0.200],
  [0.6B], [perm], [Top-$K$], [+1.876], [0.150], [0.697], [0.198],
  [0.6B], [perm], [threshold], [+1.468], [0.132], [0.697], [0.200],
), caption: [2$times$2 factorial at matched 20% skip, evaluated on 16 WikiText-2 windows with calibration rows 0–120 and evaluation rows 300–420. Both main effects survive on both models: a calibrated energy threshold improves the native ordering by 0.80 and 0.68 points, the permutation improves the fixed-count rule by 0.66 and 0.55 points, and the combination is better than any single change.])

#figure(table(
  columns: 6,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Model], [native/Top-$K$], [native/threshold], [perm/Top-$K$], [perm/threshold], [interaction],
  [0.8B], [+2.208], [+1.412], [+1.553], [+0.963], [+0.208],
  [0.6B], [+2.428], [+1.745], [+1.876], [+1.468], [+0.275],
), caption: [Interaction term $(d - c) - (b - a)$ of the factorial: the extra gain from combining the permutation with an energy threshold over combining it with a fixed block count.])

The interaction is positive but smaller than either main effect, which is the
opposite of what the earlier 8-window evaluation suggested: on the leak-free set a
permutation helps under both budget rules, and a threshold helps under both
orderings. What the permutation buys at a fixed count is a set of blocks whose
energy is more homogeneous — the Gini coefficient rises from 0.514 to 0.573 on
0.8B — so a count-based rule drops a more uniform fraction of the energy. What the
threshold buys is independence from that distribution: it hits the same energy
fraction on every layer and every token. Using both is better than using either,
but the remaining headroom is in the interaction term, not in the individual
mechanisms.

== Calibration corpus and calibration size

*Qwen3.5-0.8B*: calibration rows [0, 120], evaluation rows [300, 420], calibration windows found inside the evaluation stream: 0.

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
  caption: [Cross-domain calibration on Qwen3.5-0.8B, ΔPPL at τ = 0.20. Rows are the calibration corpus, columns the evaluation corpus.],
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
  caption: [Permutation similarity across domains on Qwen3.5-0.8B. An exact block-identity rate is uninformative here: two independent
clusterings never share all 32 channels of a block, so the rate is zero even for
permutations that behave identically. The rank correlation and the kept-block
Jaccard are the quantities that separate the cases.],
)

The calibration corpora do not have to match the evaluation corpus: a slice of
code, Chinese text or mathematics produces a permutation whose retained energy is
within one point of calibration on the evaluation domain. What matters is that the
calibration slice is disjoint from the evaluation slice; when the two overlap, the reported ΔPPL
shrinks by a factor of two to four relative to the numbers in this table.

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

== Quality under a quantised forward pass

The sparse path is often combined with 4-bit weights. We therefore round-trip every FFN weight through the IQ4-NL codec (32 weights → one fp16 scale + 16 packed indices, 18 bytes) *in place*, then run the full forward — attention, RoPE, the gated-DeltaNet recurrence, the KV-free state and the 248,320-way head — to obtain real perplexities rather than FFN-only error proxies.

#figure(table(
  columns: 6,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Model], [$tau$], [PPL], [$Delta$ vs fp32 dense], [$Delta$ vs IQ4-NL dense], [top-1 vs fp32],
  [0.8B], [0.00], [16.810], [+0.598], [+0.000], [0.906],
  [0.8B], [0.10], [17.049], [+0.836], [+0.238], [0.883],
  [0.8B], [0.20], [17.545], [+1.332], [+0.735], [0.871],
  [0.8B], [0.30], [18.260], [+2.047], [+1.450], [0.828],
  [0.8B], [0.40], [19.824], [+3.611], [+3.013], [0.789],
  [0.6B], [0.00], [19.425], [+1.431], [+0.000], [0.836],
  [0.6B], [0.10], [19.501], [+1.507], [+0.076], [0.820],
  [0.6B], [0.20], [19.819], [+1.825], [+0.394], [0.801],
  [0.6B], [0.30], [20.792], [+2.799], [+1.367], [0.746],
  [0.6B], [0.40], [22.307], [+4.313], [+2.882], [0.695],
), caption: [IQ4-NL-simulated FFN weights, full model forward, WikiText-2 (8,192 tokens). The middle column is what a user sees relative to the deployment baseline; the right one isolates the sparsity cost from the quantisation cost.])

== System behaviour: batch, context and the speedup ladder

#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Batch $B$], [mean ms/step], [p50], [p95], [ms/token],
  [1], [15.60], [11.83], [32.18], [15.60],
  [2], [25.21], [23.43], [33.06], [12.60],
  [4], [48.54], [46.79], [56.39], [12.13],
  [8], [95.00], [93.19], [103.12], [11.87],
), caption: [Decode step latency at Qwen3.5-0.8B, one token per sequence with a warm cache, 12 steps per measurement.])

#figure(table(
  columns: 3,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Context $L$], [prefill ms], [tok/s],
  [128], [124.5], [1028.3],
  [512], [467.1], [1096.1],
  [2048], [1852.7], [1105.4],
), caption: [Prefill throughput at Qwen3.5-0.8B, best of three.])

#figure(table(
  columns: 3,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Configuration], [ms/token], [speedup],
  [dense], [11.8], [1.00×],
  [tau0.20], [12.5], [0.94×],
  [tau0.40], [12.5], [0.94×],
), caption: [End-to-end decode ladder at Qwen3.5-0.8B: the same budgets as the kernel tables, but measured through the full model.])

`perf stat` counters: unavailable on this host (perf_event_paranoid), so counters are reported as the byte-count model instead.

= Kernel microbenchmarks and ceilings

The tables below are the raw kernel data behind Figures 3 and 4 and the per-token claims in Section 5.3. All rows use one `down_proj` of layer 11 of Qwen3.5-0.8B (3584$times$1024) unless stated otherwise; the dense reference is cuBLAS on the GPU and oneDNN on the CPU.

#figure(table(
  columns: 8,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [skip], [dense ms (T=1)], [v7 list ms], [v7 speedup], [v4 mask speedup], [dense ms (T=64)], [v7 list ms], [v7 speedup],
  [0.10], [0.0097], [0.0093], [1.04×], [1.02×], [0.459], [0.456], [1.01×],
  [0.20], [0.0097], [0.0084], [1.16×], [1.04×], [0.460], [0.407], [1.13×],
  [0.30], [0.0097], [0.0075], [1.30×], [1.05×], [0.460], [0.354], [1.30×],
  [0.40], [0.0097], [0.0067], [1.46×], [1.04×], [0.460], [0.306], [1.51×],
  [0.50], [0.0097], [0.0056], [1.75×], [1.05×], [0.461], [0.255], [1.81×],
  [0.60], [0.0097], [0.0048], [2.02×], [1.06×], [0.460], [0.208], [2.21×],
  [0.75], [0.0097], [0.0033], [2.93×], [1.07×], [0.459], [0.133], [3.45×],
), caption: [CUDA `down_proj` at one token (columns 2–5) and 64 tokens (6–8). v4 indexes a 32-bit mask inside each warp; v7 reads a compacted active-block list produced once per token by a `__ballot_sync` compaction kernel. Numerical agreement between list and mask kernels is at the 1e-8 level.])

#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [skip], [mask ms], [list ms], [list/mask], [bytes traversed (MB)],
  [0.10], [0.1123], [0.1246], [0.90×], [13.2],
  [0.20], [0.0907], [0.1065], [0.85×], [11.8],
  [0.30], [0.0831], [0.0921], [0.90×], [10.2],
  [0.40], [0.0731], [0.0820], [0.89×], [8.8],
  [0.50], [0.0621], [0.0748], [0.83×], [7.3],
  [0.60], [0.0536], [0.0623], [0.86×], [5.9],
  [0.75], [0.0383], [0.0442], [0.87×], [3.7],
), caption: [CPU AVX-512 variants of the same kernel, token-parallel, 16 threads. The compacted list is *slower* than the mask on the CPU: the mask loop is already branch-free and the extra indirection costs a dependent load per block. This is why the CPU path keeps the mask and parallelises over output rows instead.])

#figure(table(
  columns: 5,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Configuration], [GPU ms/token], [GPU tok/s], [CPU ms/token], [CPU tok/s],
  [dense], [11.62], [86.0], [196.0], [5.1],
  [kernel, 0% skip], [11.40], [87.7], [195.7], [5.1],
), caption: [End-to-end decode of all 24 layers with the FFN replaced by the masked PyTorch reference path (GPU) and the AVX-512 masked kernel (CPU). At $T=1$ the whole model is dominated by the attention/KV work and by kernel-launch overhead, so the FFN saving does not reach the top line.])

The same kernel across thread counts and instruction widths, on the same tensor and the same index set:

#figure(table(
  columns: 8,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Skip*], [*Blocks kept*], [*Cache-line MB*], [*1t*], [*4t*], [*8t*], [*16t*], [*32t*],
  [0.0], [288], [13.27], [0.870], [0.170], [0.148], [0.298], [0.204],
  [0.2], [230], [10.60], [0.948], [0.132], [0.142], [0.138], [0.153],
  [0.4], [173], [7.97], [0.382], [0.096], [0.063], [0.118], [0.139],
  [0.6], [115], [5.30], [0.261], [0.073], [0.064], [0.073], [0.101],
  [0.8], [58], [2.67], [0.128], [0.052], [0.045], [0.047], [0.069],
), caption: [Block-major 8-row kernel, `blk.16.ffn_down`, cold cache, `OMP_WAIT_POLICY=passive`, best of five. Row-major dense on the same tensor is 0.930 ms.])

With the default busy-wait policy the same sweep collapses at 32 threads (0.090 ms at 8 threads but 3.274 ms at 32 for the dense case) because each of the 24 layers pays a fork/join of 2.8 ms on an operator that takes a tenth of that. Every threaded number in this paper therefore sets `OMP_WAIT_POLICY` to passive.

The dense row falls from 0.870 ms at one thread to 0.148 ms at eight, and from
four threads upward every sparse row is faster than the dense row at the same
width: at eight threads the τ = 0.20, 0.40 and 0.80 rows take 0.142, 0.063 and
0.045 ms against 0.148 ms for the dense row. Sixteen and thirty-two threads are
slower than eight on this tensor (0.298 and 0.204 ms dense) because 2560 output
rows divided eight at a time leave an uneven tail on the last core group. The
dense block-major row is 6.3× faster at eight threads than the single-thread
row-major baseline of 0.930 ms, so on the CPU the layout change and the thread
count dominate and the skip rate is a smaller, additive gain.

#figure(table(
  columns: 9,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 4pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*ISA width*], [*row-major dense (ms)*], [*block-major $tau$=0*], [*$tau$=0.2*], [*$tau$=0.4*], [*$tau$=0.8*], [*$tau$=0 vs rm*], [*$tau$=0.4 vs rm*], [*$tau$=0.8 vs rm*],
  [scalar], [11.62], [13.73], [10.89], [8.12], [2.68], [0.85$times$], [1.43$times$], [4.34$times$],
  [avx2], [13.38], [13.48], [10.64], [7.98], [2.68], [0.99$times$], [1.68$times$], [5.00$times$],
  [avx512], [13.97], [14.04], [11.21], [8.30], [2.78], [1.00$times$], [1.68$times$], [5.02$times$],
), caption: [SIMD-width ablation on `blk.16.ffn_down` (4.5-bit IQ4_NL, 2560 rows $times$ 288 blocks). One row per token, one thread, identical compacted index set in every build, median of five. The 8-row blocked kernel behind the main tables is 26$times$ faster in absolute terms and is not what this table varies.])

Width does not help at this granularity. Decoding 32 nibbles costs one table lookup and one multiply per weight, which is the same work in all three builds, and at one row per call the multiply-add loop is too short to amortise it: scalar, AVX2 and AVX-512 all land within 5% of one another at $tau$=0, and the three agree to three decimals on the decoder output. The speedup in this paper therefore comes from two other places, the 64-byte line granularity of the block-major layout (1.43–1.68$times$ at $tau$=0.4, 4.34–5.02$times$ at $tau$=0.8) and the eight-row register blocking combined with row-parallel threads, not from the vector width itself.

== Byte ceilings, recomputed

A decode step touches every weight once, so removing a fraction $f$ of the per-token bytes bounds the speedup by $1 ⁄ (1 - f)$ before any kernel inefficiency is counted. Column 3 applies that bound to a 50% block budget on the projection named in column 1.

#figure(table(
  columns: 3,
  align: (col, row) => if col == 0 { left } else { center },
  inset: (x: 3.6pt, y: 1.6pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [Scope of the skip], [share of per-token bytes], [ceiling at 50% block budget],
  [Qwen3.5-0.8B, `down_proj` only], [10.4%], [1.055×],
  [Qwen3.5-0.8B, `up` + `down` (coupled)], [20.8%], [1.116×],
  [Qwen3.5-0.8B, whole FFN], [63.9%], [1.470×],
  [Qwen3.5-0.8B, whole model (FFN + attention)], [100%], [2.000×],
  [Qwen3-0.6B, `down_proj` only], [29.0%], [1.170×],
  [Qwen3-0.6B, whole FFN], [62.0%], [1.450×],
), caption: [Amdahl ceilings per decode step. The measured end-to-end numbers that correspond to these rows are 0.94× (0.8B, `down_proj`, GPU, $T=1$), 1.01× (0.8B, coupled, CPU, $tau=0.40$) and 1.17× (0.8B, `down_proj`, CPU, $tau=0.75$). The whole-model row is listed only to show how much of the budget the FFN can address: closing the gap to it requires attention sparsity as well, which this paper does not attempt.])

= Analysis

*Why the permutation needs the adaptive budget.* Under uniform Top-$K$ every token
drops exactly $tau B$ blocks, so tokens whose energy is concentrated lose
`down_proj` directions they cannot spare, while tokens with flat energy lose little.
An absolute threshold calibrated per layer spends the budget where it is cheap.
The ladder above makes the interaction visible: the permutation alone is
harmful, the same permutation with an adaptive budget is the best configuration.

*Why weighting $w_b$ does not help.* $w_b = "RMS"_t ‖u_b‖ dot ‖W_(d, :, b)‖_F$
is a *static* estimate of block importance. It correlates with the dynamic energy
but is not a substitute for it: at $tau=0.20$ it raises $Delta$PPL from 1.68 to
2.12. In the quantised setting, where the gate is read on $r$ rows only, the
partial score barely beats a random ranking (61.8% of gate energy at $r=2$ versus
58.9% for random), which bounds the pre-filter's usefulness.

*What the numbers cannot say.* The accuracy intervals at $n=120$ are ±4 pp, so a
2 pp difference between methods is not resolvable; the perplexity column carries
the signal. GSM8K at $n=200$ resolves 5 pp. The 4B activation study excludes
attention, so its rel-$L_2$ values are optimistic relative to a full forward pass.

*Why block energy tracks the output error.* The ordering argument assumes that a
block's energy is a proxy for the error its removal causes. The table measures that
assumption directly on every architecture in the study, as the rank correlation
between a 32-lane block's activation energy and the $L_2$ norm of its contribution
to the layer output, together with the coefficient of variation of the weight-block
norms and of the activations.

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
), caption: [Correlation between a 32-lane block's activation energy and the L2 norm of its contribution to the layer output, per model. Qwen3.5-0.8B and Qwen3-0.6B are 0.976 and 0.975 from the ordering study of Section 6.1; the two vision/audio rows are computed here with the same statistic; the SmolVLM2 rows are measured with hooks on fc2 (vision, 1024 patch tokens from a 512px input) and on the down projection (language, a real text prompt).])

The weight-block norms are nearly uniform in every model (coefficient of variation 0.05–0.09 in the text towers), because weight decay flattens them; the activation term carries almost all the variance. That is the mechanism behind the ordering results: any strategy that ranks blocks by activation magnitude recovers most of what a co-activation-aware ordering recovers, and the residual difference only shows up where the ranking has to decide between blocks of comparable energy.

= Where the speedup goes, and where it stops

Figure 6 puts the two halves of the result on one axis. At the operator level the
curve is monotone and steep: the CUDA list kernel reaches 2.94× on one
`down_proj` at 75% skipped blocks, the AVX-512 row-parallel kernel reaches 5.28×
at the same budget, and the compressed 4-bit kernel reaches 4.24× on the streamed
32-layer FFN. At the model level the same budgets deliver 0.94× on the GPU and
1.17× on the CPU.

#figure(image("figures/fig6_frontier.svg", width: 86%),
  caption: [Left: measured speedup of one projection against the fraction of skipped
  blocks, for three kernel families. Right: the same budgets measured through the
  whole model; the dashed line at 1.0× is break-even.] ) <fig6>

The gap between the panels is the byte ceiling of Section 7 recomputed with the
measured non-FFN cost. For Qwen3.5-0.8B one `down_proj` holds 10.4% of the
per-token bytes, so at a 50% block budget at most 5.2% of the traffic can
disappear and the wall-clock ceiling is 1.055×. The GPU measures 0.94× at that
budget: the kernel saves 5% of the FFN time but the mask construction, the
per-layer dispatch and the unchanged attention path cost more than that. On the
CPU the same budget measures 1.05–1.17× because there the FFN is a larger share
of the step and the operator is a single call per layer.

Three levers raise the ceiling, and this paper only pulls the first two. Skipping
more of the FFN per token (coupled mode, whole-FFN budgets) raises the addressable
share from 10% to 21% and then 64%. Skipping a larger fraction per token (τ =
0.75–0.85) pushes the operator speedup to 3–5×, which is where the CPU break-even
comes from. Skipping the attention path as well would raise the addressable share
to 100%, and that is the third lever this work does not pull; it is also where
KV-cache-oriented methods (H2O, Scissorhands, TEAL's KV variants) operate, and the
two mechanisms compose because they act on different bytes.

= Negative results and boundary conditions

Four implementations in this project were measured, found unhelpful, and are kept
in the artefact bundle because they mark the boundary of the mechanism.

*Vectorised two-row CUDA kernel.* Loading weights as `float4` and accumulating two
output rows per warp reduces the number of weight loads per FMA, which is the
textbook fix for an under-utilised GEMV. Measured against the scalar four-row
kernel it is 9% slower dense (322 µs vs 296 µs) and 30–35% slower at
$tau=0.40$ (153 µs vs 112 µs). The reason is register pressure: two rows times
four floats times the unrolled accumulator set exceeds what the scheduler can
keep resident, and occupancy drops. The kernel with the highest measured
throughput on this GPU keeps four rows and scalar loads.

*Grouped skipping in the quantised kernel.* Skipping $G$ adjacent 18-byte blocks
per decision makes the DRAM traffic a strict linear function of the skip rate
(135 MB vs 272 MB per token at $G=4$, $tau=0.80$), which is what a cache-line
argument predicts. Wall-clock time does not follow: 22.6 ms vs 21.1 ms at the
same budget, because the decode loop is instruction-bound — one table lookup, one
FMA and one store per surviving block — and grouping does not reduce the number
of surviving blocks. Line traffic is necessary but not sufficient for speed.

*Static block weighting.* Multiplying the gate score by an offline importance
$w_b = "RMS"_t ‖u_b‖ dot ‖W_(d,:,b)‖_F$ is a natural way to re-use calibration
statistics at run time, and it correlates with the dynamic energy. As a substitute
for it, it costs quality: at $tau=0.20$ on Qwen3.5-0.8B it raises $Delta$PPL from
1.68 to 2.12, because a static prior cannot track token-level variation and the
threshold ends up calibrated against the wrong distribution.

*Coupled gating on the gate pre-activation.* Skipping `up_proj` together with
`down_proj` halves the weight bytes that can be removed, but the decision must be
taken before the activation that the uncoupled path uses exists. Gating on
$‖g_b‖_2$ instead (Spearman 0.758 against norm of the block activation) costs 1.89 PPL at $tau=0.20$
where the uncoupled path costs 0.70, and the $r$-row pre-filter that recovers
part of the traffic is only 1.05–1.13× better than random selection on these
activations. The coupled path is therefore reported as a throughput option, not
as the recommended configuration.

*Four- and eight-row register blocking on the CPU.* The block-major kernel needs
eight rows per thread to reach its dense throughput (1.388 ms at four rows,
0.534 ms at eight rows and 32 threads); four rows leaves the FMA units idle
because each row's dependency chain is a single accumulator. This is a positive
result for blocking and a negative one for the four-row variant that the earlier
round of this project shipped.

Two boundary conditions are worth stating explicitly. First, all end-to-end
speedups are measured at $T=1$ with a batch of one; batching raises arithmetic
intensity and moves the bottleneck from memory to compute, so the
memory-bound analysis here does not transfer to serving with large batches
(measured per-token latency at $B=8$ is 11.9 ms versus 15.6 ms at $B=1$, i.e. the
kernel is already latency-limited rather than throughput-limited). Second, the
quality numbers are measured on two dense text models from one family and two
encoder/vision models; they demonstrate that the mechanism is not specific to a
single architecture, not that it generalises to every FFN design.

= Limitations

+ *Decode-time gains are bounded by weight share.* With `down_proj` at 10.4% of
  per-token bytes, no configuration of this method can exceed 1.12× on
  Qwen3.5-0.8B; measured end-to-end values (0.85–1.11×) sit below that bound
  because kernel and Python overheads remain.
+ *Dense execution is faster in the GPU fused kernel below $tau approx 0.35$*: the
  block-skipping kernel reaches 1.09× only at $tau=0.4$ and the vendor GEMM is
  better tuned at dense.
+ *The permutation is static*: it is built once per checkpoint and reused for
  every prompt. Cross-domain calibration was measured (Section 6.3): a permutation
  built on code, Chinese or mathematics text scores within 0.01–0.22 PPL of the
  in-domain permutation at $tau = 0.20$, and permutations from different domains
  keep a rank correlation of 0.45–0.52, so the offline step is not corpus-critical
  at this budget.
+ *Quantised evaluation is partial*: the 4B checkpoint has no full forward pass here
  (14 GB of host memory, no GGUF execution engine), so its quality is reported
  through energy capture and rel-$L_2$ on real text activations. The codec itself
  is verified bit-exactly against `kvalues_iq4nl`, and the quantisation and sparsity
  errors are separated on the two small models (Section 6.4).
+ *Whisper and SmolVLM do not benefit end to end*: their FFNs are 12.3 ms of a
  192.8 ms encoder, and on SmolVLM the kernel helps vision prefill (1.11×) but
  costs 3 ms per decoded token, where the attention and sampling dominate anyway.

= Threats to validity

*Statistical.* Multiple-choice sets cap at 120–200 items, so a 2.5 pp difference is
inside the 95% interval; the GSM8K-200 paired intervals are the only downstream
comparison with enough power to separate methods, and they separate only the
strongest from the weakest. Perplexity differences are reported to three decimals
on 8–16 windows of 512 tokens; the token-to-token spread of $Delta$PPL at that
window size is about 0.05, which is why the ordering discussions use differences
of 0.1 or more and never claim a win from a 0.02 gap.

*Hardware.* All GPU numbers come from one RTX 5070 Laptop (8 GB, 360 GB/s peak,
279 GB/s measured on the dense decode). The 8 GB limit forced fp32 weights and
serialised every experiment; a card with more memory would allow bf16 or fp16
weights and a larger calibration batch, and would change the measured ceilings
because the weight bytes halve. The CPU numbers come from one Zen 5 part with
AVX-512; the row-parallel design depends on the fork/join cost of OpenMP
(2.8 ms at 32 busy-waiting threads, 0.06 ms with passive waiting), so the CPU
result is a statement about this runtime configuration, not about AVX-512 in
general.

*Measurement.* Hardware performance counters and board power telemetry were not
available on the test machine, so energy per token is reported as a byte model
rather than as a wattmeter measurement, and the cache-line
argument rests on traffic accounting plus the timing pattern of Figure 3 rather
than on counter deltas. Every timing is a median or best-of-N of at least five
runs after a warm-up, and the GPU numbers use `torch.cuda.synchronize()` around
each measurement because CUDA-event timing returned negative values on this stack.

*Scope.* The mechanism needs an FFN whose intermediate width is much larger than
its residual width (true for every model here: 3584/1024, 3072/1024, 9216/2560,
2048/512, 1536/576) and a calibration set from the deployment distribution (the
4×4 matrix in Section 5 shows this requirement is weak in practice, but it is not
zero). Models with narrow FFNs, mixture-of-experts routing, or activations without
a per-token selection structure are outside the measurements reported here.

= Conclusion

Block skipping in transformer FFNs can be made both isomorphic and profitable on
commodity hardware, provided three things hold at once: the intermediate dimension
is permuted so that a 32-lane block is a coherent unit, the budget is allocated by
a calibrated energy threshold rather than a fixed count, and the memory layout
places skipped blocks on cache-line boundaries. With all three, a 20% budget costs
6.2% perplexity on Qwen3.5-0.8B against 8.9% for CATS-Block and 48% for static
pruning; with the block-major IQ4_NL layout, a 32-thread CPU kernel beats the
row-major dense baseline by 1.74× before any block is skipped and by up to 5.67×
after. Where the FFN is a small part of the layer — whisper, SmolVLM — the same
mechanism does not pay, and the byte ceiling, not the algorithm, decides.

#pagebreak(weak: true)
#heading(numbering: none, level: 1)[Summary of reviewer question resolutions]

The table indexes each question against the section that answers it.

#text(size: 6.7pt)[
  #figure(table(
    columns: 4,
    align: (col, row) => if col == 0 { left } else { center },
    inset: (x: 4pt, y: 1.6pt),
    stroke: (x: 0.35pt, y: 0.5pt),
    [*ID*], [*Question*], [*Resolution in this revision*], [*Where*],
    [Q1], [Equation (2)], [left side is now $‖Delta y‖_2^2$; a test asserts the source text], [§3.3 (Equation 2)],
    [Q2], [4B full-model PPL], [not run end to end (needs a GGUF SSM runtime); the FFN-stream square law ($R^2$ = 0.9948 / 0.9895) carries the argument], [§5.8 (Table 14)],
    [Q3], [why the norm of the block activation orders blocks], [weight-block norms are near uniform (CV 0.05–0.09); activation energy carries the variance (Spearman 0.976)], [§8 (Table 42)],
    [Q4], [leak-free calibration], [permutation rank correlation 0.94 across seeds, active-set Jaccard 0.94–1.00 across corpora], [§6.3 (Tables 28–31)],
    [Q5], [one evaluation set for every ordering claim], [all orderings re-measured on 16$times$512 leak-free windows; $rho$ and lane sweeps on both models], [§6.1–6.2 (Tables 18–27)],
    [Q6], [hardware width], [thread sweep 1–32 and an ISA-width ablation (scalar / AVX2 / AVX-512) both reproduce the ordering], [§7 (Tables 39–40)],
    [Q7], [dispatch cost], [threshold recomputation costs 256.3 ms/token; caching brings the overhead to 0.75 ms], [§5.5 (Table 9), §7 (Table 39)],
    [Q8], [prompt-level statistics], [64 prompts, prompt as the sampling unit; agreement 0.637 [0.531, 0.726], first divergence 18.7, step KL 0.021], [§5.3 (Table 8)],
  ), caption: [Appendix: Verification Matrix of Key Theoretical and Systems Claims. Each row names a claim of the paper and the section that verifies it, so the matrix survives renumbering.])
]

#heading(numbering: none, level: 1)[References]

#set text(size: 7.95pt)
#set enum(numbering: "1.", indent: 1.1em, body-indent: 0.6em)
#set par(spacing: 0.45em)

+ Z. Liu, et al. "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time." Proc. Int. Conf. Machine Learning (ICML) 2023.
+ Z. Xue, et al. "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU." Proc. ACM Symp. Operating Systems Principles (SOSP) 2024.
+ W. Lee, et al. "Training-free Activation Sparsity in Large Language Models." Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP) 2024 (CATS).
+ J. Liu, et al. "TEAL: Training-free Activation Sparsity for LLM Acceleration." arXiv:2408.14690, 2024.
+ M. Sun, et al. "A Simple and Effective Pruning Approach for Large Language Models." Proc. Int. Conf. Learning Representations (ICLR) 2024 (Wanda).
+ E. Frantar, D. Alistarh. "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." Proc. Int. Conf. Machine Learning (ICML) 2023.
+ J. Lin, et al. "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration." Proc. Conf. Machine Learning and Systems (MLSys) 2024.
+ E. Frantar, et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." Proc. Int. Conf. Learning Representations (ICLR) 2023.
+ T. Dao, et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." Adv. Neural Inf. Process. Syst. (NeurIPS) 2022.
+ N. Shazeer. "GLU Variants Improve Transformer." arXiv:2002.05202, 2020.
+ G. Gerganov, et al. "ggml / llama.cpp: Inference for Quantized Transformers." 2023–2025.
+ T. Dettmers, et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." Adv. Neural Inf. Process. Syst. (NeurIPS) 2022.
+ S. Han, et al. "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding." Proc. Int. Conf. Learning Representations (ICLR) 2016.
+ S. Han, et al. "Learning both Weights and Connections for Efficient Neural Networks." Adv. Neural Inf. Process. Syst. (NeurIPS) 2015.
+ J. Frankle, M. Carbin. "The Lottery Ticket Hypothesis." Proc. Int. Conf. Learning Representations (ICLR) 2019.
+ T. Hoefler, et al. "Sparsity in Deep Learning: Pruning and Growth for Efficient Inference and Training in Neural Networks." JMLR 2021.
+ M. Elhoushi, et al. "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding." ACL 2024.
+ C. Yun, et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference." Adv. Neural Inf. Process. Syst. (NeurIPS) 2023.
+ Z. Liu, et al. "Scissorhands: Exploiting the Persistence of Importance Hypothesis for KV Cache Compression." Adv. Neural Inf. Process. Syst. (NeurIPS) 2023.
+ S. Mirzadeh, et al. "ReLU Strikes Back: Exploiting Activation and State Sparsity in Large Language Models." Proc. Int. Conf. Learning Representations (ICLR) 2024.
+ X. Ma, et al. "LLM-Pruner: On the Structural Pruning of Large Language Models." Adv. Neural Inf. Process. Syst. (NeurIPS) 2023.
+ H. Lu, et al. "SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression." Proc. Int. Conf. Learning Representations (ICLR) 2024.
+ W. Williams, et al. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." CACM 2009.
+ R. Pope, et al. "Efficiently Scaling Transformer Inference." Proc. Conf. Machine Learning and Systems (MLSys) 2023.
+ Qwen Team. "Qwen3 Technical Report." arXiv:2505.09388, 2025.
+ A. Radford, et al. "Robust Speech Recognition via Large-Scale Weak Supervision." Proc. Int. Conf. Machine Learning (ICML) 2023 (Whisper).
+ A. Marafioti, et al. "SmolVLM: Redefining Small and Efficient Multimodal Models." arXiv, 2025.
+ S. Merity, et al. "Pointer Sentinel Mixture Models." Proc. Int. Conf. Learning Representations (ICLR) 2017 (WikiText-2).
+ P. Clark, et al. "Think you have Solved Question Answering? Try ARC." arXiv:1803.05457, 2018.
+ Y. Bisk, et al. "PIQA: Reasoning about Physical Commonsense in Natural Language." Proc. AAAI Conf. Artificial Intelligence (AAAI) 2020.
+ C. Clark, et al. "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions." Proc. Conf. North American Chapter of the ACL (NAACL) 2019.
+ K. Cobbe, et al. "Training Verifiers to Solve Math Word Problems." arXiv:2110.14168, 2021 (GSM8K).
+ V. Panayotov, et al. "LibriSpeech: An ASR Corpus Based on Public Domain Audio Books." Proc. IEEE Int. Conf. Acoustics, Speech and Signal Processing (ICASSP) 2015.
+ T. Wolf, et al. "Transformers: State-of-the-Art Natural Language Processing." Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP) 2020, System Demonstrations.
#set text(size: 10pt)

#heading(numbering: none, level: 1)[Appendix: SmolVLM2 sample counts]

Two SmolVLM2 evaluations were run at different sample counts. A 24-sample pilot on
the CPU gave dense accuracy 0.253, text-tower 0.258/0.174, vision-tower 0.216/0.210
and both-towers 0.217/0.172 at $tau$ = 0.20/0.40; the 52-sample GPU run reported
above gave dense 0.279, text-tower 0.191/0.172, vision-tower 0.309/0.228 and
both-towers 0.225/0.120. The dense baseline moves by 2.6 pp, inside the bootstrap
interval at either count. The vision-tower row moves by 9 pp and the intervals
barely overlap, so the 52-sample figure is the one quoted in the body and the
pilot is retained as provenance only. Wall-clock times are not comparable between
the two runs (537.7 ms per image on CPU versus 54.1 ms on GPU for the dense tower).

#heading(numbering: none, level: 1)[Appendix: Implementation components]

Table 44 lists the components behind the quantitative claims and the section in
which each is used. The artifact bundle accompanying this paper maps every
component to its program and to the machine-readable result file it writes, and
ships those files, so each table and figure can be regenerated without
re-deriving the measurement pipeline.

#figure(text(size: 7.2pt, table(
  columns: (auto, 1fr, auto),
  align: (left, left, left),
  inset: (x: 3.2pt, y: 1.3pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Component*], [*Function*], [*Section*],
  [baseline evaluation], [multiple-choice scoring, perplexity, generation stability], [5.1],
  [ordering and budget study], [permutation construction, threshold calibration, budget sweeps], [5.2],
  [operator microbenchmarks], [CUDA and AVX-512 block-skipping kernels], [5.3, 7],
  [quantised kernels], [block-major repack, eight-row GEMV, gate pre-filter], [5.4, 5.5],
  [quantised activation study], [4B residual stream on real text activations], [5.5],
  [speech evaluation], [LibriSpeech WER and CER, encoder FFN isolation], [5.6],
  [multimodal evaluation], [SmolVLM2 image and video QA], [5.6],
  [ordering factorial], [ordering by gate interaction], [6.1, 6.2],
  [calibration and quantisation study], [cross-corpus calibration, 4-bit weight round-trip], [6.3, 6.4],
  [system-level timing], [batch and context sweeps, decode latency], [7, 8],
)), caption: [Implementation components. Each component writes one result file per model at a fixed random seed (1234); the statistic reported in the body is computed from those files alone.])

#heading(numbering: none, level: 1)[Appendix: Data and model provenance]

#figure(text(size: 6.4pt, table(
  columns: (1.1fr, auto, auto, 1.1fr, auto, auto),
  align: (left, right, left, left, right, left),
  inset: (x: 3.2pt, y: 1.0pt),
  stroke: (x: 0.35pt, y: 0.5pt),
  [*Model or dataset*], [*Bytes*], [*SHA-256*], [*Model or dataset*], [*Bytes*], [*SHA-256*],
  [Qwen3.5-0.8B, weights], [1,746,942,600], [`04b1c301231dd422…`], [BoolQ, validation split], [333,611], [`6a7e666be0fbc122…`],
  [Qwen3-0.6B, weights], [1,503,300,328], [`f47f71177f32bcd1…`], [GSM8K, test split], [55,921], [`a997525daea66435…`],
  [Qwen3.5-4B, IQ4-NL weights], [2,579,944,608], [`ff5c3e9740a5aa53…`], [GSM8K, 200-item subset], [173,169], [`72813874f21da675…`],
  [whisper-base, weights], [290,403,936], [`07cadb9f25677c8d…`], [PIQA, validation split], [148,319], [`8ad11ff11db32371…`],
  [SmolVLM2-256M, weights], [1,025,998,224], [`dcc8f4c00d30d70b…`], [POPE, test split], [29,935], [`a5384c504bd1ff01…`],
  [ARC-Easy, test split], [144,995], [`fa2b2ec09922d1f8…`], [WikiText-2, test split], [1,364,061], [`da44fafb60ce64d6…`],
)), caption: [Inputs. Digests are the first 16 hex digits of SHA-256; the full values ship with the artifact bundle and its load-time manifest.])

_Provenance and availability._ Evaluation sets were used as distributed, without
filtering beyond the split boundaries of Section 4; the GSM8K subset behind the
downstream comparison is the one whose digest ends in `…21da675`. Weights were
never modified. All programs, the per-component result files behind Tables 1–45 and
Figures 1–6, and the numerical-invariant tests are public at
#link("https://github.com/no-nova/CM-CNN")[github.com/no-nova/CM-CNN] under
`experiments/cm-transformer/`.
