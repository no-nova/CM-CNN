== Parallelism and instruction-set width (Q6, Q7)

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

Block-major keeps its ordering across 1, 4, 8, 16 and 32 threads: the sparse rows stay below the dense row at every width, and the dense row itself is within 1.7$times$ of the row-major baseline once at least eight threads are resident.

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
