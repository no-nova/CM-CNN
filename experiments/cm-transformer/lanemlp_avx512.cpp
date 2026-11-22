// 32-lane structured block-skipping down_proj, AVX-512 (Zen 5) single thread.
// out[t, r] = sum_{b : mask[t,b]} sum_{l<32} A[t, 32b+l] * W[r, 32b+l]
// A 32-lane block is exactly 2 ZMM registers, so a skipped block is exactly
// two registers of work that are never touched.
#include <immintrin.h>
#include <vector>
#include <omp.h>
#include <cstdint>
#include <cstring>

extern "C" {

// 4 output rows at a time for ILP; I must be a multiple of 32.
void down_skip_avx512(const float* __restrict__ A, const float* __restrict__ W,
                      const unsigned char* __restrict__ M, float* __restrict__ out,
                      int T, int I, int O) {
    const int nb = I / 32;
    for (int t = 0; t < T; ++t) {
        const float* a = A + (long)t * I;
        const unsigned char* m = M + (long)t * nb;
        float* o = out + (long)t * O;
        int r = 0;
        for (; r + 4 <= O; r += 4) {
            __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            const float* w0 = W + (long)r * I;
            const float* w1 = w0 + I;
            const float* w2 = w1 + I;
            const float* w3 = w2 + I;
            for (int b = 0; b < nb; ++b) {
                if (!m[b]) continue;                 // skip the d_model x 32 tile
                const int off = b * 32;
                const __m512 a0 = _mm512_loadu_ps(a + off);
                const __m512 a1 = _mm512_loadu_ps(a + off + 16);
                c00 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w0 + off), c00);
                c01 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w0 + off + 16), c01);
                c10 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w1 + off), c10);
                c11 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w1 + off + 16), c11);
                c20 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w2 + off), c20);
                c21 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w2 + off + 16), c21);
                c30 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w3 + off), c30);
                c31 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w3 + off + 16), c31);
            }
            o[r + 0] = _mm512_reduce_add_ps(_mm512_add_ps(c00, c01));
            o[r + 1] = _mm512_reduce_add_ps(_mm512_add_ps(c10, c11));
            o[r + 2] = _mm512_reduce_add_ps(_mm512_add_ps(c20, c21));
            o[r + 3] = _mm512_reduce_add_ps(_mm512_add_ps(c30, c31));
        }
        for (; r < O; ++r) {
            __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps();
            const float* w = W + (long)r * I;
            for (int b = 0; b < nb; ++b) {
                if (!m[b]) continue;
                const int off = b * 32;
                c0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + off), _mm512_loadu_ps(w + off), c0);
                c1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + off + 16), _mm512_loadu_ps(w + off + 16), c1);
            }
            o[r] = _mm512_reduce_add_ps(_mm512_add_ps(c0, c1));
        }
    }
}

// Same math driven by a compacted active-block index list (computed once per
// token before down_proj).  The inactive blocks are never visited at all, so
// the inner loop length is K_active and the kernel scales as 1/(1-skip).
void down_skip_avx512_list(const float* __restrict__ A, const float* __restrict__ W,
                           const int* __restrict__ IDX, float* __restrict__ out,
                           int T, int I, int O, int K) {
    for (int t = 0; t < T; ++t) {
        const float* a = A + (long)t * I;
        const int* idx = IDX + (long)t * K;
        float* o = out + (long)t * O;
        int r = 0;
        for (; r + 4 <= O; r += 4) {
            __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            const float* w0 = W + (long)r * I;
            const float* w1 = w0 + I;
            const float* w2 = w1 + I;
            const float* w3 = w2 + I;
            for (int k = 0; k < K; ++k) {
                const int off = idx[k] * 32;
                const __m512 a0 = _mm512_loadu_ps(a + off);
                const __m512 a1 = _mm512_loadu_ps(a + off + 16);
                c00 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w0 + off), c00);
                c01 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w0 + off + 16), c01);
                c10 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w1 + off), c10);
                c11 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w1 + off + 16), c11);
                c20 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w2 + off), c20);
                c21 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w2 + off + 16), c21);
                c30 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w3 + off), c30);
                c31 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w3 + off + 16), c31);
            }
            o[r + 0] = _mm512_reduce_add_ps(_mm512_add_ps(c00, c01));
            o[r + 1] = _mm512_reduce_add_ps(_mm512_add_ps(c10, c11));
            o[r + 2] = _mm512_reduce_add_ps(_mm512_add_ps(c20, c21));
            o[r + 3] = _mm512_reduce_add_ps(_mm512_add_ps(c30, c31));
        }
        for (; r < O; ++r) {
            __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps();
            const float* w = W + (long)r * I;
            for (int k = 0; k < K; ++k) {
                const int off = idx[k] * 32;
                c0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + off), _mm512_loadu_ps(w + off), c0);
                c1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + off + 16), _mm512_loadu_ps(w + off + 16), c1);
            }
            o[r] = _mm512_reduce_add_ps(_mm512_add_ps(c0, c1));
        }
    }
}


// ---------------------------------------------------------------------------
// auto: mask in, index compaction + OpenMP + 8-row register blocking + prefetch
// ---------------------------------------------------------------------------
void down_skip_avx512_auto(const float* __restrict__ A, const float* __restrict__ W,
                          const unsigned char* __restrict__ M, float* __restrict__ out,
                          int T, int I, int O) {
    const int nb = I / 32;
#pragma omp parallel
    {
        int idx[2048];
#pragma omp for schedule(static)
        for (int t = 0; t < T; ++t) {
            const unsigned char* m = M + (long)t * nb;
            int K = 0;
            for (int b = 0; b < nb; ++b) idx[K] = b, K += (m[b] != 0);
            const float* a = A + (long)t * I;
            float* o = out + (long)t * O;
            int r = 0;
            for (; r + 8 <= O; r += 8) {
                __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
                __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
                __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
                __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
                __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
                __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
                __m512 c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
                __m512 c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
                const float* w0 = W + (long)r * I;
                const float* w1 = w0 + I;  const float* w2 = w1 + I;
                const float* w3 = w2 + I;  const float* w4 = w3 + I;
                const float* w5 = w4 + I;  const float* w6 = w5 + I;
                const float* w7 = w6 + I;
                for (int k = 0; k < K; ++k) {
                    const int off = idx[k] * 32;
                    if (k + 1 < K)
                        _mm_prefetch((const char*)(a + idx[k + 1] * 32), _MM_HINT_T0);
                    const __m512 a0 = _mm512_loadu_ps(a + off);
                    const __m512 a1 = _mm512_loadu_ps(a + off + 16);
                    c00 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w0 + off), c00);
                    c01 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w0 + off + 16), c01);
                    c10 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w1 + off), c10);
                    c11 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w1 + off + 16), c11);
                    c20 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w2 + off), c20);
                    c21 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w2 + off + 16), c21);
                    c30 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w3 + off), c30);
                    c31 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w3 + off + 16), c31);
                    c40 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w4 + off), c40);
                    c41 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w4 + off + 16), c41);
                    c50 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w5 + off), c50);
                    c51 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w5 + off + 16), c51);
                    c60 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w6 + off), c60);
                    c61 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w6 + off + 16), c61);
                    c70 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w7 + off), c70);
                    c71 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w7 + off + 16), c71);
                }
                o[r + 0] = _mm512_reduce_add_ps(_mm512_add_ps(c00, c01));
                o[r + 1] = _mm512_reduce_add_ps(_mm512_add_ps(c10, c11));
                o[r + 2] = _mm512_reduce_add_ps(_mm512_add_ps(c20, c21));
                o[r + 3] = _mm512_reduce_add_ps(_mm512_add_ps(c30, c31));
                o[r + 4] = _mm512_reduce_add_ps(_mm512_add_ps(c40, c41));
                o[r + 5] = _mm512_reduce_add_ps(_mm512_add_ps(c50, c51));
                o[r + 6] = _mm512_reduce_add_ps(_mm512_add_ps(c60, c61));
                o[r + 7] = _mm512_reduce_add_ps(_mm512_add_ps(c70, c71));
            }
            for (; r < O; ++r) {
                __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps();
                const float* w = W + (long)r * I;
                for (int k = 0; k < K; ++k) {
                    const int off = idx[k] * 32;
                    c0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + off),
                                         _mm512_loadu_ps(w + off), c0);
                    c1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + off + 16),
                                         _mm512_loadu_ps(w + off + 16), c1);
                }
                o[r] = _mm512_reduce_add_ps(_mm512_add_ps(c0, c1));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Defect-2 fix: parallelise over OUTPUT ROWS, not tokens.
// At T=1 the token-parallel loop above leaves 31 of 32 cores idle.  Here the
// compaction runs once (serial, T*nb is tiny), then collapse(2) spreads
// (token, 8-row group) across all cores so a single decode token is split
// over every thread.
// ---------------------------------------------------------------------------
void down_skip_avx512_auto2(const float* __restrict__ A, const float* __restrict__ W,
                            const unsigned char* __restrict__ M, float* __restrict__ out,
                            int T, int I, int O) {
    const int nb = I / 32;
    std::vector<int> IDX((size_t)T * nb);
    std::vector<int> K(T);
    for (int t = 0; t < T; ++t) {
        const unsigned char* m = M + (long)t * nb;
        int k = 0;
        int* idx = IDX.data() + (long)t * nb;
        for (int b = 0; b < nb; ++b) { idx[k] = b; k += (m[b] != 0); }
        K[t] = k;
    }
    const int RB = 8;
    const int NR = (O + RB - 1) / RB;
#pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < T; ++t) {
        for (int rg = 0; rg < NR; ++rg) {
            const float* a = A + (long)t * I;
            float* o = out + (long)t * O;
            const int* idx = IDX.data() + (long)t * nb;
            const int Kt = K[t];
            const int r0 = rg * RB;
            if (r0 + RB <= O) {
                __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
                __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
                __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
                __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
                __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps();
                __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps();
                __m512 c60 = _mm512_setzero_ps(), c61 = _mm512_setzero_ps();
                __m512 c70 = _mm512_setzero_ps(), c71 = _mm512_setzero_ps();
                const float* w0 = W + (long)r0 * I;
                const float* w1 = w0 + I;  const float* w2 = w1 + I;
                const float* w3 = w2 + I;  const float* w4 = w3 + I;
                const float* w5 = w4 + I;  const float* w6 = w5 + I;
                const float* w7 = w6 + I;
                for (int k = 0; k < Kt; ++k) {
                    const int off = idx[k] * 32;
                    const __m512 a0 = _mm512_loadu_ps(a + off);
                    const __m512 a1 = _mm512_loadu_ps(a + off + 16);
                    c00 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w0 + off), c00);
                    c01 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w0 + off + 16), c01);
                    c10 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w1 + off), c10);
                    c11 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w1 + off + 16), c11);
                    c20 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w2 + off), c20);
                    c21 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w2 + off + 16), c21);
                    c30 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w3 + off), c30);
                    c31 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w3 + off + 16), c31);
                    c40 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w4 + off), c40);
                    c41 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w4 + off + 16), c41);
                    c50 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w5 + off), c50);
                    c51 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w5 + off + 16), c51);
                    c60 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w6 + off), c60);
                    c61 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w6 + off + 16), c61);
                    c70 = _mm512_fmadd_ps(a0, _mm512_loadu_ps(w7 + off), c70);
                    c71 = _mm512_fmadd_ps(a1, _mm512_loadu_ps(w7 + off + 16), c71);
                }
                o[r0 + 0] = _mm512_reduce_add_ps(_mm512_add_ps(c00, c01));
                o[r0 + 1] = _mm512_reduce_add_ps(_mm512_add_ps(c10, c11));
                o[r0 + 2] = _mm512_reduce_add_ps(_mm512_add_ps(c20, c21));
                o[r0 + 3] = _mm512_reduce_add_ps(_mm512_add_ps(c30, c31));
                o[r0 + 4] = _mm512_reduce_add_ps(_mm512_add_ps(c40, c41));
                o[r0 + 5] = _mm512_reduce_add_ps(_mm512_add_ps(c50, c51));
                o[r0 + 6] = _mm512_reduce_add_ps(_mm512_add_ps(c60, c61));
                o[r0 + 7] = _mm512_reduce_add_ps(_mm512_add_ps(c70, c71));
            } else {
                for (int r = r0; r < O; ++r) {
                    __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps();
                    const float* w = W + (long)r * I;
                    for (int k = 0; k < Kt; ++k) {
                        const int off = idx[k] * 32;
                        c0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + off),
                                             _mm512_loadu_ps(w + off), c0);
                        c1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + off + 16),
                                             _mm512_loadu_ps(w + off + 16), c1);
                    }
                    o[r] = _mm512_reduce_add_ps(_mm512_add_ps(c0, c1));
                }
            }
        }
    }
}

}  // extern "C"
