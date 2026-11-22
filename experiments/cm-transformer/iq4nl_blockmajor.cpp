#include <cmath>
// Block-major IQ4_NL GEMV (Defect-1 fix).
//
// Row-major layout [M, nb*18] puts three-and-a-half consecutive blocks of the
// SAME row inside one 64-byte line, so skipping a block leaves its line dirty
// and no DRAM traffic is saved.  Block-major layout [nb, M*18] makes each
// block a contiguous M*18 = 46080 B (720 cache lines) region; a skipped block
// removes exactly 720 lines and every touched line is fully consumed.
//
// nibble layout (GGML IQ4_NL): 16 bytes hold 32 values, element i = low nibble
// of byte i, element 16+i = high nibble of byte i; value = d * kvalues[nibble].
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {

static inline float hsum512(__m512 v) { return _mm512_reduce_add_ps(v); }

static inline __m128 _fp16_to_fp32_broadcast(uint16_t h) {
    // fp16 -> fp32 via F16C, then broadcast by the caller
    __m128i hv = _mm_set1_epi16((short)h);
    return _mm_cvtph_ps(hv);
}

// W [M, nb*18] -> WB [nb, M*18]
void repack_blockmajor(const uint8_t* __restrict__ W, uint8_t* __restrict__ WB,
                       int M, int nb, int blk_bytes) {
    for (int b = 0; b < nb; ++b) {
        uint8_t* dst = WB + (size_t)b * M * blk_bytes;
        for (int r = 0; r < M; ++r)
            std::memcpy(dst + (size_t)r * blk_bytes,
                        W + ((size_t)r * nb + b) * blk_bytes, blk_bytes);
    }
}

// out[r] = sum_{k<K} sum_{l<32} a[idx[k]*32+l] * dequant(WB[idx[k]][r])
void gemv_blockmajor(const uint8_t* __restrict__ WB, const float* __restrict__ a,
                     float* __restrict__ out, int M, int nb, int blk_bytes,
                     const int* __restrict__ idx, int K) {
    const __m512 ktab = _mm512_setr_ps(
        -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
         1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f);
    const long row_stride = blk_bytes;
    // 4 rows at a time for ILP; each row needs 2 permutes + 2 FMA per block.
    int r = 0;
    for (; r + 4 <= M; r += 4) {
        __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps();
        __m512 c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
        for (int k = 0; k < K; ++k) {
            const uint8_t* blk = WB + (size_t)idx[k] * M * row_stride;
            const float* av = a + (size_t)idx[k] * 32;
            const __m512 a0 = _mm512_loadu_ps(av);
            const __m512 a1 = _mm512_loadu_ps(av + 16);
            for (int j = 0; j < 4; ++j) {
                const uint8_t* w = blk + (size_t)(r + j) * row_stride;
                const float d = _mm_cvtss_f32(_fp16_to_fp32_broadcast(
                    (uint16_t)(w[0] | (w[1] << 8))));
                const __m128i v = _mm_loadu_si128((const __m128i*)(w + 2));
                const __m128i lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
                const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4),
                                                 _mm_set1_epi8(0x0F));
                const __m512 f0 = _mm512_mul_ps(
                    _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(lo), ktab),
                    _mm512_set1_ps(d));
                const __m512 f1 = _mm512_mul_ps(
                    _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(hi), ktab),
                    _mm512_set1_ps(d));
                __m512* c = (j == 0) ? &c0 : (j == 1) ? &c1 : (j == 2) ? &c2 : &c3;
                *c = _mm512_fmadd_ps(a0, f0, *c);
                *c = _mm512_fmadd_ps(a1, f1, *c);
            }
        }
        out[r + 0] = _mm512_reduce_add_ps(c0);
        out[r + 1] = _mm512_reduce_add_ps(c1);
        out[r + 2] = _mm512_reduce_add_ps(c2);
        out[r + 3] = _mm512_reduce_add_ps(c3);
    }
    for (; r < M; ++r) {
        __m512 c0 = _mm512_setzero_ps();
        for (int k = 0; k < K; ++k) {
            const uint8_t* blk = WB + (size_t)idx[k] * M * row_stride;
            const float* av = a + (size_t)idx[k] * 32;
            const uint8_t* w = blk + (size_t)r * row_stride;
            const float d = _mm_cvtss_f32(_fp16_to_fp32_broadcast(
                (uint16_t)(w[0] | (w[1] << 8))));
            const __m128i v = _mm_loadu_si128((const __m128i*)(w + 2));
            const __m128i lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4),
                                             _mm_set1_epi8(0x0F));
            const __m512 f0 = _mm512_mul_ps(
                _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(lo), ktab),
                _mm512_set1_ps(d));
            const __m512 f1 = _mm512_mul_ps(
                _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(hi), ktab),
                _mm512_set1_ps(d));
            c0 = _mm512_fmadd_ps(_mm512_loadu_ps(av), f0, c0);
            c0 = _mm512_fmadd_ps(_mm512_loadu_ps(av + 16), f1, c0);
        }
        out[r] = _mm512_reduce_add_ps(c0);
    }
}

// Same as gemv_blockmajor but parallel over output rows (T=1 friendly).
void gemv_blockmajor_mt(const uint8_t* __restrict__ WB, const float* __restrict__ a,
                        float* __restrict__ out, int M, int nb, int blk_bytes,
                        const int* __restrict__ idx, int K) {
    const __m512 ktab = _mm512_setr_ps(
        -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
         1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < M; ++r) {
        __m512 c0 = _mm512_setzero_ps();
        for (int k = 0; k < K; ++k) {
            const uint8_t* blk = WB + (size_t)idx[k] * M * blk_bytes;
            const float* av = a + (size_t)idx[k] * 32;
            const uint8_t* w = blk + (size_t)r * blk_bytes;
            const float d = _mm_cvtss_f32(_fp16_to_fp32_broadcast(
                (uint16_t)(w[0] | (w[1] << 8))));
            const __m128i v = _mm_loadu_si128((const __m128i*)(w + 2));
            const __m128i lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
            const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4),
                                             _mm_set1_epi8(0x0F));
            const __m512 f0 = _mm512_mul_ps(
                _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(lo), ktab),
                _mm512_set1_ps(d));
            const __m512 f1 = _mm512_mul_ps(
                _mm512_permutexvar_ps(_mm512_cvtepu8_epi32(hi), ktab),
                _mm512_set1_ps(d));
            c0 = _mm512_fmadd_ps(_mm512_loadu_ps(av), f0, c0);
            c0 = _mm512_fmadd_ps(_mm512_loadu_ps(av + 16), f1, c0);
        }
        out[r] = _mm512_reduce_add_ps(c0);
    }
}


// ---- 8-row register blocking (Defect: dense ILP) ---------------------------
// Decoding a row costs 2 permutes + 2 FMA; with 4 rows the two permute ports are
// idle half the time.  8 rows = 16 ZMM accumulators keeps both FMA pipes and the
// permute unit busy and lets the compiler software-pipeline the loads.

static inline __m512i iq4_lo(const __m128i v) {
    return _mm512_cvtepu8_epi32(_mm_and_si128(v, _mm_set1_epi8(0x0F)));
}
static inline __m512i iq4_hi(const __m128i v) {
    return _mm512_cvtepu8_epi32(_mm_and_si128(_mm_srli_epi16(v, 4),
                                              _mm_set1_epi8(0x0F)));
}

void gemv_blockmajor8(const uint8_t* __restrict__ WB, const float* __restrict__ a,
                      float* __restrict__ out, int M, int nb, int blk_bytes,
                      const int* __restrict__ idx, int K) {
    const __m512 ktab = _mm512_setr_ps(
        -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
         1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f);
    for (int r0 = 0; r0 + 8 <= M; r0 += 8) {
        __m512 c[16];
        for (int j = 0; j < 16; ++j) c[j] = _mm512_setzero_ps();
        for (int k = 0; k < K; ++k) {
            const uint8_t* blk = WB + (size_t)idx[k] * M * blk_bytes;
            const float* av = a + (size_t)idx[k] * 32;
            const __m512 a0 = _mm512_loadu_ps(av), a1 = _mm512_loadu_ps(av + 16);
            const uint8_t* w = blk + (size_t)r0 * blk_bytes;
            for (int j = 0; j < 8; ++j, w += blk_bytes) {
                const float d = _mm_cvtss_f32(_fp16_to_fp32_broadcast(
                    (uint16_t)(w[0] | (w[1] << 8))));
                const __m128i v = _mm_loadu_si128((const __m128i*)(w + 2));
                const __m512 ds = _mm512_set1_ps(d);
                c[2 * j] = _mm512_fmadd_ps(a0, _mm512_mul_ps(
                    _mm512_permutexvar_ps(iq4_lo(v), ktab), ds), c[2 * j]);
                c[2 * j + 1] = _mm512_fmadd_ps(a1, _mm512_mul_ps(
                    _mm512_permutexvar_ps(iq4_hi(v), ktab), ds), c[2 * j + 1]);
            }
        }
        for (int j = 0; j < 8; ++j)
            out[r0 + j] = hsum512(_mm512_add_ps(c[2 * j], c[2 * j + 1]));
    }
}

// OpenMP row-parallel variant: one 8-row group per thread iteration, so a single
// decode token (T=1) still uses every core.
void gemv_blockmajor8_mt(const uint8_t* __restrict__ WB, const float* __restrict__ a,
                         float* __restrict__ out, int M, int nb, int blk_bytes,
                         const int* __restrict__ idx, int K, int nthreads) {
    const __m512 ktab = _mm512_setr_ps(
        -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
         1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f);
    const int groups = M / 8;
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int g = 0; g < groups; ++g) {
        const int r0 = g * 8;
        __m512 c[16];
        for (int j = 0; j < 16; ++j) c[j] = _mm512_setzero_ps();
        for (int k = 0; k < K; ++k) {
            const uint8_t* blk = WB + (size_t)idx[k] * M * blk_bytes;
            const float* av = a + (size_t)idx[k] * 32;
            const __m512 a0 = _mm512_loadu_ps(av), a1 = _mm512_loadu_ps(av + 16);
            const uint8_t* w = blk + (size_t)r0 * blk_bytes;
            for (int j = 0; j < 8; ++j, w += blk_bytes) {
                const float d = _mm_cvtss_f32(_fp16_to_fp32_broadcast(
                    (uint16_t)(w[0] | (w[1] << 8))));
                const __m128i v = _mm_loadu_si128((const __m128i*)(w + 2));
                const __m512 ds = _mm512_set1_ps(d);
                c[2 * j] = _mm512_fmadd_ps(a0, _mm512_mul_ps(
                    _mm512_permutexvar_ps(iq4_lo(v), ktab), ds), c[2 * j]);
                c[2 * j + 1] = _mm512_fmadd_ps(a1, _mm512_mul_ps(
                    _mm512_permutexvar_ps(iq4_hi(v), ktab), ds), c[2 * j + 1]);
            }
        }
        for (int j = 0; j < 8; ++j)
            out[r0 + j] = hsum512(_mm512_add_ps(c[2 * j], c[2 * j + 1]));
    }
}

// ---- coupled IQ4_NL: up_proj active row blocks ----------------------------
// ffn_up is [ne1=9216 rows, ne0=2560 in] of IQ4_NL: 80 nibble blocks per row
// (1440 B).  Repack gathers the 32 rows of one intermediate block so they are
// contiguous ([nb, 32*1440] = 46080 B = 720 lines), and only the active blocks
// are touched -- the same idx list the down kernel uses.
void up_rows_blockmajor(const uint8_t* __restrict__ WB, const float* __restrict__ x,
                        float* __restrict__ u, int out_rows, int nb, int n_in,
                        const int* __restrict__ idx, int K, int nthreads) {
    const __m512 ktab = _mm512_setr_ps(
        -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
         1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f);
    const int ib = n_in / 32;              // nibble blocks per row (80)
    const size_t row_bytes = (size_t)ib * 18;
    const int total = K * 32;
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int j = 0; j < total; ++j) {
        const int b = idx[j >> 5];
        const int l = j & 31;
        const uint8_t* row = WB + (size_t)b * 32 * row_bytes + (size_t)l * row_bytes;
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        const float* xp = x;
        for (int k = 0; k < ib; ++k, row += 18, xp += 32) {
            const float d = _mm_cvtss_f32(_fp16_to_fp32_broadcast(
                (uint16_t)(row[0] | (row[1] << 8))));
            const __m128i v = _mm_loadu_si128((const __m128i*)(row + 2));
            const __m512 ds = _mm512_set1_ps(d);
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(xp),
                                   _mm512_mul_ps(_mm512_permutexvar_ps(iq4_lo(v), ktab), ds),
                                   acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(xp + 16),
                                   _mm512_mul_ps(_mm512_permutexvar_ps(iq4_hi(v), ktab), ds),
                                   acc1);
        }
        u[j] = hsum512(_mm512_add_ps(acc0, acc1));
    }
}


// ---- ffn_gate pre-filter ---------------------------------------------------
// The coupled FFN is Amdahl-limited by ffn_gate, which used to be read dense.
// Cheap pre-filter: evaluate only the first `r` rows of every 32-row block
// (contiguous at the block start, r*1440 B), rank blocks by that partial energy,
// and then finish the remaining 32-r rows ONLY for the K selected blocks.  Row
// traffic drops from nb*32 to nb*r + K*(32-r).
static inline float iq4_row_dot(const float* x, const uint8_t* row, int ib,
                                __m512 ktab) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    const float* xp = x;
    for (int k = 0; k < ib; ++k, row += 18, xp += 32) {
        const float d = _mm_cvtss_f32(_fp16_to_fp32_broadcast(
            (uint16_t)(row[0] | (row[1] << 8))));
        const __m128i v = _mm_loadu_si128((const __m128i*)(row + 2));
        const __m512 ds = _mm512_set1_ps(d);
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(xp),
                               _mm512_mul_ps(_mm512_permutexvar_ps(iq4_lo(v), ktab), ds), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(xp + 16),
                               _mm512_mul_ps(_mm512_permutexvar_ps(iq4_hi(v), ktab), ds), acc1);
    }
    return hsum512(_mm512_add_ps(acc0, acc1));
}

void gate_prefilter_blockmajor(const uint8_t* __restrict__ WB, const float* __restrict__ x,
                               float* __restrict__ g, int* __restrict__ idx,
                               int* __restrict__ kp, int nb, int n_in, int r, float rate,
                               const float* __restrict__ wb2, int nthreads) {
    const __m512 ktab = _mm512_setr_ps(
        -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
         1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f);
    const int ib = n_in / 32;
    const size_t row_bytes = (size_t)ib * 18;
    const size_t blk_bytes = 32 * row_bytes;
    static float en[512];  // shared: the parallel loop writes en[b], the serial top-K reads it
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int b = 0; b < nb; ++b) {
        const uint8_t* base = WB + (size_t)b * blk_bytes;
        float e = 0.f;
        for (int l = 0; l < r; ++l) {
            const float v = iq4_row_dot(x, base + (size_t)l * row_bytes, ib, ktab);
            g[b * 32 + l] = v;
            e += v * v;
        }
        en[b] = wb2 ? e * wb2[b] : e;
    }
    const int K = (int)lroundf(nb * (1.f - rate));
    // top-K by partial energy (nb <= 512, K <= 512)
    int sel[512];
    unsigned char used[512];
    for (int b = 0; b < nb; ++b) used[b] = 0;
    for (int k = 0; k < K; ++k) sel[k] = k % nb;     // safe default
    int nsel = 0;
    for (int k = 0; k < K; ++k) {
        int best = -1; float bv = -INFINITY;
        for (int b = 0; b < nb; ++b) {
            const float e = en[b];
            if (!used[b] && e == e && e > bv) { bv = e; best = b; }
        }
        if (best < 0) break;
        used[best] = 1;
        sel[k] = best;
        nsel = k + 1;
    }
    const int Kuse = nsel > 0 ? nsel : K;
    // finish the remaining rows of the selected blocks
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int k = 0; k < Kuse; ++k) {
        const int b = sel[k];
        const uint8_t* base = WB + (size_t)b * blk_bytes;
        for (int l = r; l < 32; ++l)
            g[b * 32 + l] = iq4_row_dot(x, base + (size_t)l * row_bytes, ib, ktab);
    }
    // ascending index order for the up/down kernels
    for (int i = 1; i < Kuse; ++i) {
        const int v = sel[i];
        int j = i - 1;
        while (j >= 0 && sel[j] > v) { sel[j + 1] = sel[j]; --j; }
        sel[j + 1] = v;
    }
    for (int k = 0; k < Kuse; ++k) idx[k] = sel[k];
    kp[0] = Kuse;
}

}  // extern "C"
