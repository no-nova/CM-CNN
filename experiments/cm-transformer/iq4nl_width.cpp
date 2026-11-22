// SIMD-width ablation for the 4.5-bit IQ4_NL kernels (single-threaded).
//
// The same source is compiled three times: scalar (vectoriser disabled), AVX2
// (256-bit) and AVX-512 (512-bit).  Block decoding is identical in all three so
// the comparison isolates the width of the multiply-add loop and the registers
// it runs in.  Layouts: row-major `W[r][b][18]`, block-major `WB[b][r][18]`.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>

#if defined(USE_AVX512)
#include <immintrin.h>
#elif defined(USE_AVX2)
#include <immintrin.h>
#endif

static const float KV[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                             1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f};

static inline float h2f(const uint8_t* p) {
    uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    float v;
    if (exp == 0) v = std::ldexp((float)man, -24);
    else if (exp == 31) v = man ? NAN : INFINITY;
    else v = std::ldexp((float)(man | 0x400), (int)exp - 25);
    return sign ? -v : v;
}

// 32 nibbles -> 32 floats (identical in all builds).
static inline float decode_block(const uint8_t* blk, float* w) {
    const float d = h2f(blk);
    const uint8_t* q = blk + 2;
    for (int i = 0; i < 16; ++i) {
        w[i]      = d * KV[q[i] & 0x0f];
        w[16 + i] = d * KV[q[i] >> 4];
    }
    return d;
}

static inline float dot32(const float* a, const float* w) {
#if defined(USE_AVX512)
    __m512 acc = _mm512_setzero_ps();
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a), _mm512_loadu_ps(w), acc);
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + 16), _mm512_loadu_ps(w + 16), acc);
    return _mm512_reduce_add_ps(acc);
#elif defined(USE_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (int j = 0; j < 32; j += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(w + j), acc);
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
#else
    float acc = 0.f;
    for (int j = 0; j < 32; ++j) acc += a[j] * w[j];
    return acc;
#endif
}

extern "C" {

// Row-major dense GEMV: every row, every block.
void gemv_row_dense(const uint8_t* W, const float* a, float* out, int M, int nb) {
    float w[32];
    for (int r = 0; r < M; ++r) {
        const uint8_t* row = W + (size_t)r * nb * 18;
        float acc = 0.f;
        for (int b = 0; b < nb; ++b) {
            decode_block(row + b * 18, w);
            acc += dot32(a + b * 32, w);
        }
        out[r] = acc;
    }
}

// Block-major GEMV over a compacted block list.
void gemv_bm(const uint8_t* WB, const float* a, float* out, int M, int nb,
             const int* idx, int K) {
    float w[32];
    for (int r = 0; r < M; ++r) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) {
            int b = idx[k];
            decode_block(WB + (size_t)b * M * 18 + (size_t)r * 18, w);
            acc += dot32(a + b * 32, w);
        }
        out[r] = acc;
    }
}

}  // extern "C"
