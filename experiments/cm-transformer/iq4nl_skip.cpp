// iq4nl_skip.cpp -- AVX-512 IQ4_NL matvec with 32-weight-block skipping.
//
// Weights are kept in exactly the GGUF/dataset layout: row-major, each row is
// nblocks * 18 bytes, each block = fp16 d (2B, little endian) + 16 nibble bytes.
// Element i (0..15) = low nibble of byte i, element 16+i = high nibble of byte i.
//
// matvec: y[rows] = W[rows, nblocks*32] @ a[nblocks*32]
// The inner loop walks the *compacted* list of active block indices, so skipped
// blocks are never addressed -> no bytes read, no dequant, no FMA.
//
// Build:
//   g++ -O3 -mavx512f -mavx512bw -mavx512vl -mavx512dq -mfma -mf16c \
//       -march=native -shared -fPIC -o iq4nl_skip.so iq4nl_skip.cpp
#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace {

// kvalues_iq4nl as int32 lanes (used by the permutexvar dequant).
alignas(64) const int32_t KV32[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// prefetch distance (in active blocks) used by the skipping matvec
constexpr int PF = 4;

inline __m512 f16_to_ps(uint16_t h) {
    return _mm512_set1_ps(_cvtsh_ss(h));
}

// Decode 18-byte IQ4_NL block at q into 32 floats (two zmm: elements 0..15, 16..31).
inline void decode(const uint8_t *q, __m512 &lo, __m512 &hi,
                   const __m512i kvt, const __m512i m4) {
    uint16_t dh;
    std::memcpy(&dh, q, 2);
    const __m512 vd = f16_to_ps(dh);
    const __m128i nib = _mm_loadu_si128(reinterpret_cast<const __m128i *>(q + 2));
    const __m128i nib_hi = _mm_and_si128(_mm_srli_epi16(nib, 4),
                                         _mm_set1_epi8(0x0F));
    const __m512i ilo = _mm512_and_si512(_mm512_cvtepu8_epi32(nib), m4);
    const __m512i ihi = _mm512_cvtepu8_epi32(nib_hi);
    lo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_permutexvar_epi32(ilo, kvt)), vd);
    hi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_permutexvar_epi32(ihi, kvt)), vd);
}

// R rows at a time so the activation block (and its 2 zmm loads) is shared and
// several independent load streams keep the memory pipeline full.
template <int R>
inline void run_rows(const uint8_t *W, const float *a, float *y, int row0,
                     int nblk, const int32_t *active, int nact) {
    const size_t stride = static_cast<size_t>(nblk) * 18;
    const uint8_t *base = W + static_cast<size_t>(row0) * stride;
    const __m512i kvt = _mm512_loadu_si512(reinterpret_cast<const void *>(KV32));
    const __m512i m4 = _mm512_set1_epi32(0x0F);

    __m512 acc0[R], acc1[R];
#pragma GCC unroll 8
    for (int r = 0; r < R; ++r) {
        acc0[r] = _mm512_setzero_ps();
        acc1[r] = _mm512_setzero_ps();
    }

    for (int t = 0; t < nact; ++t) {
        const int b = active[t];
        const float *ap = a + 32 * static_cast<size_t>(b);
        const __m512 a0 = _mm512_loadu_ps(ap);
        const __m512 a1 = _mm512_loadu_ps(ap + 16);
        // software prefetch a few blocks ahead: with scattered 18-byte blocks
        // the load latency, not bandwidth, is the limit when most blocks are
        // skipped.
        const int pf = (t + PF < nact) ? active[t + PF] : b;
        const size_t pfo = static_cast<size_t>(pf) * 18;
#pragma GCC unroll 8
        for (int r = 0; r < R; ++r)
            _mm_prefetch(reinterpret_cast<const char *>(
                             base + static_cast<size_t>(r) * stride + pfo),
                         _MM_HINT_T0);
#pragma GCC unroll 8
        for (int r = 0; r < R; ++r) {
            __m512 lo, hi;
            decode(base + static_cast<size_t>(r) * stride + static_cast<size_t>(b) * 18,
                   lo, hi, kvt, m4);
            acc0[r] = _mm512_fmadd_ps(lo, a0, acc0[r]);
            acc1[r] = _mm512_fmadd_ps(hi, a1, acc1[r]);
        }
    }
#pragma GCC unroll 8
    for (int r = 0; r < R; ++r)
        y[row0 + r] = _mm512_reduce_add_ps(_mm512_add_ps(acc0[r], acc1[r]));
}

}  // namespace

extern "C" {

// Full matvec with an explicit active-block list (pass identity mask for dense).
void iq4nl_matvec(const uint8_t *W, const float *a, float *y, int rows, int nblk,
                  const int32_t *active, int nact) {
    int r = 0;
    for (; r + 4 <= rows; r += 4) run_rows<4>(W, a, y, r, nblk, active, nact);
    for (; r + 2 <= rows; r += 2) run_rows<2>(W, a, y, r, nblk, active, nact);
    for (; r < rows; ++r) run_rows<1>(W, a, y, r, nblk, active, nact);
}

// Dense baseline without the index indirection (pure sequential streaming).
void iq4nl_matvec_dense(const uint8_t *W, const float *a, float *y, int rows,
                        int nblk) {
    const size_t stride = static_cast<size_t>(nblk) * 18;
    const __m512i kvt = _mm512_loadu_si512(reinterpret_cast<const void *>(KV32));
    const __m512i m4 = _mm512_set1_epi32(0x0F);
    for (int row0 = 0; row0 + 4 <= rows; row0 += 4) {
        const uint8_t *base = W + static_cast<size_t>(row0) * stride;
        __m512 acc0[4], acc1[4];
        for (int r = 0; r < 4; ++r) {
            acc0[r] = _mm512_setzero_ps();
            acc1[r] = _mm512_setzero_ps();
        }
        const uint8_t *p[4] = {base, base + stride, base + 2 * stride,
                               base + 3 * stride};
        for (int b = 0; b < nblk; ++b) {
            const float *ap = a + 32 * static_cast<size_t>(b);
            const __m512 a0 = _mm512_loadu_ps(ap);
            const __m512 a1 = _mm512_loadu_ps(ap + 16);
#pragma GCC unroll 4
            for (int r = 0; r < 4; ++r) {
                __m512 lo, hi;
                decode(p[r], lo, hi, kvt, m4);
                acc0[r] = _mm512_fmadd_ps(lo, a0, acc0[r]);
                acc1[r] = _mm512_fmadd_ps(hi, a1, acc1[r]);
                p[r] += 18;
            }
        }
        for (int r = 0; r < 4; ++r)
            y[row0 + r] = _mm512_reduce_add_ps(_mm512_add_ps(acc0[r], acc1[r]));
    }
}

// Dequantize nblk consecutive blocks (in layout order) to nblk*32 floats.
// Used to validate the C++ decode path against the python reference.
void iq4nl_dequant(const uint8_t *q, float *out, int nblk) {
    const __m512i kvt = _mm512_loadu_si512(reinterpret_cast<const void *>(KV32));
    const __m512i m4 = _mm512_set1_epi32(0x0F);
    for (int b = 0; b < nblk; ++b) {
        __m512 lo, hi;
        decode(q + static_cast<size_t>(b) * 18, lo, hi, kvt, m4);
        _mm512_storeu_ps(out + 32 * static_cast<size_t>(b), lo);
        _mm512_storeu_ps(out + 32 * static_cast<size_t>(b) + 16, hi);
    }
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Support kernels: cache flush, streaming ceiling, traffic-only touch kernel.
// ---------------------------------------------------------------------------
extern "C" {

// NOTE: a CLFLUSHOPT-based eviction loop was tried first and measured to be a
// no-op on this machine (an 8 MB region was still served at ~140 GB/s right
// after flushing 64 MB), so cold measurements evict by capacity instead: read
// a scratch buffer that is several times the L3 size (see gguf_skip_bench.py).

// Pure sequential read bandwidth ceiling (256 bytes per iteration).
double stream_read_sum(const uint8_t *p, size_t n) {
    size_t nz = n / 256;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    const float *f = reinterpret_cast<const float *>(p);
    for (size_t i = 0; i < nz; ++i) {
        const float *q = f + i * 64;
        acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(q));
        acc1 = _mm512_add_ps(acc1, _mm512_loadu_ps(q + 16));
        acc2 = _mm512_add_ps(acc2, _mm512_loadu_ps(q + 32));
        acc3 = _mm512_add_ps(acc3, _mm512_loadu_ps(q + 48));
    }
    return static_cast<double>(_mm512_reduce_add_ps(
        _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3))));
}

// Traffic-only kernel: reads exactly the active 18-byte blocks (no dequant, no
// FMA). Time here is pure memory traffic, so it isolates the byte reduction
// that block skipping buys.
double iq4nl_touch(const uint8_t *W, int rows, int nblk, const int32_t *active,
                   int nact) {
    const size_t stride = static_cast<size_t>(nblk) * 18;
    // 4 rows at a time with independent accumulators: 4 load streams keep the
    // DRAM pipeline busy, so the number is bandwidth (not latency) bound.
    __m128i s0 = _mm_setzero_si128(), s1 = _mm_setzero_si128();
    __m128i s2 = _mm_setzero_si128(), s3 = _mm_setzero_si128();
    int r = 0;
    for (; r + 4 <= rows; r += 4) {
        const uint8_t *r0 = W + static_cast<size_t>(r) * stride;
        const uint8_t *r1 = r0 + stride, *r2 = r1 + stride, *r3 = r2 + stride;
        for (int t = 0; t < nact; ++t) {
            const size_t off = static_cast<size_t>(active[t]) * 18 + 2;
            s0 = _mm_add_epi8(s0, _mm_loadu_si128(
                                     reinterpret_cast<const __m128i *>(r0 + off)));
            s1 = _mm_add_epi8(s1, _mm_loadu_si128(
                                     reinterpret_cast<const __m128i *>(r1 + off)));
            s2 = _mm_add_epi8(s2, _mm_loadu_si128(
                                     reinterpret_cast<const __m128i *>(r2 + off)));
            s3 = _mm_add_epi8(s3, _mm_loadu_si128(
                                     reinterpret_cast<const __m128i *>(r3 + off)));
            if ((t & 15) == 15) {      // touch the fp16 scale of every 16th block
                uint16_t h;
                std::memcpy(&h, r0 + off - 2, 2);
                s0 = _mm_add_epi8(s0, _mm_set1_epi16(static_cast<short>(h)));
            }
        }
    }
    __m128i acc = _mm_add_epi8(_mm_add_epi8(s0, s1), _mm_add_epi8(s2, s3));
    for (; r < rows; ++r) {
        const uint8_t *row = W + static_cast<size_t>(r) * stride;
        for (int t = 0; t < nact; ++t) {
            const size_t off = static_cast<size_t>(active[t]) * 18 + 2;
            acc = _mm_add_epi8(acc, _mm_loadu_si128(
                                        reinterpret_cast<const __m128i *>(row + off)));
        }
    }
    const __m128i s = _mm_sad_epu8(acc, _mm_setzero_si128());
    const uint64_t s0v = static_cast<uint64_t>(_mm_cvtsi128_si64(s));
    const uint64_t s1v = static_cast<uint64_t>(_mm_extract_epi64(s, 1));
    return static_cast<double>(s0v + s1v);
}

}  // extern "C"
