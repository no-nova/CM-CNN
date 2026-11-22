// SPDX-License-Identifier: MPL-2.0
//
// CM-CNN-Accel: frozen VGG-4 (NHWC, 16-float channel blocks) for
// zero-training online Context-Mixing dynamic matrix pruning.
//
// Build:
//   g++ -O3 -march=native -mavx512f -mfma -std=c++20 -fopenmp \
//       native/cm_cnn_avx512.cpp -o experiments/cm-cnn-accel/cm_cnn_avx512
//
// Subcommands:
//   train   --train <bin> --test <bin> --val-count 200 --epochs 40
//           --out <weights.bin> --calib <calib.bin>
//   eval    --weights <weights.bin> --test <bin> [--stream <bin>]
//   bench   ... (added below)
//
// The binary is deliberately self-contained: no BLAS, no framework.  Training
// uses im2col + an AVX-512 1x16 GEMM microkernel; inference uses the same
// 16-channel blocking so gated execution can skip whole 16-channel blocks.

#include <immintrin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>

using std::string;
using std::vector;

static constexpr int IMG = 32;
static constexpr int NC = 10;

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

[[noreturn]] static void die(const string& msg) {
  std::fprintf(stderr, "error: %s\n", msg.c_str());
  std::exit(2);
}

static uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < n; i++) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

static double now_sec() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

template <typename T>
static void read_exact(std::ifstream& f, T* dst, size_t n) {
  f.read(reinterpret_cast<char*>(dst), std::streamsize(n * sizeof(T)));
  if (!f) die("short read");
}

// ---------------------------------------------------------------------------
// Dataset containers (mirrors src/lib/vision/cm-cnn-dataset.ts)
// ---------------------------------------------------------------------------

struct ImageSet {
  int count = 0, h = 0, w = 0;
  vector<uint8_t> labels;
  vector<float> px;  // count*h*w
};

struct StreamSet {
  int seqs = 0, frames = 0, h = 0, w = 0;
  vector<uint8_t> labels;
  vector<float> px;  // seqs*frames*h*w
};

static ImageSet load_images(const string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open " + path);
  char magic[8];
  read_exact(f, magic, 8);
  if (std::memcmp(magic, "CMIMG001", 8) != 0) die("bad image magic in " + path);
  uint32_t count = 0, h = 0, w = 0, ch = 0, r0 = 0, r1 = 0;
  read_exact(f, &count, 1);
  read_exact(f, &h, 1);
  read_exact(f, &w, 1);
  read_exact(f, &ch, 1);
  read_exact(f, &r0, 1);
  read_exact(f, &r1, 1);
  if (ch != 1) die("only grayscale images supported");
  ImageSet s;
  s.count = int(count);
  s.h = int(h);
  s.w = int(w);
  s.labels.resize(count);
  read_exact(f, s.labels.data(), count);
  size_t pad = (4 - (count % 4)) % 4;
  f.seekg(std::streamoff(pad), std::ios::cur);
  s.px.resize(size_t(count) * h * w);
  read_exact(f, s.px.data(), s.px.size());
  return s;
}

static StreamSet load_stream(const string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open " + path);
  char magic[8];
  read_exact(f, magic, 8);
  if (std::memcmp(magic, "CMVID001", 8) != 0) die("bad stream magic in " + path);
  uint32_t seqs = 0, frames = 0, h = 0, w = 0, ch = 0, r = 0;
  read_exact(f, &seqs, 1);
  read_exact(f, &frames, 1);
  read_exact(f, &h, 1);
  read_exact(f, &w, 1);
  read_exact(f, &ch, 1);
  read_exact(f, &r, 1);
  if (ch != 1) die("only grayscale streams supported");
  StreamSet s;
  s.seqs = int(seqs);
  s.frames = int(frames);
  s.h = int(h);
  s.w = int(w);
  s.labels.resize(seqs);
  read_exact(f, s.labels.data(), seqs);
  size_t pad = (4 - (seqs % 4)) % 4;
  f.seekg(std::streamoff(pad), std::ios::cur);
  s.px.resize(size_t(seqs) * frames * h * w);
  read_exact(f, s.px.data(), s.px.size());
  return s;
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

struct ConvLayer {
  int cin = 0, cout = 0, k = 0;  // k = 9*cin
  vector<float> w;               // training layout [k][cout]
  vector<float> wt;              // transposed [cout][k], rebuilt after Adam
  vector<float> gamma, beta;     // batch-norm affine
  vector<float> run_mean, run_var;
  // folded inference parameters
  vector<float> wf;   // [cout/16][k][16]
  vector<float> bias; // [cout]
  // gradients
  vector<float> gw, ggamma, gbeta;
  // Adam state
  vector<float> mw, vw, mg, vg, mb, vb;
};

struct Head {
  vector<float> w;  // [10][2048]
  vector<float> b;  // [10]
  vector<float> gw, gb, mw, vw, mb, vb;
};

struct Model {
  ConvLayer L[4];
  Head head;
  // batch-norm batch statistics (per forward pass)
  vector<float> bn_mean[4], bn_invstd[4];
};

static void init_model(Model& m, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  int cin[4] = {1, 32, 64, 128};
  int cout[4] = {32, 64, 128, 128};
  for (int l = 0; l < 4; l++) {
    ConvLayer& L = m.L[l];
    L.cin = cin[l];
    L.cout = cout[l];
    L.k = 9 * L.cin;
    float scale = std::sqrt(2.0f / float(L.k));
    L.w.resize(size_t(L.k) * L.cout);
    for (auto& v : L.w) v = nd(rng) * scale;
    L.wt.resize(L.w.size());
    L.gamma.assign(L.cout, 1.f);
    L.beta.assign(L.cout, 0.f);
    L.run_mean.assign(L.cout, 0.f);
    L.run_var.assign(L.cout, 1.f);
    L.wf.resize(L.w.size());
    L.bias.assign(L.cout, 0.f);
    L.gw.assign(L.w.size(), 0.f);
    L.ggamma.assign(L.cout, 0.f);
    L.gbeta.assign(L.cout, 0.f);
    L.mw.assign(L.w.size(), 0.f);
    L.vw.assign(L.w.size(), 0.f);
    L.mg.assign(L.cout, 0.f);
    L.vg.assign(L.cout, 0.f);
    L.mb.assign(L.cout, 0.f);
    L.vb.assign(L.cout, 0.f);
    m.bn_mean[l].assign(L.cout, 0.f);
    m.bn_invstd[l].assign(L.cout, 1.f);
  }
  m.head.w.resize(size_t(NC) * 2048);
  float hs = std::sqrt(1.0f / 2048.f);
  for (auto& v : m.head.w) v = nd(rng) * hs;
  m.head.b.assign(NC, 0.f);
  m.head.gw.assign(m.head.w.size(), 0.f);
  m.head.gb.assign(NC, 0.f);
  m.head.mw.assign(m.head.w.size(), 0.f);
  m.head.vw.assign(m.head.w.size(), 0.f);
  m.head.mb.assign(NC, 0.f);
  m.head.vb.assign(NC, 0.f);
}

static void transpose_weights(ConvLayer& L) {
  for (int c = 0; c < L.cout; c++)
    for (int k = 0; k < L.k; k++) L.wt[size_t(c) * L.k + k] = L.w[size_t(k) * L.cout + c];
}

// ---------------------------------------------------------------------------
// v3: isomorphic co-activation channel clustering
// ---------------------------------------------------------------------------
// Sort output channels by folded bias (duty-cycle proxy) descending.  The 16
// always-on channels land in block 0; sparse foreground channels are packed
// into blocks 1..B-1.  The same permutation is applied to the next layer's
// input-channel axis (and to the classifier rows), so the function is exactly
// unchanged: it is a relabelling of channel identities only.
static vector<vector<int>> clustered_perm(const Model& m,
                                          const vector<float>* duty = nullptr) {
  vector<vector<int>> perm(4);
  for (int l = 0; l < 4; l++) {
    const int C = m.L[l].cout;
    vector<int> idx(C);
    for (int i = 0; i < C; i++) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        return m.L[l].bias[a] > m.L[l].bias[b];
      });
    if (l == 3 && std::getenv("CM_ASC4"))
      std::reverse(idx.begin(), idx.end());
    if (l == 3 && std::getenv("CM_RAND4")) {
      uint64_t rs = std::stoull(std::getenv("CM_RAND4"));
      for (int i = C - 1; i > 0; i--) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; std::swap(idx[i], idx[rs % (i + 1)]); }
    }
    if (l == 3 && std::getenv("CM_MID4"))
      std::reverse(idx.begin() + 16, idx.end());
    perm[l] = std::move(idx);
  }
  return perm;
}

static void permute_model(const Model& src, Model& dst, const vector<vector<int>>& perm) {
  dst = src;
  for (int l = 0; l < 4; l++) {
    const ConvLayer& S = src.L[l];
    ConvLayer& D = dst.L[l];
    const int K = S.k, cin = S.cin, C = S.cout;
    vector<float> nw(S.wf.size(), 0.f);
    for (int ncb = 0; ncb < C / 16; ncb++)
      for (int tap = 0; tap < 9; tap++)
        for (int nci = 0; nci < cin; nci++) {
          const int oci = (l == 0) ? 0 : perm[l - 1][nci];
          for (int lane = 0; lane < 16; lane++) {
            const int oc = perm[l][ncb * 16 + lane];
            nw[(size_t(ncb) * K + tap * cin + nci) * 16 + lane] =
                S.wf[(size_t(oc / 16) * K + tap * cin + oci) * 16 + (oc % 16)];
          }
        }
    D.wf.swap(nw);
    for (int nc = 0; nc < C; nc++) D.bias[nc] = S.bias[perm[l][nc]];
    for (int nc = 0; nc < C; nc++) {
      if (!S.gamma.empty()) D.gamma[nc] = S.gamma[perm[l][nc]];
      if (!S.beta.empty()) D.beta[nc] = S.beta[perm[l][nc]];
    }
  }
  for (int k = 0; k < NC; k++)
    for (int j = 0; j < 2048; j++) {
      const int cell = (j / 128) * 128;
      dst.head.w[size_t(k) * 2048 + j] =
          src.head.w[size_t(k) * 2048 + cell + perm[3][j % 128]];
    }
}

// v3 block context: activity bucket of causal W/N/NW neighbours (0,1,2-4,5+),
// input-patch energy and (for video) temporal bits.  Indexed by block id.
static inline int act_bucket(int n) { return n == 0 ? 0 : (n == 1 ? 1 : (n <= 4 ? 2 : 3)); }

// ---------------------------------------------------------------------------
// Forward / backward primitives
// ---------------------------------------------------------------------------

// im2col with optional ReLU on the previous activation.
// prev is spatially padded by 1, [Hp][Wp][cin].  xcol is [H*W][9*cin].
static void im2col(const float* prev, int H, int W, int cin, bool relu,
                   float* xcol) {
  const int Wp = W + 2;
  const int K = 9 * cin;
#pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      float* dst = xcol + (size_t(y) * W + x) * K;
      for (int ky = 0; ky < 3; ky++) {
        for (int kx = 0; kx < 3; kx++) {
          const float* src = prev + ((size_t(y + ky) * Wp + (x + kx))) * cin;
          float* d = dst + (size_t(ky) * 3 + kx) * cin;
          if (relu) {
            for (int c = 0; c < cin; c++) d[c] = src[c] > 0.f ? src[c] : 0.f;
          } else {
            std::memcpy(d, src, size_t(cin) * sizeof(float));
          }
        }
      }
    }
  }
}

// C[M][N] = A[M][K] * B[K][N].  N is a multiple of 16.
static void gemm_forward(const float* A, const float* B, float* C, int M, int K, int N) {
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; m++) {
    const float* a = A + size_t(m) * K;
    float* c = C + size_t(m) * N;
    for (int n = 0; n < N; n += 16) {
      __m512 acc = _mm512_setzero_ps();
      for (int k = 0; k < K; k++) {
        const __m512 b = _mm512_loadu_ps(B + size_t(k) * N + n);
        acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]), b, acc);
      }
      _mm512_storeu_ps(c + n, acc);
    }
  }
}

// dW[k][c] += sum_m xcol[m][k] * dC[m][c]
static void conv_dw(const float* xcol, const float* dC, int M, int K, int N, float* dW) {
#pragma omp parallel for schedule(static)
  for (int k = 0; k < K; k++) {
    float* dw = dW + size_t(k) * N;
    for (int n = 0; n < N; n += 16) {
      __m512 acc = _mm512_loadu_ps(dw + n);  // accumulate across images in batch
      for (int m = 0; m < M; m++) {
        const float xv = xcol[size_t(m) * K + k];
        acc = _mm512_fmadd_ps(_mm512_set1_ps(xv), _mm512_loadu_ps(dC + size_t(m) * N + n), acc);
      }
      _mm512_storeu_ps(dw + n, acc);
    }
  }
}

// dX[m][ci] = sum_{tap, c} wt[c][tap*cin+ci] * dC[m][c]
// wt is [cout][k]; cin is a multiple of 16.
static void conv_dx(const float* dC, const ConvLayer& L, int M, float* dX) {
  const int cin = L.cin, cout = L.cout;
  const int K = L.k;
  std::fill(dX, dX + size_t(M) * cin, 0.f);
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; m++) {
    const float* dz = dC + size_t(m) * cout;
    float* dx = dX + size_t(m) * cin;
    for (int cb = 0; cb < cin; cb += 16) {
      for (int tap = 0; tap < 9; tap++) {
        const int koff = tap * cin + cb;
        __m512 acc = _mm512_setzero_ps();
        for (int c = 0; c < cout; c++) {
          const __m512 wv = _mm512_loadu_ps(&L.wt[size_t(c) * K + koff]);
          acc = _mm512_fmadd_ps(_mm512_set1_ps(dz[c]), wv, acc);
        }
        _mm512_storeu_ps(dx + cb, _mm512_add_ps(_mm512_loadu_ps(dx + cb), acc));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Batch norm (training)
// ---------------------------------------------------------------------------

// Normalise conv output in place (padded z buffer, valid region only) and
// record batch statistics.
static void bn_forward_inplace(float* z, int H, int W, int C, ConvLayer& L,
                               vector<float>& mean_out, vector<float>& invstd_out,
                               bool update_running = true) {
  const int M = H * W;
  const int Wp = W + 2;
  mean_out.assign(C, 0.f);
  vector<float> var(C, 0.f);
#pragma omp parallel for schedule(static)
  for (int c = 0; c < C; c++) {
    double s = 0, s2 = 0;
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        float v = z[(size_t(y + 1) * Wp + (x + 1)) * C + c];
        s += v;
        s2 += double(v) * v;
      }
    double mean = s / M;
    double varv = std::max(1e-8, s2 / M - mean * mean);
    mean_out[c] = float(mean);
    var[c] = float(varv);
    if (update_running) {
      L.run_mean[c] = 0.9f * L.run_mean[c] + 0.1f * float(mean);
      L.run_var[c] = 0.9f * L.run_var[c] + 0.1f * float(varv * M / std::max(1, M - 1));
    }
  }
#pragma omp parallel for schedule(static)
  for (int c = 0; c < C; c++) {
    const float inv = 1.f / std::sqrt(var[c] + 1e-5f);
    invstd_out[c] = inv;
    const float g = L.gamma[c], b = L.beta[c];
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        float& v = z[(size_t(y + 1) * Wp + (x + 1)) * C + c];
        v = (v - mean_out[c]) * inv * g + b;
      }
  }
}

// Given dZ (normalized-space grad) and stored z (= gamma*xhat+beta), produce
// dC (conv-output grad) and accumulate dgamma/dbeta.
static void bn_backward(const float* dZ, const float* z, int H, int W, int C,
                        ConvLayer& L, const vector<float>& mean, const vector<float>& invstd,
                        float* dC) {
  const int M = H * W;
  const int Wp = W + 2;
#pragma omp parallel for schedule(static)
  for (int c = 0; c < C; c++) {
    const float g = L.gamma[c];
    const float inv = invstd[c];
    const float beta = L.beta[c];
    // xhat = (z - beta) / gamma; dgamma = sum(dY * xhat); dbeta = sum(dY).
    double sdz = 0, sdzx = 0, dg = 0, db = 0;
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        size_t idx = (size_t(y + 1) * Wp + (x + 1)) * C + c;
        float dz = dZ[(size_t(y) * W + x) * C + c];
        float xhat = (z[idx] - beta) / g;
        sdz += dz;
        sdzx += dz * xhat;
        dg += dz * xhat;
        db += dz;
      }
    L.ggamma[c] += float(dg);
    L.gbeta[c] += float(db);
    const float mean_dz = float(sdz / M);
    const float mean_dzx = float(sdzx / M);
    const float dg_scaled = g;
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        size_t idx = (size_t(y + 1) * Wp + (x + 1)) * C + c;
        float xhat = (z[idx] - beta) / g;
        float d = dZ[(size_t(y) * W + x) * C + c] * dg_scaled;
        dC[(size_t(y) * W + x) * C + c] = inv * (d - dg_scaled * mean_dz - xhat * dg_scaled * mean_dzx);
      }
  }
}

// ---------------------------------------------------------------------------
// Whole-network forward for a batch
// ---------------------------------------------------------------------------

struct Batch {
  int B = 0;
  int H[4] = {32, 32, 16, 16};
  int W[4] = {32, 32, 16, 16};
  int C[4] = {32, 64, 128, 128};
  // padded activation buffers: z0 (input), z1..z4
  vector<float> z0, z1, z2, z3, z4, pool;
  vector<uint8_t> pool_arg;  // argmax within 2x2, per pool output
  vector<float> pooled;      // [B][2048]
  vector<float> logits;      // [B][10]
  vector<float> xcol[4];     // cached im2col for backward
};

static int padded_size(int H, int W, int C) { return (H + 2) * (W + 2) * C; }

static void alloc_batch(Batch& b, int B) {
  b.B = B;
  b.z0.assign(size_t(B) * padded_size(32, 32, 1), 0.f);
  b.z1.assign(size_t(B) * padded_size(32, 32, 32), 0.f);
  b.z2.assign(size_t(B) * padded_size(32, 32, 64), 0.f);
  b.pool.assign(size_t(B) * padded_size(16, 16, 64), 0.f);
  b.z3.assign(size_t(B) * padded_size(16, 16, 128), 0.f);
  b.z4.assign(size_t(B) * padded_size(16, 16, 128), 0.f);
  b.pool_arg.assign(size_t(B) * 16 * 16 * 64, 0);
  b.pooled.assign(size_t(B) * 2048, 0.f);
  b.logits.assign(size_t(B) * NC, 0.f);
  b.xcol[0].assign(size_t(B) * 32 * 32 * 9, 0.f);
  b.xcol[1].assign(size_t(B) * 32 * 32 * 288, 0.f);
  b.xcol[2].assign(size_t(B) * 16 * 16 * 576, 0.f);
  b.xcol[3].assign(size_t(B) * 16 * 16 * 1152, 0.f);
}

// conv into valid region of a padded z buffer.
static void conv_forward(const ConvLayer& L, const float* prev, int H, int W,
                         bool relu_prev, float* out_padded, int /*outH*/, int /*outW*/,
                         float* xcol) {
  const int M = H * W, K = L.k, C = L.cout;
  im2col(prev, H, W, L.cin, relu_prev, xcol);
  // temporary contiguous result, then scatter into padded layout
  vector<float> tmp(size_t(M) * C);
  gemm_forward(xcol, L.w.data(), tmp.data(), M, K, C);
  const int Wp = W + 2;
#pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++)
      std::memcpy(out_padded + ((size_t(y + 1) * Wp + (x + 1)) * C),
                  tmp.data() + (size_t(y) * W + x) * C, size_t(C) * sizeof(float));
}

static void pool_forward(const float* z2, int B, int /*batch_offset unused*/,
                         float* pool_out, uint8_t* arg) {
  const int H = 32, W = 32, C = 64;
  const int Wp = W + 2;
  // per image
#pragma omp parallel for collapse(2) schedule(static)
  for (int b = 0; b < B; b++)
    for (int c = 0; c < C; c++)
      for (int py = 0; py < 16; py++)
        for (int px = 0; px < 16; px++) {
          float best = -1e30f;
          int bi = 0;
          for (int j = 0; j < 4; j++) {
            int yy = 1 + py * 2 + (j >> 1);
            int xx = 1 + px * 2 + (j & 1);
            float v = z2[(size_t(b) * H * W * C) + (size_t(yy) * Wp + xx) * C + c];
            if (v > best) { best = v; bi = j; }
          }
          int yy = 1 + py * 2 + (bi >> 1);
          int xx = 1 + px * 2 + (bi & 1);
          size_t src = size_t(b) * padded_size(32, 32, 64) + (size_t(yy) * Wp + xx) * C + c;
          pool_out[(size_t(b) * padded_size(16, 16, 64)) + (size_t(py + 1) * 18 + (px + 1)) * C + c] = z2[src];
          arg[(size_t(b) * 16 * 16 + py * 16 + px) * C + c] = uint8_t(bi);
        }
}

static void head_forward(Batch& b, Model& m) {
  // grid 4x4 average pool of relu(z4), then linear.
#pragma omp parallel for schedule(static)
  for (int bi = 0; bi < b.B; bi++) {
    const float* z4 = b.z4.data() + size_t(bi) * padded_size(16, 16, 128);
    float* pe = b.pooled.data() + size_t(bi) * 2048;
    for (int qy = 0; qy < 4; qy++)
      for (int qx = 0; qx < 4; qx++)
        for (int c = 0; c < 128; c++) {
          float s = 0;
          for (int y = 0; y < 4; y++)
            for (int x = 0; x < 4; x++) {
              float v = z4[(size_t(qy * 4 + y + 1) * 18 + (qx * 4 + x + 1)) * 128 + c];
              s += v > 0 ? v : 0.f;
            }
          pe[((qy * 4 + qx) * 128) + c] = s * (1.f / 16.f);
        }
    for (int k = 0; k < NC; k++) {
      __m512 a2 = _mm512_setzero_ps();
      for (int j = 0; j < 2048; j += 16)
        a2 = _mm512_fmadd_ps(_mm512_loadu_ps(&m.head.w[size_t(k) * 2048 + j]),
                             _mm512_loadu_ps(pe + j), a2);
      float sum = _mm512_reduce_add_ps(a2) + m.head.b[k];
      b.logits[size_t(bi) * NC + k] = sum;
    }
  }
}

// Full forward for a batch of images (px = B*1024 floats, [0,1]).
static void forward_batch(Batch& b, Model& m, const float* px, bool train) {
  const int sz0 = padded_size(32, 32, 1), sz1 = padded_size(32, 32, 32);
  const int sz2 = padded_size(32, 32, 64);
  const int szp = padded_size(16, 16, 64);
  const int sz3 = padded_size(16, 16, 128), sz4 = sz3;
  // input buffers
  for (int bi = 0; bi < b.B; bi++) {
    float* z0 = b.z0.data() + size_t(bi) * sz0;
    std::fill(z0, z0 + sz0, 0.f);
    const float* src = px + size_t(bi) * 1024;
    for (int y = 0; y < 32; y++)
      for (int x = 0; x < 32; x++) z0[(size_t(y + 1) * 34 + (x + 1))] = src[y * 32 + x];
  }
  // conv1..4, per image (conv kernels operate on a single image's HxW field)
  for (int bi = 0; bi < b.B; bi++) {
    conv_forward(m.L[0], b.z0.data() + size_t(bi) * sz0, 32, 32, false,
                 b.z1.data() + size_t(bi) * sz1, 32, 32, b.xcol[0].data() + size_t(bi) * 32 * 32 * 9);
    bn_forward_inplace(b.z1.data() + size_t(bi) * sz1, 32, 32, 32, m.L[0],
                       m.bn_mean[0], m.bn_invstd[0], train);
  }
  for (int bi = 0; bi < b.B; bi++) {
    conv_forward(m.L[1], b.z1.data() + size_t(bi) * sz1, 32, 32, true,
                 b.z2.data() + size_t(bi) * sz2, 32, 32, b.xcol[1].data() + size_t(bi) * 32 * 32 * 288);
    bn_forward_inplace(b.z2.data() + size_t(bi) * sz2, 32, 32, 64, m.L[1],
                       m.bn_mean[1], m.bn_invstd[1], train);
  }
  pool_forward(b.z2.data(), b.B, 0, b.pool.data(), b.pool_arg.data());
  for (int bi = 0; bi < b.B; bi++) {
    conv_forward(m.L[2], b.pool.data() + size_t(bi) * szp, 16, 16, true,
                 b.z3.data() + size_t(bi) * sz3, 16, 16, b.xcol[2].data() + size_t(bi) * 16 * 16 * 576);
    bn_forward_inplace(b.z3.data() + size_t(bi) * sz3, 16, 16, 128, m.L[2],
                       m.bn_mean[2], m.bn_invstd[2], train);
  }
  for (int bi = 0; bi < b.B; bi++) {
    conv_forward(m.L[3], b.z3.data() + size_t(bi) * sz3, 16, 16, true,
                 b.z4.data() + size_t(bi) * sz4, 16, 16, b.xcol[3].data() + size_t(bi) * 16 * 16 * 1152);
    bn_forward_inplace(b.z4.data() + size_t(bi) * sz4, 16, 16, 128, m.L[3],
                       m.bn_mean[3], m.bn_invstd[3], train);
  }
  head_forward(b, m);
}

// ---------------------------------------------------------------------------
// Loss / gradient of the head
// ---------------------------------------------------------------------------

static float softmax_ce(const float* logits, int label, float* dlogits, float smooth = 0.05f) {
  float mx = -1e30f;
  for (int i = 0; i < NC; i++) mx = std::max(mx, logits[i]);
  double sum = 0;
  float p[NC];
  for (int i = 0; i < NC; i++) { p[i] = std::exp(logits[i] - mx); sum += p[i]; }
  for (int i = 0; i < NC; i++) p[i] = float(p[i] / sum);
  float loss = 0;
  for (int i = 0; i < NC; i++) {
    float t = (i == label) ? (1.f - smooth) : (smooth / (NC - 1));
    loss -= t * std::log(std::max(1e-9f, p[i]));
    dlogits[i] = p[i] - t;
  }
  return loss;
}

// ---------------------------------------------------------------------------
// Training step for one image (gradient accumulation)
// ---------------------------------------------------------------------------

static void backward_image(Batch& b, Model& m, int bi, int label, float* dpool,
                           float* dlogits, float* dz, float* dc, float* dx, float* dconv) {
  const float* pe = b.pooled.data() + size_t(bi) * 2048;
  float* gpe = dpool;  // [2048]
  // head gradients
  for (int k = 0; k < NC; k++) {
    for (int j = 0; j < 2048; j++) m.head.gw[size_t(k) * 2048 + j] += dlogits[k] * pe[j];
    m.head.gb[k] += dlogits[k];
  }
  for (int j = 0; j < 2048; j++) {
    float s = 0;
    for (int k = 0; k < NC; k++) s += m.head.w[size_t(k) * 2048 + j] * dlogits[k];
    gpe[j] = s;
  }

  const int sz2 = padded_size(32, 32, 64);
  const int sz3 = padded_size(16, 16, 128);
  const int sz4 = sz3;
  const int sz1 = padded_size(32, 32, 32);
  const int sz0 = padded_size(32, 32, 1);
  (void)sz0;
  float* z1 = b.z1.data() + size_t(bi) * sz1;
  float* z2 = b.z2.data() + size_t(bi) * sz2;
  float* z3 = b.z3.data() + size_t(bi) * sz3;
  float* z4 = b.z4.data() + size_t(bi) * sz4;

  // dZ4 = d(pooled)/d(z4) * relu'(z4)
  std::fill(dz, dz + size_t(16) * 16 * 128, 0.f);
  for (int qy = 0; qy < 4; qy++)
    for (int qx = 0; qx < 4; qx++)
      for (int c = 0; c < 128; c++) {
        float g = gpe[(qy * 4 + qx) * 128 + c] * (1.f / 16.f);
        for (int y = 0; y < 4; y++)
          for (int x = 0; x < 4; x++) {
            int yy = qy * 4 + y, xx = qx * 4 + x;
            float& d = dz[(size_t(yy) * 16 + xx) * 128 + c];
            d = g * (z4[(size_t(yy + 1) * 18 + (xx + 1)) * 128 + c] > 0 ? 1.f : 0.f);
          }
      }

  // conv4 backward
  bn_backward(dz, z4, 16, 16, 128, m.L[3], m.bn_mean[3], m.bn_invstd[3], dc);
  conv_dw(b.xcol[3].data() + size_t(bi) * 16 * 16 * 1152, dc, 256, 1152, 128, m.L[3].gw.data());
  conv_dx(dc, m.L[3], 256, dx);  // [256][128], grad wrt z3
  // relu'(z3)
  for (int y = 0; y < 16; y++)
    for (int x = 0; x < 16; x++)
      for (int c = 0; c < 128; c++)
        if (z3[(size_t(y + 1) * 18 + (x + 1)) * 128 + c] <= 0)
          dx[(size_t(y) * 16 + x) * 128 + c] = 0;

  // conv3 backward: dZ3 -> dC3 -> dW3, dPool
  bn_backward(dx, z3, 16, 16, 128, m.L[2], m.bn_mean[2], m.bn_invstd[2], dc);
  conv_dw(b.xcol[2].data() + size_t(bi) * 16 * 16 * 576, dc, 256, 576, 128, m.L[2].gw.data());
  conv_dx(dc, m.L[2], 256, dx);  // [256][64], grad wrt pool output

  // maxpool backward into dZ2 (pre-ReLU)
  std::fill(dz, dz + size_t(32) * 32 * 64, 0.f);
  for (int py = 0; py < 16; py++)
    for (int px = 0; px < 16; px++)
      for (int c = 0; c < 64; c++) {
        float pv = b.pool[(size_t(bi) * 16 * 16 * 64) + (size_t(py + 1) * 18 + (px + 1)) * 64 + c];
        if (pv <= 0) continue;
        int arg = b.pool_arg[(size_t(bi) * 16 * 16 + py * 16 + px) * 64 + c];
        int yy = 1 + py * 2 + (arg >> 1);
        int xx = 1 + px * 2 + (arg & 1);
        dz[(size_t(yy - 1) * 32 + (xx - 1)) * 64 + c] += dx[(size_t(py) * 16 + px) * 64 + c];
      }
  // relu'(z2)
  for (int y = 0; y < 32; y++)
    for (int x = 0; x < 32; x++)
      for (int c = 0; c < 64; c++)
        if (z2[(size_t(y + 1) * 34 + (x + 1)) * 64 + c] <= 0)
          dz[(size_t(y) * 32 + x) * 64 + c] = 0;

  // conv2 backward
  bn_backward(dz, z2, 32, 32, 64, m.L[1], m.bn_mean[1], m.bn_invstd[1], dc);
  conv_dw(b.xcol[1].data() + size_t(bi) * 32 * 32 * 288, dc, 1024, 288, 64, m.L[1].gw.data());
  conv_dx(dc, m.L[1], 1024, dx);
  for (int y = 0; y < 32; y++)
    for (int x = 0; x < 32; x++)
      for (int c = 0; c < 32; c++)
        if (z1[(size_t(y + 1) * 34 + (x + 1)) * 32 + c] <= 0)
          dx[(size_t(y) * 32 + x) * 32 + c] = 0;

  // conv1 backward
  bn_backward(dx, z1, 32, 32, 32, m.L[0], m.bn_mean[0], m.bn_invstd[0], dc);
  conv_dw(b.xcol[0].data() + size_t(bi) * 32 * 32 * 9, dc, 1024, 9, 32, m.L[0].gw.data());
  (void)dconv;
  (void)label;
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------

static void adam_vec(vector<float>& p, vector<float>& g, vector<float>& m, vector<float>& v,
                     float lr, float wd, float b1, float b2, float eps, int t, float gscale) {
  const float bc1 = 1.f - std::pow(b1, float(t));
  const float bc2 = 1.f - std::pow(b2, float(t));
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < p.size(); i++) {
    float gi = g[i] * gscale + wd * p[i];
    m[i] = b1 * m[i] + (1 - b1) * gi;
    v[i] = b2 * v[i] + (1 - b2) * gi * gi;
    p[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
    g[i] = 0;
  }
}

static void adam_model(Model& m, float lr, float wd, int t, float gscale) {
  for (int l = 0; l < 4; l++) {
    ConvLayer& L = m.L[l];
    adam_vec(L.w, L.gw, L.mw, L.vw, lr, wd, 0.9f, 0.999f, 1e-8f, t, gscale);
    adam_vec(L.gamma, L.ggamma, L.mg, L.vg, lr, 0.f, 0.9f, 0.999f, 1e-8f, t, gscale);
    adam_vec(L.beta, L.gbeta, L.mb, L.vb, lr, 0.f, 0.9f, 0.999f, 1e-8f, t, gscale);
    transpose_weights(L);
  }
  adam_vec(m.head.w, m.head.gw, m.head.mw, m.head.vw, lr, wd, 0.9f, 0.999f, 1e-8f, t, gscale);
  adam_vec(m.head.b, m.head.gb, m.head.mb, m.head.vb, lr, 0.f, 0.9f, 0.999f, 1e-8f, t, gscale);
}

// ---------------------------------------------------------------------------
// Fold BN into conv weights; dead-zone statistics
// ---------------------------------------------------------------------------

static void fold_bn(Model& m) {
  for (int l = 0; l < 4; l++) {
    ConvLayer& L = m.L[l];
    for (int c = 0; c < L.cout; c++) {
      float scale = L.gamma[c] / std::sqrt(L.run_var[c] + 1e-5f);
      for (int k = 0; k < L.k; k++) L.w[size_t(k) * L.cout + c] *= scale;
      L.bias[c] = L.beta[c] - L.run_mean[c] * scale;
    }
    // pack into [cout/16][k][16]
    for (int cb = 0; cb < L.cout / 16; cb++)
      for (int k = 0; k < L.k; k++)
        for (int lane = 0; lane < 16; lane++)
          L.wf[(size_t(cb) * L.k + k) * 16 + lane] = L.w[size_t(k) * L.cout + cb * 16 + lane];
  }
}

// ---------------------------------------------------------------------------
// Dense inference from folded weights (used for accuracy + dead rates)
// ---------------------------------------------------------------------------

struct DenseStats {
  double acc = 0;
  double dead[4] = {0, 0, 0, 0};
  double total_pre[4] = {0, 0, 0, 0};
  // oracle sparsity structure (ground truth, analysis only)
  double blk_dead[4] = {0, 0, 0, 0};
  double blk_total[4] = {0, 0, 0, 0};
  double pix_dead[4] = {0, 0, 0, 0};
  double pix_total[4] = {0, 0, 0, 0};
  double half_dead[4] = {0, 0, 0, 0};
  double half_total[4] = {0, 0, 0, 0};
};

struct InferBufs {
  vector<float> z0, z1, z2, pool, z3, z4;
  vector<uint8_t> pool_arg;
  vector<float> xcol;  // 1024 * 1152 scratch for dense im2col
};

static void alloc_infer(InferBufs& b) {
  b.z0.assign(padded_size(32, 32, 1), 0.f);
  b.z1.assign(padded_size(32, 32, 32), 0.f);
  b.z2.assign(padded_size(32, 32, 64), 0.f);
  b.pool.assign(padded_size(16, 16, 64), 0.f);
  b.z3.assign(padded_size(16, 16, 128), 0.f);
  b.z4.assign(padded_size(16, 16, 128), 0.f);
  b.pool_arg.assign(16 * 16 * 64, 0);
  b.xcol.assign(size_t(1024) * 1152, 0.f);
}

// Dense conv with folded weights.  Optional dead-zone counters and active map.
static void conv_dense(const ConvLayer& L, const float* prev, int H, int W, bool relu_prev,
                       float* out, double* dead_count, double* total_count, uint8_t* active_map,
                       float* xcol, double* blk_dead = nullptr, double* blk_total = nullptr,
                       double* pix_dead = nullptr, double* pix_total = nullptr,
                       double* half_dead = nullptr, double* half_total = nullptr) {
  const int Wp = W + 2;
  const int C = L.cout;
  const int K = L.k;
  const int M = H * W;
  im2col(prev, H, W, L.cin, relu_prev, xcol);
  double local_dead = 0, local_total = 0, l_blk_dead = 0, l_blk_total = 0, l_pix_dead = 0,
         l_pix_total = 0, l_half_dead = 0, l_half_total = 0;
#pragma omp parallel for schedule(static) reduction(+ : local_dead, local_total, l_blk_dead, l_blk_total, l_pix_dead, l_pix_total)
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      bool pixel_any = false;
      const float* a = xcol + (size_t(y) * W + x) * K;
      float* o = out + ((size_t(y + 1) * Wp + (x + 1)) * C);
      for (int cb = 0; cb < C / 16; cb++) {
        __m512 acc = _mm512_loadu_ps(&L.bias[cb * 16]);
        const float* wbase = &L.wf[(size_t(cb) * K) * 16];
        for (int k = 0; k < K; k++) {
          acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]), _mm512_loadu_ps(wbase + size_t(k) * 16), acc);
        }
        // dead-zone stats
        __mmask16 pos = _mm512_cmp_ps_mask(acc, _mm512_setzero_ps(), _CMP_GT_OQ);
        local_dead += 16 - _mm_popcnt_u32((unsigned)pos);
        local_total += 16;
        l_blk_total += 1.0;
        if (pos == 0) l_blk_dead += 1.0;
        l_half_total += 2.0;
        if ((pos & 0xff) == 0) l_half_dead += 1.0;
        if ((pos & 0xff00) == 0) l_half_dead += 1.0;
        else pixel_any = true;
        if (active_map) {
          for (int lane = 0; lane < 16; lane++) {
            if (!(pos & (1u << lane))) active_map[(size_t(y) * W + x) * C + cb * 16 + lane] = 0;
            else active_map[(size_t(y) * W + x) * C + cb * 16 + lane] = 1;
          }
        }
        __m512 r = _mm512_max_ps(acc, _mm512_setzero_ps());
        _mm512_storeu_ps(o + cb * 16, r);
      }
      l_pix_total += 1.0;
      if (!pixel_any) l_pix_dead += 1.0;
    }
  }
  if (dead_count) *dead_count += local_dead;
  if (total_count) *total_count += local_total;
  if (blk_dead) *blk_dead += l_blk_dead;
  if (blk_total) *blk_total += l_blk_total;
  if (pix_dead) *pix_dead += l_pix_dead;
  if (pix_total) *pix_total += l_pix_total;
  if (half_dead) *half_dead += l_half_dead;
  if (half_total) *half_total += l_half_total;
}


static void pool_dense(const float* z2, float* pool, uint8_t* arg) {
  const int C = 64, Wp = 34;
#pragma omp parallel for schedule(static)
  for (int py = 0; py < 16; py++)
    for (int px = 0; px < 16; px++)
      for (int c = 0; c < C; c++) {
        float best = -1e30f;
        int bi = 0;
        for (int j = 0; j < 4; j++) {
          int yy = 1 + py * 2 + (j >> 1);
          int xx = 1 + px * 2 + (j & 1);
          float v = z2[(size_t(yy) * Wp + xx) * C + c];
          if (v > best) { best = v; bi = j; }
        }
        int yy = 1 + py * 2 + (bi >> 1);
        int xx = 1 + px * 2 + (bi & 1);
        pool[(size_t(py + 1) * 18 + (px + 1)) * C + c] = z2[(size_t(yy) * Wp + xx) * C + c];
        arg[(py * 16 + px) * C + c] = uint8_t(bi);
      }
}

static void dense_forward(const Model& m, const float* px, InferBufs& b, float* logits,
                          DenseStats* stats, uint8_t* a2map = nullptr, uint8_t* a3map = nullptr,
                          uint8_t* a4map = nullptr) {
  std::fill(b.z0.begin(), b.z0.end(), 0.f);
  for (int y = 0; y < 32; y++)
    for (int x = 0; x < 32; x++) b.z0[(size_t(y + 1) * 34 + (x + 1))] = px[y * 32 + x];
  conv_dense(m.L[0], b.z0.data(), 32, 32, false, b.z1.data(),
             stats ? &stats->dead[0] : nullptr, stats ? &stats->total_pre[0] : nullptr, nullptr,
             b.xcol.data(), stats ? &stats->blk_dead[0] : nullptr,
             stats ? &stats->blk_total[0] : nullptr, stats ? &stats->pix_dead[0] : nullptr,
             stats ? &stats->pix_total[0] : nullptr, stats ? &stats->half_dead[0] : nullptr,
             stats ? &stats->half_total[0] : nullptr);
  conv_dense(m.L[1], b.z1.data(), 32, 32, true, b.z2.data(),
             stats ? &stats->dead[1] : nullptr, stats ? &stats->total_pre[1] : nullptr, a2map,
             b.xcol.data(), stats ? &stats->blk_dead[1] : nullptr,
             stats ? &stats->blk_total[1] : nullptr, stats ? &stats->pix_dead[1] : nullptr,
             stats ? &stats->pix_total[1] : nullptr, stats ? &stats->half_dead[1] : nullptr,
             stats ? &stats->half_total[1] : nullptr);
  pool_dense(b.z2.data(), b.pool.data(), b.pool_arg.data());
  conv_dense(m.L[2], b.pool.data(), 16, 16, true, b.z3.data(),
             stats ? &stats->dead[2] : nullptr, stats ? &stats->total_pre[2] : nullptr, a3map,
             b.xcol.data(), stats ? &stats->blk_dead[2] : nullptr,
             stats ? &stats->blk_total[2] : nullptr, stats ? &stats->pix_dead[2] : nullptr,
             stats ? &stats->pix_total[2] : nullptr, stats ? &stats->half_dead[2] : nullptr,
             stats ? &stats->half_total[2] : nullptr);
  conv_dense(m.L[3], b.z3.data(), 16, 16, true, b.z4.data(),
             stats ? &stats->dead[3] : nullptr, stats ? &stats->total_pre[3] : nullptr, a4map,
             b.xcol.data(), stats ? &stats->blk_dead[3] : nullptr,
             stats ? &stats->blk_total[3] : nullptr, stats ? &stats->pix_dead[3] : nullptr,
             stats ? &stats->pix_total[3] : nullptr, stats ? &stats->half_dead[3] : nullptr,
             stats ? &stats->half_total[3] : nullptr);
  // head
  float pe[2048];
  for (int qy = 0; qy < 4; qy++)
    for (int qx = 0; qx < 4; qx++)
      for (int c = 0; c < 128; c++) {
        float s = 0;
        for (int y = 0; y < 4; y++)
          for (int x = 0; x < 4; x++) {
            float v = b.z4[(size_t(qy * 4 + y + 1) * 18 + (qx * 4 + x + 1)) * 128 + c];
            s += v > 0 ? v : 0.f;
          }
        pe[(qy * 4 + qx) * 128 + c] = s * (1.f / 16.f);
      }
  for (int k = 0; k < NC; k++) {
    __m512 acc = _mm512_setzero_ps();
    for (int j = 0; j < 2048; j += 16)
      acc = _mm512_fmadd_ps(_mm512_loadu_ps(&m.head.w[size_t(k) * 2048 + j]),
                            _mm512_loadu_ps(pe + j), acc);
    logits[k] = _mm512_reduce_add_ps(acc) + m.head.b[k];
  }
}

// Pack channels into 8-lane fibers by greedy co-activation cosine affinity on dev.
static vector<vector<int>> cosine_perm(const Model& m, const ImageSet& dev, InferBufs& b) {
  struct Bits { vector<uint64_t> w; int cnt = 0; };
  vector<Bits> act[4];
  for (int l = 0; l < 4; l++) act[l].resize(m.L[l].cout);
  int pos = 0, stride = 2;
  for (int i = 0; i < dev.count; i++) {
    float lg[NC];
    dense_forward(m, dev.px.data() + size_t(i) * 1024, b, lg, nullptr);
    const float* zs[3] = { b.z2.data(), b.z3.data(), b.z4.data() };
    for (int li = 1; li <= 3; li++) {
      int H = (li == 1) ? 32 : 16, W = H, C = m.L[li].cout, Wp = W + 2;
      for (int y = 0; y < H; y += stride) for (int x = 0; x < W; x += stride) {
        for (int c = 0; c < C; c++) {
          if (zs[li - 1][(size_t(y + 1) * Wp + (x + 1)) * C + c] > 0) {
            Bits& B = act[li][c];
            if (int(B.w.size()) <= pos / 64) B.w.resize(pos / 64 + 1, 0);
            B.w[pos / 64] |= 1ull << (pos % 64); B.cnt++;
          }
        }
        pos++;
      }
    }
  }
  vector<vector<int>> perm(4);
  for (int i = 0; i < 4; i++) { perm[i].resize(m.L[i].cout); for (int j = 0; j < m.L[i].cout; j++) perm[i][j] = j; }
  const int nw = pos / 64 + 1;
  for (int l = 1; l <= 3; l++) {
    int C = m.L[l].cout;
    for (auto& B : act[l]) B.w.resize(nw, 0);
    vector<char> used(C, 0);
    vector<int> order; order.reserve(C);
    for (int g = 0; g < C / 8; g++) {
      int seed = -1, bestc = -1;
      for (int c = 0; c < C; c++) if (!used[c] && act[l][c].cnt > bestc) { bestc = act[l][c].cnt; seed = c; }
      used[seed] = 1; order.push_back(seed);
      for (int k = 1; k < 8; k++) {
        int pick = -1; double best = -1;
        const Bits& S = act[l][seed];
        for (int c = 0; c < C; c++) if (!used[c]) {
          uint64_t dot = 0;
          for (int w = 0; w < nw; w++) dot += __builtin_popcountll(S.w[w] & act[l][c].w[w]);
          double co = double(dot) / std::sqrt(double(S.cnt) * double(act[l][c].cnt) + 1e-9);
          if (co > best) { best = co; pick = c; }
        }
        used[pick] = 1; order.push_back(pick);
      }
    }
    perm[l] = std::move(order);
  }
  return perm;
}

// ---------------------------------------------------------------------------
// Weight file I/O
// ---------------------------------------------------------------------------

struct FileHeader {
  char magic[8];
  uint32_t version;
  uint32_t cin[4], cout[4];
  uint32_t epochs;
  float dead[4];
  float train_acc;
  uint32_t reserved[8];
};

static void save_weights(const string& path, const Model& m, const DenseStats& calib) {
  std::ofstream f(path, std::ios::binary);
  if (!f) die("cannot write " + path);
  FileHeader h{};
  std::memcpy(h.magic, "CMVGG04", 7);
  h.version = 1;
  int cin[4] = {1, 32, 64, 128}, cout[4] = {32, 64, 128, 128};
  for (int l = 0; l < 4; l++) {
    h.cin[l] = uint32_t(cin[l]);
    h.cout[l] = uint32_t(cout[l]);
    h.dead[l] = float(calib.dead[l] / std::max(1.0, calib.total_pre[l]));
  }
  f.write(reinterpret_cast<const char*>(&h), sizeof(h));
  for (int l = 0; l < 4; l++) {
    f.write(reinterpret_cast<const char*>(m.L[l].wf.data()),
            std::streamsize(m.L[l].wf.size() * sizeof(float)));
    f.write(reinterpret_cast<const char*>(m.L[l].bias.data()),
            std::streamsize(m.L[l].bias.size() * sizeof(float)));
  }
  f.write(reinterpret_cast<const char*>(m.head.w.data()),
          std::streamsize(m.head.w.size() * sizeof(float)));
  f.write(reinterpret_cast<const char*>(m.head.b.data()),
          std::streamsize(m.head.b.size() * sizeof(float)));
}

static void load_weights(const string& path, Model& m) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open " + path);
  FileHeader h{};
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (std::memcmp(h.magic, "CMVGG04", 7) != 0) die("bad weights magic");
  int cin[4] = {1, 32, 64, 128}, cout[4] = {32, 64, 128, 128};
  for (int l = 0; l < 4; l++) {
    ConvLayer& L = m.L[l];
    L.cin = cin[l];
    L.cout = cout[l];
    L.k = 9 * L.cin;
    L.wf.resize(size_t(L.cout) * L.k);
    L.bias.resize(L.cout);
    L.beta.assign(L.cout, 0.f);
    L.gamma.assign(L.cout, 1.f);
    f.read(reinterpret_cast<char*>(L.wf.data()), std::streamsize(L.wf.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(L.bias.data()), std::streamsize(L.bias.size() * sizeof(float)));
  }
  m.head.w.resize(size_t(NC) * 2048);
  m.head.b.resize(NC);
  f.read(reinterpret_cast<char*>(m.head.w.data()), std::streamsize(m.head.w.size() * sizeof(float)));
  f.read(reinterpret_cast<char*>(m.head.b.data()), std::streamsize(m.head.b.size() * sizeof(float)));
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

static void zero_grads(Model& m) {
  for (int l = 0; l < 4; l++) {
    std::fill(m.L[l].gw.begin(), m.L[l].gw.end(), 0.f);
    std::fill(m.L[l].ggamma.begin(), m.L[l].ggamma.end(), 0.f);
    std::fill(m.L[l].gbeta.begin(), m.L[l].gbeta.end(), 0.f);
  }
  std::fill(m.head.gw.begin(), m.head.gw.end(), 0.f);
  std::fill(m.head.gb.begin(), m.head.gb.end(), 0.f);
}

static int argmax10(const float* v) {
  int best = 0;
  for (int i = 1; i < NC; i++) if (v[i] > v[best]) best = i;
  return best;
}

static double eval_accuracy(Model& m, const ImageSet& set, int begin, int end, DenseStats* stats,
                            uint8_t* a2 = nullptr, uint8_t* a3 = nullptr, uint8_t* a4 = nullptr) {
  InferBufs b;
  alloc_infer(b);
  int correct = 0;
  for (int i = begin; i < end; i++) {
    float logits[NC];
    dense_forward(m, set.px.data() + size_t(i) * 1024, b, logits, stats, a2, a3, a4);
    if (argmax10(logits) == set.labels[i]) correct++;
  }
  return double(correct) / std::max(1, end - begin);
}

static void apply_dead_control(Model& m, const Batch& b, const double target[4], float eta) {
  for (int l = 0; l < 4; l++) {
    int H = b.H[l], W = b.W[l], C = b.C[l];
    const vector<float>& z = (l == 0) ? b.z1 : (l == 1) ? b.z2 : (l == 2) ? b.z3 : b.z4;
    const int Wp = W + 2;
    for (int c = 0; c < C; c++) {
      long dead = 0, tot = 0;
      for (int bi = 0; bi < b.B; bi++) {
        const float* zp = z.data() + size_t(bi) * padded_size(H, W, C);
        for (int y = 0; y < H; y++)
          for (int x = 0; x < W; x++) {
            dead += (zp[(size_t(y + 1) * Wp + (x + 1)) * C + c] <= 0.f);
            tot++;
          }
      }
      float rate = float(dead) / std::max(1L, tot);
      m.L[l].beta[c] -= eta * float(target[l] - rate);
    }
  }
}

static void train_command(int argc, char** argv) {
  string train_path, test_path, out_path, calib_path, init_path;
  int val_count = 200, epochs = 40, batch = 16;
  float lr = 2e-3f;
  float dead_eta = 0.f;
  double dead_target[4] = {0, 0, 0, 0};
  uint64_t seed = 12345;
  for (int i = 2; i < argc; i++) {
    string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) die("missing value for " + a); return string(argv[++i]); };
    if (a == "--train") train_path = next();
    else if (a == "--test") test_path = next();
    else if (a == "--out") out_path = next();
    else if (a == "--calib") calib_path = next();
    else if (a == "--val-count") val_count = std::stoi(next());
    else if (a == "--epochs") epochs = std::stoi(next());
    else if (a == "--batch") batch = std::stoi(next());
    else if (a == "--lr") lr = std::stof(next());
    else if (a == "--seed") seed = std::stoull(next());
    else if (a == "--init") init_path = next();
    else if (a == "--dead-eta") dead_eta = std::stof(next());
    else if (a == "--dead-targets") {
      string v = next();
      size_t p = 0;
      for (int l = 0; l < 4; l++) {
        size_t q = v.find(',', p);
        dead_target[l] = std::stod(v.substr(p, q == string::npos ? string::npos : q - p));
        p = (q == string::npos) ? v.size() : q + 1;
      }
    } else die("unknown arg " + a);
  }
  if (train_path.empty() || out_path.empty()) die("train requires --train and --out");
  ImageSet tr = load_images(train_path);
  ImageSet te;
  if (!test_path.empty()) te = load_images(test_path);
  if (val_count > tr.count) val_count = tr.count;

  Model m;
  init_model(m, seed);
  if (!init_path.empty()) {
    load_weights(init_path, m);
    for (int l = 0; l < 4; l++) {
      ConvLayer& L = m.L[l];
      for (int cb = 0; cb < L.cout / 16; cb++)
        for (int k = 0; k < L.k; k++)
          for (int lane = 0; lane < 16; lane++)
            L.w[size_t(k) * L.cout + cb * 16 + lane] = L.wf[(size_t(cb) * L.k + k) * 16 + lane];
      L.gamma.assign(L.cout, 1.f);
      L.beta = L.bias;
      L.run_mean.assign(L.cout, 0.f);
      L.run_var.assign(L.cout, 1.f);
    }
    std::printf("init from %s\n", init_path.c_str());
  }
  Batch b;
  alloc_batch(b, batch);

  std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ull);
  vector<int> order(tr.count);
  std::iota(order.begin(), order.end(), 0);
  std::vector<float> augment(1024);
  vector<float> dpool(2048), dlogits(NC), dz(size_t(32) * 32 * 128), dc(size_t(32) * 32 * 128),
      dx(size_t(32) * 32 * 128), dconv(1024);
  int step = 0;
  const float wd = 5e-4f;
  (void)calib_path;

  for (int ep = 0; ep < epochs; ep++) {
    std::shuffle(order.begin(), order.end(), rng);
    double loss_sum = 0;
    int seen = 0;
    for (int start = 0; start < tr.count; start += batch) {
      int B = std::min(batch, tr.count - start);
      b.B = B;  // buffers were allocated for max batch
      zero_grads(m);
      // build augmented batch
      vector<float> batch_px(size_t(B) * 1024);
      for (int i = 0; i < B; i++) {
        int idx = order[start + i];
        const float* src = tr.px.data() + size_t(idx) * 1024;
        int sdx = int(rng() % 7) - 3;
        int sdy = int(rng() % 7) - 3;
        float gain = 0.85f + 0.3f * float(rng() % 1000) / 1000.f;
        int cx = int(rng() % 32), cy = int(rng() % 32), cs = int(rng() % 6) + 5;
        bool cut = (rng() % 100) < 50;
        for (int y = 0; y < 32; y++)
          for (int x = 0; x < 32; x++) {
            int sy = y - sdy, sx = x - sdx;
            float v = 0;
            if (sy >= 0 && sy < 32 && sx >= 0 && sx < 32) v = src[sy * 32 + sx] * gain;
            if (cut && std::abs(x - cx) < cs && std::abs(y - cy) < cs) v = 0.f;
            batch_px[size_t(i) * 1024 + y * 32 + x] = v;
          }
      }
      forward_batch(b, m, batch_px.data(), true);
      // loss + backward per image
      for (int i = 0; i < B; i++) {
        int idx = order[start + i];
        float dlog[NC];
        float loss = softmax_ce(b.logits.data() + size_t(i) * NC, tr.labels[idx], dlog);
        loss_sum += loss;
        seen++;
        backward_image(b, m, i, tr.labels[idx], dpool.data(), dlog, dz.data(), dc.data(), dx.data(),
                       dconv.data());
      }
      step++;
      float lr_t = lr * 0.5f * (1.f + std::cos(float(M_PI) * float(ep) / float(epochs)));
      lr_t = std::max(lr_t, lr * 0.02f);
      // gradient scale: 1/B
      adam_model(m, lr_t, wd, step, 1.f / float(B));
      if (dead_eta > 0.f) apply_dead_control(m, b, dead_target, dead_eta);
    }
    // running BN stats are updated inside bn_forward? not yet; emulate from batch stats.
    double val_acc = 0;
    if (!te.count) {
      // no test set: use a fixed slice of train (leaky, only for progress)
      val_acc = eval_accuracy(m, tr, 0, std::min(val_count, tr.count), nullptr);
    }
    std::printf("epoch %d loss %.4f val %.4f\n", ep, loss_sum / std::max(1, seen), val_acc);
    std::fflush(stdout);
  }

  fold_bn(m);
  DenseStats stats{};
  if (te.count) {
    double acc = eval_accuracy(m, te, 0, te.count, &stats);
    std::printf("dense test acc %.4f dead %.4f %.4f %.4f %.4f\n", acc,
                stats.dead[0] / stats.total_pre[0], stats.dead[1] / stats.total_pre[1],
                stats.dead[2] / stats.total_pre[2], stats.dead[3] / stats.total_pre[3]);
    m.head.b[0] += 0;  // keep
    save_weights(out_path, m, stats);
    std::printf("saved %s\n", out_path.c_str());
  } else {
    save_weights(out_path, m, stats);
  }
}

// ---------------------------------------------------------------------------
// Online Context-Mixing gate
// ---------------------------------------------------------------------------
//
// State is a 16-bit probability + 16-bit observation count per
// (layer, channel, context-hash).  A context is a 12-bit integer assembled
// from CAUSAL quantities only:
//   bits 0..3  activation of W, N, NW, NE neighbours of the same channel
//   bits 4..5  margin bucket of the W neighbour (deep neg / near 0 / weak / strong)
//   bits 6..7  centre-pixel input-energy bucket
//   bits 8..9  activity of the preceding 4 output channels at the same pixel
//   bits 10..11 temporal: active at t-1 + 3x3 motion flag at t-1 (video only)
//
// Whole 16-channel AVX-512 blocks are skipped only when all 16 lanes agree.
// Stage B computes the centre 1x1xCin tap first (1/9 of the block MACs) and
// aborts the remaining 8 taps when the centre is below the online 99.5th
// percentile bound of the surrounding taps.

static constexpr int NHASH = 1 << 12;
static constexpr int NB = 8;  // max 16-channel blocks per layer (Conv3/4)
static constexpr int NHIST = 64;

enum GateMode {
  GM_DENSE = 0, GM_RANDOM = 1, GM_CHANNEL = 2, GM_SPATIAL = 3, GM_FULL = 4, GM_TEMPORAL = 5,
  // v2: vectorize across C_in, gate each output channel independently
  GM_INPUT_SPARSE = 6, GM_OUT_A = 7, GM_OUT_AB = 8, GM_DUAL = 9, GM_RANDOM_CH = 10,
  GM_TEMPORAL_DUAL = 11,
  // v3: keep v1 C_out-vectorised exact scalar-tap kernel, cluster channels so
  // whole 16-lane blocks become dead, gate blocks with a per-block CM table.
  GM_V3_A = 12, GM_V3_FULL = 13, GM_V3_TEMPORAL = 14, GM_BLOCK_RANDOM = 15,
  GM_V3_TAP_ONLY = 16, GM_V3_STAGEB_ONLY = 17, GM_RANDOM_UNC = 18,
  GM_RANDOM_DENSE = 19
};

struct GateConfig {
  int mode = GM_FULL;
  int th_skip = 900;    // < 2.75% predicted active -> hard skip
  int th_full = 16384;  // < 50% -> uncertain (centre-tap test eligible)
  uint32_t anchor_mask = 15;  // pixel anchor when ((y*W+x) & mask) == 0
  bool stage_b = true;
  float margin_scale = 1.25f;
  int min_count = 4;
  int min_hist = 256;
  float random_q[3] = {0.4f, 0.4f, 0.4f};
  float ch_target_skip[3] = {0.55f, 0.55f, 0.55f};
  float random_ch_q[3] = {0.5f, 0.5f, 0.5f};
  int th_skip_l[3] = {-1, -1, -1};  // per-layer override; -1 = use th_skip
  int th_block = 30000;             // v3: block p(all-dead) threshold (of 32768)
  int min_block_count = 8;
};

struct GateLayer {
  int H = 0, W = 0, Cin = 0, Cout = 0, K = 0;
  vector<uint32_t> tab;      // [NHASH][Cout], p in low 16, count in high 16
  vector<uint16_t> hist;     // [Cout][NHIST] of |surrounding taps|
  vector<uint32_t> nsamp;    // [Cout]
  vector<float> mb;          // [Cout] online bound
  vector<uint8_t> prev_act;  // [H*W*Cout]
  vector<uint8_t> prev_pix;  // [H*W]
  vector<float> wc;          // v2 prepacked [Cout][9][Cin], tap 0 = centre
  uint64_t full_blocks = 0, center_blocks = 0, skip_blocks = 0;
  uint64_t full_blocks_l[3] = {0, 0, 0}, center_blocks_l[3] = {0, 0, 0}, skip_blocks_l[3] = {0, 0, 0};
  uint64_t full_macs = 0, center_macs = 0;
  uint64_t out_evals = 0, out_skips = 0, stageb_aborts = 0;
  uint64_t anchor_pred_skip = 0, anchor_false_skip = 0;
  uint32_t img_seen = 0, img_skip = 0;
  // v3 per-block (16 output channels) online CM table and Stage-B ratio bound.
  vector<uint32_t> btab;   // [NB][NHASH], p(all-dead) low 16, count high 16
  vector<uint32_t> bhist;  // [NB][NHIST] histogram of |rest| / surround_L1
  vector<uint32_t> bnsamp; // [NB]
  vector<float> sb;        // [NB] online Stage-B bound on |rest| / L1
};

struct HeatMaps {
  vector<uint8_t> input;         // 32*32
  vector<uint16_t> act2, exe2;    // 32*32 x 64ch executed/active counts
  vector<uint16_t> act3, exe3;    // 16*16
  vector<uint16_t> act4, exe4;    // 16*16
};

struct Gate {
  GateLayer g[3];  // conv2, conv3, conv4
  GateConfig cfg;
  uint32_t imgs = 0;
  float e_mean[3] = {0.f, 0.f, 0.f};
  uint64_t rng = 0x243f6a8885a308d3ull;
  uint64_t executed_macs = 0, dense_macs = 0;
  uint64_t blocks_full = 0, blocks_center = 0, blocks_skip = 0;
  float ch_thr[3] = {0.05f, 0.05f, 0.05f};
};

static inline uint32_t xorshift64(uint64_t& s) {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return uint32_t(s >> 32);
}

static inline float rnd01(uint64_t& s) { return float(xorshift64(s) & 0xffffff) / float(0x1000000); }

static inline void tab_update(uint32_t& v, bool active) {
  int p = int(v & 0xffff);
  int cnt = int(v >> 16);
  p += ((active ? 32768 : 0) - p) >> 5;
  if (p < 0) p = 0;
  if (p > 32768) p = 32768;
  if (cnt < 65535) cnt++;
  v = (uint32_t(cnt) << 16) | uint32_t(p);
}

static void gate_init(Gate& gt, const GateConfig& cfg) {
  gt.cfg = cfg;
  gt.executed_macs = gt.dense_macs = 0;
  gt.blocks_full = gt.blocks_center = gt.blocks_skip = 0;
  gt.imgs = 0;
  int H[3] = {32, 32, 16}, W[3] = {32, 32, 16};
  int Cin[3] = {32, 64, 128}, Cout[3] = {64, 128, 128};
  for (int i = 0; i < 3; i++) {
    GateLayer& G = gt.g[i];
    G.H = H[i];
    G.W = W[i];
    G.Cin = Cin[i];
    G.Cout = Cout[i];
    G.K = 9 * Cin[i];
    G.btab.assign(size_t(NB) * NHASH, 16384u);
    G.bhist.assign(size_t(NB) * NHIST, 0);
    G.bnsamp.assign(NB, 0);
    G.sb.assign(NB, 1.0f);
    G.tab.assign(size_t(NHASH) * Cout[i], 0u);
    // optimistic prior: start at P(active)=0.5 so the first image is evaluated
    // densely and the table learns from data instead of starving itself.
    for (auto& v : G.tab) v = (0u << 16) | 16384u;
    G.hist.assign(size_t(Cout[i]) * NHIST, 0);
    G.nsamp.assign(Cout[i], 0);
    G.mb.assign(Cout[i], std::numeric_limits<float>::infinity());
    G.prev_act.assign(size_t(H[i]) * W[i] * Cout[i], 0);
    G.prev_pix.assign(size_t(H[i]) * W[i], 0);
    G.full_blocks = G.center_blocks = G.skip_blocks = 0;
    G.full_macs = G.center_macs = 0;
    G.out_evals = G.out_skips = G.stageb_aborts = 0;
    G.anchor_pred_skip = G.anchor_false_skip = 0;
    G.wc.clear();
    gt.e_mean[i] = 0.f;
  }
}


// v2: prepack folded weights as [Cout][9][Cin] with tap 0 = centre tap.
static void gate_prepack(Gate& gt, const Model& m) {
  static const int src_tap[9] = {4, 0, 1, 2, 3, 5, 6, 7, 8};
  for (int li = 0; li < 3; li++) {
    const ConvLayer& L = m.L[li + 1];
    GateLayer& G = gt.g[li];
    G.wc.assign(size_t(L.cout) * 9 * L.cin, 0.f);
    for (int c = 0; c < L.cout; c++) {
      const int cb = c / 16, lane = c % 16;
      for (int t = 0; t < 9; t++) {
        const float* src = &L.wf[(size_t(cb) * L.k + src_tap[t] * L.cin) * 16 + lane];
        float* dst = &G.wc[(size_t(c) * 9 + t) * L.cin];
        for (int ci = 0; ci < L.cin; ci++) dst[ci] = src[size_t(ci) * 16];
      }
    }
  }
}

static void gate_refresh_bounds(Gate& gt) {
  for (int i = 0; i < 3; i++) {
    GateLayer& G = gt.g[i];
    for (int c = 0; c < G.Cout; c++) {
      uint32_t n = G.nsamp[c];
      if (n < uint32_t(gt.cfg.min_hist)) {
        G.mb[c] = std::numeric_limits<float>::infinity();
        continue;
      }
      double target = 0.995 * double(n);
      double cum = 0;
      int bin = NHIST - 1;
      for (int b = 0; b < NHIST; b++) {
        cum += G.hist[size_t(c) * NHIST + b];
        if (cum >= target) { bin = b; break; }
      }
      G.mb[c] = (float(bin + 1) * 0.5f) * gt.cfg.margin_scale;
    }
  }
}

// v3 Stage-B bound: online high quantile of |surround| / L1(surround input).
static void gate_refresh_block_bounds(Gate& gt) {
  for (int i = 0; i < 3; i++) {
    GateLayer& G = gt.g[i];
    for (int b = 0; b < NB; b++) {
      uint64_t n = G.bnsamp[b];
      if (n < 2048) { G.sb[b] = 1.0f; continue; }
      double target = 0.999 * double(n);
      double cum = 0;
      int bin = NHIST - 1;
      for (int k = 0; k < NHIST; k++) {
        cum += G.bhist[size_t(b) * NHIST + k];
        if (cum >= target) { bin = k; break; }
      }
      float q = (float(bin) + 1.0f) / float(NHIST);
      G.sb[b] = std::min(1.0f, q * gt.cfg.margin_scale);
    }
  }
}

static inline int gate_cmp_bucket(float v, float mb) {
  if (v <= -mb) return 0;
  if (v < 0.f) return 1;
  if (v < mb) return 2;
  return 3;
}

// Hash for one channel.  z is the padded post-ReLU output of this layer
// (causal neighbours only).  Returns 12-bit context.
static inline int gate_hash(const GateLayer& G, const float* z, int y, int x, int c,
                            int patch_b, int chan_ctx, int temporal_bits) {
  const int Wp = G.W + 2;
  const int C = G.Cout;
  float w = z[(size_t(y + 1) * Wp + x) * C + c];
  float n = z[(size_t(y) * Wp + (x + 1)) * C + c];
  float nw = z[(size_t(y) * Wp + x) * C + c];
  float ne = z[(size_t(y) * Wp + (x + 2)) * C + c];
  int h = (w > 0.f) | ((n > 0.f) << 1) | ((nw > 0.f) << 2) | ((ne > 0.f) << 3);
  h |= gate_cmp_bucket(w, G.mb[c]) << 4;
  h |= (patch_b & 3) << 6;
  h |= (chan_ctx & 3) << 8;
  h |= (temporal_bits & 3) << 10;
  return h;
}

// Gated convolution for one layer.  `prev` is the padded input; `out` is the
// padded output (pre-zeroed by the caller).  Returns executed MACs.
static uint64_t gated_conv(Gate& gt, int li, const ConvLayer& L, const float* prev,
                           int H, int W, bool relu_prev, float* out,
                           uint16_t* act_map, uint16_t* exec_map) {
  GateLayer& G = gt.g[li];
  const int Wp = W + 2;
  const int C = L.cout;
  const int Cin = L.cin;
  const int K = L.k;
  const int md = gt.cfg.mode;
  const bool v3 = (md >= GM_V3_A);          // v3 loop layout / block bookkeeping
  const bool v3a = (md == GM_V3_A || md == GM_V3_FULL || md == GM_V3_TEMPORAL);
  const bool v3b = (md == GM_V3_FULL || md == GM_V3_TEMPORAL || md == GM_V3_STAGEB_ONLY);
  const bool temporal = (gt.cfg.mode == GM_TEMPORAL || gt.cfg.mode == GM_V3_TEMPORAL);
  const bool spatial_ctx = (gt.cfg.mode == GM_SPATIAL || gt.cfg.mode == GM_FULL ||
                            gt.cfg.mode == GM_TEMPORAL || gt.cfg.mode == GM_INPUT_SPARSE ||
                            gt.cfg.mode >= GM_OUT_A);
  const bool use_table = (gt.cfg.mode != GM_RANDOM && gt.cfg.mode < GM_V3_A);
  const bool tap_sparse = (gt.cfg.mode == GM_SPATIAL || gt.cfg.mode == GM_FULL ||
                           gt.cfg.mode == GM_TEMPORAL || gt.cfg.mode == GM_INPUT_SPARSE ||
                           (gt.cfg.mode >= GM_OUT_A && gt.cfg.mode != GM_RANDOM_DENSE));
  const bool channel_mode = (gt.cfg.mode == GM_CHANNEL);
  uint64_t macs = 0;

  int idx[1152];
  float vals[1152];
  int cidx[128];
  float cvals[128];

  for (int y = 0; y < H; y++) {
    int chan_ctx = 0;
    for (int x = 0; x < W; x++) {
      const float* pc = prev + ((size_t(y + 1) * Wp + (x + 1)) * Cin);
      // input-energy bucket for this pixel
      float e = 0.f;
      for (int ci = 0; ci < Cin; ci++) e += std::fabs(pc[ci]);
      e /= float(Cin);
      float& em = gt.e_mean[li];
      if (em <= 0.f) em = e;
      else em = 0.995f * em + 0.005f * e;
      int patch_b = 0;
      if (spatial_ctx) {
        float r = e / std::max(1e-6f, em);
        patch_b = (r < 0.15f) ? 0 : (r < 1.0f) ? 1 : (r < 3.0f) ? 2 : 3;
      }
      const bool anchor = (((y * W + x) & int(gt.cfg.anchor_mask)) == 0);
      const int motion = temporal ? (G.prev_pix[size_t(y) * W + x] ? 1 : 0) : 0;

      // Exact zero-tap compaction for the CM modes.
      int nz = 0, nzc = 0;
      float surround_l1 = 0.f;
      if (tap_sparse) {
        const __m512i lane_ids = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
        for (int tap = 0; tap < 9; tap++) {
          const int ky = tap / 3 - 1, kx = tap % 3 - 1;
          const float* pp = prev + ((size_t(y + 1 + ky) * Wp + (x + 1 + kx)) * Cin);
          const int base = tap * Cin;
          int* dst_i = (tap == 4) ? cidx : idx;
          float* dst_v = (tap == 4) ? cvals : vals;
          int& nn = (tap == 4) ? nzc : nz;
          for (int ci = 0; ci < Cin; ci += 16) {
            __m512 v = _mm512_loadu_ps(pp + ci);
            __mmask16 m = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_NEQ_OQ);
            _mm512_mask_compressstoreu_ps(dst_v + nn, m, v);
            __m512i iv = _mm512_add_epi32(_mm512_set1_epi32(base + ci), lane_ids);
            _mm512_mask_compressstoreu_epi32(dst_i + nn, m, iv);
            nn += __builtin_popcount(unsigned(m));
          }
        }
      }
      if (v3) {
        for (int t = 0; t < nz; t++) surround_l1 += std::fabs(vals[t]);
      }

      for (int cb = 0; cb < C / 16; cb++) {
        const int c0 = cb * 16;
        G.img_seen++;
        int hs[16];
        int ps[16];
        int cnts[16];
        bool all_hard = !anchor;
        bool finite_all = true;
        if (use_table) {
          const size_t Wpc = size_t(W + 2) * C;
          const float* zw = out + (size_t(y + 1) * Wpc) + size_t(x) * C + c0;
          const float* zn = out + (size_t(y) * Wpc) + size_t(x + 1) * C + c0;
          const float* znw = out + (size_t(y) * Wpc) + size_t(x) * C + c0;
          const float* zne = out + (size_t(y) * Wpc) + size_t(x + 2) * C + c0;
          __m512 wv = _mm512_loadu_ps(zw);
          __m512 nv = _mm512_loadu_ps(zn);
          __m512 nwv = _mm512_loadu_ps(znw);
          __m512 nev = _mm512_loadu_ps(zne);
          __m512 mbv = _mm512_loadu_ps(&G.mb[c0]);
          __m512 negmb = _mm512_sub_ps(_mm512_setzero_ps(), mbv);
          __mmask16 mw = _mm512_cmp_ps_mask(wv, _mm512_setzero_ps(), _CMP_GT_OQ);
          __mmask16 mn = _mm512_cmp_ps_mask(nv, _mm512_setzero_ps(), _CMP_GT_OQ);
          __mmask16 mnw = _mm512_cmp_ps_mask(nwv, _mm512_setzero_ps(), _CMP_GT_OQ);
          __mmask16 mne = _mm512_cmp_ps_mask(nev, _mm512_setzero_ps(), _CMP_GT_OQ);
          __mmask16 b0 = _mm512_cmp_ps_mask(wv, negmb, _CMP_LE_OQ);
          __mmask16 b1 = _mm512_cmp_ps_mask(wv, _mm512_setzero_ps(), _CMP_LT_OQ);
          __mmask16 b2 = _mm512_cmp_ps_mask(wv, mbv, _CMP_LT_OQ);
          for (int lane = 0; lane < 16; lane++) {
            const int c = c0 + lane;
            int h = spatial_ctx ? int(((mw >> lane) & 1) | (((mn >> lane) & 1) << 1) |
                                      (((mnw >> lane) & 1) << 2) | (((mne >> lane) & 1) << 3)) : 0;
            int bucket = ((b0 >> lane) & 1) ? 0 : ((b1 >> lane) & 1) ? 1 : ((b2 >> lane) & 1) ? 2 : 3;
            h |= bucket << 4;
            h |= (patch_b & 3) << 6;
            h |= (chan_ctx & 3) << 8;
            if (temporal) {
              int tb = G.prev_act[(size_t(y) * W + x) * C + c] ? 1 : 0;
              h |= (tb | (motion << 1)) << 10;
            }
            hs[lane] = h;
            uint32_t v = G.tab[size_t(h) * C + c];
            ps[lane] = int(v & 0xffff);
            cnts[lane] = int(v >> 16);
            bool hard = cnts[lane] >= gt.cfg.min_count && ps[lane] < gt.cfg.th_skip;
            if (!hard) all_hard = false;
            if (cnts[lane] < gt.cfg.min_count || ps[lane] < gt.cfg.th_full) finite_all = false;
          }
        }

        // Stage A: exact bypass when the whole 3x3 patch is zero.
        if (tap_sparse && nz == 0 && nzc == 0) {
          __m512 acc = _mm512_loadu_ps(&L.bias[c0]);
          if (use_table) {
            for (int lane = 0; lane < 16; lane++) {
              const int c = c0 + lane;
              bool active = ((float*)&acc)[lane] > 0.f;
              tab_update(G.tab[size_t(hs[lane]) * C + c], active);
            }
          }
          __m512 r = _mm512_max_ps(acc, _mm512_setzero_ps());
          _mm512_storeu_ps(out + ((size_t(y + 1) * Wp + (x + 1)) * C) + c0, r);
          if (act_map) {
            for (int lane = 0; lane < 16; lane++)
              if (((float*)&r)[lane] > 0.f) act_map[size_t(y) * W + x]++;
          }
          continue;
        }

        // v3: per-block (16-lane) CM decision.  The channel permutation puts
        // always-on channels in block 0 and sparse channels in blocks >= 1.
        int bh = 0;
        if (v3) {
          const size_t Wpc = size_t(W + 2) * C;
          const float* bw = out + (size_t(y + 1) * Wpc) + size_t(x) * C + c0;
          const float* bn = out + (size_t(y) * Wpc) + size_t(x + 1) * C + c0;
          const float* bnw = out + (size_t(y) * Wpc) + size_t(x) * C + c0;
          const float* bne = out + (size_t(y) * Wpc) + size_t(x + 2) * C + c0;
          auto bcount = [&](const float* p) -> int {
            __mmask16 m = _mm512_cmp_ps_mask(_mm512_loadu_ps(p), _mm512_setzero_ps(), _CMP_GT_OQ);
            return __builtin_popcount(unsigned(m));
          };
          bh = act_bucket(bcount(bw)) | (act_bucket(bcount(bn)) << 2) |
               (act_bucket(bcount(bnw)) << 4) | ((patch_b & 3) << 6);
          if (temporal) {
            int any = 0;
            for (int lane = 0; lane < 16; lane++)
              any |= G.prev_act[(size_t(y) * W + x) * C + c0 + lane];
            bh |= any << 8;
            bh |= motion << 9;
          }
          bool bskip = false;
          if (md == GM_BLOCK_RANDOM || md == GM_RANDOM_UNC) {
            if (!anchor && rnd01(gt.rng) < gt.cfg.random_ch_q[li]) bskip = true;
          } else if (v3a) {
            uint32_t bv = G.btab[size_t(cb) * NHASH + bh];
            int bp = int(bv & 0xffff), bc = int(bv >> 16);
            if (!anchor && bc >= gt.cfg.min_block_count && bp >= gt.cfg.th_block) bskip = true;
          }
          if (bskip) {
            gt.blocks_skip++;
            G.skip_blocks++;
            G.img_skip++;
            G.out_evals += 16;
            G.out_skips += 16;
            _mm512_storeu_ps(out + ((size_t(y + 1) * Wp + (x + 1)) * C) + c0,
                             _mm512_setzero_ps());
            continue;
          }
        }

        if (!v3 && use_table && !anchor && all_hard) {
          gt.blocks_skip++;
          G.skip_blocks++;
          G.img_skip++;
          _mm512_storeu_ps(out + ((size_t(y + 1) * Wp + (x + 1)) * C) + c0, _mm512_setzero_ps());
          continue;
        }
        if (gt.cfg.mode == GM_RANDOM && !anchor) {
          if (rnd01(gt.rng) > gt.cfg.random_q[li]) {
            gt.blocks_skip++;
            G.skip_blocks++;
            G.img_skip++;
            _mm512_storeu_ps(out + ((size_t(y + 1) * Wp + (x + 1)) * C) + c0, _mm512_setzero_ps());
            continue;
          }
        }
        if (channel_mode && !anchor) {
          float meanp = 0.f;
          for (int lane = 0; lane < 16; lane++) meanp += float(ps[lane]);
          meanp /= (16.f * 32768.f);
          if (meanp < gt.ch_thr[li]) {
            gt.blocks_skip++;
            G.skip_blocks++;
            G.img_skip++;
            _mm512_storeu_ps(out + ((size_t(y + 1) * Wp + (x + 1)) * C) + c0, _mm512_setzero_ps());
            continue;
          }
        }

        if (md == GM_RANDOM_DENSE && !anchor &&
            rnd01(gt.rng) < gt.cfg.random_ch_q[li]) {
          gt.blocks_skip++;
          G.skip_blocks++;
          G.img_skip++;
          _mm512_storeu_ps(out + ((size_t(y + 1) * Wp + (x + 1)) * C) + c0,
                           _mm512_setzero_ps());
          continue;
        }

        __m512 acc = _mm512_loadu_ps(&L.bias[c0]);
        __m512 center;
        uint64_t center_macs = 0;

        if (tap_sparse) {
          for (int t = 0; t < nzc; t++)
            acc = _mm512_fmadd_ps(_mm512_set1_ps(cvals[t]),
                                  _mm512_loadu_ps(&L.wf[(size_t(cb) * K + cidx[t]) * 16]), acc);
          center_macs = 16ull * uint64_t(nzc);
          center = acc;
          bool aborted = false;
          if (gt.cfg.stage_b && !anchor &&
              (gt.cfg.mode == GM_FULL || gt.cfg.mode == GM_TEMPORAL)) {
            __m512 mbv = _mm512_loadu_ps(&G.mb[c0]);
            __mmask16 finite = _mm512_cmp_ps_mask(mbv, _mm512_set1_ps(1e29f), _CMP_LT_OQ);
            if (finite == 0xffff) {
              __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), mbv);
              __mmask16 below = _mm512_cmp_ps_mask(acc, neg, _CMP_LT_OQ);
              if (below == 0xffff) aborted = true;
            }
          }
          if (!aborted && v3b && gt.cfg.stage_b &&
              (md == GM_V3_STAGEB_ONLY || !anchor) && surround_l1 > 1e-6f) {
            // Split-tap Stage B: the centre tap alone already puts every lane
            // below -(online |rest|/L1 bound).  Abort taps 1..8.
            const float bound = G.sb[cb] * surround_l1;
            if (bound > 0.f &&
                _mm512_cmp_ps_mask(acc, _mm512_set1_ps(-bound), _CMP_LE_OQ) == 0xffff)
              aborted = true;
          }
          if (aborted) {
            macs += center_macs;
            gt.blocks_center++;
            G.center_blocks++;
            G.center_macs += center_macs;
            G.stageb_aborts++;
            G.out_evals += 16;
            G.out_skips += 16;
            if (use_table)
              for (int lane = 0; lane < 16; lane++)
                tab_update(G.tab[size_t(hs[lane]) * C + c0 + lane], false);
            _mm512_storeu_ps(out + ((size_t(y + 1) * Wp + (x + 1)) * C) + c0, _mm512_setzero_ps());
            continue;
          }
          __m512 a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
          int t = 0;
          for (; t + 4 <= nz; t += 4) {
            acc = _mm512_fmadd_ps(_mm512_set1_ps(vals[t]),
                                  _mm512_loadu_ps(&L.wf[(size_t(cb) * K + idx[t]) * 16]), acc);
            a1 = _mm512_fmadd_ps(_mm512_set1_ps(vals[t + 1]),
                                 _mm512_loadu_ps(&L.wf[(size_t(cb) * K + idx[t + 1]) * 16]), a1);
            a2 = _mm512_fmadd_ps(_mm512_set1_ps(vals[t + 2]),
                                 _mm512_loadu_ps(&L.wf[(size_t(cb) * K + idx[t + 2]) * 16]), a2);
            a3 = _mm512_fmadd_ps(_mm512_set1_ps(vals[t + 3]),
                                 _mm512_loadu_ps(&L.wf[(size_t(cb) * K + idx[t + 3]) * 16]), a3);
          }
          for (; t < nz; t++)
            acc = _mm512_fmadd_ps(_mm512_set1_ps(vals[t]),
                                  _mm512_loadu_ps(&L.wf[(size_t(cb) * K + idx[t]) * 16]), acc);
          acc = _mm512_add_ps(acc, _mm512_add_ps(a1, _mm512_add_ps(a2, a3)));
          macs += 16ull * uint64_t(nz + nzc);
        } else {
          // dense-tap baseline kernel (no exact zero-tap bypass)
          const float* wc = &L.wf[(size_t(cb) * K + 4 * Cin) * 16];
          for (int ci = 0; ci < Cin; ci++) {
            const float v = relu_prev && pc[ci] <= 0.f ? 0.f : pc[ci];
            acc = _mm512_fmadd_ps(_mm512_set1_ps(v), _mm512_loadu_ps(wc + size_t(ci) * 16), acc);
          }
          center = acc;
          bool aborted = false;
          if (gt.cfg.stage_b && !anchor &&
              (gt.cfg.mode == GM_FULL || gt.cfg.mode == GM_TEMPORAL)) {
            __m512 mbv = _mm512_loadu_ps(&G.mb[c0]);
            __mmask16 finite = _mm512_cmp_ps_mask(mbv, _mm512_set1_ps(1e29f), _CMP_LT_OQ);
            if (finite == 0xffff) {
              __m512 neg = _mm512_sub_ps(_mm512_setzero_ps(), mbv);
              __mmask16 below = _mm512_cmp_ps_mask(acc, neg, _CMP_LT_OQ);
              if (below == 0xffff) aborted = true;
            }
          }
          if (aborted) {
            macs += 16ull * Cin;
            gt.blocks_center++;
            G.center_blocks++;
            G.center_macs += 16ull * Cin;
            _mm512_storeu_ps(out + ((size_t(y + 1) * Wp + (x + 1)) * C) + c0, _mm512_setzero_ps());
            continue;
          }
          for (int tap = 0; tap < 9; tap++) {
            if (tap == 4) continue;
            const int ky = tap / 3 - 1, kx = tap % 3 - 1;
            const float* pp = prev + ((size_t(y + 1 + ky) * Wp + (x + 1 + kx)) * Cin);
            const float* wp = &L.wf[(size_t(cb) * K + tap * Cin) * 16];
            for (int ci = 0; ci < Cin; ci++) {
              const float xv = relu_prev && pp[ci] <= 0.f ? 0.f : pp[ci];
              acc = _mm512_fmadd_ps(_mm512_set1_ps(xv), _mm512_loadu_ps(wp + size_t(ci) * 16), acc);
            }
          }
          macs += 16ull * K;
        }

        gt.blocks_full++;
        G.full_blocks++;
        G.full_macs += 16ull * (tap_sparse ? uint64_t(nz + nzc) : uint64_t(K));
        if (use_table) {
          for (int lane = 0; lane < 16; lane++) {
            const int c = c0 + lane;
            float zv = ((float*)&acc)[lane];
            bool active = zv > 0.f;
            tab_update(G.tab[size_t(hs[lane]) * C + c], active);
            if (nz > 0 || !tap_sparse) {
              float rest = zv - ((float*)&center)[lane];
              int bin = int(std::fabs(rest) * 2.f);
              if (bin >= NHIST) bin = NHIST - 1;
              if (G.hist[size_t(c) * NHIST + bin] < 65535) G.hist[size_t(c) * NHIST + bin]++;
              if (G.nsamp[c] < 0xffffffffu) G.nsamp[c]++;
            }
          }
        }
        __m512 r = _mm512_max_ps(acc, _mm512_setzero_ps());
        _mm512_storeu_ps(out + ((size_t(y + 1) * Wp + (x + 1)) * C) + c0, r);
        if (v3) {
          const __mmask16 alive = _mm512_cmp_ps_mask(r, _mm512_setzero_ps(), _CMP_GT_OQ);
          const bool all_dead = (alive == 0);
          tab_update(G.btab[size_t(cb) * NHASH + bh], all_dead);
          if (surround_l1 > 1e-6f && anchor) {
            for (int lane = 0; lane < 16; lane++) {
              float rest = ((float*)&acc)[lane] - ((float*)&center)[lane];
              float ratio = std::fabs(rest) / surround_l1;
              int bin = int(ratio * float(NHIST - 1) + 0.5f);
              if (bin < 0) bin = 0;
              if (bin >= NHIST) bin = NHIST - 1;
              if (G.bhist[size_t(cb) * NHIST + bin] < 0xffffffffu)
                G.bhist[size_t(cb) * NHIST + bin]++;
            }
            G.bnsamp[cb] += 16;
          }
          G.out_evals += 16;
          G.out_skips += all_dead ? 16 : 0;
        }
        if (exec_map) {
          // approximate per-pixel executed channel count (blocks are all-or-nothing)
          if (tap_sparse)
            exec_map[size_t(y) * W + x] += uint16_t(std::min(65535, nz + nzc));
          else exec_map[size_t(y) * W + x] += 16;
        }
        if (act_map) {
          for (int lane = 0; lane < 16; lane++)
            if (((float*)&r)[lane] > 0.f) act_map[size_t(y) * W + x]++;
        }
        {
          int a = 0;
          for (int lane = 12; lane < 16; lane++) if (((float*)&r)[lane] > 0.f) a++;
          chan_ctx = a >= 3 ? 3 : a;
        }
      }
    }
  }

  if (temporal) {
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        int any = 0;
        for (int c = 0; c < C; c++) {
          float v = out[(size_t(y + 1) * Wp + (x + 1)) * C + c];
          uint8_t a = v > 0.f ? 1 : 0;
          G.prev_act[(size_t(y) * W + x) * C + c] = a;
          any |= a;
        }
        G.prev_pix[size_t(y) * W + x] = uint8_t(any);
      }
  }
  return macs;
}


// ---------------------------------------------------------------------------
// v2 kernel: AVX-512 over C_in, one scalar output channel at a time.
// Stage A skips a whole (y,x,c_out) using the online CM table; Stage B runs the
// centre 1x1xC_in tap and aborts taps 1..8 against the online surround bound;
// Stage C (dual modes) uses vector-level compaction of all-zero C_in chunks.
// ---------------------------------------------------------------------------
static uint64_t gated_conv_v2(Gate& gt, int li, const ConvLayer& L, const float* prev,
                              int H, int W, bool relu_prev, float* out,
                              uint16_t* act_map, uint16_t* exec_map) {
  GateLayer& G = gt.g[li];
  const int Wp = W + 2;
  const int C = L.cout, Cin = L.cin, Cv = Cin / 16;
  const int mode = gt.cfg.mode;
  const bool temporal = (mode == GM_TEMPORAL_DUAL);
  const bool compact = (mode == GM_DUAL || mode == GM_TEMPORAL_DUAL);
  const bool stage_b = gt.cfg.stage_b && (mode == GM_OUT_AB || mode == GM_DUAL || mode == GM_TEMPORAL_DUAL);
  const bool use_table = (mode != GM_RANDOM_CH);
  const bool random_mode = (mode == GM_RANDOM_CH);
  static const int oy[9] = {0, -1, -1, -1, 0, 0, 1, 1, 1};
  static const int ox[9] = {0, -1, 0, 1, -1, 1, -1, 0, 1};
  uint64_t macs = 0;

  int hbase[128];
  int vidx[9][8];
  int nvec[9];

  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      const float* pc = prev + ((size_t(y + 1) * Wp + (x + 1)) * Cin);

      // patch energy bucket (vectorised over C_in)
      __m512 es = _mm512_setzero_ps();
      const __m512i absmask = _mm512_set1_epi32(0x7fffffff);
      for (int k = 0; k < Cv; k++) {
        __m512 v = _mm512_loadu_ps(pc + k * 16);
        es = _mm512_add_ps(es, _mm512_castsi512_ps(
            _mm512_and_si512(_mm512_castps_si512(v), absmask)));
      }
      const float e = _mm512_reduce_add_ps(es) / float(Cin);
      float& em = gt.e_mean[li];
      if (em <= 0.f) em = e;
      else em = 0.995f * em + 0.005f * e;
      const float r = e / std::max(1e-6f, em);
      const int patch_b = (r < 0.15f) ? 0 : (r < 1.0f) ? 1 : (r < 3.0f) ? 2 : 3;
      const int motion = temporal ? (G.prev_pix[size_t(y) * W + x] ? 1 : 0) : 0;

      // 12-bit context base for every channel (vectorised over 16 channels)
      for (int cb = 0; cb < C / 16; cb++) {
        const size_t Wpc = size_t(W + 2) * C;
        const float* zw = out + size_t(y + 1) * Wpc + size_t(x) * C + cb * 16;
        const float* zn = out + size_t(y) * Wpc + size_t(x + 1) * C + cb * 16;
        const float* znw = out + size_t(y) * Wpc + size_t(x) * C + cb * 16;
        const float* zne = out + size_t(y) * Wpc + size_t(x + 2) * C + cb * 16;
        __m512 wv = _mm512_loadu_ps(zw), nv = _mm512_loadu_ps(zn);
        __m512 nwv = _mm512_loadu_ps(znw), nev = _mm512_loadu_ps(zne);
        __m512 mbv = _mm512_loadu_ps(&G.mb[cb * 16]);
        __m512 negmb = _mm512_sub_ps(_mm512_setzero_ps(), mbv);
        __mmask16 mw = _mm512_cmp_ps_mask(wv, _mm512_setzero_ps(), _CMP_GT_OQ);
        __mmask16 mn = _mm512_cmp_ps_mask(nv, _mm512_setzero_ps(), _CMP_GT_OQ);
        __mmask16 mnw = _mm512_cmp_ps_mask(nwv, _mm512_setzero_ps(), _CMP_GT_OQ);
        __mmask16 mne = _mm512_cmp_ps_mask(nev, _mm512_setzero_ps(), _CMP_GT_OQ);
        __mmask16 b0 = _mm512_cmp_ps_mask(wv, negmb, _CMP_LE_OQ);
        __mmask16 b1 = _mm512_cmp_ps_mask(wv, _mm512_setzero_ps(), _CMP_LT_OQ);
        __mmask16 b2 = _mm512_cmp_ps_mask(wv, mbv, _CMP_LT_OQ);
        for (int lane = 0; lane < 16; lane++) {
          const int c = cb * 16 + lane;
          int h = int(((mw >> lane) & 1) | (((mn >> lane) & 1) << 1) |
                      (((mnw >> lane) & 1) << 2) | (((mne >> lane) & 1) << 3));
          h |= (((b0 >> lane) & 1) ? 0 : ((b1 >> lane) & 1) ? 1 : ((b2 >> lane) & 1) ? 2 : 3) << 4;
          h |= (patch_b & 3) << 6;
          if (temporal) {
            int tb = G.prev_act[(size_t(y) * W + x) * C + c] ? 1 : 0;
            h |= (tb | (motion << 1)) << 10;
          }
          hbase[c] = h;
        }
      }

      // vector-level non-zero compaction per tap
      if (compact) {
        for (int t = 0; t < 9; t++) {
          const float* pp = prev + ((size_t(y + 1 + oy[t]) * Wp + (x + 1 + ox[t])) * Cin);
          int nn = 0;
          for (int k = 0; k < Cv; k++) {
            __m512 v = _mm512_loadu_ps(pp + k * 16);
            __mmask16 m = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_NEQ_UQ);
            if (m) vidx[t][nn++] = k;
          }
          nvec[t] = nn;
        }
      }

      int last4 = 0;
      for (int c = 0; c < C; c++) {
        const bool anchor = ((((y * W + x) ^ c) & int(gt.cfg.anchor_mask)) == 0);
        int h = hbase[c] | ((__builtin_popcount(unsigned(last4 & 15)) & 3) << 8);
        uint32_t& slot = G.tab[size_t(h) * C + c];
        const int p = int(slot & 0xffff), cnt = int(slot >> 16);
        bool skip = false;
        const int thl = gt.cfg.th_skip_l[li] >= 0 ? gt.cfg.th_skip_l[li] : gt.cfg.th_skip;
        const bool pred_dead = use_table && cnt >= gt.cfg.min_count && p < thl;
        if (pred_dead && anchor) G.anchor_pred_skip++;
        if (pred_dead && !anchor) skip = true;
        if (random_mode && !anchor && rnd01(gt.rng) < gt.cfg.random_ch_q[li]) skip = true;
        G.out_evals++;
        if (skip) {
          G.out_skips++;
          out[((size_t(y + 1) * Wp + (x + 1)) * C) + c] = 0.f;
          last4 = (last4 << 1) & 15;
          continue;
        }

        // Stage B: centre tap first.
        const float* wc0 = &G.wc[(size_t(c) * 9 + 0) * Cin];
        __m512 acc = _mm512_setzero_ps();
        const int n0 = compact ? nvec[0] : Cv;
        for (int j = 0; j < n0; j++) {
          const int k = compact ? vidx[0][j] : j;
          acc = _mm512_fmadd_ps(_mm512_loadu_ps(pc + k * 16), _mm512_loadu_ps(wc0 + k * 16), acc);
        }
        macs += 16ull * uint64_t(n0);
        const float zc = _mm512_reduce_add_ps(acc) + L.bias[c];
        bool aborted = false;
        if (stage_b && !anchor && G.nsamp[c] >= uint32_t(gt.cfg.min_hist) &&
            G.mb[c] < 1e29f && zc < -G.mb[c])
          aborted = true;

        float z = 0.f;
        if (aborted) {
          G.stageb_aborts++;
        } else {
          __m512 a0 = acc, a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(),
                 a3 = _mm512_setzero_ps();
          for (int t = 1; t < 9; t++) {
            const float* wc = &G.wc[(size_t(c) * 9 + t) * Cin];
            const float* pp = prev + ((size_t(y + 1 + oy[t]) * Wp + (x + 1 + ox[t])) * Cin);
            const int nn = compact ? nvec[t] : Cv;
            for (int j = 0; j < nn; j++) {
              const int k = compact ? vidx[t][j] : j;
              __m512 vx = _mm512_loadu_ps(pp + k * 16);
              __m512 vw = _mm512_loadu_ps(wc + k * 16);
              switch (t & 3) {
                case 1: a1 = _mm512_fmadd_ps(vx, vw, a1); break;
                case 2: a2 = _mm512_fmadd_ps(vx, vw, a2); break;
                case 3: a3 = _mm512_fmadd_ps(vx, vw, a3); break;
                default: a0 = _mm512_fmadd_ps(vx, vw, a0); break;
              }
            }
            macs += 16ull * uint64_t(nn);
          }
          acc = _mm512_add_ps(a0, _mm512_add_ps(a1, _mm512_add_ps(a2, a3)));
          z = _mm512_reduce_add_ps(acc) + L.bias[c];
          const float rest = z - zc;
          int bin = int(std::fabs(rest) * 2.f);
          if (bin >= NHIST) bin = NHIST - 1;
          if (G.hist[size_t(c) * NHIST + bin] < 65535) G.hist[size_t(c) * NHIST + bin]++;
          if (G.nsamp[c] < 0xffffffffu) G.nsamp[c]++;
        }

        const float rz = z > 0.f ? z : 0.f;
        if (pred_dead && anchor && rz > 0.f) G.anchor_false_skip++;
        out[((size_t(y + 1) * Wp + (x + 1)) * C) + c] = rz;
        last4 = ((last4 << 1) | (rz > 0.f ? 1 : 0)) & 15;
        if (use_table) tab_update(slot, rz > 0.f);
        if (act_map && rz > 0.f) act_map[size_t(y) * W + x]++;
        if (exec_map) exec_map[size_t(y) * W + x]++;
      }
    }
  }

  if (temporal) {
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        int any = 0;
        for (int c = 0; c < C; c++) {
          float v = out[((size_t(y + 1) * Wp + (x + 1)) * C) + c];
          uint8_t a = v > 0.f ? 1 : 0;
          G.prev_act[(size_t(y) * W + x) * C + c] = a;
          any |= a;
        }
        G.prev_pix[size_t(y) * W + x] = uint8_t(any);
      }
  }
  G.full_macs += macs;
  return macs;
}

static void head_from_z4(const Model& m, const float* z4, float* logits) {
  float pe[2048];
  for (int qy = 0; qy < 4; qy++)
    for (int qx = 0; qx < 4; qx++)
      for (int c = 0; c < 128; c++) {
        float s = 0;
        for (int y = 0; y < 4; y++)
          for (int x = 0; x < 4; x++) {
            float v = z4[(size_t(qy * 4 + y + 1) * 18 + (qx * 4 + x + 1)) * 128 + c];
            s += v > 0 ? v : 0.f;
          }
        pe[(qy * 4 + qx) * 128 + c] = s * (1.f / 16.f);
      }
  for (int k = 0; k < NC; k++) {
    __m512 acc = _mm512_setzero_ps();
    for (int j = 0; j < 2048; j += 16)
      acc = _mm512_fmadd_ps(_mm512_loadu_ps(&m.head.w[size_t(k) * 2048 + j]),
                            _mm512_loadu_ps(pe + j), acc);
    logits[k] = _mm512_reduce_add_ps(acc) + m.head.b[k];
  }
}

// v3 debug: dense forward up to z4 (for permutation verification).
static void forward_z4(const Model& m, const float* px, InferBufs& b, float* z4out) {
  std::fill(b.z0.begin(), b.z0.end(), 0.f);
  for (int y = 0; y < 32; y++)
    for (int x = 0; x < 32; x++) b.z0[(size_t(y + 1) * 34 + (x + 1))] = px[y * 32 + x];
  conv_dense(m.L[0], b.z0.data(), 32, 32, false, b.z1.data(), nullptr, nullptr, nullptr, b.xcol.data());
  conv_dense(m.L[1], b.z1.data(), 32, 32, true, b.z2.data(), nullptr, nullptr, nullptr, b.xcol.data());
  pool_dense(b.z2.data(), b.pool.data(), b.pool_arg.data());
  conv_dense(m.L[2], b.pool.data(), 16, 16, true, b.z3.data(), nullptr, nullptr, nullptr, b.xcol.data());
  conv_dense(m.L[3], b.z3.data(), 16, 16, true, b.z4.data(), nullptr, nullptr, nullptr, b.xcol.data());
  for (int y = 0; y < 16; y++)
    for (int x = 0; x < 16; x++)
      for (int c = 0; c < 128; c++)
        z4out[(size_t(y) * 16 + x) * 128 + c] = b.z4[(size_t(y + 1) * 18 + (x + 1)) * 128 + c];
}

// One gated image (conv1 always dense).  Returns executed MACs.
static uint64_t gated_forward(const Model& m, const float* px, InferBufs& b, Gate& gt,
                              float* logits, HeatMaps* hm) {
  const int sz0 = padded_size(32, 32, 1), sz1 = padded_size(32, 32, 32);
  const int sz2 = padded_size(32, 32, 64), szp = padded_size(16, 16, 64);
  const int sz3 = padded_size(16, 16, 128), sz4 = sz3;
  for (int y = 0; y < 32; y++)
    for (int x = 0; x < 32; x++) b.z0[(size_t(y + 1) * 34 + (x + 1))] = px[y * 32 + x];
  (void)sz0; (void)sz1; (void)sz2; (void)szp; (void)sz3; (void)sz4;

  // conv1 dense (0.39% of MACs)
  conv_dense(m.L[0], b.z0.data(), 32, 32, false, b.z1.data(), nullptr, nullptr, nullptr,
             b.xcol.data());
  uint64_t macs = 294912ull;

  uint16_t* a2 = hm ? hm->act2.data() : nullptr;
  uint16_t* e2 = hm ? hm->exe2.data() : nullptr;
  uint16_t* a3 = hm ? hm->act3.data() : nullptr;
  uint16_t* e3 = hm ? hm->exe3.data() : nullptr;
  uint16_t* a4 = hm ? hm->act4.data() : nullptr;
  uint16_t* e4 = hm ? hm->exe4.data() : nullptr;
  if (hm) {
    std::fill(hm->act2.begin(), hm->act2.end(), 0);
    std::fill(hm->exe2.begin(), hm->exe2.end(), 0);
    std::fill(hm->act3.begin(), hm->act3.end(), 0);
    std::fill(hm->exe3.begin(), hm->exe3.end(), 0);
    std::fill(hm->act4.begin(), hm->act4.end(), 0);
    std::fill(hm->exe4.begin(), hm->exe4.end(), 0);
  }

  const bool v2 = gt.cfg.mode >= GM_OUT_A && gt.cfg.mode <= GM_TEMPORAL_DUAL;
  auto gconv = [&](int li, const ConvLayer& L, const float* pv, int H, int W, float* ov,
                   uint16_t* am, uint16_t* em) -> uint64_t {
    // per-channel output gating is applied to Conv4 only; earlier layers use the
    // exact tap-sparse kernel because zero-filling predicted-dead features there
    // cascades into large errors (see docs).
    const bool use_v2 = v2 && li == 2;
    return use_v2 ? gated_conv_v2(gt, li, L, pv, H, W, true, ov, am, em)
                  : gated_conv(gt, li, L, pv, H, W, true, ov, am, em);
  };
  macs += gconv(0, m.L[1], b.z1.data(), 32, 32, b.z2.data(), a2, e2);
  pool_dense(b.z2.data(), b.pool.data(), b.pool_arg.data());
  macs += gconv(1, m.L[2], b.pool.data(), 16, 16, b.z3.data(), a3, e3);
  macs += gconv(2, m.L[3], b.z3.data(), 16, 16, b.z4.data(), a4, e4);
  if (v2) gate_refresh_bounds(gt);
  if (gt.cfg.mode >= GM_V3_A) gate_refresh_block_bounds(gt);
  head_from_z4(m, b.z4.data(), logits);
  for (int i = 0; i < 3; i++) {
    GateLayer& G = gt.g[i];
    if (gt.cfg.mode == GM_CHANNEL && G.img_seen) {
      float rate = float(G.img_skip) / float(G.img_seen);
      float& thr = gt.ch_thr[i];
      if (rate < gt.cfg.ch_target_skip[i]) thr = std::min(0.9f, thr * 1.03f);
      else thr = std::max(0.001f, thr * 0.97f);
    }
    G.img_seen = G.img_skip = 0;
  }
  return macs;
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

static const char* kRamp = " .:-=+*#%@";

static string ascii_map(const uint16_t* v, int H, int W, int maxv) {
  string s;
  if (maxv <= 0) {
    for (int i = 0; i < H * W; i++) maxv = std::max(maxv, int(v[i]));
    if (maxv <= 0) maxv = 1;
  }
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      float f = maxv > 0 ? float(v[y * W + x]) / float(maxv) : 0.f;
      int idx = int(f * 9.f + 0.5f);
      if (idx < 0) idx = 0;
      if (idx > 9) idx = 9;
      s += kRamp[idx];
    }
    s += '\n';
  }
  return s;
}

static string ascii_u8(const uint8_t* v, int H, int W) {
  string s;
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      float f = float(v[y * W + x]) / 255.f;
      int idx = int(f * 9.99f);
      if (idx < 0) idx = 0;
      if (idx > 9) idx = 9;
      s += kRamp[idx];
    }
    s += '\n';
  }
  return s;
}

struct ModeResult {
  string name;
  double acc0 = 0, acc_final = 0;
  double pass0_ms = 0, median_ms = 0, best_ms = 0;
  double macs_per_image0 = 0, macs_per_image_final = 0;
  double elim0 = 0, elim_final = 0;
  uint64_t layer_macs[4] = {0, 0, 0, 0};
  uint64_t full_blocks = 0, center_blocks = 0, skip_blocks = 0;
  uint64_t full_blocks_l[3] = {0, 0, 0}, center_blocks_l[3] = {0, 0, 0}, skip_blocks_l[3] = {0, 0, 0};
  uint64_t out_channels = 0, out_skips = 0, stageb_aborts = 0;
  double out_skip_rate[3] = {0, 0, 0}, stageb_rate[3] = {0, 0, 0};
  double logit_rmse = 0, conv4_rel_l2 = 0, dense_agreement = 0;
  vector<double> pass_ms;
};

static double time_dense_pass(const Model& m, const ImageSet& set, InferBufs& b, int* correct,
                              DenseStats* stats) {
  double t0 = now_sec();
  int ok = 0;
  for (int i = 0; i < set.count; i++) {
    float logits[NC];
    dense_forward(m, set.px.data() + size_t(i) * 1024, b, logits, stats);
    if (argmax10(logits) == set.labels[i]) ok++;
  }
  double t1 = now_sec();
  if (correct) *correct = ok;
  return t1 - t0;
}

static double time_gated_pass(const Model& m, const ImageSet& set, InferBufs& b, Gate& gt,
                              int* correct, uint64_t* macs, uint64_t layer_macs[4]) {
  double t0 = now_sec();
  int ok = 0;
  uint64_t total = 0;
  uint64_t lm[4] = {294912, 0, 0, 0};
  for (int i = 0; i < set.count; i++) {
    const uint64_t before = gt.g[0].full_macs + gt.g[0].center_macs +
                            gt.g[1].full_macs + gt.g[1].center_macs +
                            gt.g[2].full_macs + gt.g[2].center_macs;
    float logits[NC];
    total += gated_forward(m, set.px.data() + size_t(i) * 1024, b, gt, logits, nullptr);
    const uint64_t after = gt.g[0].full_macs + gt.g[0].center_macs +
                           gt.g[1].full_macs + gt.g[1].center_macs +
                           gt.g[2].full_macs + gt.g[2].center_macs;
    (void)before;
    (void)after;
    if (argmax10(logits) == set.labels[i]) ok++;
  }
  double t1 = now_sec();
  if (correct) *correct = ok;
  if (macs) *macs = total;
  if (layer_macs) {
    lm[1] = gt.g[0].full_macs + gt.g[0].center_macs;
    lm[2] = gt.g[1].full_macs + gt.g[1].center_macs;
    lm[3] = gt.g[2].full_macs + gt.g[2].center_macs;
    lm[0] = 294912ull;
    for (int l = 0; l < 4; l++) layer_macs[l] = lm[l];
  }
  return t1 - t0;
}

static double median(vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static string json_escape(const string& s) {
  string o;
  for (char c : s) {
    if (c == '"') o += "\\\"";
    else if (c == '\\') o += "\\\\";
    else o += c;
  }
  return o;
}

// Zero-training in-memory calibration: per-channel firing duty on dev images.
static void measure_duty(const Model& m, const ImageSet& set, int count,
                         vector<float> duty[4]) {
  for (int l = 0; l < 4; l++) duty[l].assign(128, 0.f);
  InferBufs b;
  alloc_infer(b);
  for (int i = 0; i < count; i++) {
    std::fill(b.z0.begin(), b.z0.end(), 0.f);
    for (int y = 0; y < 32; y++)
      for (int x = 0; x < 32; x++) b.z0[(size_t(y + 1) * 34 + (x + 1))] = set.px[size_t(i) * 1024 + y * 32 + x];
    conv_dense(m.L[0], b.z0.data(), 32, 32, false, b.z1.data(), nullptr, nullptr, nullptr, b.xcol.data());
    const float* zs[4] = {b.z1.data(), b.z2.data(), b.z3.data(), b.z4.data()};
    const int HH[4] = {32, 32, 16, 16};
    const int CC[4] = {32, 64, 128, 128};
    for (int y = 0; y < HH[0]; y++)
      for (int x = 0; x < HH[0]; x++)
        for (int c = 0; c < CC[0]; c++)
          if (zs[0][(size_t(y + 1) * 34 + (x + 1)) * CC[0] + c] > 0.f) duty[0][c]++;
    conv_dense(m.L[1], b.z1.data(), 32, 32, true, b.z2.data(), nullptr, nullptr, nullptr, b.xcol.data());
    for (int y = 0; y < HH[1]; y++)
      for (int x = 0; x < HH[1]; x++)
        for (int c = 0; c < CC[1]; c++)
          if (zs[1][(size_t(y + 1) * 34 + (x + 1)) * CC[1] + c] > 0.f) duty[1][c]++;
    pool_dense(b.z2.data(), b.pool.data(), b.pool_arg.data());
    conv_dense(m.L[2], b.pool.data(), 16, 16, true, b.z3.data(), nullptr, nullptr, nullptr, b.xcol.data());
    for (int y = 0; y < HH[2]; y++)
      for (int x = 0; x < HH[2]; x++)
        for (int c = 0; c < CC[2]; c++)
          if (zs[2][(size_t(y + 1) * 18 + (x + 1)) * CC[2] + c] > 0.f) duty[2][c]++;
    conv_dense(m.L[3], b.z3.data(), 16, 16, true, b.z4.data(), nullptr, nullptr, nullptr, b.xcol.data());
    for (int y = 0; y < HH[3]; y++)
      for (int x = 0; x < HH[3]; x++)
        for (int c = 0; c < CC[3]; c++)
          if (zs[3][(size_t(y + 1) * 18 + (x + 1)) * CC[3] + c] > 0.f) duty[3][c]++;
  }
}


static void bench_command(int argc, char** argv) {
  string wpath, test_path, stream_path, json_path, heat_path, dev_path;
  int reps = 20;
  GateConfig cfg;
  for (int i = 2; i < argc; i++) {
    string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) die("missing value for " + a); return string(argv[++i]); };
    if (a == "--weights") wpath = next();
    else if (a == "--test") test_path = next();
    else if (a == "--stream") stream_path = next();
    else if (a == "--json") json_path = next();
    else if (a == "--heatmaps") heat_path = next();
    else if (a == "--reps") reps = std::stoi(next());
    else if (a == "--th-skip") cfg.th_skip = std::stoi(next());
    else if (a == "--th-skip-l") {
      string v = next();
      int k = 0;
      size_t p0 = 0;
      while (k < 3) {
        size_t p1 = v.find(',', p0);
        cfg.th_skip_l[k++] = std::stoi(v.substr(p0, p1 == string::npos ? string::npos : p1 - p0));
        if (p1 == string::npos) break;
        p0 = p1 + 1;
      }
    }
    else if (a == "--th-full") cfg.th_full = std::stoi(next());
    else if (a == "--anchor-mask") cfg.anchor_mask = uint32_t(std::stoul(next()));
    else if (a == "--no-stage-b") cfg.stage_b = false;
    else if (a == "--margin-scale") cfg.margin_scale = std::stof(next());
    else if (a == "--dev") dev_path = next();
    else if (a == "--th-block") cfg.th_block = std::stoi(next());
    else if (a == "--min-block-count") cfg.min_block_count = std::stoi(next());
    else if (a == "--min-count") cfg.min_count = std::stoi(next());
    else die("unknown arg " + a);
  }
  if (wpath.empty() || test_path.empty()) die("bench requires --weights and --test");
  omp_set_num_threads(1);
  Model m;
  load_weights(wpath, m);
  ImageSet te = load_images(test_path);
  ImageSet dev;
  if (!dev_path.empty()) dev = load_images(dev_path);
  StreamSet stream;
  if (!stream_path.empty()) stream = load_stream(stream_path);

  InferBufs b;
  alloc_infer(b);
  ImageSet dummy;
  (void)dummy;

  // ---- dense baseline ----
  DenseStats ds{};
  int correct = 0;
  time_dense_pass(m, te, b, &correct, &ds);  // warmup + accuracy
  vector<double> dense_ms;
  for (int r = 0; r < reps; r++) {
    int c2 = 0;
    double t = time_dense_pass(m, te, b, &c2, nullptr);
    dense_ms.push_back(t * 1000.0 / te.count);
  }
  const double dense_med = median(dense_ms);
  const double dense_acc = double(correct) / te.count;
  std::printf("dense acc %.4f median_ms %.4f\n", dense_acc, dense_med);
  std::fflush(stdout);

  ModeResult dense_res;
  dense_res.name = "dense-avx512";
  dense_res.acc0 = dense_res.acc_final = dense_acc;
  dense_res.pass0_ms = dense_ms.empty() ? 0 : dense_ms[0];
  dense_res.median_ms = dense_med;
  dense_res.best_ms = *std::min_element(dense_ms.begin(), dense_ms.end());
  dense_res.macs_per_image0 = dense_res.macs_per_image_final = 75792384.0;
  dense_res.elim0 = dense_res.elim_final = 0;
  dense_res.layer_macs[0] = 294912; dense_res.layer_macs[1] = 18874368;
  dense_res.layer_macs[2] = 18874368; dense_res.layer_macs[3] = 37748736;

  vector<ModeResult> modes;
  modes.push_back(dense_res);

  // ---- v3: isomorphic co-activation clustering + per-block CM ----
  Model pm;
  vector<vector<int>> perm = clustered_perm(m);
  permute_model(m, pm, perm);
  DenseStats cst;
  int cok = 0;
  for (int i = 0; i < te.count; i++) {
    float lg[NC];
    dense_forward(pm, te.px.data() + size_t(i) * 1024, b, lg, &cst);
    if (argmax10(lg) == te.labels[i]) cok++;
  }
  const double cluster_acc = double(cok) / te.count;
  std::printf("v3 clustered dense acc %.4f block16_dead conv2/3/4 %.4f/%.4f/%.4f "
              "block8_dead %.4f/%.4f/%.4f\n",
              cluster_acc, cst.blk_dead[1] / cst.blk_total[1],
              cst.blk_dead[2] / cst.blk_total[2], cst.blk_dead[3] / cst.blk_total[3],
              cst.half_dead[1] / cst.half_total[1], cst.half_dead[2] / cst.half_total[2],
              cst.half_dead[3] / cst.half_total[3]);
  std::fflush(stdout);

  // ---- v4: 8-lane cosine co-activation clustering on the same frozen weights ----
  Model cosm;
  vector<vector<int>> cperm;
  if (dev.count > 0) {
    cperm = cosine_perm(m, dev, b);
    permute_model(m, cosm, cperm);
    DenseStats ccst; int ccok = 0;
    for (int i = 0; i < te.count; i++) {
      float lg[NC];
      dense_forward(cosm, te.px.data() + size_t(i) * 1024, b, lg, &ccst);
      if (argmax10(lg) == te.labels[i]) ccok++;
    }
    std::printf("v4 cosine dense acc %.4f block8_dead conv2/3/4 %.4f/%.4f/%.4f\n",
                double(ccok) / te.count, ccst.half_dead[1] / ccst.half_total[1],
                ccst.half_dead[2] / ccst.half_total[2], ccst.half_dead[3] / ccst.half_total[3]);
    std::fflush(stdout);
  }

  // Dense references for fidelity metrics (dense logits are permutation-invariant;
  // Conv4 tensors are compared in each model's own channel order).
  vector<float> ref_logits(size_t(te.count) * NC);
  vector<float> ref_z4_m(size_t(te.count) * 32768), ref_z4_pm(size_t(te.count) * 32768);
  for (int i = 0; i < te.count; i++) {
    float lg[NC];
    dense_forward(m, te.px.data() + size_t(i) * 1024, b, lg, nullptr);
    for (int k = 0; k < NC; k++) ref_logits[size_t(i) * NC + k] = lg[k];
    const float* z = b.z4.data();
    for (int y = 0; y < 16; y++)
      for (int x = 0; x < 16; x++)
        for (int c = 0; c < 128; c++)
          ref_z4_m[(size_t(i) * 256 + y * 16 + x) * 128 + c] =
              z[((size_t(y + 1) * 18 + (x + 1)) * 128) + c];
    dense_forward(pm, te.px.data() + size_t(i) * 1024, b, lg, nullptr);
    const float* zp = b.z4.data();
    for (int y = 0; y < 16; y++)
      for (int x = 0; x < 16; x++)
        for (int c = 0; c < 128; c++)
          ref_z4_pm[(size_t(i) * 256 + y * 16 + x) * 128 + c] =
              zp[((size_t(y + 1) * 18 + (x + 1)) * 128) + c];
  }

  auto run_mode = [&](const string& name, int mode, const Model& mm,
                      bool clustered) -> ModeResult {
    GateConfig c = cfg;
    c.mode = mode;
    Gate gt;
    gate_init(gt, c);
    if (mode >= GM_OUT_A) gate_prepack(gt, mm);
    ModeResult r;
    r.name = name;
    int ok = 0;
    uint64_t macs = 0, lm[4] = {0, 0, 0, 0};
    double t = time_gated_pass(mm, te, b, gt, &ok, &macs, lm);
    r.acc0 = double(ok) / te.count;
    r.pass0_ms = t * 1000.0 / te.count;
    r.macs_per_image0 = double(macs) / te.count;
    r.elim0 = 1.0 - r.macs_per_image0 / 75792384.0;
    for (int l = 0; l < 4; l++) r.layer_macs[l] = (l == 0) ? lm[0] : lm[l] / te.count;
    r.pass_ms.push_back(r.pass0_ms);
    for (int rep = 1; rep < reps; rep++) {
      int ok2 = 0;
      double t2 = time_gated_pass(mm, te, b, gt, &ok2, &macs, lm);
      r.pass_ms.push_back(t2 * 1000.0 / te.count);
    }
    // Fidelity pass: compare every logit and the 16x16x128 Conv4 tensor with
    // exact dense inference in the same channel space.
    int okf = 0, agree = 0;
    uint64_t macsf = 0;
    double se_logit = 0, se_z4 = 0, se_ref = 0;
    const vector<float>& rz = clustered ? ref_z4_pm : ref_z4_m;
    for (int i = 0; i < te.count; i++) {
      float lg[NC];
      macsf += gated_forward(mm, te.px.data() + size_t(i) * 1024, b, gt, lg, nullptr);
      const float* dz = b.z4.data();
      for (int k = 0; k < NC; k++) {
        const double d = double(lg[k]) - double(ref_logits[size_t(i) * NC + k]);
        se_logit += d * d;
      }
      for (int q = 0; q < 256; q++)
        for (int c = 0; c < 128; c++) {
          const size_t off = ((size_t(q / 16 + 1) * 18 + (q % 16 + 1)) * 128) + c;
          const double d = double(dz[off]) - double(rz[(size_t(i) * 256 + q) * 128 + c]);
          const double v = double(rz[(size_t(i) * 256 + q) * 128 + c]);
          se_z4 += d * d;
          se_ref += v * v;
        }
      if (argmax10(lg) == argmax10(&ref_logits[size_t(i) * NC])) agree++;
      if (argmax10(lg) == te.labels[i]) okf++;
    }
    r.logit_rmse = std::sqrt(se_logit / (double(te.count) * NC));
    r.conv4_rel_l2 = se_ref > 0 ? std::sqrt(se_z4 / se_ref) : 0.0;
    r.dense_agreement = double(agree) / te.count;
    r.acc_final = double(okf) / te.count;
    r.macs_per_image_final = double(macsf) / te.count;
    r.elim_final = 1.0 - r.macs_per_image_final / 75792384.0;
    vector<double> all(r.pass_ms.begin(), r.pass_ms.end());
    r.median_ms = median(all);
    r.best_ms = *std::min_element(all.begin(), all.end());
    r.full_blocks = gt.g[0].full_blocks + gt.g[1].full_blocks + gt.g[2].full_blocks;
    r.center_blocks = gt.g[0].center_blocks + gt.g[1].center_blocks + gt.g[2].center_blocks;
    r.skip_blocks = gt.g[0].skip_blocks + gt.g[1].skip_blocks + gt.g[2].skip_blocks;
    for (int l = 0; l < 3; l++) {
      r.full_blocks_l[l] = gt.g[l].full_blocks;
      r.center_blocks_l[l] = gt.g[l].center_blocks;
      r.skip_blocks_l[l] = gt.g[l].skip_blocks;
    }
    for (int l = 0; l < 3; l++) {
      r.out_channels += gt.g[l].out_evals;
      r.out_skips += gt.g[l].out_skips;
      r.stageb_aborts += gt.g[l].stageb_aborts;
      const double n = double(gt.g[l].out_evals);
      r.out_skip_rate[l] = n > 0 ? double(gt.g[l].out_skips) / n : 0.0;
      r.stageb_rate[l] = n > 0 ? double(gt.g[l].stageb_aborts) / n : 0.0;
    }
    uint64_t aps = 0, afs = 0;
    for (int l = 0; l < 3; l++) { aps += gt.g[l].anchor_pred_skip; afs += gt.g[l].anchor_false_skip; }
    std::printf("%s acc %.4f acc_final %.4f macs %.0f elim %.2f%% med_ms %.4f speedup %.3f"
                " skiprate %.2f/%.2f/%.2f anchor_false %.4f rmse %.6f z4rel %.6f agree %.4f\n",
                name.c_str(), r.acc0, r.acc_final, r.macs_per_image0, r.elim0 * 100, r.median_ms,
                dense_med / r.median_ms, r.out_skip_rate[0], r.out_skip_rate[1], r.out_skip_rate[2],
                aps ? double(afs) / double(aps) : 0.0, r.logit_rmse, r.conv4_rel_l2,
                r.dense_agreement);
    std::fflush(stdout);
    return r;
  };

  // ---- v3.1 waterfall: M1..M5, then two random baselines ----
  // Pure input compaction: disable the v1 per-channel CM skips so M1 and M2
  // have identical executed-tap counts (the legacy 15.948M number included them).
  const int saved_th_skip = cfg.th_skip;
  cfg.th_skip = 0;
  cfg.th_full = 0;
  ModeResult m1 = run_mode("M1: v1-input-sparse-only", GM_INPUT_SPARSE, m, false);
  cfg.th_skip = saved_th_skip;
  ModeResult m2 = run_mode("M2: v3-clustered-tap-only", GM_V3_TAP_ONLY, pm, true);
  ModeResult m3 = run_mode("M3: v3-clustered-cm-stageA-only", GM_V3_A, pm, true);
  ModeResult m4 = run_mode("M4: v3-clustered-stageB-only", GM_V3_STAGEB_ONLY, pm, true);
  ModeResult m5 = run_mode("M5: v3-clustered-cm-full-turbo", GM_V3_FULL, pm, true);
  ModeResult mcos = dev.count > 0 ? run_mode("v4-8lane-cosine-cm", GM_V3_FULL, cosm, true) : ModeResult{};

  // Match M5's per-layer block-skip fraction among non-anchor blocks.
  for (int l = 0; l < 3; l++) {
    const double total = double(m5.full_blocks_l[l] + m5.center_blocks_l[l] + m5.skip_blocks_l[l]);
    const double q = total > 0 ? double(m5.skip_blocks_l[l]) / (total * 15.0 / 16.0) : 0.0;
    cfg.random_ch_q[l] = float(std::min(0.98, std::max(0.0, q)));
  }
  ModeResult rnd_blk = run_mode("block-matched-random-skip", GM_BLOCK_RANDOM, pm, true);
  // Dense-tap random channel-block skip at M5's total MAC budget: no input
  // compaction and no clustering, so every skipped block is real lost work.
  {
    const double target = (m5.macs_per_image_final - 294912.0) / (75792384.0 - 294912.0);
    const double q = 1.0 - (target - 1.0 / 16.0) / (15.0 / 16.0);
    for (int l = 0; l < 3; l++) cfg.random_ch_q[l] = float(q);
  }
  ModeResult rnd_unc = run_mode("unclustered-random-skip", GM_RANDOM_DENSE, m, false);

  modes.push_back(m1);
  modes.push_back(m2);
  modes.push_back(m3);
  modes.push_back(m4);
  modes.push_back(m5);
  if (dev.count > 0) modes.push_back(mcos);
  modes.push_back(rnd_blk);
  modes.push_back(rnd_unc);

  // ---- temporal stream ----
  struct StreamResult {
    double acc = 0, ms_per_frame = 0, macs_per_frame = 0, elim = 0;
    vector<double> pass_ms;  // ms/frame
  } sd, v3t, v3s, v4t;

  if (stream.seqs > 0) {
    const int total_frames = stream.seqs * stream.frames;
    auto dense_stream_pass = [&](int* correct, double* secs) {
      double t0 = now_sec();
      int ok = 0;
      for (int i = 0; i < total_frames; i++) {
        float logits[NC];
        dense_forward(m, stream.px.data() + size_t(i) * 1024, b, logits, nullptr);
        if (argmax10(logits) == stream.labels[i / stream.frames]) ok++;
      }
      *correct = ok;
      *secs = now_sec() - t0;
    };
    int c = 0;
    double secs = 0;
    dense_stream_pass(&c, &secs);  // warmup
    sd.acc = double(c) / total_frames;
    sd.ms_per_frame = secs * 1000.0 / total_frames;
    vector<double> dm;
    for (int r = 0; r < std::min(reps, 10); r++) {
      dense_stream_pass(&c, &secs);
      dm.push_back(secs * 1000.0 / total_frames);
    }
    sd.ms_per_frame = median(dm);
    std::printf("stream dense acc %.4f ms/frame %.4f\n", sd.acc, sd.ms_per_frame);
    std::fflush(stdout);

    auto run_stream_mode = [&](const string& name, int mode, const Model& mm) -> StreamResult {
      GateConfig c2 = cfg;
      c2.mode = mode;
      Gate gt;
      gate_init(gt, c2);
      if (mode >= GM_OUT_A) gate_prepack(gt, mm);
      StreamResult r;
      int ok = 0;
      uint64_t macs = 0, lm[4] = {0, 0, 0, 0};
      double t0 = now_sec();
      for (int s = 0; s < stream.seqs; s++) {
        // reset temporal history at each sequence
        for (int i = 0; i < 3; i++) {
          std::fill(gt.g[i].prev_act.begin(), gt.g[i].prev_act.end(), 0);
          std::fill(gt.g[i].prev_pix.begin(), gt.g[i].prev_pix.end(), 0);
        }
        for (int f = 0; f < stream.frames; f++) {
          int idx = s * stream.frames + f;
          float logits[NC];
          macs += gated_forward(mm, stream.px.data() + size_t(idx) * 1024, b, gt, logits, nullptr);
          if (argmax10(logits) == stream.labels[s]) ok++;
        }
      }
      double secs0 = now_sec() - t0;
      r.acc = double(ok) / total_frames;
      r.macs_per_frame = double(macs) / total_frames;
      r.elim = 1.0 - r.macs_per_frame / 75792384.0;
      r.pass_ms.push_back(secs0 * 1000.0 / total_frames);
      for (int rep = 1; rep < std::min(reps, 10); rep++) {
        int ok2 = 0;
        uint64_t m2 = 0;
        double t1 = now_sec();
        for (int s = 0; s < stream.seqs; s++) {
          std::fill(gt.g[0].prev_act.begin(), gt.g[0].prev_act.end(), 0);
          std::fill(gt.g[1].prev_act.begin(), gt.g[1].prev_act.end(), 0);
          std::fill(gt.g[2].prev_act.begin(), gt.g[2].prev_act.end(), 0);
          std::fill(gt.g[0].prev_pix.begin(), gt.g[0].prev_pix.end(), 0);
          std::fill(gt.g[1].prev_pix.begin(), gt.g[1].prev_pix.end(), 0);
          std::fill(gt.g[2].prev_pix.begin(), gt.g[2].prev_pix.end(), 0);
          for (int f = 0; f < stream.frames; f++) {
            int idx = s * stream.frames + f;
            float logits[NC];
            m2 += gated_forward(mm, stream.px.data() + size_t(idx) * 1024, b, gt, logits, nullptr);
            if (argmax10(logits) == stream.labels[s]) ok2++;
          }
        }
        double t2 = now_sec() - t1;
        r.pass_ms.push_back(t2 * 1000.0 / total_frames);
        (void)m2;
      }
      r.ms_per_frame = median(r.pass_ms);
      std::printf("stream %s acc %.4f elim %.2f%% ms/frame %.4f speedup %.3f\n", name.c_str(),
                  r.acc, r.elim * 100, r.ms_per_frame, sd.ms_per_frame / r.ms_per_frame);
      std::fflush(stdout);
      return r;
    };
    v3t = run_stream_mode("v3-temporal-stream", GM_V3_TEMPORAL, pm);
    v3s = run_stream_mode("v3-full-no-temporal", GM_V3_FULL, pm);
    if (dev.count > 0) v4t = run_stream_mode("v4-cosine-temporal", GM_V3_TEMPORAL, cosm);
    (void)v3s;
    (void)v3t;
  }

  // ---- JSON ----
  std::ostringstream js;
  js << "{\n";
  js << "  \"config\": {\"reps\": " << reps << ", \"threads\": 1, \"th_skip\": " << cfg.th_skip
     << ", \"th_full\": " << cfg.th_full << ", \"th_skip_frac\": " << (double(cfg.th_skip) / 32768.0)
     << ", \"th_full_frac\": " << (double(cfg.th_full) / 32768.0) << ", \"anchor_mask\": "
     << cfg.anchor_mask << ", \"stage_b\": " << (cfg.stage_b ? "true" : "false")
     << ", \"margin_scale\": " << cfg.margin_scale << ", \"min_count\": " << cfg.min_count
     << ", \"th_skip_l\": [" << cfg.th_skip_l[0] << ", " << cfg.th_skip_l[1] << ", "
     << cfg.th_skip_l[2] << "], \"th_block\": " << cfg.th_block << ", \"random_ch_q\": [" << cfg.random_ch_q[0] << ", " << cfg.random_ch_q[1] << ", "
     << cfg.random_ch_q[2] << "]},\n";
  js << "  \"dense\": {\"accuracy\": " << dense_acc << ", \"dead_rates\": ["
     << ds.dead[0] / ds.total_pre[0] << ", " << ds.dead[1] / ds.total_pre[1] << ", "
     << ds.dead[2] / ds.total_pre[2] << ", " << ds.dead[3] / ds.total_pre[3]
     << "], \"oracle_block16_dead\": [" << ds.blk_dead[0] / ds.blk_total[0] << ", "
     << ds.blk_dead[1] / ds.blk_total[1] << ", " << ds.blk_dead[2] / ds.blk_total[2] << ", "
     << ds.blk_dead[3] / ds.blk_total[3] << "], \"oracle_pixel_dead\": ["
     << ds.pix_dead[0] / ds.pix_total[0] << ", " << ds.pix_dead[1] / ds.pix_total[1] << ", "
     << ds.pix_dead[2] / ds.pix_total[2] << ", " << ds.pix_dead[3] / ds.pix_total[3]
     << "], \"macs_per_image\": 75792384, \"median_ms\": " << dense_med
     << ", \"images_per_sec\": " << (1000.0 / dense_med) << ", \"pass_ms\": [";
  for (size_t i = 0; i < dense_ms.size(); i++) js << (i ? ", " : "") << dense_ms[i];
  js << "]},\n";
  js << "  \"clustered\": {\"accuracy\": " << cluster_acc << ", \"dead_rates\": ["
     << cst.dead[0] / cst.total_pre[0] << ", " << cst.dead[1] / cst.total_pre[1] << ", "
     << cst.dead[2] / cst.total_pre[2] << ", " << cst.dead[3] / cst.total_pre[3]
     << "], \"oracle_block16_dead\": [" << cst.blk_dead[0] / cst.blk_total[0] << ", "
     << cst.blk_dead[1] / cst.blk_total[1] << ", " << cst.blk_dead[2] / cst.blk_total[2] << ", "
     << cst.blk_dead[3] / cst.blk_total[3] << "], \"block8_dead\": ["
     << cst.half_dead[0] / cst.half_total[0] << ", " << cst.half_dead[1] / cst.half_total[1]
     << ", " << cst.half_dead[2] / cst.half_total[2] << ", "
     << cst.half_dead[3] / cst.half_total[3] << "]},\n";
  js << "  \"modes\": {\n";
  for (size_t i = 0; i < modes.size(); i++) {
    const ModeResult& r = modes[i];
    js << "    \"" << json_escape(r.name) << "\": {\"accuracy\": " << r.acc0
       << ", \"accuracy_final\": " << r.acc_final << ", \"macs_per_image\": " << r.macs_per_image0
       << ", \"macs_per_image_final\": " << r.macs_per_image_final << ", \"mac_elimination\": "
       << r.elim0 << ", \"mac_elimination_final\": " << r.elim_final
       << ", \"pass0_ms\": " << r.pass0_ms << ", \"median_ms\": " << r.median_ms
       << ", \"best_ms\": " << r.best_ms << ", \"speedup\": " << (dense_med / r.median_ms)
       << ", \"speedup_pass0\": " << (dense_med / r.pass0_ms) << ", \"layer_macs\": ["
       << r.layer_macs[0] << ", " << r.layer_macs[1] << ", " << r.layer_macs[2] << ", "
       << r.layer_macs[3] << "], \"blocks\": {\"full\": " << r.full_blocks
       << ", \"center\": " << r.center_blocks << ", \"skip\": " << r.skip_blocks << "}, \"out_channels\": "
       << r.out_channels << ", \"out_skips\": " << r.out_skips << ", \"stageb_aborts\": "
       << r.stageb_aborts << ", \"out_skip_rate\": [" << r.out_skip_rate[0] << ", "
       << r.out_skip_rate[1] << ", " << r.out_skip_rate[2] << "], \"stageb_rate\": ["
       << r.stageb_rate[0] << ", " << r.stageb_rate[1] << ", " << r.stageb_rate[2]
       << "], \"logit_rmse\": " << r.logit_rmse << ", \"conv4_rel_l2\": " << r.conv4_rel_l2
       << ", \"dense_agreement\": " << r.dense_agreement << "}"
       << (i + 1 < modes.size() ? "," : "") << "\n";
    if (i == 0) {
      // insert full mode's random budget into config-looking field below
    }
  }
  js << "  },\n";
  const auto ratio = [](double a, double b) { return b > 0 ? a / b : 0.0; };
  js << "  \"stream\": {\"frames\": " << (stream.seqs * stream.frames) << ", \"dense\": {\"accuracy\": "
     << sd.acc << ", \"ms_per_frame\": " << sd.ms_per_frame << ", \"macs_per_frame\": 75792384},\n";
  if (stream.seqs > 0) {
    js << "    \"v3-temporal-stream\": {\"accuracy\": " << v3t.acc << ", \"ms_per_frame\": "
       << v3t.ms_per_frame << ", \"macs_per_frame\": " << v3t.macs_per_frame
       << ", \"mac_elimination\": " << v3t.elim << ", \"speedup\": "
       << ratio(sd.ms_per_frame, v3t.ms_per_frame) << "},\n";
    js << "    \"v4-cosine-temporal\": {\"accuracy\": " << v4t.acc << ", \"ms_per_frame\": "
       << v4t.ms_per_frame << ", \"macs_per_frame\": " << v4t.macs_per_frame
       << ", \"mac_elimination\": " << v4t.elim << ", \"speedup\": "
       << ratio(sd.ms_per_frame, v4t.ms_per_frame) << "},\n";
    js << "    \"v3-full-no-temporal\": {\"accuracy\": " << v3s.acc << ", \"ms_per_frame\": "
       << v3s.ms_per_frame << ", \"macs_per_frame\": " << v3s.macs_per_frame
       << ", \"mac_elimination\": " << v3s.elim << ", \"speedup\": "
       << ratio(sd.ms_per_frame, v3s.ms_per_frame) << "}\n";
  } else {
    js << "    \"v3-temporal-stream\": null, \"v3-full-no-temporal\": null\n";
  }
  js << "  }\n}\n";

  if (!json_path.empty()) {
    std::ofstream f(json_path);
    if (!f) die("cannot write " + json_path);
    f << js.str();
    std::printf("wrote %s\n", json_path.c_str());
  } else {
    std::printf("%s", js.str().c_str());
  }

  // ---- heatmaps ----
  if (!heat_path.empty()) {
    std::ofstream f(heat_path);
    if (!f) die("cannot write " + heat_path);
    // run a converged full-turbo gate over the test set to produce stable maps
    GateConfig hc = cfg;
    hc.mode = GM_V3_FULL;
    Gate gt;
    gate_init(gt, hc);
    gate_prepack(gt, pm);
    for (int rep = 0; rep < 2; rep++)
      for (int i = 0; i < te.count; i++) {
        float logits[NC];
        gated_forward(pm, te.px.data() + size_t(i) * 1024, b, gt, logits, nullptr);
      }
    InferBufs db;
    alloc_infer(db);
    for (int sample = 0; sample < 2 && sample < te.count; sample++) {
      HeatMaps hm;
      hm.input.assign(1024, 0);
      hm.act2.assign(1024, 0); hm.exe2.assign(1024, 0);
      hm.act3.assign(256, 0); hm.exe3.assign(256, 0);
      hm.act4.assign(256, 0); hm.exe4.assign(256, 0);
      for (int i = 0; i < 1024; i++) {
        float v = te.px[size_t(sample) * 1024 + i];
        hm.input[i] = uint8_t(std::min(1.f, std::max(0.f, v)) * 255.f);
      }
      float logits[NC];
      gated_forward(pm, te.px.data() + size_t(sample) * 1024, b, gt, logits, &hm);
      // dense active maps
      vector<uint8_t> a2(1024 * 64, 0), a3(256 * 128, 0), a4(256 * 128, 0);
      float dl[NC];
      DenseStats dsmp{};
      dense_forward(m, te.px.data() + size_t(sample) * 1024, db, dl, &dsmp, a2.data(), a3.data(),
                    a4.data());
      vector<uint16_t> d2(1024, 0), d3(256, 0), d4(256, 0);
      for (int y = 0; y < 32; y++)
        for (int x = 0; x < 32; x++)
          for (int c = 0; c < 64; c++) d2[y * 32 + x] += a2[(y * 32 + x) * 64 + c];
      for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++) {
          for (int c = 0; c < 128; c++) {
            d3[y * 16 + x] += a3[(y * 16 + x) * 128 + c];
            d4[y * 16 + x] += a4[(y * 16 + x) * 128 + c];
          }
        }
      f << "### sample " << sample << " label " << int(te.labels[sample]) << "\n";
      f << "INPUT\n" << ascii_u8(hm.input.data(), 32, 32);
      f << "DENSE_CONV2_ACTIVE\n" << ascii_map(d2.data(), 32, 32, 64);
      f << "CM_CONV2_EXECUTED\n" << ascii_map(hm.exe2.data(), 32, 32, 0);
      f << "CM_CONV2_ACTIVE\n" << ascii_map(hm.act2.data(), 32, 32, 64);
      f << "DENSE_CONV3_ACTIVE\n" << ascii_map(d3.data(), 16, 16, 128);
      f << "CM_CONV3_EXECUTED\n" << ascii_map(hm.exe3.data(), 16, 16, 0);
      f << "CM_CONV3_ACTIVE\n" << ascii_map(hm.act3.data(), 16, 16, 128);
      f << "DENSE_CONV4_ACTIVE\n" << ascii_map(d4.data(), 16, 16, 128);
      f << "CM_CONV4_EXECUTED\n" << ascii_map(hm.exe4.data(), 16, 16, 0);
    }
    std::printf("wrote %s\n", heat_path.c_str());
  }
}

// ---------------------------------------------------------------------------
// Eval command (dense only for now)
// ---------------------------------------------------------------------------

static void eval_command(int argc, char** argv) {
  string wpath, test_path;
  for (int i = 2; i < argc; i++) {
    string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) die("missing value for " + a); return string(argv[++i]); };
    if (a == "--weights") wpath = next();
    else if (a == "--test") test_path = next();
    else die("unknown arg " + a);
  }
  if (wpath.empty() || test_path.empty()) die("eval requires --weights and --test");
  Model m;
  load_weights(wpath, m);
  ImageSet te = load_images(test_path);
  DenseStats stats{};
  double acc = eval_accuracy(m, te, 0, te.count, &stats);
  std::printf("acc %.6f dead", acc);
  for (int l = 0; l < 4; l++) std::printf(" %.5f", stats.dead[l] / stats.total_pre[l]);
  std::printf("\n");
}


// ---------------------------------------------------------------------------
// Bias calibration: shift each layer's folded bias so the natural ReLU dead
// rate matches a target, without touching a single weight.
// ---------------------------------------------------------------------------
static double measure_dead(Model& m, const ImageSet& set, int li, DenseStats* out = nullptr) {
  InferBufs b;
  alloc_infer(b);
  DenseStats st{};
  for (int i = 0; i < set.count; i++) {
    float logits[NC];
    dense_forward(m, set.px.data() + size_t(i) * 1024, b, logits, &st);
  }
  if (out) *out = st;
  return st.dead[li] / std::max(1.0, st.total_pre[li]);
}

static void calibrate_command(int argc, char** argv) {
  string wpath, out_path, calib_path, test_path;
  double targets[4] = {0.55, 0.68, 0.70, 0.72};
  int iters = 18;
  for (int i = 2; i < argc; i++) {
    string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) die("missing value for " + a); return string(argv[++i]); };
    if (a == "--weights") wpath = next();
    else if (a == "--out") out_path = next();
    else if (a == "--calib") calib_path = next();
    else if (a == "--test") test_path = next();
    else if (a == "--iters") iters = std::stoi(next());
    else if (a == "--targets") {
      string v = next();
      size_t p = 0;
      for (int l = 0; l < 4; l++) {
        size_t q = v.find(',', p);
        targets[l] = std::stod(v.substr(p, q == string::npos ? string::npos : q - p));
        p = (q == string::npos) ? v.size() : q + 1;
      }
    } else die("unknown arg " + a);
  }
  if (wpath.empty() || out_path.empty() || calib_path.empty())
    die("calibrate requires --weights --out --calib");
  Model m;
  load_weights(wpath, m);
  ImageSet cal = load_images(calib_path);
  for (int l = 0; l < 4; l++) {
    vector<float> orig = m.L[l].bias;
    double base = measure_dead(m, cal, l);
    std::printf("calib L%d base_dead %.4f target %.3f\n", l + 1, base, targets[l]);
    std::fflush(stdout);
    if (base >= targets[l]) {
      std::printf("calib L%d already sparse\n", l + 1);
      continue;
    }
    auto apply = [&](double d) {
      for (int c = 0; c < m.L[l].cout; c++) m.L[l].bias[c] = orig[c] - float(d);
    };
    apply(0);
    double lo = 0, hi = 1.0;
    apply(hi);
    while (measure_dead(m, cal, l) < targets[l] && hi < 512.0) {
      lo = hi;
      hi *= 2.0;
      apply(hi);
    }
    for (int it = 0; it < iters; it++) {
      double mid = 0.5 * (lo + hi);
      apply(mid);
      double d = measure_dead(m, cal, l);
      if (d < targets[l]) lo = mid; else hi = mid;
    }
    apply(hi);
    double fin = measure_dead(m, cal, l);
    std::printf("calib L%d shift %.4f dead %.4f\n", l + 1, hi, fin);
    std::fflush(stdout);
  }
  DenseStats st{};
  measure_dead(m, cal, 0, &st);
  std::printf("calib final dead %.4f %.4f %.4f %.4f\n",
              st.dead[0] / st.total_pre[0], st.dead[1] / st.total_pre[1],
              st.dead[2] / st.total_pre[2], st.dead[3] / st.total_pre[3]);
  if (!test_path.empty()) {
    ImageSet te = load_images(test_path);
    double acc = eval_accuracy(m, te, 0, te.count, &st);
    std::printf("calib test acc %.4f dead %.4f %.4f %.4f %.4f\n", acc,
                st.dead[0] / st.total_pre[0], st.dead[1] / st.total_pre[1],
                st.dead[2] / st.total_pre[2], st.dead[3] / st.total_pre[3]);
  }
  save_weights(out_path, m, st);
  std::printf("saved %s\n", out_path.c_str());
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s train|eval ...\n", argv[0]);
    return 2;
  }
  string cmd = argv[1];
  if (cmd == "train") train_command(argc, argv);
  else if (cmd == "eval") eval_command(argc, argv);
  else if (cmd == "bench") bench_command(argc, argv);
  else if (cmd == "calibrate") calibrate_command(argc, argv);
  else die("unknown command " + cmd);
  return 0;
}
