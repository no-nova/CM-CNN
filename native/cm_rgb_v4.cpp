// SPDX-License-Identifier: MPL-2.0
//
// CM-CNN-RGB-v4: full-canvas RGB VGG-6 + 8-lane co-activation CM gate.
// Also benchmarks the same engine on the frozen Phase-A VGG-4 weights.
//
// Build:
//   g++ -O3 -march=native -mavx512f -mfma -std=c++20 -fopenmp \
//       native/cm_rgb_v4.cpp -o experiments/cm-cnn-rgb-v4/cm_rgb_v4
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
static constexpr int NC = 10;
static constexpr int MAXL = 6;

[[noreturn]] static void die(const string& m) { std::fprintf(stderr, "error: %s\n", m.c_str()); std::exit(2); }
static double now_sec() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template <typename T> static void rd(std::ifstream& f, T* p, size_t n) {
  f.read(reinterpret_cast<char*>(p), std::streamsize(n * sizeof(T)));
  if (!f) die("short read");
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------
struct ImageSet {
  int count = 0, h = 0, w = 0, c = 0;
  vector<uint8_t> labels;
  vector<float> px;
};

static ImageSet load_images(const string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open " + path);
  char magic[9] = {};
  f.read(magic, 8);
  ImageSet s;
  uint32_t v[4] = {};
  rd(f, v, 4);
  s.count = int(v[0]); s.h = int(v[1]); s.w = int(v[2]); s.c = int(v[3]);
  if (string(magic) == "CMIMG001") s.c = 1;
  else if (string(magic) != "CMRGB001") die("bad image magic");
  s.labels.resize(s.count);
  rd(f, s.labels.data(), s.labels.size());
  size_t off = 32 + s.count;
  off += (4 - (s.count % 4)) % 4;
  f.seekg(std::streamoff(off));
  s.px.resize(size_t(s.count) * s.h * s.w * s.c);
  rd(f, s.px.data(), s.px.size());
  return s;
}

struct StreamSet {
  int seqs = 0, frames = 0, h = 0, w = 0, c = 0;
  vector<uint8_t> labels;
  vector<float> px;
};

static StreamSet load_stream(const string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open " + path);
  char magic[9] = {};
  f.read(magic, 8);
  StreamSet s;
  uint32_t v[5] = {};
  rd(f, v, 5);
  s.seqs = int(v[0]); s.frames = int(v[1]); s.h = int(v[2]); s.w = int(v[3]); s.c = int(v[4]);
  if (string(magic) == "CMVID001") s.c = 1;
  else if (string(magic) != "CMRGBV01") die("bad stream magic");
  s.labels.resize(s.seqs);
  rd(f, s.labels.data(), s.labels.size());
  size_t off = 32 + s.seqs;
  off += (4 - (s.seqs % 4)) % 4;
  f.seekg(std::streamoff(off));
  s.px.resize(size_t(s.seqs) * s.frames * s.h * s.w * s.c);
  rd(f, s.px.data(), s.px.size());
  return s;
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
struct Layer {
  int cin = 0, cout = 0, h = 0, wd = 0, k = 0;
  bool pool_after = false;
  vector<float> w, wt, gamma, beta, run_mean, run_var;
  vector<float> wf;   // [cout/16][9][cin][16]
  vector<float> wf8;  // [cout/8][9][cin][8]
  vector<float> bias;
  vector<float> gw, ggamma, gbeta, mw, vw, mg, vg, mb, vb;
};

struct Head {
  int in = 0;
  vector<float> w, b;
  vector<float> gw, gb, mw, vw, mb, vb;
};

struct Model {
  int nl = 6;
  Layer L[MAXL];
  Head head;
  vector<float> bn_mean[MAXL], bn_invstd[MAXL];
};

static void init_model(Model& m, int nl, const int* cin, const int* cout, const int* dims, const bool* pool, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  m.nl = nl;
  int last_c = 0, last_h = 0;
  for (int l = 0; l < nl; l++) {
    Layer& L = m.L[l];
    L.cin = cin[l]; L.cout = cout[l]; L.h = dims[l]; L.wd = dims[l];
    L.k = 9 * L.cin;
    L.pool_after = pool[l];
    float sc = std::sqrt(2.0f / float(L.k));
    L.w.resize(size_t(L.k) * L.cout);
    for (auto& v : L.w) v = nd(rng) * sc;
    L.wt.resize(L.w.size());
    L.gamma.assign(L.cout, 1.f); L.beta.assign(L.cout, 0.f);
    L.run_mean.assign(L.cout, 0.f); L.run_var.assign(L.cout, 1.f);
    m.bn_mean[l].assign(L.cout, 0.f); m.bn_invstd[l].assign(L.cout, 1.f);
    L.wf.resize(size_t(L.cout) * L.k);
    L.wf8.resize(size_t(L.cout) * L.k);
    L.bias.assign(L.cout, 0.f);
    L.gw.assign(L.w.size(), 0.f); L.ggamma.assign(L.cout, 0.f); L.gbeta.assign(L.cout, 0.f);
    L.mw.assign(L.w.size(), 0.f); L.vw.assign(L.w.size(), 0.f);
    L.mg.assign(L.cout, 0.f); L.vg.assign(L.cout, 0.f);
    L.mb.assign(L.cout, 0.f); L.vb.assign(L.cout, 0.f);
    m.bn_mean[l].assign(L.cout, 0.f); m.bn_invstd[l].assign(L.cout, 1.f);
    last_c = L.cout; last_h = L.h;
    if (L.pool_after) last_h /= 2;
  }
  const int grid = last_h / 4;
  m.head.in = grid * grid * last_c;
  m.head.w.assign(size_t(NC) * m.head.in, 0.f);
  float hs = std::sqrt(1.f / float(m.head.in));
  for (auto& v : m.head.w) v = nd(rng) * hs;
  m.head.b.assign(NC, 0.f); m.head.gw.assign(m.head.w.size(), 0.f); m.head.gb.assign(NC, 0.f);
  m.head.mw.assign(m.head.w.size(), 0.f); m.head.vw.assign(m.head.w.size(), 0.f);
  m.head.mb.assign(NC, 0.f); m.head.vb.assign(NC, 0.f);
}

static void transpose_weights(Layer& L) {
  for (int c = 0; c < L.cout; c++)
    for (int k = 0; k < L.k; k++) L.wt[size_t(c) * L.k + k] = L.w[size_t(k) * L.cout + c];
}

// ---------------------------------------------------------------------------
// Training kernels
// ---------------------------------------------------------------------------
static int psz(int h, int w, int c) { return (h + 2) * (w + 2) * c; }

static void im2col(const float* prev, int H, int W, int cin, bool relu, float* xcol) {
  const int Wp = W + 2, K = 9 * cin;
#pragma omp parallel for schedule(static)
  for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
    float* dst = xcol + (size_t(y) * W + x) * K;
    for (int ky = 0; ky < 3; ky++) for (int kx = 0; kx < 3; kx++) {
      const float* src = prev + ((size_t(y + ky) * Wp + (x + kx))) * cin;
      float* d = dst + (size_t(ky) * 3 + kx) * cin;
      if (relu) for (int c = 0; c < cin; c++) d[c] = src[c] > 0 ? src[c] : 0.f;
      else std::memcpy(d, src, size_t(cin) * sizeof(float));
    }
  }
}

static void gemm(const float* A, const float* B, float* C, int M, int K, int N) {
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; m++) {
    const float* a = A + size_t(m) * K;
    float* c = C + size_t(m) * N;
    for (int n = 0; n < N; n += 16) {
      __m512 acc = _mm512_setzero_ps();
      for (int k = 0; k < K; k++) acc = _mm512_fmadd_ps(_mm512_set1_ps(a[k]), _mm512_loadu_ps(B + size_t(k) * N + n), acc);
      _mm512_storeu_ps(c + n, acc);
    }
  }
}

static void conv_dw(const float* xcol, const float* dC, int M, int K, int N, float* dW) {
#pragma omp parallel for schedule(static)
  for (int k = 0; k < K; k++) {
    float* dw = dW + size_t(k) * N;
    for (int n = 0; n < N; n += 16) {
      __m512 acc = _mm512_loadu_ps(dw + n);
      for (int m = 0; m < M; m++) acc = _mm512_fmadd_ps(_mm512_set1_ps(xcol[size_t(m) * K + k]), _mm512_loadu_ps(dC + size_t(m) * N + n), acc);
      _mm512_storeu_ps(dw + n, acc);
    }
  }
}

static void conv_dx(const float* dC, const Layer& L, int M, float* dX) {
  const int cin = L.cin, cout = L.cout, K = L.k;
  std::fill(dX, dX + size_t(M) * cin, 0.f);
#pragma omp parallel for schedule(static)
  for (int m = 0; m < M; m++) {
    const float* dz = dC + size_t(m) * cout;
    float* dx = dX + size_t(m) * cin;
    for (int cb = 0; cb < cin; cb += 16) {
      for (int tap = 0; tap < 9; tap++) {
        const int koff = tap * cin + cb;
        __m512 acc = _mm512_setzero_ps();
        for (int c = 0; c < cout; c++) acc = _mm512_fmadd_ps(_mm512_set1_ps(dz[c]), _mm512_loadu_ps(&L.wt[size_t(c) * K + koff]), acc);
        _mm512_storeu_ps(dx + cb, _mm512_add_ps(_mm512_loadu_ps(dx + cb), acc));
      }
    }
  }
}

static void bn_forward(float* z, int H, int W, int C, Layer& L, vector<float>& mean, vector<float>& inv, bool update) {
  const int M = H * W, Wp = W + 2;
  if (std::getenv("CM_NOBN")) {
    for (int c = 0; c < C; c++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
      float& v = z[(size_t(y + 1) * Wp + (x + 1)) * C + c];
      v = v * L.gamma[c] + L.beta[c];
    }
    return;
  }
  mean.assign(C, 0.f); vector<float> var(C, 0.f);
  const bool inst = !update && std::getenv("CM_INST");
  if (!update && !inst) {
    mean = L.run_mean; var = L.run_var;
  } else {
#pragma omp parallel for schedule(static)
  for (int c = 0; c < C; c++) {
    double s = 0, s2 = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
      float v = z[(size_t(y + 1) * Wp + (x + 1)) * C + c]; s += v; s2 += double(v) * v;
    }
    double mu = s / M, vv = std::max(1e-8, s2 / M - mu * mu);
    mean[c] = float(mu); var[c] = float(vv);
    if (update) { L.run_mean[c] = 0.9f * L.run_mean[c] + 0.1f * float(mu); L.run_var[c] = 0.9f * L.run_var[c] + 0.1f * float(vv * M / std::max(1, M - 1)); }
  }
  }
#pragma omp parallel for schedule(static)
  for (int c = 0; c < C; c++) {
    const float iv = 1.f / std::sqrt(var[c] + 1e-5f); inv[c] = iv;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
      float& v = z[(size_t(y + 1) * Wp + (x + 1)) * C + c];
      v = (v - mean[c]) * iv * L.gamma[c] + L.beta[c];
    }
  }
}

static void bn_backward(const float* dZ, const float* z, int H, int W, int C, Layer& L,
                        const vector<float>& mean, const vector<float>& inv, float* dC) {
  (void)mean;
  const int M = H * W, Wp = W + 2;
#pragma omp parallel for schedule(static)
  for (int c = 0; c < C; c++) {
    const float g = L.gamma[c], be = L.beta[c], iv = inv[c];
    double sdz = 0, sdzx = 0;
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
      size_t idx = (size_t(y + 1) * Wp + (x + 1)) * C + c;
      float dz = dZ[(size_t(y) * W + x) * C + c];
      float xhat = (z[idx] - be) / g;
      sdz += dz; sdzx += dz * xhat; L.ggamma[c] += dz * xhat; L.gbeta[c] += dz;
    }
    const float mdz = float(sdz / M), mdzx = float(sdzx / M);
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
      size_t idx = (size_t(y + 1) * Wp + (x + 1)) * C + c;
      float xhat = (z[idx] - be) / g;
      float d = dZ[(size_t(y) * W + x) * C + c] * g;
      dC[(size_t(y) * W + x) * C + c] = iv * (d - g * mdz - xhat * g * mdzx);
    }
  }
}

// ---------------------------------------------------------------------------
// Batch buffers + forward/backward
// ---------------------------------------------------------------------------
struct Batch {
  int B = 0, nl = 0, H[MAXL], W[MAXL], C[MAXL];
  vector<float> z0, z[MAXL], pool[MAXL], pooled, logits, xcol[MAXL];
  vector<uint8_t> arg[MAXL];
};

static void alloc_batch(Batch& b, const Model& m, int B) {
  b.B = B; b.nl = m.nl;
  int cin = m.L[0].cin;
  b.z0.assign(size_t(B) * psz(32, 32, cin), 0.f);
  for (int l = 0; l < m.nl; l++) {
    b.H[l] = m.L[l].h; b.W[l] = m.L[l].wd; b.C[l] = m.L[l].cout;
    b.z[l].assign(size_t(B) * psz(b.H[l], b.W[l], b.C[l]), 0.f);
    b.xcol[l].assign(size_t(B) * b.H[l] * b.W[l] * m.L[l].k, 0.f);
    if (m.L[l].pool_after) {
      int ph = b.H[l] / 2, pw = b.W[l] / 2;
      b.pool[l].assign(size_t(B) * psz(ph, pw, b.C[l]), 0.f);
      b.arg[l].assign(size_t(B) * ph * pw * b.C[l], 0);
    }
  }
  b.pooled.assign(size_t(B) * m.head.in, 0.f);
  b.logits.assign(size_t(B) * NC, 0.f);
}

static void conv_into(const Layer& L, const float* prev, bool relu, float* out, float* xcol) {
  const int M = L.h * L.wd;
  im2col(prev, L.h, L.wd, L.cin, relu, xcol);
  vector<float> tmp(size_t(M) * L.cout);
  gemm(xcol, L.w.data(), tmp.data(), M, L.k, L.cout);
  const int Wp = L.wd + 2;
#pragma omp parallel for schedule(static)
  for (int y = 0; y < L.h; y++) for (int x = 0; x < L.wd; x++)
    std::memcpy(out + ((size_t(y + 1) * Wp + (x + 1)) * L.cout), tmp.data() + (size_t(y) * L.wd + x) * L.cout, size_t(L.cout) * sizeof(float));
}

static void pool_fwd(const float* z, int H, int W, int C, float* pool, uint8_t* arg) {
  const int Wp = W + 2, ph = H / 2, pw = W / 2;
#pragma omp parallel for collapse(2) schedule(static)
  for (int py = 0; py < ph; py++) for (int px = 0; px < pw; px++) for (int c = 0; c < C; c++) {
    float best = -1e30f; int bi = 0;
    for (int j = 0; j < 4; j++) {
      int yy = 1 + py * 2 + (j >> 1), xx = 1 + px * 2 + (j & 1);
      float v = z[(size_t(yy) * Wp + xx) * C + c];
      if (v > best) { best = v; bi = j; }
    }
    int yy = 1 + py * 2 + (bi >> 1), xx = 1 + px * 2 + (bi & 1);
    pool[(size_t(py + 1) * (pw + 2) + (px + 1)) * C + c] = z[(size_t(yy) * Wp + xx) * C + c];
    arg[(size_t(py) * pw + px) * C + c] = uint8_t(bi);
  }
}

static void head_fwd(const Model& m, const float* zlast, int H, float* pooled, float* logits) {
  const int C = m.L[m.nl - 1].cout;
  const int grid = H / 4, cell = 4;
  const int Wp = H + 2;
  for (int qy = 0; qy < grid; qy++) for (int qx = 0; qx < grid; qx++) for (int c = 0; c < C; c++) {
    float s = 0;
    for (int y = 0; y < cell; y++) for (int x = 0; x < cell; x++) {
      float v = zlast[(size_t(qy * cell + y + 1) * Wp + (qx * cell + x + 1)) * C + c];
      s += v > 0 ? v : 0.f;
    }
    pooled[(qy * grid + qx) * C + c] = s * (1.f / float(cell * cell));
  }
  const int IN = m.head.in;
  for (int k = 0; k < NC; k++) {
    __m512 a = _mm512_setzero_ps();
    for (int j = 0; j < IN; j += 16) a = _mm512_fmadd_ps(_mm512_loadu_ps(&m.head.w[size_t(k) * IN + j]), _mm512_loadu_ps(pooled + j), a);
    logits[k] = _mm512_reduce_add_ps(a) + m.head.b[k];
  }
}

static void forward_one(Model& m, const float* px, Batch& b, int bi, bool train) {
  const int cin = m.L[0].cin;
  float* z0 = b.z0.data() + size_t(bi) * psz(32, 32, cin);
  std::fill(z0, z0 + psz(32, 32, cin), 0.f);
  for (int y = 0; y < 32; y++) for (int x = 0; x < 32; x++)
    for (int c = 0; c < cin; c++) z0[(size_t(y + 1) * 34 + (x + 1)) * cin + c] = px[(y * 32 + x) * cin + c];
  for (int l = 0; l < m.nl; l++) {
    const Layer& L = m.L[l];
    const float* prev = l == 0 ? z0 : (m.L[l - 1].pool_after ? b.pool[l - 1].data() + size_t(bi) * psz(b.H[l - 1] / 2, b.W[l - 1] / 2, b.C[l - 1]) : b.z[l - 1].data() + size_t(bi) * psz(b.H[l - 1], b.W[l - 1], b.C[l - 1]));
    float* z = b.z[l].data() + size_t(bi) * psz(b.H[l], b.W[l], b.C[l]);
    conv_into(L, prev, l != 0, z, b.xcol[l].data() + size_t(bi) * b.H[l] * b.W[l] * L.k);
    bn_forward(z, b.H[l], b.W[l], L.cout, m.L[l], m.bn_mean[l], m.bn_invstd[l], train);
    if (L.pool_after) {
      float* pp = b.pool[l].data() + size_t(bi) * psz(b.H[l] / 2, b.W[l] / 2, b.C[l]);
      uint8_t* ar = b.arg[l].data() + size_t(bi) * (b.H[l] / 2) * (b.W[l] / 2) * b.C[l];
      pool_fwd(z, b.H[l], b.W[l], b.C[l], pp, ar);
    }
  }
  const int li = m.nl - 1;
  float* zp = b.z[li].data() + size_t(bi) * psz(b.H[li], b.W[li], b.C[li]);
  head_fwd(m, zp, b.H[li], b.pooled.data() + size_t(bi) * m.head.in, b.logits.data() + size_t(bi) * NC);
}

static float softmax_ce(const float* logits, int label, float* dl) {
  float mx = -1e30f;
  for (int i = 0; i < NC; i++) mx = std::max(mx, logits[i]);
  float sum = 0, p[NC];
  for (int i = 0; i < NC; i++) { p[i] = std::exp(logits[i] - mx); sum += p[i]; }
  for (int i = 0; i < NC; i++) p[i] /= sum;
  const float eps = 0.05f;
  for (int i = 0; i < NC; i++) dl[i] = p[i] - ((i == label) ? 1.f - eps : 0.f) - eps / NC;
  return -std::log(std::max(1e-9f, p[label]));
}

static void backward_one(Batch& b, Model& m, int bi, const float* dlogits, vector<float>& dz, vector<float>& dc, vector<float>& dx) {
  const int IN = m.head.in;
  const float* pe = b.pooled.data() + size_t(bi) * IN;
  vector<float> gpe(IN, 0.f);
  for (int k = 0; k < NC; k++) {
    for (int j = 0; j < IN; j++) m.head.gw[size_t(k) * IN + j] += dlogits[k] * pe[j];
    m.head.gb[k] += dlogits[k];
  }
  for (int j = 0; j < IN; j++) {
    float s = 0;
    for (int k = 0; k < NC; k++) s += m.head.w[size_t(k) * IN + j] * dlogits[k];
    gpe[j] = s;
  }
  const int li = m.nl - 1;
  const int C = b.C[li], H = b.H[li], Wd = b.W[li], grid = H / 4, cell = 4, Wp = H + 2;
  float* zl = b.z[li].data() + size_t(bi) * psz(H, Wd, C);
  std::fill(dz.begin(), dz.end(), 0.f);
  for (int qy = 0; qy < grid; qy++) for (int qx = 0; qx < grid; qx++) for (int c = 0; c < C; c++) {
    float g = gpe[(qy * grid + qx) * C + c] / float(cell * cell);
    for (int y = 0; y < cell; y++) for (int x = 0; x < cell; x++) {
      int yy = qy * cell + y, xx = qx * cell + x;
      float& d = dz[size_t(yy * Wd + xx) * C + c];
      d = g * (zl[(size_t(yy + 1) * Wp + (xx + 1)) * C + c] > 0 ? 1.f : 0.f);
    }
  }
  for (int l = m.nl - 1; l >= 0; l--) {
    Layer& L = m.L[l];
    bn_backward(dz.data(), b.z[l].data() + size_t(bi) * psz(b.H[l], b.W[l], b.C[l]), b.H[l], b.W[l], b.C[l], m.L[l], m.bn_mean[l], m.bn_invstd[l], dc.data());
    const int M = b.H[l] * b.W[l];
    conv_dw(b.xcol[l].data() + size_t(bi) * M * L.k, dc.data(), M, L.k, L.cout, L.gw.data());
    if (l > 0) {
      conv_dx(dc.data(), L, M, dx.data());  // grad wrt prev layer output
      if (m.L[l - 1].pool_after) {
        // route pool grad into dZ of l-1
        const int ph = m.L[l - 1].h / 2, pw = m.L[l - 1].wd / 2, pc = m.L[l - 1].cout;
        std::fill(dz.begin(), dz.end(), 0.f);
        const uint8_t* ar = b.arg[l - 1].data() + size_t(bi) * ph * pw * pc;
        for (int py = 0; py < ph; py++) for (int px = 0; px < pw; px++) for (int c = 0; c < pc; c++) {
          int arg = ar[(size_t(py) * pw + px) * pc + c];
          int yy = py * 2 + (arg >> 1), xx = px * 2 + (arg & 1);
          dz[size_t(yy * m.L[l - 1].wd + xx) * pc + c] += dx[size_t(py * pw + px) * pc + c];
        }
      } else {
        std::copy(dx.begin(), dx.begin() + size_t(M) * L.cin, dz.begin());
      }
      // relu mask of previous layer
      const Layer& P = m.L[l - 1];
      const float* zp = b.z[l - 1].data() + size_t(bi) * psz(P.h, P.wd, P.cout);
      const int Wp2 = P.wd + 2;
      for (int y = 0; y < P.h; y++) for (int x = 0; x < P.wd; x++) for (int c = 0; c < P.cout; c++)
        if (zp[(size_t(y + 1) * Wp2 + (x + 1)) * P.cout + c] <= 0) dz[size_t(y * P.wd + x) * P.cout + c] = 0;
    }
  }
}

// ---------------------------------------------------------------------------
// Adam / fold / save / load
// ---------------------------------------------------------------------------
static void adam_vec(vector<float>& p, vector<float>& g, vector<float>& m, vector<float>& v, float lr, float wd, int t) {
  const float b1 = 0.9f, b2 = 0.999f, bc1 = 1 - std::pow(b1, float(t)), bc2 = 1 - std::pow(b2, float(t));
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < p.size(); i++) {
    float gi = g[i] + wd * p[i];
    m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi;
    p[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + 1e-8f); g[i] = 0;
  }
}

static void adam_model(Model& m, float lr, float wd, int t) {
  for (int l = 0; l < m.nl; l++) {
    Layer& L = m.L[l];
    adam_vec(L.w, L.gw, L.mw, L.vw, lr, wd, t);
    adam_vec(L.gamma, L.ggamma, L.mg, L.vg, lr, 0.f, t);
    adam_vec(L.beta, L.gbeta, L.mb, L.vb, lr, 0.f, t);
    transpose_weights(L);
  }
  adam_vec(m.head.w, m.head.gw, m.head.mw, m.head.vw, lr, wd, t);
  adam_vec(m.head.b, m.head.gb, m.head.mb, m.head.vb, lr, 0.f, t);
}

static void pack_weights(Model& m);

static void fold_bn(Model& m) {
  for (int l = 0; l < m.nl; l++) {
    Layer& L = m.L[l];
    for (int c = 0; c < L.cout; c++) {
      float sc = L.gamma[c] / std::sqrt(L.run_var[c] + 1e-5f);
      for (int k = 0; k < L.k; k++) L.w[size_t(k) * L.cout + c] *= sc;
      L.bias[c] = L.beta[c] - L.run_mean[c] * sc;
      L.gamma[c] = 1.f; L.beta[c] = 0.f;
    }
  }
  pack_weights(m);
}

static void pack_weights(Model& m) {
  for (int l = 0; l < m.nl; l++) {
    Layer& L = m.L[l];
    for (int cb = 0; cb < L.cout; cb += 16) for (int k = 0; k < L.k; k++) for (int lane = 0; lane < 16; lane++)
      L.wf[(size_t(cb / 16) * L.k + k) * 16 + lane] = L.w[size_t(k) * L.cout + cb + lane];
    for (int hb = 0; hb < L.cout; hb += 8) for (int k = 0; k < L.k; k++) for (int lane = 0; lane < 8; lane++)
      L.wf8[(size_t(hb / 8) * L.k + k) * 8 + lane] = L.w[size_t(k) * L.cout + hb + lane];
  }
}

struct Header {
  char magic[8];
  uint32_t nl, cin[MAXL], cout[MAXL], dims[MAXL], pool[MAXL], head_in;
  uint32_t epochs;
  float dead[MAXL], train_acc;
  uint32_t reserved[8];
};

static void save_weights(const string& path, const Model& m) {
  std::ofstream f(path, std::ios::binary);
  if (!f) die("cannot write " + path);
  Header h{};
  std::memcpy(h.magic, "CMRGB007", 8);
  h.nl = m.nl; h.head_in = m.head.in; h.epochs = 1;
  for (int l = 0; l < m.nl; l++) {
    h.cin[l] = m.L[l].cin; h.cout[l] = m.L[l].cout; h.dims[l] = m.L[l].h; h.pool[l] = m.L[l].pool_after;
    h.dead[l] = 0;
  }
  f.write(reinterpret_cast<const char*>(&h), sizeof(h));
  for (int l = 0; l < m.nl; l++) {
    f.write(reinterpret_cast<const char*>(m.L[l].wf.data()), std::streamsize(m.L[l].wf.size() * sizeof(float)));
    f.write(reinterpret_cast<const char*>(m.L[l].bias.data()), std::streamsize(m.L[l].bias.size() * sizeof(float)));
    f.write(reinterpret_cast<const char*>(m.L[l].gamma.data()), std::streamsize(m.L[l].gamma.size() * sizeof(float)));
    f.write(reinterpret_cast<const char*>(m.L[l].beta.data()), std::streamsize(m.L[l].beta.size() * sizeof(float)));
  }
  f.write(reinterpret_cast<const char*>(m.head.w.data()), std::streamsize(m.head.w.size() * sizeof(float)));
  f.write(reinterpret_cast<const char*>(m.head.b.data()), std::streamsize(m.head.b.size() * sizeof(float)));
}

static void load_weights(const string& path, Model& m) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open " + path);
  Header h{};
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (std::memcmp(h.magic, "CMRGB007", 8) != 0) die("bad RGB weights magic");
  m.nl = int(h.nl);
  for (int l = 0; l < m.nl; l++) {
    Layer& L = m.L[l];
    L.cin = int(h.cin[l]); L.cout = int(h.cout[l]); L.h = int(h.dims[l]); L.wd = int(h.dims[l]);
    L.k = 9 * L.cin; L.pool_after = h.pool[l];
    L.wf.resize(size_t(L.cout) * L.k); L.wf8.resize(size_t(L.cout) * L.k);
    L.bias.resize(L.cout); L.bias.assign(L.cout, 0.f);
    f.read(reinterpret_cast<char*>(L.wf.data()), std::streamsize(L.wf.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(L.bias.data()), std::streamsize(L.bias.size() * sizeof(float)));
    L.gamma.resize(L.cout); L.beta.resize(L.cout);
    L.run_mean.assign(L.cout, 0.f); L.run_var.assign(L.cout, 1.f);
    m.bn_mean[l].assign(L.cout, 0.f); m.bn_invstd[l].assign(L.cout, 1.f);
    f.read(reinterpret_cast<char*>(L.gamma.data()), std::streamsize(L.gamma.size() * sizeof(float)));
    f.read(reinterpret_cast<char*>(L.beta.data()), std::streamsize(L.beta.size() * sizeof(float)));
    L.w.assign(size_t(L.k) * L.cout, 0.f);
    for (int cb = 0; cb < L.cout; cb += 16) for (int k = 0; k < L.k; k++) for (int lane = 0; lane < 16; lane++)
      L.w[size_t(k) * L.cout + cb + lane] = L.wf[(size_t(cb / 16) * L.k + k) * 16 + lane];
    pack_weights(m);
  }
  m.head.in = int(h.head_in);
  m.head.w.resize(size_t(NC) * m.head.in); m.head.b.resize(NC);
  f.read(reinterpret_cast<char*>(m.head.w.data()), std::streamsize(m.head.w.size() * sizeof(float)));
  f.read(reinterpret_cast<char*>(m.head.b.data()), std::streamsize(m.head.b.size() * sizeof(float)));
}

// ---------------------------------------------------------------------------
// Dense inference + stats
// ---------------------------------------------------------------------------
struct Bufs {
  vector<float> z[MAXL], pool;
  vector<uint8_t> arg;
};

static void alloc_bufs(Bufs& b, const Model& m) {
  for (int l = 0; l < m.nl; l++) b.z[l].assign(psz(m.L[l].h, m.L[l].wd, m.L[l].cout), 0.f);
  int last = m.nl - 1;
  if (m.L[last].pool_after) b.pool.assign(psz(m.L[last].h / 2, m.L[last].wd / 2, m.L[last].cout), 0.f);
  b.arg.assign(size_t(32) * 32 * 256, 0);
}

static void conv_dense(const Layer& L, const float* prev, bool relu, float* out) {
  const int H = L.h, W = L.wd, cin = L.cin, cout = L.cout, Wp = W + 2;
#pragma omp parallel for collapse(2) schedule(static)
  for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
    for (int cb = 0; cb < cout; cb += 16) {
      __m512 acc = _mm512_loadu_ps(&L.bias[cb]);
      for (int tap = 0; tap < 9; tap++) {
        int yy = y + (tap / 3), xx = x + (tap % 3);
        const float* src = prev + ((size_t(yy) * Wp + xx)) * cin;
        const float* wp = &L.wf[(size_t(cb / 16) * L.k + tap * cin) * 16];
        for (int ci = 0; ci < cin; ci++) {
          float v = src[ci];
          if (relu && v <= 0) continue;
          acc = _mm512_fmadd_ps(_mm512_set1_ps(v), _mm512_loadu_ps(wp + size_t(ci) * 16), acc);
        }
      }
      _mm512_storeu_ps(out + ((size_t(y + 1) * Wp + (x + 1)) * cout) + cb, acc);
    }
  }
}

static void dense_forward(const Model& m, const float* px, Bufs& b, float* logits, float* feat = nullptr) {
  const int cin = m.L[0].cin;
  std::vector<float> in(psz(32, 32, cin), 0.f);
  for (int y = 0; y < 32; y++) for (int x = 0; x < 32; x++) for (int c = 0; c < cin; c++) in[(size_t(y + 1) * 34 + (x + 1)) * cin + c] = px[(y * 32 + x) * cin + c];
  const float* prev = in.data();
  for (int l = 0; l < m.nl; l++) {
    conv_dense(m.L[l], prev, l != 0, b.z[l].data());
    if (!std::getenv("CM_NOBN")) {
      const Layer& L = m.L[l]; int C = L.cout, Wp = L.wd + 2, M = L.h * L.wd;
      std::vector<double> su(C, 0.0), sq(C, 0.0);
      for (int y = 0; y < L.h; y++) for (int x = 0; x < L.wd; x++) for (int c = 0; c < C; c++) {
        double v = b.z[l][(size_t(y + 1) * Wp + (x + 1)) * C + c]; su[c] += v; sq[c] += v * v;
      }
      for (int c = 0; c < C; c++) {
        double mu = su[c] / M, vv = std::max(1e-8, sq[c] / M - mu * mu);
        float iv = 1.f / std::sqrt(float(vv) + 1e-5f);
        for (int y = 0; y < L.h; y++) for (int x = 0; x < L.wd; x++) {
          float& v = b.z[l][(size_t(y + 1) * Wp + (x + 1)) * C + c];
          v = (v - float(mu)) * iv * L.gamma[c] + L.beta[c];
        }
      }
    }
    if (m.L[l].pool_after) {
      int ph = m.L[l].h / 2, pw = m.L[l].wd / 2;
      b.pool.assign(psz(ph, pw, m.L[l].cout), 0.f);
      b.arg.assign(size_t(ph) * pw * m.L[l].cout, 0);
      pool_fwd(b.z[l].data(), m.L[l].h, m.L[l].wd, m.L[l].cout, b.pool.data(), b.arg.data());
      prev = b.pool.data();
    } else prev = b.z[l].data();
  }
  vector<float> pooled(m.head.in, 0.f);
  head_fwd(m, b.z[m.nl - 1].data(), m.L[m.nl - 1].h, pooled.data(), logits);
  if (feat) std::memcpy(feat, pooled.data(), sizeof(float) * m.head.in);
}

static double eval_acc(const Model& m, const ImageSet& s, int from, int to) {
  Bufs b; alloc_bufs(b, m);
  int ok = 0;
  for (int i = from; i < to; i++) {
    float logits[NC];
    dense_forward(m, &s.px[size_t(i) * s.h * s.w * s.c], b, logits);
    int best = 0;
    for (int k = 1; k < NC; k++) if (logits[k] > logits[best]) best = k;
    if (best == s.labels[i]) ok++;
  }
  return double(ok) / std::max(1, to - from);
}

static void dead_stats(const Model& m, const ImageSet& s, double* dead, double* total, int maxn = -1) {
  Bufs b; alloc_bufs(b, m);
  int n = maxn > 0 ? std::min(maxn, s.count) : s.count;
  for (int l = 0; l < m.nl; l++) { dead[l] = 0; total[l] = 0; }
  for (int i = 0; i < n; i++) {
    float logits[NC];
    dense_forward(m, &s.px[size_t(i) * s.h * s.w * s.c], b, logits);
    for (int l = 0; l < m.nl; l++) {
      const Layer& L = m.L[l];
      const float* z = b.z[l].data();
      for (int y = 0; y < L.h; y++) for (int x = 0; x < L.wd; x++) for (int c = 0; c < L.cout; c++) {
        total[l] += 1;
        if (z[(size_t(y + 1) * (L.wd + 2) + (x + 1)) * L.cout + c] <= 0) dead[l] += 1;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Training command
// ---------------------------------------------------------------------------
static int arch_nl = 6;
static void arch_dims(int nl, int* cin, int* cout, int* dim, bool* pool) {
  if (nl == 6) {
    int ci[6] = {3, 32, 64, 128, 128, 256};
    int co[6] = {32, 64, 128, 128, 256, 256};
    int di[6] = {32, 32, 16, 16, 8, 8};
    bool po[6] = {false, true, false, true, false, false};
    for (int i = 0; i < 6; i++) { cin[i] = ci[i]; cout[i] = co[i]; dim[i] = di[i]; pool[i] = po[i]; }
  } else {
    int ci[4] = {1, 32, 64, 128};
    int co[4] = {32, 64, 128, 128};
    int di[4] = {32, 32, 16, 16};
    bool po[4] = {false, true, false, false};
    for (int i = 0; i < 4; i++) { cin[i] = ci[i]; cout[i] = co[i]; dim[i] = di[i]; pool[i] = po[i]; }
  }
}

static void train_command(int argc, char** argv) {
  string trp, tep, outp, calp;
  int epochs = 80, B = 32;
  float lr = 3e-3f;
  for (int i = 2; i < argc; i++) {
    string a = argv[i];
    auto nx = [&]() { if (i + 1 >= argc) die("missing " + a); return string(argv[++i]); };
    if (a == "--train") trp = nx(); else if (a == "--test") tep = nx(); else if (a == "--out") outp = nx();
    else if (a == "--epochs") epochs = std::stoi(nx()); else if (a == "--batch") B = std::stoi(nx());
    else if (a == "--lr") lr = std::stof(nx()); else if (a == "--calib") calp = nx();
    else die("unknown " + a);
  }
  ImageSet tr = load_images(trp), te = load_images(tep);
  Model m;
  int cin[MAXL], cout[MAXL], dim[MAXL]; bool pool[MAXL];
  arch_dims(arch_nl, cin, cout, dim, pool);
  init_model(m, arch_nl, cin, cout, dim, pool, 0x1234abcdULL);
  Batch b; alloc_batch(b, m, B);
  std::mt19937 rng(0xc0ffee);
  vector<float> dz, dc, dx, dl(NC);
  size_t maxz = 0, maxc = 0;
  for (int l = 0; l < m.nl; l++) { maxz = std::max(maxz, size_t(b.H[l]) * b.W[l] * b.C[l]); maxc = std::max(maxc, size_t(b.H[l]) * b.W[l] * b.C[l]); }
  dz.resize(maxz); dc.resize(maxz); dx.resize(maxz);
  int t = 0;
  {
    Batch dbg; alloc_batch(dbg, m, 1);
    forward_one(m, &tr.px[0], dbg, 0, false);
    double m0=0,v0=0; int n0=32*32*3;
    for(int i=0;i<n0;i++){m0+=tr.px[i];v0+=tr.px[i]*tr.px[i];} m0/=n0; v0=v0/n0-m0*m0;
    double m1=0,v1=0; int n1=32*32*32;
    for(int y=0;y<32;y++)for(int x=0;x<32;x++)for(int c=0;c<32;c++){float q=dbg.z[0][(size_t(y+1)*34+(x+1))*32+c];m1+=q;v1+=q*q;}
    m1/=n1;v1=v1/n1-m1*m1;
    std::printf("stat input std %.4f conv1 std %.4f\n", std::sqrt(v0), std::sqrt(v1));
  }
  for (int ep = 1; ep <= epochs; ep++) {
    vector<int> ord(tr.count);
    std::iota(ord.begin(), ord.end(), 0);
    std::shuffle(ord.begin(), ord.end(), rng);
    double loss = 0, el = now_sec();
    float lre = lr * (ep > epochs * 3 / 4 ? 0.25f : ep > epochs / 2 ? 0.5f : 1.f);
    for (int s = 0; s < tr.count; s += B) {
      int nb = std::min(B, tr.count - s);
      std::fill(m.head.gw.begin(), m.head.gw.end(), 0.f); std::fill(m.head.gb.begin(), m.head.gb.end(), 0.f);
      for (int l = 0; l < m.nl; l++) {
        std::fill(m.L[l].gw.begin(), m.L[l].gw.end(), 0.f);
        std::fill(m.L[l].ggamma.begin(), m.L[l].ggamma.end(), 0.f);
        std::fill(m.L[l].gbeta.begin(), m.L[l].gbeta.end(), 0.f);
      }
      for (int bi = 0; bi < nb; bi++) {
        int idx = ord[s + bi];
        // shift + brightness augmentation
        vector<float> aug(size_t(32) * 32 * tr.c);
        int sx = std::getenv("CM_NOAUG") ? 0 : int(rng() % 5) - 2;
        int sy = std::getenv("CM_NOAUG") ? 0 : int(rng() % 5) - 2;
        float gain = std::getenv("CM_NOAUG") ? 1.f : 0.88f + 0.24f * float(rng() % 1000) / 1000.f;
        float bias = std::getenv("CM_NOAUG") ? 0.f : 0.02f * (float(rng() % 1000) / 1000.f - 0.5f);
        const float* src = &tr.px[size_t(idx) * 32 * 32 * tr.c];
        for (int y = 0; y < 32; y++) for (int x = 0; x < 32; x++) {
          int yy = std::clamp(y + sy, 0, 31), xx = std::clamp(x + sx, 0, 31);
          for (int c = 0; c < tr.c; c++) aug[(y * 32 + x) * tr.c + c] = std::clamp(src[(yy * 32 + xx) * tr.c + c] * gain + bias, 0.f, 1.f);
        }
        forward_one(m, aug.data(), b, bi, true);
        loss += softmax_ce(&b.logits[size_t(bi) * NC], tr.labels[idx], dl.data());
        backward_one(b, m, bi, dl.data(), dz, dc, dx);
      }
      if (t == 0 || t == 5) {
        double g0=0,g1=0; for(auto v:m.L[0].gw)g0+=v*v; for(auto v:m.L[1].gw)g1+=v*v;
        std::printf("gradnorm L0 %.6f L1 %.6f dc %.6f\n", std::sqrt(g0), std::sqrt(g1), 0.0);
      }
      float gs = 1.f / float(nb);
      for (int l = 0; l < m.nl; l++) {
        for (auto& v : m.L[l].gw) v *= gs;
        for (auto& v : m.L[l].ggamma) v *= gs;
        for (auto& v : m.L[l].gbeta) v *= gs;
      }
      for (auto& v : m.head.gw) v *= gs;
      for (auto& v : m.head.gb) v *= gs;
      adam_model(m, lre, 1e-4f, ++t);
    }
    if (ep % 5 == 0 || ep == 1) {
      double a = eval_acc(m, tr, 0, std::min(300, tr.count));
      double n0=0,n1=0,n2=0; for(auto v:m.L[0].w)n0+=v*v; for(auto v:m.L[1].w)n1+=v*v; for(auto v:m.L[3].w)n2+=v*v;
      std::printf("ep %d loss %.4f train300 %.4f %.1fs wn %.3f %.3f %.3f\n", ep, loss / tr.count, a, now_sec() - el, std::sqrt(n0),std::sqrt(n1),std::sqrt(n2));
      std::fflush(stdout);
    }
  }
  // Running BN statistics accumulated during training are the inference
  // statistics; fold them directly into the conv weights.
  if (std::getenv("CM_INST")) {
    Batch eb; alloc_batch(eb, m, 1);
    int ok = 0;
    for (int i = 0; i < te.count; i++) {
      forward_one(m, &te.px[size_t(i) * 32 * 32 * te.c], eb, 0, false);
      int best = 0; for (int k = 1; k < NC; k++) if (eb.logits[k] > eb.logits[best]) best = k;
      if (best == te.labels[i]) ok++;
    }
    std::printf("instance-norm test acc %.4f\n", double(ok) / te.count);
  }
  {
    Batch eb; alloc_batch(eb, m, 1);
    forward_one(m, &te.px[0], eb, 0, false);
    float lg[NC]; Bufs db; alloc_bufs(db, m); dense_forward(m, &te.px[0], db, lg);
    double md=0; for(int k=0;k<NC;k++) md=std::max(md,(double)std::fabs(eb.logits[k]-lg[k]));
    std::printf("probe maxdiff %.5f\n", md);
    for(int l=0;l<m.nl;l++){double d=0;int Wp=m.L[l].wd+2;
      for(int y=0;y<m.L[l].h;y++)for(int x=0;x<m.L[l].wd;x++)for(int c=0;c<m.L[l].cout;c++){
        float q=eb.z[l][(size_t(y+1)*Wp+(x+1))*m.L[l].cout+c], r=db.z[l][(size_t(y+1)*Wp+(x+1))*m.L[l].cout+c]; d=std::max(d,(double)std::fabs(q-r));}
      std::printf("  z%d maxdiff %.5f\n", l+1, d);}
  }
  pack_weights(m);
  double acc = eval_acc(m, te, 0, te.count);
  std::printf("folded test acc %.4f\n", acc);
  double dead[MAXL], total[MAXL];
  dead_stats(m, te, dead, total);
  for (int l = 0; l < m.nl; l++) std::printf("dead L%d %.4f\n", l + 1, dead[l] / total[l]);
  save_weights(outp, m);
  std::printf("saved %s\n", outp.c_str());
}


static void fithead_command(int argc, char** argv) {
  string wp, tp, op; double lam = 1e-2;
  for (int i = 2; i < argc; i++) {
    string a = argv[i]; auto nx = [&]() { return string(argv[++i]); };
    if (a == "--weights") wp = nx(); else if (a == "--train") tp = nx();
    else if (a == "--out") op = nx(); else if (a == "--lambda") lam = std::stod(nx()); else die("unknown " + a);
  }
  Model m; load_weights(wp, m);
  ImageSet tr = load_images(tp);
  const int F = m.head.in;
  std::vector<double> A(size_t(F) * F, 0.0), B(size_t(F) * NC, 0.0);
  Bufs b; alloc_bufs(b, m);
  std::vector<float> feat(F);
  for (int i = 0; i < tr.count; i++) {
    float dummy[NC]; dense_forward(m, &tr.px[size_t(i) * tr.h * tr.w * tr.c], b, dummy, feat.data());
    int y = tr.labels[i];
    for (int j = 0; j < F; j++) {
      double fj = feat[j];
      for (int k = 0; k < F; k++) A[size_t(j) * F + k] += fj * feat[k];
      B[size_t(j) * NC + y] += fj;
    }
  }
  for (int j = 0; j < F; j++) A[size_t(j) * F + j] += lam * tr.count;
  // Cholesky A = L L^T
  std::vector<double> Lc(size_t(F) * F, 0.0);
  for (int i = 0; i < F; i++) for (int j = 0; j <= i; j++) {
    double s = A[size_t(i) * F + j];
    for (int k = 0; k < j; k++) s -= Lc[size_t(i) * F + k] * Lc[size_t(j) * F + k];
    if (i == j) Lc[size_t(i) * F + j] = std::sqrt(std::max(1e-12, s));
    else Lc[size_t(i) * F + j] = s / Lc[size_t(j) * F + j];
  }
  std::vector<double> X(size_t(F) * NC, 0.0);
  for (int col = 0; col < NC; col++) {
    std::vector<double> z(F), x(F);
    for (int i = 0; i < F; i++) { double s = B[size_t(i) * NC + col]; for (int k = 0; k < i; k++) s -= Lc[size_t(i) * F + k] * z[k]; z[i] = s / Lc[size_t(i) * F + i]; }
    for (int i = F - 1; i >= 0; i--) { double s = z[i]; for (int k = i + 1; k < F; k++) s -= Lc[size_t(k) * F + i] * x[k]; x[i] = s / Lc[size_t(i) * F + i]; }
    for (int i = 0; i < F; i++) X[size_t(i) * NC + col] = x[i];
  }
  for (int k = 0; k < NC; k++) { for (int j = 0; j < F; j++) m.head.w[size_t(k) * F + j] = float(X[size_t(j) * NC + k]); m.head.b[k] = 0.f; }
  pack_weights(m);
  save_weights(op, m);
  std::printf("fithead saved %s lambda %.4f\n", op.c_str(), lam);
}

static void eval_command(int argc, char** argv) {
  string wp, tp;
  for (int i = 2; i < argc; i++) {
    string a = argv[i];
    auto nx = [&]() { return string(argv[++i]); };
    if (a == "--weights") wp = nx(); else if (a == "--test") tp = nx(); else die("unknown " + a);
  }
  Model m; load_weights(wp, m);
  ImageSet te = load_images(tp);
  Bufs b; alloc_bufs(b, m);
  int ok = 0; int conf[10][10] = {};
  { Batch eb; alloc_batch(eb, m, 1); forward_one(m, &te.px[0], eb, 0, false);
    float lg[NC]; Bufs dbg; alloc_bufs(dbg, m); dense_forward(m, &te.px[0], dbg, lg);
    for(int l=0;l<m.nl;l++){double d=0;int Wp=m.L[l].wd+2;
      for(int y=0;y<m.L[l].h;y++)for(int x=0;x<m.L[l].wd;x++)for(int c=0;c<m.L[l].cout;c++){
        float q=eb.z[l][(size_t(y+1)*Wp+(x+1))*m.L[l].cout+c], r=dbg.z[l][(size_t(y+1)*Wp+(x+1))*m.L[l].cout+c]; d=std::max(d,(double)std::fabs(q-r));}
      std::printf("eval z%d maxdiff %.5f\n", l+1, d);} }
  for (int i = 0; i < te.count; i++) {
    float logits[NC]; dense_forward(m, &te.px[size_t(i) * te.h * te.w * te.c], b, logits);
    int best = 0; for (int k = 1; k < NC; k++) if (logits[k] > logits[best]) best = k;
    conf[te.labels[i]][best]++;
    if (best == te.labels[i]) ok++;
    if (i < 3) { std::printf("i%d label %d pred %d logits", i, te.labels[i], best); for (float v : logits) std::printf(" %.2f", v); std::printf("\n"); }
  }
  std::printf("acc %.4f\n", double(ok) / te.count);
  for (int i = 0; i < NC; i++) { std::printf("true %d:", i); for (int j = 0; j < NC; j++) std::printf(" %d", conf[i][j]); std::printf("\n"); }
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s train|eval ...\n", argv[0]); return 2; }
  arch_nl = 6;
  string cmd = argv[1];
  if (cmd == "train") train_command(argc, argv);
  else if (cmd == "eval") eval_command(argc, argv);
  else if (cmd == "fithead") fithead_command(argc, argv);
  else { std::fprintf(stderr, "unknown command\n"); return 2; }
  return 0;
}
