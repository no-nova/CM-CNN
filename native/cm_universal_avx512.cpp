// CM-CNN Universal AVX-512 inference engine (ResNet-18 / MobileNetV2, 32x32x3 RGB).
//
// Nodes are a small DAG (input, conv, depthwise, residual add, global-avg-pool, linear).
// Modes:
//   M0 dense-avx512        : im2col + AVX-512 GEMM, no sparsity of any kind
//   M1 input-sparse-only   : exact non-zero scalar input compaction (no permutation)
//   M2 clustered-tap-only  : M1 on an isomorphically permuted (8-lane clustered) model
//   M3 cm-v4-static        : M2 + 8-lane online CM fiber gate (static context)
//   M4 cm-v4-temporal      : M3 + temporal context bits on the RGB video stream
//   block-matched-random-skip / unclustered-random-skip : controls at M3's MAC budget
//
// Build:
//   g++ -O3 -march=native -mavx512f -mavx512vl -mavx512bw -mfma -std=c++20 \
//       native/cm_universal_avx512.cpp -o experiments/cm-cnn-universal/cm_universal_avx512
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <immintrin.h>

using namespace std;

static double now_sec() {
  return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}
[[noreturn]] static void die(const string& s) { fprintf(stderr, "error: %s\n", s.c_str()); exit(1); }

// ---------------------------------------------------------------------------
// dataset
// ---------------------------------------------------------------------------
struct ImageSet { int n = 0, h = 0, w = 0, c = 0; vector<uint8_t> lab; vector<float> px; };
struct StreamSet { int seqs = 0, frames = 0, h = 0, w = 0, c = 0; vector<uint8_t> lab; vector<float> px; };

static ImageSet load_images(const string& p) {
  ifstream f(p, ios::binary);
  if (!f) die("open " + p);
  char mg[8]; f.read(mg, 8);
  if (memcmp(mg, "CMRGB001", 8)) die("image magic " + p);
  uint32_t n, h, w, c; f.read((char*)&n, 4); f.read((char*)&h, 4); f.read((char*)&w, 4); f.read((char*)&c, 4);
  ImageSet s; s.n = n; s.h = h; s.w = w; s.c = c;
  f.seekg(32, ios::beg);
  s.lab.resize(n); f.read((char*)s.lab.data(), n);
  f.seekg(off_t(32 + n + ((4 - n % 4) % 4)), ios::beg);
  s.px.resize(size_t(n) * h * w * c); f.read((char*)s.px.data(), s.px.size() * 4);
  if (!f) die("short read " + p);
  return s;
}

static StreamSet load_stream(const string& p) {
  ifstream f(p, ios::binary);
  if (!f) die("open " + p);
  char mg[8]; f.read(mg, 8);
  if (memcmp(mg, "CMRGBV01", 8)) die("stream magic " + p);
  uint32_t seqs, fr, h, w, c; f.read((char*)&seqs, 4); f.read((char*)&fr, 4); f.read((char*)&h, 4); f.read((char*)&w, 4); f.read((char*)&c, 4);
  StreamSet s; s.seqs = seqs; s.frames = fr; s.h = h; s.w = w; s.c = c;
  f.seekg(32, ios::beg);
  s.lab.resize(seqs); f.read((char*)s.lab.data(), seqs);
  f.seekg(off_t(32 + seqs + ((4 - seqs % 4) % 4)), ios::beg);
  s.px.resize(size_t(seqs) * fr * h * w * c); f.read((char*)s.px.data(), s.px.size() * 4);
  if (!f) die("short read " + p);
  return s;
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------
enum { OP_IN = 0, OP_CONV = 1, OP_DW = 2, OP_ADD = 3, OP_GAP = 4, OP_FC = 5 };
enum Mode { M_DENSE = 0, M_SPARSE = 1, M_CLUSTER = 2, M_CM = 3, M_TEMPORAL = 4, M_RAND_BLOCK = 5, M_RAND_UNC = 6, M_LEARN = 7, M2Z = 8, M3Z = 9 };

struct Node {
  int op = 0, i1 = 0, i2 = 0, ci = 0, co = 0, h = 0, w = 0, st = 1, g = 1, relu = 0, k = 1;
  int cp = 0, F = 0, kdim = 0;
  vector<float> wf;   // [kdim][cp]   conv: kdim = taps*ci (tap major), dw: kdim = taps, fc: kdim = ci
  vector<float> b;    // [cp]
};

struct Model { int type = 0, n = 0, cls = 10; vector<Node> nd; };

static int cpad(int c) { return (c + 15) & ~15; }

static Model load_model(const string& p) {
  ifstream f(p, ios::binary);
  if (!f) die("open " + p);
  char mg[8]; f.read(mg, 8);
  if (memcmp(mg, "CMUNIV02", 8)) die("model magic " + p);
  uint32_t t, n, cls;
  f.read((char*)&t, 4); f.read((char*)&n, 4); f.read((char*)&cls, 4);
  Model m; m.type = t; m.n = n; m.cls = cls; m.nd.resize(n);
  for (int i = 0; i < n; i++) {
    int hd[12]; f.read((char*)hd, 48);
    Node& q = m.nd[i];
    q.op = hd[0]; q.i1 = hd[1]; q.i2 = hd[2]; q.ci = hd[3]; q.co = hd[4];
    q.h = hd[5]; q.w = hd[6]; q.st = hd[7]; q.g = hd[8]; q.relu = hd[9]; q.k = hd[10];
    q.cp = cpad(q.co); q.F = q.cp / 8;
    if (q.op == OP_CONV) {
      const int taps = q.k;                 // 9 or 1
      q.kdim = taps * q.ci;
      vector<float> raw(size_t(q.co) * q.ci * taps);
      f.read((char*)raw.data(), raw.size() * 4);
      q.wf.assign(size_t(q.kdim) * q.cp, 0.f);
      for (int tp = 0; tp < taps; tp++)
        for (int ci = 0; ci < q.ci; ci++)
          for (int co = 0; co < q.co; co++)
            q.wf[size_t(tp * q.ci + ci) * q.cp + co] = raw[(size_t(co) * q.ci + ci) * taps + tp];
    } else if (q.op == OP_DW) {
      q.kdim = 9;
      vector<float> raw(size_t(q.co) * 9);
      f.read((char*)raw.data(), raw.size() * 4);
      q.wf.assign(size_t(9) * q.cp, 0.f);
      for (int tp = 0; tp < 9; tp++)
        for (int co = 0; co < q.co; co++) q.wf[size_t(tp) * q.cp + co] = raw[size_t(co) * 9 + tp];
    } else if (q.op == OP_FC) {
      q.kdim = q.ci;
      vector<float> raw(size_t(q.co) * q.ci);
      f.read((char*)raw.data(), raw.size() * 4);
      q.wf.assign(size_t(q.ci) * q.cp, 0.f);
      for (int kk = 0; kk < q.ci; kk++)
        for (int co = 0; co < q.co; co++) q.wf[size_t(kk) * q.cp + co] = raw[size_t(co) * q.ci + kk];
    }
    if (q.op == OP_CONV || q.op == OP_DW || q.op == OP_FC) {
      q.b.assign(q.cp, 0.f);
      f.read((char*)q.b.data(), q.co * 4);
    }
  }
  return m;
}

// ---------------------------------------------------------------------------
// tensors (channel-last, spatially padded by 1 with zeros on the border)
// ---------------------------------------------------------------------------
struct Ten {
  int h = 0, w = 0, c = 0, cp = 0, F = 0;
  vector<float> d;   // (h+2)*(w+2)*cp
  const float* pix(int y, int x) const { return &d[(size_t(y + 1) * (w + 2) + (x + 1)) * cp]; }
  float* pix(int y, int x) { return &d[(size_t(y + 1) * (w + 2) + (x + 1)) * cp]; }
};

static void alloc_ten(Ten& t, int h, int w, int c) {
  t.h = h; t.w = w; t.c = c; t.cp = cpad(c); t.F = t.cp / 8;
  t.d.assign(size_t(h + 2) * (w + 2) * t.cp, 0.f);
}

static inline float actv(float v, int r) {
  if (r == 1) return v > 0 ? v : 0.f;
  if (r == 2) return v < 0 ? 0.f : (v > 6.f ? 6.f : v);
  return v;
}

// ---------------------------------------------------------------------------
// channel-space unification: isomorphic permutation constraints
// ---------------------------------------------------------------------------
struct UF {
  vector<int> p;
  explicit UF(int n) : p(n) { iota(p.begin(), p.end(), 0); }
  int find(int a) { while (p[a] != a) { p[a] = p[p[a]]; a = p[a]; } return a; }
  void uni(int a, int b) { a = find(a); b = find(b); if (a != b) p[b] = a; }
};

// spaces: node->space root.  Constraints:
//   add      : both operands and the result share one channel space
//   dw       : input and output share one space (depthwise is channel-diagonal)
//   gap      : channel-wise, input and output share one space
static vector<int> build_spaces(const Model& m, const vector<int>& space_sample, int& nsp) {
  UF uf(m.n);
  for (int i = 1; i < m.n; i++) {
    const Node& q = m.nd[i];
    if (q.op == OP_ADD) { uf.uni(q.i1, i); uf.uni(q.i2, i); }
    else if (q.op == OP_DW) uf.uni(q.i1, i);
    else if (q.op == OP_GAP) uf.uni(q.i1, i);
  }
  vector<int> sp(m.n);
  for (int i = 0; i < m.n; i++) sp[i] = uf.find(i);
  // canonical renumber
  vector<int> map_(m.n, -1);
  nsp = 0;
  for (int i = 0; i < m.n; i++) { if (map_[sp[i]] < 0) map_[sp[i]] = nsp++; sp[i] = map_[sp[i]]; }
  (void)space_sample;
  return sp;
}

// ---------------------------------------------------------------------------
// activation sampling for clustering
// ---------------------------------------------------------------------------
struct Bits { vector<uint64_t> w; int cnt = 0; };

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------
struct GateCfg {
  int mincnt = 6;
  int th = 216;            // skip when dead_count/total >= th/256
  int anchor_mask = 15;
  float guard_act = 0.80f;
  int gate_start = 2;      // gate only layers at/after this topological index
  int NH = 256;            // context slots
};

struct Gate {
  GateCfg cfg;
  vector<uint16_t> tab;        // [node][fiber][NH]
  vector<size_t> taboff;       // per node
  vector<float> actema;        // [node][fiber]
  vector<size_t> fiboff;       // per node
  vector<vector<uint8_t>> actmap;   // [node] -> (h+2)*(w+2)*F
  vector<vector<uint8_t>> prevmap;  // temporal
  vector<float> skipq;         // per node random-control probability
  vector<uint8_t> tabth;       // per node calibrated skip threshold (0 = gate off)
  uint64_t rng = 0x9E3779B97F4A7C15ull;
};

static inline float rnd01(uint64_t& s) {
  s ^= s << 13; s ^= s >> 7; s ^= s << 17;
  return float((s >> 40) & 0xffffff) / float(1u << 24);
}

struct Ctr {
  double macs = 0;
  long long nz = 0, nztot = 0;   // non-zero input scalars / all input scalars
  long long fibers = 0, fiber_skips = 0, learned = 0;
  std::shared_ptr<std::vector<double>> lskip, lfib;   // per-layer skipped / total fibers
};

struct Engine {
  Model m;
  vector<Ten> t;
  vector<int> space_of;
  vector<vector<int>> space_perm;   // [space][old channel] = old channel at new position
  vector<vector<int>> inv_perm;     // [space][old channel] -> new position
  Gate gate;
  vector<float> zbuf;               // GEMM output scratch
  vector<float> icol;               // im2col scratch
  vector<float> vals;               // nz values
  vector<int> idx;                  // nz k-indices
  vector<double> layer_macs;
  vector<double> ntime;
  vector<double> lmac;   // per-node MACs of the last pass (any mode)
  // dead-zone calibration (work item 2c) + residual-shortcut context (2b)
  vector<float> eps;     // per-node dead-zone radius applied at store time
  vector<double> rms;    // per-node output RMS measured on the calibration set
  vector<int> sc_node;   // per-node residual partner (tensor whose ADD consumes us)
  bool deadzone = false;
  bool measure_rms = false;
  long long tz = 0, tztot = 0;   // nonzero input scalars / total (density probe)
  bool timing = false;   // per-node wall-clock accumulation (kernel dispatch calibration)
  vector<int> kern;      // per-node kernel: 0 = 8-pixel/1-group tile, 1 = 2x2x4-group tile, 2 = packed 2x2x4
  vector<vector<uint8_t>> fibdead;  // per-node bitmap: fiber skipped as dead at a pixel (coupled DW skip)
  vector<int> ukey;      // per-tile union of active im2col rows (sorted)
  vector<uint8_t> umask; // which of the PTS pixels has that row
  vector<int> upos;      // position of that row inside each pixel's nz list
  int nzs = 0;                      // per-pixel stride inside vals/idx (4 pixel slots)
};

// channel count of a node's output tensor (add/pool carry their activation's width)
static int node_ch(const Model& m, int i) {
  const Node& q = m.nd[i];
  if (q.op == OP_ADD || q.op == OP_GAP) return node_ch(m, q.i1);
  return q.co;
}

static void alloc_engine(Engine& e, const Model& m) {
  e.m = m;
  e.t.resize(m.n);
  e.layer_macs.assign(m.n, 0.0);
  size_t maxicol = 0, maxzb = 0;
  for (int i = 1; i < m.n; i++) {
    const Node& q = m.nd[i];
    if (q.op == OP_IN) continue;
    if (q.op == OP_CONV) {
      alloc_ten(e.t[i], q.h, q.w, q.co);
      maxicol = max(maxicol, size_t(q.h) * q.w * q.k * q.ci);
      maxzb = max(maxzb, size_t(q.h) * q.w * q.cp);
    } else if (q.op == OP_DW) {
      alloc_ten(e.t[i], q.h, q.w, q.co);
    } else if (q.op == OP_ADD) {
      const Node& a = m.nd[q.i1];
      alloc_ten(e.t[i], a.h, a.w, node_ch(m, q.i1));
    } else if (q.op == OP_GAP || q.op == OP_FC) {
      alloc_ten(e.t[i], 1, 1, node_ch(m, i));
    }
  }
  alloc_ten(e.t[0], 32, 32, 3);
  e.icol.assign(maxicol + 16, 0.f);
  e.zbuf.assign(maxzb + 16, 0.f);
  size_t maxnz = 0;
  for (int i = 1; i < m.n; i++) {
    const Node& q = m.nd[i];
    if (q.op == OP_CONV) maxnz = max(maxnz, size_t(q.k) * q.ci);
  }
  e.eps.assign(m.n, 0.f);
  e.kern.assign(m.n, 0);
  e.fibdead.assign(m.n, {});
  for (int i = 1; i < m.n; i++) {
    const Node& q = m.nd[i];
    if (q.op != OP_CONV && q.op != OP_DW) continue;
    const size_t bits = size_t(q.h + 2) * (q.w + 2) * q.F;   // padded positions are indexed
    e.fibdead[i].assign((bits + 7) / 8, 0);
  }
  e.rms.assign(m.n, 0.0);
  e.sc_node.assign(m.n, -1);
  {
    size_t nkmax = 0;
    for (int i = 1; i < m.n; i++) {
      const Node& q = m.nd[i];
      if (q.op == OP_CONV) nkmax = max(nkmax, size_t(q.k) * q.ci);
    }
    e.ukey.assign(nkmax + 16, 0);
    e.umask.assign(nkmax + 16, 0);
    e.upos.assign((nkmax + 16) * 4, 0);
  }
  e.nzs = int(maxnz + 16);
  e.vals.assign(size_t(8) * e.nzs, 0.f);
  e.idx.assign(size_t(8) * e.nzs, 0);
}

// ---- gate helpers -----------------------------------------------------------
static void gate_init(Gate& g, const Model& m, const GateCfg& cfg) {
  g.cfg = cfg;
  g.taboff.assign(m.n, 0);
  g.fiboff.assign(m.n, 0);
  g.actmap.assign(m.n, {});
  g.prevmap.assign(m.n, {});
  g.skipq.assign(m.n, 0.f);
  g.tabth.assign(m.n, (uint8_t)(cfg.th > 0 && cfg.th < 256 ? cfg.th : 216));
  size_t off = 0, fo = 0;
  for (int i = 0; i < m.n; i++) {
    const Node& q = m.nd[i];
    g.taboff[i] = off; g.fiboff[i] = fo;
    if (q.op == OP_CONV || q.op == OP_DW) {
      off += size_t(q.F) * cfg.NH;
      fo += q.F;
      g.actmap[i].assign(size_t(q.h + 2) * (q.w + 2) * q.F, 0);
      g.prevmap[i].assign(size_t(q.h + 2) * (q.w + 2) * q.F, 0);
    }
  }
  g.tab.assign(off, 0);
  g.actema.assign(fo, 1.f);
}

static inline void gate_learn(Gate& g, int li, int f, int hh, bool dead) {
  uint16_t& e = g.tab[g.taboff[li] + size_t(f) * g.cfg.NH + hh];
  int tot = e >> 8, dd = e & 0xff;
  if (tot < 255) { tot++; if (dead) dd++; }
  e = uint16_t((tot << 8) | (dd & 0xff));
}

static inline bool gate_pred(const Gate& g, int li, int f, int hh) {
  const int th = g.tabth[li];
  if (!th) return false;
  uint16_t e = g.tab[g.taboff[li] + size_t(f) * g.cfg.NH + hh];
  const int tot = e >> 8, dd = e & 0xff;
  return tot >= g.cfg.mincnt && dd * 256 >= th * tot;
}

// Per-layer threshold calibration from the learned (visits, dead) statistics.
// For a candidate threshold th a table entry fires when dd*256 >= th*tot; scanning th
// from loose to strict gives the loosest threshold whose predicted-dead instances are
// truly dead with precision >= target_prec, subject to a per-layer skip-rate cap.
static void gate_calibrate_one(Gate& g, const Model& m, int i, double target_prec, double max_rate) {
  {
    const Node& q = m.nd[i];
    if (q.op != OP_CONV && q.op != OP_DW) return;
    const size_t base = g.taboff[i], nent = size_t(q.F) * g.cfg.NH;
    double tot_all = 0;
    for (size_t e = 0; e < nent; e++) tot_all += (g.tab[base + e] >> 8);
    int best = 0;
    if (tot_all >= 1) {
      for (int th = 8; th <= 248; th += 8) {
        double pv = 0, pd = 0;
        for (size_t e = 0; e < nent; e++) {
          const uint16_t v = g.tab[base + e];
          const int tot = v >> 8, dd = v & 0xff;
          if (tot < g.cfg.mincnt) continue;
          if (dd * 256 >= th * tot) { pv += tot; pd += dd; }
        }
        if (pv <= 0) continue;
        if (pd / pv >= target_prec && pv / tot_all <= max_rate) { best = th; break; }
      }
    }
    g.tabth[i] = (uint8_t)best;
  }
}

static void gate_calibrate(Gate& g, const Model& m, double target_prec, double max_rate) {
  for (int i = 1; i < m.n; i++) gate_calibrate_one(g, m, i, target_prec, max_rate);
}

// context: 4 causal neighbour activity bits + patch-energy bucket + 2 temporal bits
static inline int gate_ctx(const Gate& g, int li, int f, int y, int x, int W, int F,
                           int ebucket, bool temporal, int scbits = 0) {
  const uint8_t* am = g.actmap[li].data();
  const size_t row = size_t(W + 2);
  const size_t base_cur = size_t(y + 1) * row + (x + 1);
  int aw = am[(base_cur - 1) * F + f] & 1;
  int an = am[(base_cur - row) * F + f] & 1;
  int anw = am[(base_cur - row - 1) * F + f] & 1;
  int ane = am[(base_cur - row + 1) * F + f] & 1;
  int h = aw | (an << 1) | (anw << 2) | (ane << 3) | ((ebucket & 3) << 4);
  if (!temporal) h |= (scbits & 3) << 6;   // 2b: shortcut activity bits
  if (temporal) {
    const uint8_t* pm = g.prevmap[li].data();
    int pa = pm[base_cur * F + f] & 1;
    int pd = (pm[base_cur * F + f] >> 1) & 1;
    h |= (pa << 6) | (pd << 7);
  }
  return h & (g.cfg.NH - 1);
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------
// PT spatial pixels per tile; the tile's P_TILE accumulators stay in ZMM
// registers while one 64 B weight row is reused by every active pixel.
#define PT 8
#define PTS 4
#define SPARSE_GB 4
#define MAXBLK 640
#define MAXFIB 160

// |x| without _mm512_abs_ps (absent from some GCC versions)
static inline __m512 vabs(__m512 v) {
  return _mm512_and_ps(v, _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)));
}

// activation (+ optional dead-zone truncation: |x| < eps becomes exact zero)
static inline __m512 finish_act(__m512 acc, int relu, float eps) {
  if (relu == 1) acc = _mm512_max_ps(acc, _mm512_setzero_ps());
  else if (relu == 2) acc = _mm512_min_ps(_mm512_max_ps(acc, _mm512_setzero_ps()), _mm512_set1_ps(6.f));
  if (eps > 0.f)
    acc = _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(vabs(acc), _mm512_set1_ps(eps), _CMP_GE_OQ), acc);
  return acc;
}

// Dense 3x3 / 1x1 conv, tile-major and weight-resident.
//
//   for g (16 output channels)            <- one ZMM of outputs
//     for tile of PT pixels               <- PT ZMM accumulators
//       for tap, for 16-channel input block
//         for j in 0..15: wj = W[k]; acc[p] += ib[p][..] * wj  for all p
//
// The weight row W[k] (64 B, output channels g..g+15) is loaded once and used
// PT times, and no im2col buffer is materialised at all (the previous kernel
// built an M x K im2col image and re-read it co/16 = 32 times, i.e. up to
// 32 x 18 MB of extra L3 traffic per layer).
static void conv_dense(Engine& e, const Node& q, const Ten& in, Ten& out, double& macs,
                       const float* epsp) {
  const int taps = q.k, ci = q.ci, co = q.co;
  const float* W = q.wf.data();
  const float eps = epsp ? *epsp : 0.f;
  const float* b = q.b.data();
  for (int g = 0; g < co; g += 16) {
    const __m512 bias = _mm512_loadu_ps(b + g);
    for (int y = 0; y < q.h; y++)
      for (int x0 = 0; x0 < q.w; x0 += PT) {
        const int np = std::min(PT, q.w - x0);
        __m512 acc[PT];
        for (int p = 0; p < PT; p++) acc[p] = bias;
        for (int tp = 0; tp < taps; tp++) {
          const int dy = (taps == 9) ? (tp / 3 - 1) : 0;
          const int dx = (taps == 9) ? (tp % 3 - 1) : 0;
          const float* ib[PT];
          for (int p = 0; p < np; p++) ib[p] = in.pix(y * q.st + dy, (x0 + p) * q.st + dx);
          for (int p = np; p < PT; p++) ib[p] = ib[np - 1];   // in-bounds dummy lanes
          const float* wrow = W + size_t(tp) * ci * co + g;
          int cb = 0;
          for (; cb + 16 <= ci; cb += 16) {
            for (int j = 0; j < 16; j++) {
              const __m512 wj = _mm512_loadu_ps(wrow + size_t(cb + j) * co);
#pragma GCC unroll 8
              for (int p = 0; p < PT; p++)
                acc[p] = _mm512_fmadd_ps(_mm512_set1_ps(ib[p][cb + j]), wj, acc[p]);
            }
          }
          for (; cb < ci; cb++) {   // only the 3-channel stem reaches here
            const __m512 wj = _mm512_loadu_ps(wrow + size_t(cb) * co);
#pragma GCC unroll 8
            for (int p = 0; p < PT; p++)
              acc[p] = _mm512_fmadd_ps(_mm512_set1_ps(ib[p][cb]), wj, acc[p]);
          }
        }
        for (int p = 0; p < np; p++)
          _mm512_storeu_ps(out.pix(y, x0 + p) + g, finish_act(acc[p], q.relu, eps));
      }
  }
  macs += double(q.h) * q.w * double(taps) * ci * co;
}

// depthwise 3x3 (channel diagonal), dense
// ---------------------------------------------------------------------------
// Patch-packed shared-weight tile kernel (work order item 1).
//
// A 2x2 spatial tile is processed against FOUR output groups (64 channels) at
// once, so the sixteen ZMM accumulators live in registers and — the whole point
// — every broadcast input value is loaded ONCE and reused by all four groups:
//
//   per input channel j:  4 weight-row loads + 4 scalar broadcasts  -> 16 FMAs
//                         = 0.5 load uops per FMA (the 8x1-group tiling needs
//                           1.125, which is where the old kernel's cost was)
//
// packed=true additionally drops every (tap, 16-channel block) whose four pixels
// are *all* zero: the block list is built once per tile (pass 1) and all
// `co/64` group passes replay it, so the mask work is amortised over the groups.
static void conv_tile(Engine& e, const Node& q, const Ten& in, Ten& out, double& macs,
                      const float* epsp, bool packed) {
  const int taps = q.k, ci = q.ci, co = q.co;
  const float* W = q.wf.data();
  const float eps = epsp ? *epsp : 0.f;
  const float* b = q.b.data();
  const int nblk = (ci + 15) / 16;
  if (nblk * taps > MAXBLK) return;
  int sel_t[MAXBLK], sel_cb[MAXBLK];
  double macc = 0;
  // 16 accumulators with *compile-time* indices: the generic `acc[gg][p]` form
  // spills every ZMM to the stack as soon as gg/p are runtime values.
#define TILE_BODY(WB)                                                              \
  {                                                                                \
    const float* wp = (WB);                                                        \
    for (int j = 0; j < nb; j++) {                                                 \
      const __m512 w0 = _mm512_loadu_ps(wp + size_t(j) * co);                       \
      const __m512 w1 = _mm512_loadu_ps(wp + size_t(j) * co + 16);                  \
      const __m512 w2 = _mm512_loadu_ps(wp + size_t(j) * co + 32);                  \
      const __m512 w3 = _mm512_loadu_ps(wp + size_t(j) * co + 48);                  \
      const __m512 s0 = _mm512_set1_ps(p0[j]);                                      \
      const __m512 s1 = _mm512_set1_ps(p1[j]);                                      \
      const __m512 s2 = _mm512_set1_ps(p2[j]);                                      \
      const __m512 s3 = _mm512_set1_ps(p3[j]);                                      \
      a00 = _mm512_fmadd_ps(w0, s0, a00); a01 = _mm512_fmadd_ps(w0, s1, a01);        \
      a02 = _mm512_fmadd_ps(w0, s2, a02); a03 = _mm512_fmadd_ps(w0, s3, a03);        \
      a10 = _mm512_fmadd_ps(w1, s0, a10); a11 = _mm512_fmadd_ps(w1, s1, a11);        \
      a12 = _mm512_fmadd_ps(w1, s2, a12); a13 = _mm512_fmadd_ps(w1, s3, a13);        \
      a20 = _mm512_fmadd_ps(w2, s0, a20); a21 = _mm512_fmadd_ps(w2, s1, a21);        \
      a22 = _mm512_fmadd_ps(w2, s2, a22); a23 = _mm512_fmadd_ps(w2, s3, a23);        \
      a30 = _mm512_fmadd_ps(w3, s0, a30); a31 = _mm512_fmadd_ps(w3, s1, a31);        \
      a32 = _mm512_fmadd_ps(w3, s2, a32); a33 = _mm512_fmadd_ps(w3, s3, a33);        \
    }                                                                                \
  }
  for (int y0 = 0; y0 < q.h; y0 += 2)
    for (int x0 = 0; x0 < q.w; x0 += 2) {
      int pys[4], pxs[4], np = 0;
      for (int p = 0; p < 4; p++) {
        const int dy2 = p >> 1, dx2 = p & 1;
        if (y0 + dy2 < q.h && x0 + dx2 < q.w) { pys[np] = y0 + dy2; pxs[np] = x0 + dx2; np++; }
      }
      if (e.measure_rms) {   // input-density probe: exact non-zero scalars
        for (int p = 0; p < np; p++) {
          const float* v0 = in.pix(pys[p] * q.st, pxs[p] * q.st);
          int c = 0;
          for (; c + 16 <= ci; c += 16) {
            const __mmask16 m = _mm512_cmp_ps_mask(_mm512_loadu_ps(v0 + c), _mm512_setzero_ps(), _CMP_NEQ_OQ);
            e.tz += __builtin_popcount((unsigned)m);
          }
          for (; c < ci; c++) if (v0[c] != 0.f) e.tz++;
          e.tztot += ci;
        }
      }
      int nsel = 0;
      if (packed) {
        for (int tp = 0; tp < taps; tp++) {
          const int dy = (taps == 9) ? (tp / 3 - 1) : 0;
          const int dx = (taps == 9) ? (tp % 3 - 1) : 0;
          const float* qv[4];
          for (int p = 0; p < np; p++) qv[p] = in.pix(pys[p] * q.st + dy, pxs[p] * q.st + dx);
          for (int p = np; p < 4; p++) qv[p] = qv[np - 1];
          for (int cbi = 0; cbi < nblk; cbi++) {
            const int cb = cbi * 16;
            const int nb = std::min(16, ci - cb);
            const __mmask16 kmask = (__mmask16)(nb >= 16 ? 0xffffu : ((1u << nb) - 1u));
            __mmask16 live = 0;
            for (int p = 0; p < 4; p++)
              live |= _mm512_cmp_ps_mask(_mm512_maskz_loadu_ps(kmask, qv[p] + cb),
                                         _mm512_setzero_ps(), _CMP_NEQ_OQ);
            if (live) { sel_t[nsel] = tp; sel_cb[nsel] = cb; nsel++; }
          }
        }
        if (!nsel) continue;
      }
      for (int g0 = 0; g0 < co; g0 += 64) {
        const int ng = std::min(4, (co - g0) / 16);
        const int nblock = packed ? nsel : taps * nblk;
        if (np == 4 && ng == 4) {
          const __m512 bias0 = _mm512_loadu_ps(b + g0);
          const __m512 bias1 = _mm512_loadu_ps(b + g0 + 16);
          const __m512 bias2 = _mm512_loadu_ps(b + g0 + 32);
          const __m512 bias3 = _mm512_loadu_ps(b + g0 + 48);
          __m512 a00 = bias0, a01 = bias0, a02 = bias0, a03 = bias0;
          __m512 a10 = bias1, a11 = bias1, a12 = bias1, a13 = bias1;
          __m512 a20 = bias2, a21 = bias2, a22 = bias2, a23 = bias2;
          __m512 a30 = bias3, a31 = bias3, a32 = bias3, a33 = bias3;
          for (int bi = 0; bi < nblock; bi++) {
            const int tp = packed ? sel_t[bi] : (bi / nblk);
            const int cb = packed ? sel_cb[bi] : ((bi % nblk) * 16);
            const int dy = (taps == 9) ? (tp / 3 - 1) : 0;
            const int dx = (taps == 9) ? (tp % 3 - 1) : 0;
            const float* p0 = in.pix(pys[0] * q.st + dy, pxs[0] * q.st + dx) + cb;
            const float* p1 = in.pix(pys[1] * q.st + dy, pxs[1] * q.st + dx) + cb;
            const float* p2 = in.pix(pys[2] * q.st + dy, pxs[2] * q.st + dx) + cb;
            const float* p3 = in.pix(pys[3] * q.st + dy, pxs[3] * q.st + dx) + cb;
            const int nb = std::min(16, ci - cb);
            TILE_BODY(W + (size_t(tp) * ci + cb) * co + g0)
            macc += double(4) * nb * 4 * 16;
          }
          _mm512_storeu_ps(out.pix(pys[0], pxs[0]) + g0, finish_act(a00, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[1], pxs[1]) + g0, finish_act(a01, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[2], pxs[2]) + g0, finish_act(a02, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[3], pxs[3]) + g0, finish_act(a03, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[0], pxs[0]) + g0 + 16, finish_act(a10, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[1], pxs[1]) + g0 + 16, finish_act(a11, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[2], pxs[2]) + g0 + 16, finish_act(a12, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[3], pxs[3]) + g0 + 16, finish_act(a13, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[0], pxs[0]) + g0 + 32, finish_act(a20, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[1], pxs[1]) + g0 + 32, finish_act(a21, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[2], pxs[2]) + g0 + 32, finish_act(a22, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[3], pxs[3]) + g0 + 32, finish_act(a23, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[0], pxs[0]) + g0 + 48, finish_act(a30, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[1], pxs[1]) + g0 + 48, finish_act(a31, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[2], pxs[2]) + g0 + 48, finish_act(a32, q.relu, eps));
          _mm512_storeu_ps(out.pix(pys[3], pxs[3]) + g0 + 48, finish_act(a33, q.relu, eps));
        } else {
          __m512 acc[4][4];
          for (int gg = 0; gg < 4; gg++) {
            const __m512 bias = _mm512_loadu_ps(b + g0 + std::min(gg, ng - 1) * 16);
            for (int p = 0; p < 4; p++) acc[gg][p] = bias;
          }
          for (int bi = 0; bi < nblock; bi++) {
            const int tp = packed ? sel_t[bi] : (bi / nblk);
            const int cb = packed ? sel_cb[bi] : ((bi % nblk) * 16);
            const int dy = (taps == 9) ? (tp / 3 - 1) : 0;
            const int dx = (taps == 9) ? (tp % 3 - 1) : 0;
            const float* qv[4];
            for (int p = 0; p < np; p++) qv[p] = in.pix(pys[p] * q.st + dy, pxs[p] * q.st + dx) + cb;
            for (int p = np; p < 4; p++) qv[p] = qv[np - 1];
            const int nb = std::min(16, ci - cb);
            const float* wb = W + (size_t(tp) * ci + cb) * co + g0;
            for (int j = 0; j < nb; j++)
              for (int gg = 0; gg < ng; gg++) {
                const __m512 wv = _mm512_loadu_ps(wb + size_t(j) * co + gg * 16);
                for (int p = 0; p < 4; p++)
                  acc[gg][p] = _mm512_fmadd_ps(wv, _mm512_set1_ps(qv[p][j]), acc[gg][p]);
              }
            macc += double(4) * nb * ng * 16;
          }
          for (int gg = 0; gg < ng; gg++)
            for (int p = 0; p < np; p++)
              _mm512_storeu_ps(out.pix(pys[p], pxs[p]) + g0 + gg * 16, finish_act(acc[gg][p], q.relu, eps));
        }
      }
    }
#undef TILE_BODY
  macs += macc;
}

static void dw_dense(const Node& q, const Ten& in, Ten& out, double& macs) {
  const int cp = q.cp, co = q.co;
  for (int y = 0; y < q.h; y++)
    for (int x = 0; x < q.w; x++) {
      float* dst = out.pix(y, x);
      for (int cb = 0; cb < cp; cb += 16) {
        __m512 acc = _mm512_loadu_ps(&q.b[cb]);
        for (int tp = 0; tp < 9; tp++) {
          const float* p = in.pix(y * q.st + tp / 3 - 1, x * q.st + tp % 3 - 1) + cb;
          acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), _mm512_loadu_ps(&q.wf[size_t(tp) * cp + cb]), acc);
        }
        if (q.relu == 1) acc = _mm512_max_ps(acc, _mm512_setzero_ps());
        else if (q.relu == 2) acc = _mm512_min_ps(_mm512_max_ps(acc, _mm512_setzero_ps()), _mm512_set1_ps(6.f));
        _mm512_storeu_ps(dst + cb, acc);
      }
    }
  macs += double(q.h) * q.w * cp * 9;
}

// gather the exact non-zero scalars of a conv input patch, grouped by tap.
// returns nz count; off[t]..off[t+1] is the range of tap t; pmax = max |value|.
#define MAXNB 96

static const __m512i IOTA16 = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

static inline int gather_nz(const Ten& in, int y, int x, int st, int taps, int ci,
                            float* vals, int* idx, int* off, float& pmax) {
  int n = 0;
  pmax = 0.f;
  for (int tp = 0; tp < taps; tp++) {
    off[tp] = n;
    const int dy = (taps == 9) ? (tp / 3 - 1) : 0;
    const int dx = (taps == 9) ? (tp % 3 - 1) : 0;
    const float* p = in.pix(y * st + dy, x * st + dx);
    int kb = 0;
    for (; kb + 16 <= ci; kb += 16) {
      const __m512 v = _mm512_loadu_ps(p + kb);
      const __mmask16 m = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_NEQ_UQ);
      if (!m) continue;
      _mm512_mask_compressstoreu_ps(vals + n, m, v);
      {
        const __m512i io = _mm512_add_epi32(_mm512_set1_epi32(tp * ci + kb), IOTA16);
        _mm512_mask_compressstoreu_epi32(idx + n, m, io);
        n += __builtin_popcount((unsigned)m);
      }
      const __m512 a = _mm512_abs_ps(v);
      pmax = fmaxf(pmax, _mm512_reduce_max_ps(a));
    }
    if (kb < ci) {
      const __mmask16 m = (__mmask16)((1u << (ci - kb)) - 1);
      const __m512 v = _mm512_maskz_loadu_ps(m, p + kb);
      const __mmask16 nzm = _mm512_mask_cmp_ps_mask(m, v, _mm512_setzero_ps(), _CMP_NEQ_UQ);
      if (nzm) {
        _mm512_mask_compressstoreu_ps(vals + n, nzm, v);
        {
          const __m512i io = _mm512_add_epi32(_mm512_set1_epi32(tp * ci + kb), IOTA16);
          _mm512_mask_compressstoreu_epi32(idx + n, nzm, io);
          n += __builtin_popcount((unsigned)nzm);
        }
        pmax = fmaxf(pmax, _mm512_reduce_max_ps(_mm512_and_ps(v, _mm512_set1_ps(-0.0f))));
      }
    }
  }
  off[taps] = n;
  return n;
}

// sparse (compact + optional gated) conv, 16-lane with 8-lane half-fiber fallback
static void conv_sparse(Engine& e, const Node& q, const Ten& in, Ten& out, int mode, int li,
                        bool learning, int temporal, Ctr& ct) {
  const int taps = q.k, ci = q.ci, cp = q.cp, co = q.co, F = q.F;
  const bool use_gate = (mode == M_CM || mode == M_TEMPORAL);
  const bool use_rand = (mode == M_RAND_BLOCK || mode == M_RAND_UNC);
  const bool gated = use_gate || use_rand || (learning && (mode == M_LEARN));
  // Layers whose calibrated threshold is 0 never fire: skip context + table work.
  // NOTE: this must NOT disable the random controls (they need no table, only the
  // per-layer rate), so the shortcuts are keyed on *gate activity*, not on `gated`.
  const bool gate_active = use_gate && li < int(e.gate.tabth.size()) && e.gate.tabth[li] != 0 &&
                           li >= e.gate.cfg.gate_start;
  const bool layer_ctx = gate_active || use_rand || learning;
  uint8_t* am = gated ? e.gate.actmap[li].data() : nullptr;
  uint8_t* fdm = (li < int(e.fibdead.size()) && !e.fibdead[li].empty()) ? e.fibdead[li].data() : nullptr;
  const size_t row = size_t(q.w + 2);
  const float qq = e.gate.skipq[li];
  const int S = e.nzs;
  float* VALS = e.vals.data();
  int* IDX = e.idx.data();
  const int nb = cp / 16;
  const int fi_off = e.gate.fiboff[li];
  double* lsk = (ct.lskip && li < int(ct.lskip->size())) ? &(*ct.lskip)[li] : nullptr;
  double* lfb = lsk ? &(*ct.lfib)[li] : nullptr;
  const int cfg_gate_start = e.gate.cfg.gate_start;
  const unsigned anchor_mask = (unsigned)e.gate.cfg.anchor_mask;
  const bool anchor_gate = use_gate || use_rand;

  if (nb > MAXNB) {   // never happens for the shipped models; keep a safe fallback
    for (int y = 0; y < q.h; y++)
      for (int x = 0; x < q.w; x++) {
        int off[10]; float pmax;
        const int n = gather_nz(in, y, x, q.st, taps, ci, VALS, IDX, off, pmax);
        for (int cb = 0; cb < cp; cb += 16) {
          __m512 acc = _mm512_loadu_ps(&q.b[cb]);
          for (int j = 0; j < n; j++)
            acc = _mm512_fmadd_ps(_mm512_set1_ps(VALS[j]), _mm512_loadu_ps(&q.wf[size_t(IDX[j]) * cp + cb]), acc);
          ct.macs += 16.0 * n;
          if (q.relu == 1) acc = _mm512_max_ps(acc, _mm512_setzero_ps());
          else if (q.relu == 2) acc = _mm512_min_ps(_mm512_max_ps(acc, _mm512_setzero_ps()), _mm512_set1_ps(6.f));
          _mm512_storeu_ps(out.pix(y, x) + cb, acc);
        }
      }
    return;
  }

  // ---- tile-major, weight-resident sparse kernel -----------------------------
  //
  // Per spatial tile of PT pixels and per block of GB output groups (GB*16
  // channels), the accumulators stay in registers for the whole K walk, and the
  // GB weight rows of each K step are loaded once and reused by every active
  // pixel of the tile.  Zero input scalars are never multiplied: a pixel "has"
  // im2col row k only if k appears in its compacted (sorted) non-zero list, so
  // the walk skips dead channels for free.
  // ---------------------------------------------------------------------------
  const int nk = taps * ci;
  const float ez = e.deadzone ? e.eps[li] : 0.f;
  const int GB = SPARSE_GB;                      // output groups live per pass (64 channels)
  const int coBlk = GB * 16;
  int N[PTS];
  uint8_t dmap[PTS][MAXFIB];
  int cmap[PTS][MAXFIB];

  for (int y = 0; y < q.h; y++)
    for (int x0 = 0; x0 < q.w; x0 += PTS) {
      const int np = std::min(PTS, q.w - x0);
      int ebt[PTS];
      for (int p = 0; p < np; p++) {
        int off[10]; float pmax;
        N[p] = gather_nz(in, y, x0 + p, q.st, taps, ci, VALS + p * S, IDX + p * S, off, pmax);
        ebt[p] = (N[p] == 0) ? 0 : (N[p] * 8 < nk) ? 1 : (N[p] * 2 < nk) ? 2 : 3;
        ct.nz += N[p];
        ct.nztot += nk;
      }
      // ---- one gate decision per (pixel, fiber) for the whole tile ----
      {
        const int nfib = co / 8;
        const bool want_ctx = gate_active || learning;
        for (int p = 0; p < np; p++) {
          for (int f = 0; f < nfib; f++) {
            int h = 0;
            if (gated && layer_ctx && want_ctx) {
              int scb = 0;   // 2b: residual shortcut activity for this fiber
              if (e.sc_node.size() > size_t(li) && e.sc_node[li] >= 0) {
                const auto& st = e.t[e.sc_node[li]];
                const float* sp = st.pix(y, x0 + p) + f * 8;
                float mx = sp[0];
                for (int c = 1; c < 8; c++) mx = sp[c] > mx ? sp[c] : mx;
                scb = (mx <= 0.f ? 1 : 0) | (mx > 1.f ? 2 : 0);
              }
              h = gate_ctx(e.gate, li, f, y, x0 + p, q.w, F, ebt[p], temporal, scb);
            }
            uint8_t d = 0;
            const bool anchor = anchor_gate &&
                ((((unsigned)(y * q.w + x0 + p) ^ (unsigned)f) & anchor_mask) == 0);
            if ((use_gate || use_rand) && !anchor && li >= cfg_gate_start) {
              if (use_gate) {
                if (gate_active && gate_pred(e.gate, li, f, h) &&
                    e.gate.actema[fi_off + f] < e.gate.cfg.guard_act) d = 1;
              } else {
                if (rnd01(e.gate.rng) < qq) d = 1;
              }
            }
            dmap[p][f] = d;
            cmap[p][f] = h;
          }
          for (int f = nfib; f < MAXFIB; f++) { dmap[p][f] = 0; cmap[p][f] = 0; }
        }
      }
      // publish which fibers this layer predicted dead: the coupled depthwise
      // stage of an InvertedResidual can then skip the same 8-lane fiber whole.
      if (fdm) {
        for (int p = 0; p < np; p++) {
          const size_t posr = size_t(y + 1) * row + size_t(x0 + p + 1);
          for (int f = 0; f < co / 8; f++) {
            const uint8_t dd = dmap[p][f];
            if (!dd) continue;
            for (int t = 0; t < 2; t++) {
              const int fib = f * 2 + t;
              if (fib >= F) break;
              if ((dd >> t) & 1) {
                const size_t bit = posr * F + fib;
                fdm[bit >> 3] |= (uint8_t)(1u << (bit & 7));
              }
            }
          }
        }
      }
      for (int cb0 = 0; cb0 < co; cb0 += coBlk) {
        const int ng = std::min(GB, (co - cb0) / 16);

        // ---- fully-unrolled fast path: every pixel and every group live ----
        if (np == PTS && ng == GB) {
          __m512 acc[PTS][GB];
#pragma GCC unroll 4
          for (int p = 0; p < PTS; p++)
#pragma GCC unroll 4
            for (int g = 0; g < GB; g++) acc[p][g] = _mm512_loadu_ps(&q.b[cb0 + g * 16]);
          int cur[PTS] = {0, 0, 0, 0};
          long long fmac = 0;
          for (int k = 0; k < nk; k++) {
            int act = 0;
            int pos[PTS];
#pragma GCC unroll 4
            for (int p = 0; p < PTS; p++) {
              const int c = cur[p];
              if (c < N[p] && IDX[p * S + c] == k) { pos[p] = c; cur[p] = c + 1; act |= 1 << p; }
            }
            if (!act) continue;
            const __m512 w0 = _mm512_loadu_ps(&q.wf[size_t(k) * cp + cb0 + 0]);
            const __m512 w1 = _mm512_loadu_ps(&q.wf[size_t(k) * cp + cb0 + 16]);
            const __m512 w2 = _mm512_loadu_ps(&q.wf[size_t(k) * cp + cb0 + 32]);
            const __m512 w3 = _mm512_loadu_ps(&q.wf[size_t(k) * cp + cb0 + 48]);
              if (act & 1) { const __m512 v = _mm512_set1_ps(VALS[0 * S + pos[0]]);
                acc[0][0] = _mm512_fmadd_ps(v, w0, acc[0][0]);
                acc[0][1] = _mm512_fmadd_ps(v, w1, acc[0][1]);
                acc[0][2] = _mm512_fmadd_ps(v, w2, acc[0][2]);
                acc[0][3] = _mm512_fmadd_ps(v, w3, acc[0][3]);
              }
              if (act & 2) { const __m512 v = _mm512_set1_ps(VALS[1 * S + pos[1]]);
                acc[1][0] = _mm512_fmadd_ps(v, w0, acc[1][0]);
                acc[1][1] = _mm512_fmadd_ps(v, w1, acc[1][1]);
                acc[1][2] = _mm512_fmadd_ps(v, w2, acc[1][2]);
                acc[1][3] = _mm512_fmadd_ps(v, w3, acc[1][3]);
              }
              if (act & 4) { const __m512 v = _mm512_set1_ps(VALS[2 * S + pos[2]]);
                acc[2][0] = _mm512_fmadd_ps(v, w0, acc[2][0]);
                acc[2][1] = _mm512_fmadd_ps(v, w1, acc[2][1]);
                acc[2][2] = _mm512_fmadd_ps(v, w2, acc[2][2]);
                acc[2][3] = _mm512_fmadd_ps(v, w3, acc[2][3]);
              }
              if (act & 8) { const __m512 v = _mm512_set1_ps(VALS[3 * S + pos[3]]);
                acc[3][0] = _mm512_fmadd_ps(v, w0, acc[3][0]);
                acc[3][1] = _mm512_fmadd_ps(v, w1, acc[3][1]);
                acc[3][2] = _mm512_fmadd_ps(v, w2, acc[3][2]);
                acc[3][3] = _mm512_fmadd_ps(v, w3, acc[3][3]);
              }
            fmac += __builtin_popcount((unsigned)act) * 4;
          }
          ct.macs += 16.0 * double(fmac);
          for (int p = 0; p < PTS; p++) {
            const int x = x0 + p;
            const size_t posr = size_t(y + 1) * row + (x + 1);
            for (int g = 0; g < GB; g++) {
              const int cb = cb0 + g * 16;
              const int f0 = cb / 8, f1 = f0 + 1;
              const uint8_t d = dmap[p][f0];
              __m512 v = acc[p][g];
              if (q.relu == 1) v = _mm512_max_ps(v, _mm512_setzero_ps());
              else if (q.relu == 2) v = _mm512_min_ps(_mm512_max_ps(v, _mm512_setzero_ps()), _mm512_set1_ps(6.f));
              if (ez > 0.f) v = finish_act(v, 0, ez);
              if (gated) {
                if (d & 1) v = _mm512_maskz_mov_ps((__mmask16)0xff00, v);
                if (dmap[p][f1] & 1) v = _mm512_maskz_mov_ps((__mmask16)0x00ff, v);
              }
              _mm512_storeu_ps(out.pix(y, x) + cb, v);
              if (gated) {
                const __mmask16 m = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GT_OQ);
                const bool b0 = (m & 0xff) != 0, b1 = (m & 0xff00) != 0;
                am[posr * F + f0] = b0 ? 1 : 2;
                am[posr * F + f1] = b1 ? 1 : 2;
                e.gate.actema[fi_off + f0] = 0.95f * e.gate.actema[fi_off + f0] + 0.05f * (b0 ? 1.f : 0.f);
                e.gate.actema[fi_off + f1] = 0.95f * e.gate.actema[fi_off + f1] + 0.05f * (b1 ? 1.f : 0.f);
                if (lfb) *lfb += 2;
                ct.fibers += 2;
                if (d & 1) { ct.fiber_skips++; if (lsk) *lsk += 1; }
                if (dmap[p][f1] & 1) { ct.fiber_skips++; if (lsk) *lsk += 1; }
                if (learning) {
                  gate_learn(e.gate, li, f0, cmap[p][f0] & 0xff, !b0);
                  gate_learn(e.gate, li, f1, cmap[p][f1] & 0xff, !b1);
                }
              }
            }
          }
          continue;
        }

        __m512 acc[PTS][GB];
#pragma GCC unroll 4
        for (int p = 0; p < PTS; p++)
#pragma GCC unroll 4
          for (int g = 0; g < GB; g++)
            acc[p][g] = (p < np && g < ng) ? _mm512_loadu_ps(&q.b[cb0 + g * 16])
                                          : _mm512_setzero_ps();
        int cur[PTS];
        long long fmac = 0;
#pragma GCC unroll 4
        for (int p = 0; p < PTS; p++) cur[p] = 0;
        for (int k = 0; k < nk; k++) {
          int act = 0;
          int pos[PTS];
#pragma GCC unroll 4
          for (int p = 0; p < PTS; p++) {
            if (p >= np) break;
            const int c = cur[p];
            if (c < N[p] && IDX[p * S + c] == k) { pos[p] = c; cur[p] = c + 1; act |= 1 << p; }
          }
          if (!act) continue;
          __m512 w[GB];
#pragma GCC unroll 4
          for (int g = 0; g < GB; g++)
            if (g < ng) w[g] = _mm512_loadu_ps(&q.wf[size_t(k) * cp + cb0 + g * 16]);
#pragma GCC unroll 4
          for (int p = 0; p < PTS; p++) {
            if (p >= np) break;
            if (!((act >> p) & 1)) continue;
            const __m512 v = _mm512_set1_ps(VALS[p * S + pos[p]]);
#pragma GCC unroll 4
            for (int g = 0; g < GB; g++)
              if (g < ng) acc[p][g] = _mm512_fmadd_ps(v, w[g], acc[p][g]);
          }
          fmac += __builtin_popcount((unsigned)act) * ng;
        }
        ct.macs += 16.0 * double(fmac);
        for (int p = 0; p < np; p++) {
          const int x = x0 + p;
          const size_t posr = size_t(y + 1) * row + (x + 1);
          for (int g = 0; g < ng; g++) {
            const int cb = cb0 + g * 16;
            const int f0 = cb / 8, f1 = f0 + 1;
            const uint8_t d = dmap[p][f0];
            __m512 v = acc[p][g];
            if (q.relu == 1) v = _mm512_max_ps(v, _mm512_setzero_ps());
            else if (q.relu == 2) v = _mm512_min_ps(_mm512_max_ps(v, _mm512_setzero_ps()), _mm512_set1_ps(6.f));
            if (ez > 0.f) v = finish_act(v, 0, ez);
            if (gated) {
              if (d & 1) v = _mm512_maskz_mov_ps((__mmask16)0xff00, v);
              if (dmap[p][f1] & 1) v = _mm512_maskz_mov_ps((__mmask16)0x00ff, v);
            }
            _mm512_storeu_ps(out.pix(y, x) + cb, v);
            if (gated) {
              const __mmask16 m = _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GT_OQ);
              const bool b0 = (m & 0xff) != 0, b1 = (m & 0xff00) != 0;
              am[posr * F + f0] = b0 ? 1 : 2;
              am[posr * F + f1] = b1 ? 1 : 2;
              e.gate.actema[fi_off + f0] = 0.95f * e.gate.actema[fi_off + f0] + 0.05f * (b0 ? 1.f : 0.f);
              e.gate.actema[fi_off + f1] = 0.95f * e.gate.actema[fi_off + f1] + 0.05f * (b1 ? 1.f : 0.f);
              if (lfb) *lfb += 2;
              ct.fibers += 2;
              if (d & 1) { ct.fiber_skips++; if (lsk) *lsk += 1; }
              if (dmap[p][f1] & 1) { ct.fiber_skips++; if (lsk) *lsk += 1; }
              if (learning) {
                gate_learn(e.gate, li, f0, cmap[p][f0] & 0xff, !b0);
                gate_learn(e.gate, li, f1, cmap[p][f1] & 0xff, !b1);
              }
            }
          }
        }
      }
    }
}

static void dw_sparse(Engine& e, const Node& q, const Ten& in, Ten& out, int mode, int li, bool learning, Ctr& ct) {
  const int cp = q.cp, F = q.F;
  const int temporal = (mode == M_TEMPORAL) ? 1 : 0;
  const bool use_gate = (mode == M_CM || mode == M_TEMPORAL);
  const bool use_rand = (mode == M_RAND_BLOCK || mode == M_RAND_UNC);
  const bool gated = use_gate || use_rand || (mode == M_LEARN);
  const bool gate_active = use_gate && li < int(e.gate.tabth.size()) && e.gate.tabth[li] != 0 &&
                           li >= e.gate.cfg.gate_start;
  const bool layer_ctx = gate_active || use_rand || (mode == M_LEARN);
  uint8_t* am = gated ? e.gate.actmap[li].data() : nullptr;
  uint8_t* fdm = (li < int(e.fibdead.size()) && !e.fibdead[li].empty()) ? e.fibdead[li].data() : nullptr;
  const size_t row = size_t(q.w + 2);
  const int fi_off = e.gate.fiboff[li];
  const float qq = e.gate.skipq[li];
  const unsigned anchor_mask = (unsigned)e.gate.cfg.anchor_mask;
  double* lsk = (ct.lskip && li < int(ct.lskip->size())) ? &(*ct.lskip)[li] : nullptr;
  double* lfb = lsk ? &(*ct.lfib)[li] : nullptr;
  const Ten& tin = e.t[q.i1];
  const int tF = tin.cp / 8;
  const uint8_t* fdsrc =
      (q.i1 >= 0 && q.i1 < int(e.fibdead.size()) && !e.fibdead[q.i1].empty() &&
       tin.w == q.w * q.st && tin.h == q.h * q.st) ? e.fibdead[q.i1].data() : nullptr;
  const size_t trin = size_t(tin.w + 2);
  const float ez = e.deadzone ? e.eps[li] : 0.f;

  for (int y = 0; y < q.h; y++)
    for (int x = 0; x < q.w; x++) {
      float* dst = out.pix(y, x);
      const size_t pos = size_t(y + 1) * row + (x + 1);
      int live_all = 0, tap_nz = 0;
      for (int f = 0; f < F; f++) {
        const int c0 = f * 8;
        bool dead = false;
        int hh = 0;
        // --- coupled sandwich skip: the 1x1 Expand already zeroed this fiber ---
        if (fdsrc && f < tF) {
          const size_t pin = size_t(y * q.st + 1) * trin + size_t(x * q.st + 1);
          const size_t bit = pin * size_t(tF) + size_t(f);
          if (fdsrc[bit >> 3] & (uint8_t)(1u << (bit & 7))) {
            __m256 bb = _mm256_loadu_ps(&q.b[c0]);
            if (q.relu == 1) bb = _mm256_max_ps(bb, _mm256_setzero_ps());
            else if (q.relu == 2) bb = _mm256_min_ps(_mm256_max_ps(bb, _mm256_setzero_ps()), _mm256_set1_ps(6.f));
            if (ez > 0.f) {
              const __m256 av = _mm256_and_ps(bb, _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)));
              bb = _mm256_maskz_mov_ps(_mm256_cmp_ps_mask(av, _mm256_set1_ps(ez), _CMP_GE_OQ), bb);
            }
            _mm256_storeu_ps(dst + c0, bb);
            if (gated) am[pos * F + f] = 2;
            ct.fiber_skips++;
            if (lsk) { *lsk += 1; *lfb += 1; }
            continue;
          }
        }
        if (gated && layer_ctx) {
          const int eb = 0;   // depthwise context: neighbour activity only
          if (gate_active || mode == M_LEARN) hh = gate_ctx(e.gate, li, f, y, x, q.w, F, eb, temporal != 0);
          if (use_gate || use_rand) {
            const bool anchor = ((((unsigned)(y * q.w + x) ^ (unsigned)f) & anchor_mask) == 0);
            if (!anchor && li >= e.gate.cfg.gate_start) {
              if (use_gate)
                dead = gate_active && gate_pred(e.gate, li, f, hh) && e.gate.actema[fi_off + f] < e.gate.cfg.guard_act;
              else
                dead = rnd01(e.gate.rng) < qq;
            }
          }
        }
        if (dead) {
          _mm256_storeu_ps(dst + c0, _mm256_setzero_ps());
          am[pos * F + f] = 2;
          ct.fiber_skips++;
          if (lsk) { *lsk += 1; *lfb += 1; }
          if (learning) gate_learn(e.gate, li, f, hh, true);
          continue;
        }
        __m256 acc = _mm256_loadu_ps(&q.b[c0]);
        int live = 0;
        for (int tp = 0; tp < 9; tp++) {
          const float* p = in.pix(y * q.st + tp / 3 - 1, x * q.st + tp % 3 - 1) + c0;
          const __m256 v = _mm256_loadu_ps(p);
          const __mmask8 m = _mm256_cmp_ps_mask(v, _mm256_setzero_ps(), _CMP_NEQ_UQ);
          if (!m) continue;
          acc = _mm256_mask3_fmadd_ps(v, _mm256_loadu_ps(&q.wf[size_t(tp) * cp + c0]), acc, m);  // dst = k ? v*w + acc : acc
          live += __builtin_popcount((unsigned)m);
        }
        ct.macs += double(live);   // one MAC per live (tap, lane)
        if (q.relu == 1) acc = _mm256_max_ps(acc, _mm256_setzero_ps());
        else if (q.relu == 2) acc = _mm256_min_ps(_mm256_max_ps(acc, _mm256_setzero_ps()), _mm256_set1_ps(6.f));
        _mm256_storeu_ps(dst + c0, acc);
        if (gated) {
          const bool a = _mm256_cmp_ps_mask(acc, _mm256_setzero_ps(), _CMP_GT_OQ) != 0;
          am[pos * F + f] = a ? 1 : 2;
          e.gate.actema[fi_off + f] = 0.95f * e.gate.actema[fi_off + f] + 0.05f * (a ? 1.f : 0.f);
          if (lfb) *lfb += 1;
          if (learning) gate_learn(e.gate, li, f, hh, !a);
        }
        live_all += live;
        for (int tp = 0; tp < 9; tp++) tap_nz += 0;
      }
      (void)live_all; (void)tap_nz;
    }
}

static void add_relu(const Node& q, const Ten& a, const Ten& b, Ten& out) {
  const size_t n = out.d.size();
  for (size_t i = 0; i < n; i += 16) {
    __m512 v = _mm512_add_ps(_mm512_loadu_ps(&a.d[i]), _mm512_loadu_ps(&b.d[i]));
    if (q.relu == 1) v = _mm512_max_ps(v, _mm512_setzero_ps());
    else if (q.relu == 2) v = _mm512_min_ps(_mm512_max_ps(v, _mm512_setzero_ps()), _mm512_set1_ps(6.f));
    _mm512_storeu_ps(&out.d[i], v);
  }
}

static void gap(const Ten& a, Ten& out) {
  const int cp = a.cp;
  out.d.assign(out.d.size(), 0.f);
  float* o = out.pix(0, 0);
  const float inv = 1.f / float(a.h * a.w);
  for (int c = 0; c < cp; c += 16) {
    __m512 s = _mm512_setzero_ps();
    for (int y = 0; y < a.h; y++)
      for (int x = 0; x < a.w; x++) s = _mm512_add_ps(s, _mm512_loadu_ps(a.pix(y, x) + c));
    _mm512_storeu_ps(o + c, _mm512_mul_ps(s, _mm512_set1_ps(inv)));
  }
}

static void fc(const Node& q, const Ten& a, Ten& out, double& macs) {
  const float* x = a.pix(0, 0);
  for (int n0 = 0; n0 < q.cp; n0 += 16) {
    __m512 acc = _mm512_loadu_ps(&q.b[n0]);
    for (int kk = 0; kk < q.ci; kk++)
      acc = _mm512_fmadd_ps(_mm512_set1_ps(x[kk]), _mm512_loadu_ps(&q.wf[size_t(kk) * q.cp + n0]), acc);
    _mm512_storeu_ps(out.pix(0, 0) + n0, acc);
  }
  macs += double(q.ci) * q.co;
}

// ---------------------------------------------------------------------------
// forward
// ---------------------------------------------------------------------------
static void forward(Engine& e, const float* px, int mode, int temporal, Ctr& ct, float* logits,
                    float* feat, int feat_c) {
  const Model& m = e.m;
  // node 0: input (px is NHWC 32*32*3)
  {
    Ten& t0 = e.t[0];
    std::fill(t0.d.begin(), t0.d.end(), 0.f);
    for (int y = 0; y < 32; y++)
      for (int x = 0; x < 32; x++) {
        const float* s = px + (size_t(y) * 32 + x) * 3;
        float* d = t0.pix(y, x);
        d[0] = s[0]; d[1] = s[1]; d[2] = s[2];
      }
  }
  const bool learning = (mode == M_LEARN);
  static const bool prof_env = getenv("CM_PROF") != nullptr;
  const bool prof = prof_env || e.timing;
  if (prof) { if (int(e.ntime.size()) < m.n) e.ntime.assign(m.n, 0.0); }
  for (int i = 1; i < m.n; i++) {
    const Node& q = m.nd[i];
    Ten& out = e.t[i];
    const double tt0 = prof ? now_sec() : 0.0;
    const double mac0 = ct.macs;
    if (q.op == OP_CONV) {
      if (q.k == 9) {
        const int kk = (i < int(e.kern.size())) ? e.kern[i] : 0;
        const float* ez = e.deadzone ? &e.eps[i] : nullptr;
        if (mode == M_DENSE) {
          conv_tile(e, q, e.t[q.i1], out, ct.macs, ez, false);
        } else if (kk == 1) {
          conv_tile(e, q, e.t[q.i1], out, ct.macs, ez, false);
        } else if (kk == 2) {
          conv_tile(e, q, e.t[q.i1], out, ct.macs, ez, true);
        } else {
          conv_sparse(e, q, e.t[q.i1], out, mode, i, learning, temporal, ct);
        }
      } else {
        const int kk = (i < int(e.kern.size())) ? e.kern[i] : 0;
        const float* ez = e.deadzone ? &e.eps[i] : nullptr;
        if (mode == M_DENSE) {
          conv_tile(e, q, e.t[q.i1], out, ct.macs, ez, false);
        } else if (kk == 1) {
          conv_tile(e, q, e.t[q.i1], out, ct.macs, ez, false);
        } else if (kk == 2) {
          conv_tile(e, q, e.t[q.i1], out, ct.macs, ez, true);
        } else {
          conv_sparse(e, q, e.t[q.i1], out, mode, i, learning, temporal, ct);
        }
      }
    } else if (q.op == OP_DW) {
      if (mode == M_DENSE) dw_dense(q, e.t[q.i1], out, ct.macs);
      else dw_sparse(e, q, e.t[q.i1], out, mode, i, learning, ct);
    } else if (q.op == OP_ADD) {
      add_relu(q, e.t[q.i1], e.t[q.i2], out);
      if (mode == M_LEARN || mode == M_CM || mode == M_TEMPORAL) {
        // activation map for the residual bus comes from the inputs; the add is not gated
      }
    } else if (q.op == OP_GAP) {
      gap(e.t[q.i1], out);
      if (feat) {
        const Ten& f = e.t[i];
        const int c = min(feat_c, f.c);
        for (int cc = 0; cc < c; cc++) feat[cc] = f.pix(0, 0)[cc];
      }
    } else if (q.op == OP_FC) {
      fc(q, e.t[q.i1], out, ct.macs);
    } else die("bad op");
    if (i < int(e.lmac.size())) e.lmac[i] += ct.macs - mac0;
    if (e.measure_rms) {
      const Ten& o = e.t[i];
      double ss = 0;
      for (int yy = 0; yy < q.h; yy++)
        for (int xx = 0; xx < q.w; xx++) {
          const float* v = o.pix(yy, xx);
          for (int c = 0; c < q.cp; c++) ss += double(v[c]) * v[c];
        }
      e.rms[i] += ss / double(q.h * q.w * q.cp);
    }
    if (prof) e.ntime[i] += now_sec() - tt0;
    if (getenv("CMDUMP")) {
      const Ten& tt = e.t[i];
      fprintf(stderr, "n%02d op%d %dx%dx%d:", i, q.op, tt.h, tt.w, tt.c);
      if (tt.h && tt.w) { const float* p = tt.pix(0, 0); for (int c = 0; c < tt.c && c < 4; c++) fprintf(stderr, " %.6f", p[c]); }
      fprintf(stderr, "\n");
    }
    if (ct.macs > 0 && i == m.n - 1) {}
  }
  const Ten& o = e.t[m.n - 1];
  for (int k = 0; k < m.cls; k++) logits[k] = o.pix(0, 0)[k];
  (void)feat_c;
}

// ---------------------------------------------------------------------------
// clustering (8-lane cosine co-activation over the unified channel spaces)
// ---------------------------------------------------------------------------
struct Stats { vector<double> dead_rate, half_dead; double macs = 0; };

static void permute_model(Model& dst, const Model& src, const vector<int>& space_of,
                          const vector<vector<int>>& perm, int nsp) {
  dst = src;
  // per-space inverse: old channel -> new position
  vector<vector<int>> inv(nsp);
  for (int s = 0; s < nsp; s++) {
    inv[s].assign(perm[s].size(), 0);
    for (size_t i = 0; i < perm[s].size(); i++) inv[s][perm[s][i]] = (int)i;
  }
  for (int i = 1; i < dst.n; i++) {
    Node& q = dst.nd[i];
    const Node& s = src.nd[i];
    if (q.op == OP_CONV) {
      const int taps = q.k;
      vector<float> nw(size_t(q.kdim) * q.cp, 0.f);
      const vector<int>& pin = perm[space_of[q.i1]];          // old channel at each new position
      const vector<int>& pout = perm[space_of[i]];
      const int cin_ = q.ci, co_ = q.co;
      for (int tp = 0; tp < taps; tp++)
        for (int ni = 0; ni < cin_; ni++) {
          const int oi = pin[ni];
          for (int no = 0; no < co_; no++)
            nw[size_t(tp * cin_ + ni) * q.cp + no] = s.wf[size_t(tp * cin_ + oi) * q.cp + pout[no]];
        }
      q.wf.swap(nw);
      for (int no = 0; no < co_; no++) q.b[no] = s.b[pout[no]];
    } else if (q.op == OP_DW) {
      // depthwise: same space on both sides, so permute taps only
      vector<float> nw(size_t(9) * q.cp, 0.f);
      const vector<int>& p = perm[space_of[i]];
      for (int tp = 0; tp < 9; tp++)
        for (int no = 0; no < q.co; no++) nw[size_t(tp) * q.cp + no] = s.wf[size_t(tp) * q.cp + p[no]];
      q.wf.swap(nw);
      for (int no = 0; no < q.co; no++) q.b[no] = s.b[p[no]];
    } else if (q.op == OP_FC) {
      vector<float> nw(size_t(q.ci) * q.cp, 0.f);
      const vector<int>& pin = perm[space_of[q.i1]];
      for (int ni = 0; ni < q.ci; ni++)
        for (int no = 0; no < q.co; no++) nw[size_t(ni) * q.cp + no] = s.wf[size_t(pin[ni]) * q.cp + no];
      q.wf.swap(nw);
    }
  }
}

// ---------------------------------------------------------------------------
// benchmark harness
// ---------------------------------------------------------------------------
static void argmax(const float* v, int n, int& best) {
  best = 0;
  for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
}

struct ModeOut {
  string name;
  double acc = 0, macs = 0, ms = 0, rmse = 0, feat_rel = 0, agree = 0;
  double fiber_dead = 0, fiber_skip = 0;
  vector<double> layer_macs;
  vector<double> ntime;
  vector<double> dead_rate;
};

static double median(vector<double> v) {
  sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  string wp, tp, dp, sp, jp, lp, model_name = "resnet18";
  int reps = 20, reps_stream = 0;
  bool verify = false;
  GateCfg gcfg;
  for (int i = 1; i < argc; i++) {
    string a = argv[i];
    auto nx = [&]() { if (i + 1 >= argc) die("missing value for " + a); return string(argv[++i]); };
    if (a == "--weights") wp = nx();
    else if (a == "--test") tp = nx();
    else if (a == "--dev") dp = nx();
    else if (a == "--train") lp = nx();
    else if (a == "--stream") sp = nx();
    else if (a == "--json") jp = nx();
    else if (a == "--model") model_name = nx();
    else if (a == "--reps") reps = stoi(nx());
    else if (a == "--reps-stream") reps_stream = stoi(nx());
    else if (a == "--min-count") gcfg.mincnt = stoi(nx());
    else if (a == "--th") gcfg.th = stoi(nx());
    else if (a == "--anchor-mask") gcfg.anchor_mask = stoi(nx());
    else if (a == "--guard-act") gcfg.guard_act = stof(nx());
    else if (a == "--gate-start") gcfg.gate_start = stoi(nx());
    else if (a == "--verify") verify = true;
    else die("unknown arg " + a);
  }
  if (wp.empty() || tp.empty()) die("need --weights and --test");
  if (!jp.empty()) { if (!freopen(jp.c_str(), "w", stdout)) die("cannot write " + jp); }
  if (reps_stream <= 0) reps_stream = reps;

  ImageSet test = load_images(tp);
  ImageSet dev;
  if (!dp.empty()) dev = load_images(dp);
  ImageSet train;
  if (!lp.empty()) train = load_images(lp);
  StreamSet stream;
  if (!sp.empty()) stream = load_stream(sp);

  const Model orig = load_model(wp);
  const int cls = orig.cls;
  printf("{\n  \"model\": \"%s\", \"type\": %d, \"cls\": %d, \"nodes\": %d,\n", model_name.c_str(), orig.type, cls, orig.n);

  // ---- spaces + clustering -------------------------------------------------
  int nsp = 0;
  vector<int> space_of = build_spaces(orig, {}, nsp);
  // sample node per space = highest node index in that space
  vector<int> sample(orig.n, -1);
  for (int i = orig.n - 1; i >= 0; i--) sample[space_of[i]] = i;
  vector<int> spc(nsp, 0);
  for (int i = 0; i < orig.n; i++) spc[space_of[i]] = node_ch(orig, i);

  auto setup_sc = [](Engine& e) {   // 2b: residual partner per node
    for (int i = 1; i < e.m.n; i++) {
      const Node& q = e.m.nd[i];
      if (q.op == OP_ADD && e.m.nd[q.i1].op == OP_CONV) e.sc_node[q.i1] = q.i2;
    }
  };
  Engine e0;
  alloc_engine(e0, orig);
  setup_sc(e0);
  gate_init(e0.gate, orig, gcfg);
  vector<vector<Bits>> act(nsp);
  for (int s = 0; s < nsp; s++) act[s].resize(spc[s]);
  int pos = 0;
  const int maxpos = 1024;
  const ImageSet& cluster_set = train.n ? train : dev;
  if (cluster_set.n) {
    for (int im = 0; im < cluster_set.n && pos < maxpos; im++) {
      Ctr ct;
      float lg[64], ft[16];
      forward(e0, cluster_set.px.data() + size_t(im) * 3 * 32 * 32, M_DENSE, 0, ct, lg, nullptr, 0);
      for (int s = 0; s < nsp; s++) {
        const int ni = sample[s];
        if (ni <= 0 || spc[s] <= 0) continue;
        const Ten& t = e0.t[ni];
        for (int y = 0; y < t.h; y += 2)
          for (int x = 0; x < t.w; x += 2) {
            const float* p = t.pix(y, x);
            for (int c = 0; c < spc[s]; c++) {
              if (p[c] > 0) {
                Bits& B = act[s][c];
                const int wd = pos / 64;
                if ((int)B.w.size() <= wd) B.w.resize(wd + 1, 0);
                B.w[wd] |= 1ull << (pos % 64);
                B.cnt++;
              }
            }
            pos++;
            if (pos >= maxpos) break;
          }
      }
    }
  }
  const int nwords = pos / 64 + 1;
  vector<vector<int>> perm(nsp);
  for (int s = 0; s < nsp; s++) {
    const int C = spc[s];
    perm[s].resize(C);
    for (int i = 0; i < C; i++) perm[s][i] = i;
    if (C % 8) continue;
    for (auto& B : act[s]) B.w.resize(nwords, 0);
    vector<char> used(C, 0);
    vector<int> order;
    order.reserve(C);
    for (int g = 0; g < C / 8; g++) {
      int seed = -1, bestc = -1;
      for (int c = 0; c < C; c++) if (!used[c] && act[s][c].cnt > bestc) { bestc = act[s][c].cnt; seed = c; }
      if (seed < 0) break;
      used[seed] = 1; order.push_back(seed);
      for (int kk = 1; kk < 8; kk++) {
        int pick = -1; double best = -2;
        const Bits& S = act[s][seed];
        for (int c = 0; c < C; c++) if (!used[c]) {
          uint64_t dot = 0;
          for (int wd = 0; wd < nwords; wd++) dot += __builtin_popcountll(S.w[wd] & act[s][c].w[wd]);
          const double co = double(dot) / sqrt(double(S.cnt) * double(act[s][c].cnt) + 1e-9);
          if (co > best) { best = co; pick = c; }
        }
        if (pick < 0) { for (int c = 0; c < C; c++) if (!used[c]) { pick = c; break; } }
        used[pick] = 1; order.push_back(pick);
      }
    }
    if ((int)order.size() == C) perm[s] = order;
  }

  Model clus;
  permute_model(clus, orig, space_of, perm, nsp);
  Engine ec;
  alloc_engine(ec, clus);
  gate_init(ec.gate, clus, gcfg);

  // ---- mode runner ---------------------------------------------------------
  struct Res { double acc, ms, macs, rmse, feat, agree, dead, skip, dens; vector<double> layer_macs, dead_rate; };
  int featw = 0;
  for (int i = 0; i < orig.n; i++) if (orig.nd[i].op == OP_GAP) featw = max(featw, node_ch(orig, i));
  featw = max(1, min(featw, 1408));
  vector<float> ref_logits(size_t(test.n) * cls), ref_feat(size_t(test.n) * size_t(featw));
  vector<ModeOut> outs;

  auto lskip = std::make_shared<vector<double>>(ec.m.n, 0.0);
  auto lfib = std::make_shared<vector<double>>(ec.m.n, 0.0);
  auto run_static = [&](Engine& e, int mode, bool learning_pass, bool temporal, int reps_,
                        vector<double>* msout = nullptr, bool warm = true) -> Res {
    Res r{};
    r.layer_macs.assign(e.m.n, 0.0);
    e.lmac.assign(e.m.n, 0.0);
    Ctr ct;
    vector<double> pass_ms;
    vector<float> lg(cls);
    int correct = 0;
    // approximate modes may use the dev-calibrated dead zone; M0/M1/M2 stay exact
    // the dev-calibrated dead zone is an *approximate* transform: only the Z
    // modes use it, so M0/M1/M2/M3/M4 and the random controls stay exact and
    // remain directly comparable with the committed results.
    e.deadzone = (mode == M2Z || mode == M3Z);
    e.gate.rng = 0x9E3779B97F4A7C15ull;
    if (warm && !learning_pass) {  // untimed warm-up: caches, branch predictors, boost state
      for (int i = 0; i < test.n; i++) {
        Ctr cw;
        forward(e, test.px.data() + size_t(i) * 3 * 32 * 32, mode, temporal, cw, lg.data(), nullptr, 0);
      }
      e.gate.rng = 0x9E3779B97F4A7C15ull;
    }
    if (reps_ <= 0) return r;  // warm-up only
    {   // untimed input-density probe over a small slice (tile kernels count exact zero scalars)
      e.measure_rms = true;
      for (int i = 0; i < std::min(20, test.n); i++) {
        Ctr c1;
        forward(e, test.px.data() + size_t(i) * 3 * 32 * 32, mode, temporal, c1, lg.data(), nullptr, 0);
        ct.nz += c1.nz; ct.nztot += c1.nztot;
      }
      ct.nz += e.tz; ct.nztot += e.tztot;
      e.tz = 0; e.tztot = 0;
      e.measure_rms = false;
    }
    for (int rep = 0; rep < reps_; rep++) {
      const double t0 = now_sec();
      int ok = 0;
      for (int i = 0; i < test.n; i++) {
        Ctr c1;
        c1.lskip = lskip; c1.lfib = lfib;
        forward(e, test.px.data() + size_t(i) * 3 * 32 * 32, mode, temporal, c1, lg.data(), nullptr, 0);
        ct.macs += c1.macs; ct.fibers += c1.fibers; ct.fiber_skips += c1.fiber_skips;
        ct.nz += c1.nz; ct.nztot += c1.nztot;
        int b; argmax(lg.data(), cls, b);
        if (b == test.lab[i]) ok++;
      }
      pass_ms.push_back((now_sec() - t0) * 1000.0);
      if (msout) msout->push_back(pass_ms.back());
      r.acc = double(ok) / test.n;
      if (learning_pass) break;
    }
    r.ms = (msout ? median(*msout) : median(pass_ms)) / double(test.n);
    if (getenv("CM_PROF")) {
      double tot = 0;
      for (int i = 1; i < e.m.n; i++) tot += e.ntime[i];
      fprintf(stderr, "-- PROFILE mode %d total %.2f ms/img\n", mode, tot / (double(reps_) * test.n) * 1000.0);
      for (int i = 1; i < e.m.n; i++) {
        const Node& q = e.m.nd[i];
        if (q.op != OP_CONV && q.op != OP_DW) continue;
        const double ms = e.ntime[i] / (double(reps_) * test.n) * 1000.0;
        fprintf(stderr, "   n%02d op%d %dx%dx%d k%d->%d  %.3f ms  %8.2f MMAC  %6.2f GMAC/s\n", i, q.op, q.h, q.w, q.ci, q.k, q.co, ms,
                r.layer_macs[i] / 1e6, r.layer_macs[i] / max(1e-9, ms) / 1e6);
      }
      e.ntime.assign(e.m.n, 0.0);
    }
    for (int i = 0; i < e.m.n && i < int(r.layer_macs.size()) && i < int(e.lmac.size()); i++)
      r.layer_macs[i] = e.lmac[i] / (reps_ * double(test.n));
    e.lmac.assign(e.m.n, 0.0);
    r.macs = ct.macs / (reps_ * double(test.n));
    r.dead = ct.fibers ? double(ct.fiber_skips) / double(ct.fibers) : 0.0;
    r.dens = ct.nztot ? double(ct.nz) / double(ct.nztot) : 0.0;
    (void)correct;
    return r;
  };

  int gap_node = -1;
  for (int i = 0; i < orig.n; i++) if (orig.nd[i].op == OP_GAP) gap_node = i;

  auto measure_fidelity = [&](Engine& e, int mode, int temporal, Res& r, bool clustered) {
    // the fidelity pass must run under exactly the mode's own dead-zone setting,
    // otherwise it inherits whatever flag the last timed mode left on the engine
    e.deadzone = (mode == M2Z || mode == M3Z);
    e.gate.rng = 0x9E3779B97F4A7C15ull;
    vector<float> lg(cls), back(featw, 0.f);
    double se = 0, sf = 0, nf = 0;
    int agree = 0;
    for (int i = 0; i < test.n; i++) {
      Ctr c1;
      float feat[1408];
      forward(e, test.px.data() + size_t(i) * 3 * 32 * 32, mode, temporal, c1, lg.data(), feat, featw);
      (void)0;
      if (clustered) {  // map the clustered pooled vector back into reference channel order
        const vector<int>& pm = perm[space_of[gap_node]];
        if (int(pm.size()) == featw) { for (int c2 = 0; c2 < featw; c2++) back[pm[c2]] = feat[c2]; memcpy(feat, back.data(), sizeof(float) * featw); }
      }
      int b; argmax(lg.data(), cls, b);
      int b0; argmax(&ref_logits[size_t(i) * cls], cls, b0);
      if (b == b0) agree++;
      for (int k = 0; k < cls; k++) { const double d = double(lg[k]) - ref_logits[size_t(i) * cls + k]; se += d * d; }
      double ff = 0, rr = 0;
      for (int c = 0; c < featw; c++) { const double d = feat[c] - ref_feat[size_t(i) * featw + c]; ff += d * d; rr += double(ref_feat[size_t(i) * featw + c]) * ref_feat[size_t(i) * featw + c]; }
      sf += ff; nf += rr;
    }
    r.rmse = sqrt(se / (double(test.n) * cls));
    r.feat = sqrt(sf / max(1e-30, nf));
    r.agree = double(agree) / test.n;
  };

  // ---- M0 dense ------------------------------------------------------------
  Res m0 = run_static(e0, M_DENSE, false, 0, reps);
  {
    vector<float> lg(cls);
    for (int i = 0; i < test.n; i++) {
      Ctr c1;
      float feat[512];
      forward(e0, test.px.data() + size_t(i) * 3 * 32 * 32, M_DENSE, 0, c1, lg.data(), feat, 512);
      memcpy(&ref_logits[size_t(i) * cls], lg.data(), sizeof(float) * cls);
      memcpy(&ref_feat[size_t(i) * 512], feat, sizeof(float) * 512);
      for (int nnode = 1; nnode < orig.n; nnode++) m0.layer_macs[nnode] += 0;
    }
  }
  // per-layer dense MACs
  {
    vector<double> lm(orig.n, 0.0);
    for (int i = 1; i < orig.n; i++) {
      const Node& q = orig.nd[i];
      if (q.op == OP_CONV) lm[i] = double(q.h) * q.w * q.k * q.ci * q.co;
      else if (q.op == OP_DW) lm[i] = double(q.h) * q.w * q.cp * 9;
      else if (q.op == OP_FC) lm[i] = double(q.ci) * q.co;
    }
    m0.layer_macs = lm;
  }
  double dense_macs = 0;
  for (double v : m0.layer_macs) dense_macs += v;

  printf("  \"dense\": {\"accuracy\": %.6f, \"macs_per_image\": %.0f, \"median_ms\": %.4f, \"layer_macs\": [",
         m0.acc, dense_macs, m0.ms);
  for (int i = 1; i < orig.n; i++) printf("%s%.0f", i > 1 ? ", " : "", m0.layer_macs[i]);
  printf("]},\n");

  // ---- 8-lane dead rates on the clustered model (ground truth, dense pass) --
  vector<double> block8_dead(orig.n, 0.0), half16_dead(orig.n, 0.0);
  {
    vector<float> lg(cls);
    for (int i = 0; i < test.n; i++) {
      Ctr c1;
      forward(ec, test.px.data() + size_t(i) * 3 * 32 * 32, M_DENSE, 0, c1, lg.data(), nullptr, 0);
      for (int ni = 1; ni < ec.m.n; ni++) {
        const Node& q = ec.m.nd[ni];
        if (q.op != OP_CONV && q.op != OP_DW) continue;
        const Ten& t = ec.t[ni];
        long long dead_f = 0, tot_f = 0, dead_b = 0, tot_b = 0;
        for (int y = 0; y < t.h; y++)
          for (int x = 0; x < t.w; x++) {
            const float* p = t.pix(y, x);
            for (int f = 0; f < t.F; f++) {
              bool any = false;
              for (int l = 0; l < 8; l++) if (p[f * 8 + l] > 0) any = true;
              dead_f += any ? 0 : 1; tot_f++;
            }
            for (int cb = 0; cb < t.cp; cb += 16) {
              bool any = false;
              for (int l = 0; l < 16; l++) if (p[cb + l] > 0) any = true;
              dead_b += any ? 0 : 1; tot_b++;
            }
          }
        block8_dead[ni] += double(dead_f) / max(1LL, tot_f);
        half16_dead[ni] += double(dead_b) / max(1LL, tot_b);
      }
    }
    for (int ni = 1; ni < ec.m.n; ni++) { block8_dead[ni] /= test.n; half16_dead[ni] /= test.n; }
  }
  printf("  \"cluster\": {\"block8_dead\": [");
  {
    bool first = true;
    for (int i = 1; i < ec.m.n; i++) {
      const Node& q = ec.m.nd[i];
      if (q.op != OP_CONV && q.op != OP_DW) continue;
      printf("%s%.6f", first ? "" : ", ", block8_dead[i]);
      first = false;
    }
  }
  printf("], \"block16_dead\": [");
  {
    bool first = true;
    for (int i = 1; i < ec.m.n; i++) {
      const Node& q = ec.m.nd[i];
      if (q.op != OP_CONV && q.op != OP_DW) continue;
      printf("%s%.6f", first ? "" : ", ", half16_dead[i]);
      first = false;
    }
  }
  printf("], \"accuracy\": %.6f},\n", m0.acc);

  if (getenv("CM_CMP") && test.n) {
    Engine ea, eb;
    alloc_engine(ea, orig);
    alloc_engine(eb, orig);
    gate_init(eb.gate, orig, gcfg);
    Ctr c1, c2;
    float la[64], lb[64];
    forward(ea, test.px.data(), M_DENSE, 0, c1, la, nullptr, 0);
    forward(eb, test.px.data(), M_SPARSE, 0, c2, lb, nullptr, 0);
    for (int i = 1; i < orig.n; i++) {
      const Ten& A = ea.t[i];
      const Ten& B = eb.t[i];
      if (A.d.size() != B.d.size()) { fprintf(stderr, "n%d size mismatch\n", i); continue; }
      double md = 0; size_t at = 0;
      for (size_t k = 0; k < A.d.size(); k++) { const double d = fabs(double(A.d[k]) - double(B.d[k])); if (d > md) { md = d; at = k; } }
      fprintf(stderr, "n%02d op%d %dx%dx%d maxdiff %.6g at %zu (%.6f vs %.6f)\n", i, orig.nd[i].op, A.h, A.w, A.c, md, at, A.d[at], B.d[at]);
    }
    fprintf(stderr, "logits dense %.6f sparse %.6f\n", la[0], lb[0]);
  }

  // ---- mode sweep ----------------------------------------------------------
  struct ModeSpec { int mode; const char* name; bool clustered; bool learn; int temporal; };
  vector<ModeSpec> specs = {
      {M_DENSE, "M0: dense-avx512", false, false, 0},
      {M_SPARSE, "M1: input-sparse-only", false, false, 0},
      {M_CLUSTER, "M2: clustered-tap-only", true, false, 0},
      {M2Z, "M2z: clustered-deadzone", true, false, 0},
      {M_CM, "M3: cm-v4-static", true, false, 0},
      {M3Z, "M3z: cm-deadzone", true, false, 0},
  };

  // learn gate tables (unsupervised: activations only), then calibrate per-layer thresholds
  const ImageSet& learn_set = train.n ? train : dev;
  const double prec_target = getenv("CM_PREC") ? atof(getenv("CM_PREC")) : 0.995;
  const double rate_cap = getenv("CM_RATE") ? atof(getenv("CM_RATE")) : 0.50;
  double m3_skip_rate = 0;
  // (2a) pre-warm the CM tables on dev first: each (layer, fiber, context) cell
  // starts from dev's measured all-dead rate instead of the 0.5 cold start.
  if (dev.n) {
    vector<float> lg(cls);
    for (int im = 0; im < dev.n; im++) {
      Ctr c1;
      forward(ec, dev.px.data() + size_t(im) * 3 * 32 * 32, M_LEARN, 0, c1, lg.data(), nullptr, 0);
    }
    if (verify) fprintf(stderr, "CM tables pre-warmed on dev (%d images)\n", dev.n);
  }
  if (learn_set.n) {
    vector<float> lg(cls);
    Engine& el = ec;
    for (int im = 0; im < learn_set.n; im++) {
      Ctr c1;
      forward(el, learn_set.px.data() + size_t(im) * 3 * 32 * 32, M_LEARN, 0, c1, lg.data(), nullptr, 0);
    }
  }
  gate_calibrate(ec.gate, ec.m, prec_target, rate_cap);

  // ---- (2c) per-layer dead-zone calibration, dev only -----------------------
  // eps_l = eta * RMS_l(dev); eta is the largest value whose dev accuracy loss
  // stays within 0.5 pp.  Applied only to the approximate modes (M3/M4/random/M2z).
  double eps_eta = 0.0, eps_dev_acc = 0.0, base_dev_acc = 0.0;
  vector<string> deadzone_report;
  if (dev.n) {
    vector<float> lg(cls);
    // (a) per-node output RMS on dev, measured with the exact clustered model
    ec.measure_rms = true;
    for (int im = 0; im < dev.n; im++) {
      Ctr c1;
      forward(ec, dev.px.data() + size_t(im) * 3 * 32 * 32, M_CLUSTER, 0, c1, lg.data(), nullptr, 0);
    }
    ec.measure_rms = false;
    for (int i = 1; i < ec.m.n; i++) ec.rms[i] /= double(dev.n);
    auto dev_accuracy = [&]() {
      int ok = 0;
      for (int im = 0; im < dev.n; im++) {
        Ctr c1;
        forward(ec, dev.px.data() + size_t(im) * 3 * 32 * 32, M_CLUSTER, 0, c1, lg.data(), nullptr, 0);
        int b; argmax(lg.data(), cls, b);
        if (b == dev.lab[im]) ok++;
      }
      return double(ok) / double(dev.n);
    };
    // ---- (3) per-layer greedy coordinate ascent: eta_l then tau_l -----------
    // Order: deepest first (largest node index), because the tail layers carry
    // most of the MACs and a dead zone there is worth the most.
    if (getenv("CM_SKIP_CAL") == nullptr) {
      const int calN = std::min(dev.n, 120);
      auto eval_pass = [&](int mode, double* acc_out) -> double {
        int ok = 0;
        double my = 0;
        for (int im = 0; im < calN; im++) {
          Ctr c1;
          forward(ec, dev.px.data() + size_t(im) * 3 * 32 * 32, mode, 0, c1, lg.data(), nullptr, 0);
          int b; argmax(lg.data(), cls, b);
          if (b == dev.lab[im]) ok++;
          my += c1.macs;
        }
        *acc_out = double(ok) / double(calN);
        return my / double(calN);
      };
      for (int i = 1; i < ec.m.n; i++) ec.eps[i] = 0.f;
      ec.deadzone = false;
      double base_acc = 0;
      eval_pass(M_CLUSTER, &base_acc);
      const double budget = 0.0035;  // 0.35 pp total dev budget
      vector<int> ord;
      for (int i = 1; i < ec.m.n; i++)
        if (ec.m.nd[i].op == OP_CONV || ec.m.nd[i].op == OP_DW) ord.push_back(i);
      std::sort(ord.begin(), ord.end(), [](int a, int b) { return a > b; });
      const double eta_grid[] = {0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80};
      const double tau_grid[] = {0.85, 0.90, 0.94, 0.97, 0.995};
      vector<string> lines;
      ec.deadzone = true;
      for (int i : ord) {
        const double sig = std::sqrt(std::max(1e-12, ec.rms[i]));
        float best_eta = 0.f;
        double best_macs = 1e300, a = 0;
        for (double eta : eta_grid) {
          ec.eps[i] = float(eta * sig);
          const double m_i = eval_pass(M2Z, &a);
          // only pay accuracy for a dead zone that actually removes work:
          // require a >=0.5% MAC reduction over the best accepted config so far
          if (a >= base_acc - budget && m_i < best_macs * 0.995) { best_macs = m_i; best_eta = float(eta); }
        }
        ec.eps[i] = float(best_eta * sig);
        int best_th = 0;
        for (double tau : tau_grid) {
          gate_calibrate_one(ec.gate, ec.m, i, tau, 0.50);
          const int th = ec.gate.tabth[i];
          if (!th) { if (best_th == 0) best_th = 0; continue; }
          ec.gate.tabth[i] = (uint8_t)th;
          const double m_i = eval_pass(M3Z, &a);
          if (a >= base_acc - budget && m_i < best_macs) { best_macs = m_i; best_th = th; }
        }
        ec.gate.tabth[i] = (uint8_t)best_th;
        char buf[128];
        snprintf(buf, sizeof(buf), "%d:eta=%.2f,eps=%.5f,tabth=%d", i, best_eta, ec.eps[i], best_th);
        lines.push_back(buf);
      }
      ec.deadzone = false;
      e0.eps = ec.eps;
      e0.gate.tabth = ec.gate.tabth;
      eps_dev_acc = base_acc;
      if (verify) {
        double full = 0;
        int ok = 0;
        for (int im = 0; im < dev.n; im++) {
          Ctr c1;
          forward(ec, dev.px.data() + size_t(im) * 3 * 32 * 32, M3Z, 1, c1, lg.data(), nullptr, 0);
          int b; argmax(lg.data(), cls, b);
          if (b == dev.lab[im]) ok++;
        }
        full = double(ok) / double(dev.n);
        fprintf(stderr, "per-layer calibration on %d dev images (exact %.6f), full-dev M3Z %.6f:\n  ",
                calN, base_acc, full);
        for (size_t k = 0; k < lines.size(); k++) {
          fprintf(stderr, "%s%s", lines[k].c_str(), (k % 6 == 5) ? "\n  " : "  ");
        }
        fprintf(stderr, "\n");
      }
      deadzone_report = lines;
    }
    // ---- per-layer kernel dispatch (work order item 1) ---------------------
    // Time each candidate kernel per layer on a dev slice; the argmin per node
    // wins.  Layers whose block-active rate is high (shallow 3x3, depthwise)
    // end up on the dense tile kernel, small-channel 1x1 layers on the pair
    // (tap, 16-channel block) packed kernel, the rest on exact compaction.
    {
      const int calN = std::min(dev.n, 90);
      if (const char* fk = getenv("CM_KERN")) {
        const int k = atoi(fk);
        for (int i = 1; i < ec.m.n; i++) if (ec.m.nd[i].op == OP_CONV) { ec.kern[i] = k; e0.kern[i] = k; }
        if (verify) fprintf(stderr, "kernel dispatch FORCED to %d\n", k);
      } else {
      vector<double> tk[3];
      ec.deadzone = true;   // dispatch under the same (approximate) mode M3 will use
      for (int k = 0; k < 3; k++) {
        for (int i = 1; i < ec.m.n; i++)
          if (ec.m.nd[i].op == OP_CONV) ec.kern[i] = k;
        ec.ntime.assign(ec.m.n, 0.0);
        ec.timing = true;
        for (int im = 0; im < calN; im++) {
          Ctr c1;
          forward(ec, dev.px.data() + size_t(im) * 3 * 32 * 32, M2Z, 0, c1, lg.data(), nullptr, 0);
        }
        ec.timing = false;
        tk[k] = ec.ntime;
      }
      ec.deadzone = false;
      int nk[3] = {0, 0, 0};
      for (int i = 1; i < ec.m.n; i++) {
        if (ec.m.nd[i].op != OP_CONV) continue;
        // require a clear (>15%) win before leaving the register-tiled dense kernel
        int best = 1;
        if (tk[2][i] < 0.85 * tk[1][i]) best = 2;
        if (tk[0][i] < 0.85 * tk[1][i] && (best == 1 || tk[0][i] < 0.85 * tk[2][i])) best = 0;
        nk[best]++;
        ec.kern[i] = best;
        e0.kern[i] = best;
      }
      if (verify)
        fprintf(stderr, "kernel dispatch (%d conv layers): compacted %d, tile-dense %d, packed %d\n",
                nk[0] + nk[1] + nk[2], nk[0], nk[1], nk[2]);
      }
    }
  }
  if (verify) {
    fprintf(stderr, "gate thresholds:");
    for (int ni = 1; ni < ec.m.n; ni++) fprintf(stderr, " %d", (int)ec.gate.tabth[ni]);
    fprintf(stderr, "\n");
  }

  // Modes are timed ROUND-ROBIN across reps: this machine throttles within a
  // run, so sweeping one mode for all reps and then the next made a mode's
  // measured cost depend on its position in the sweep (M0 12.0 vs M2 16.5 ms
  // for identical work). Interleaving gives every mode the same thermal/cache
  // history, then the per-mode median is taken.
  vector<Res> res(specs.size());
  vector<vector<double>> msA(specs.size());
  for (size_t si = 0; si < specs.size(); si++)
    run_static(specs[si].clustered ? ec : e0, specs[si].mode, false, specs[si].temporal, 0);
  for (int rep = 0; rep < reps; rep++)
    for (size_t si = 0; si < specs.size(); si++) {
      Engine& e = specs[si].clustered ? ec : e0;
      res[si] = run_static(e, specs[si].mode, false, specs[si].temporal, 1, &msA[si], false);
    }
  for (size_t si = 0; si < specs.size(); si++) {
    Engine& e = specs[si].clustered ? ec : e0;
    res[si].ms = median(msA[si]) / double(test.n);
    measure_fidelity(e, specs[si].mode, specs[si].temporal, res[si], specs[si].clustered);
    if (specs[si].mode == M_CM) m3_skip_rate = res[si].dead;
  }
  m0 = res[0];  // dense baseline from the same interleaved group (identical work)

  // random controls use M3's measured per-layer fiber-skip rate
  vector<double> layer_skip(ec.m.n, 0.0);
  for (int ni = 1; ni < ec.m.n; ni++) {
    const Node& q = ec.m.nd[ni];
    if (q.op != OP_CONV && q.op != OP_DW) continue;
    if ((*lfib)[ni] > 0) layer_skip[ni] = std::max(0.02, (*lskip)[ni] / (*lfib)[ni]);
  }
  if (verify) {
    fprintf(stderr, "per-layer M3 skip rate:");
    for (int ni = 1; ni < ec.m.n; ni++) fprintf(stderr, " %.3f", layer_skip[ni]);
    fprintf(stderr, "\n");
  }
  gate_init(e0.gate, e0.m, gcfg);   // resets e0.skipq, so (re)assign it below
  for (int ni = 1; ni < ec.m.n; ni++) ec.gate.skipq[ni] = (float)layer_skip[ni];
  for (int ni = 1; ni < e0.m.n; ni++) e0.gate.skipq[ni] = (float)layer_skip[ni];

  vector<Res> rres(2);
  {
    const int rp = max(1, reps / 2);
    vector<vector<double>> msB(2);
    run_static(ec, M_RAND_BLOCK, false, 0, 0);
    run_static(e0, M_RAND_UNC, false, 0, 0);
    for (int rep = 0; rep < rp; rep++) {
      rres[0] = run_static(ec, M_RAND_BLOCK, false, 0, 1, &msB[0], false);
      rres[1] = run_static(e0, M_RAND_UNC, false, 0, 1, &msB[1], false);
    }
    rres[0].ms = median(msB[0]) / double(test.n);
    rres[1].ms = median(msB[1]) / double(test.n);
  }
  measure_fidelity(ec, M_RAND_BLOCK, 0, rres[0], true);
  measure_fidelity(e0, M_RAND_UNC, 0, rres[1], false);

  printf("  \"deadzone\": {\"dev_acc\": %.6f, \"per_layer\": [", eps_dev_acc);
  for (size_t k = 0; k < deadzone_report.size(); k++)
    printf("%s\"%s\"", k ? ", " : "", deadzone_report[k].c_str());
  printf("], \"eps\": [");
  for (int i = 1; i < ec.m.n; i++) printf("%s%.6e", i > 1 ? ", " : "", ec.eps[i]);
  printf("], \"kern\": [");
  for (int i = 1; i < ec.m.n; i++) printf("%s%d", i > 1 ? ", " : "", ec.kern[i]);
  printf("]},\n");
  printf("  \"modes\": {\n");
  auto emit = [&](const char* name, const Res& r, bool last) {
    printf("    \"%s\": {\"accuracy\": %.6f, \"macs_per_image\": %.0f, \"mac_elimination\": %.6f, "
           "\"median_ms\": %.4f, \"speedup\": %.4f, \"logit_rmse\": %.6e, \"feat_rel_l2\": %.6e, "
           "\"dense_agreement\": %.6f, \"fiber_skip_rate\": %.6f, \"input_density\": %.6f, \"layer_macs\": [",
           name, r.acc, r.macs, 1.0 - r.macs / dense_macs, r.ms, m0.ms / r.ms, r.rmse, r.feat, r.agree, r.dead, r.dens);
    for (int i = 1; i < ec.m.n; i++) printf("%s%.0f", i > 1 ? ", " : "", r.layer_macs[i]);
    printf("]}%s\n", last ? "" : ",");
  };
  // layer macs are only tracked for the clustered engine; recompute dense-per-mode is not needed
  for (size_t si = 1; si < specs.size(); si++) emit(specs[si].name, res[si], false);
  emit("block-matched-random-skip", rres[0], false);
  emit("unclustered-random-skip", rres[1], true);
  printf("  },\n");

  // ---- stream --------------------------------------------------------------
  double stream_acc = 0, stream_ms = 0, stream_macs = 0, stream_acc4 = 0, stream_ms4 = 0, stream_macs4 = 0;
  int stream_frames = 0;
  if (stream.seqs) {
    stream_frames = stream.seqs * stream.frames;
    auto run_stream = [&](Engine& e, int mode, int temporal, int reps_, double& acc, double& ms, double& macs) {
      vector<float> lg(cls);
      vector<double> pass_ms;
      uint64_t rng_save = e.gate.rng;
      for (int rep = 0; rep < reps_; rep++) {
        e.gate.rng = rng_save;
        Ctr ct;
        long long ok = 0;
        for (int s = 0; s < stream.seqs; s++) {
          // reset temporal state at sequence start
          for (int ni = 0; ni < e.m.n; ni++)
            if (!e.gate.prevmap[ni].empty()) std::fill(e.gate.prevmap[ni].begin(), e.gate.prevmap[ni].end(), 0);
          for (int fr = 0; fr < stream.frames; fr++) {
            const float* px = stream.px.data() + ((size_t(s) * stream.frames + fr) * 32 * 32 * 3);
            Ctr c1;
            forward(e, px, mode, temporal, c1, lg.data(), nullptr, 0);
            ct.macs += c1.macs;
            int b; argmax(lg.data(), cls, b);
            if (b == stream.lab[s]) ok++;
            if (temporal)
              for (int ni = 0; ni < e.m.n; ni++)
                if (!e.gate.prevmap[ni].empty()) {
                  const auto& am = e.gate.actmap[ni];
                  auto& pm = e.gate.prevmap[ni];
                  for (size_t j = 0; j < am.size(); j++) pm[j] = (uint8_t)((am[j] == 1 ? 1 : 0) | ((am[j] == 2) ? 2 : 0));
                }
          }
        }
        pass_ms.push_back((now_sec() - 0) * 0 + 0.0);   // placeholder replaced below
        acc = double(ok) / stream_frames;
        macs = ct.macs / double(stream_frames);
      }
      ms = 0;
      (void)pass_ms;
    };
    // timed stream loop (separate, so timing is clean)
    auto time_stream = [&](Engine& e, int mode, int temporal, int reps_, double& acc, double& ms, double& macs) {
      vector<float> lg(cls);
      vector<double> pass_ms;
      for (int rep = -1; rep < reps_; rep++) {  // rep -1 = untimed warm-up
        const double t0 = now_sec();
        Ctr ct;
        long long ok = 0;
        for (int s = 0; s < stream.seqs; s++) {
          for (int ni = 0; ni < e.m.n; ni++)
            if (!e.gate.prevmap[ni].empty()) std::fill(e.gate.prevmap[ni].begin(), e.gate.prevmap[ni].end(), 0);
          for (int fr = 0; fr < stream.frames; fr++) {
            const float* px = stream.px.data() + ((size_t(s) * stream.frames + fr) * 32 * 32 * 3);
            Ctr c1;
            forward(e, px, mode, temporal, c1, lg.data(), nullptr, 0);
            ct.macs += c1.macs;
            int b; argmax(lg.data(), cls, b);
            if (b == stream.lab[s]) ok++;
            if (temporal)
              for (int ni = 0; ni < e.m.n; ni++)
                if (!e.gate.prevmap[ni].empty()) {
                  const auto& am = e.gate.actmap[ni];
                  auto& pm = e.gate.prevmap[ni];
                  for (size_t j = 0; j < am.size(); j++) pm[j] = (uint8_t)((am[j] == 1 ? 1 : 0) | ((am[j] == 2) ? 2 : 0));
                }
          }
        }
        if (rep >= 0) pass_ms.push_back((now_sec() - t0) * 1000.0);
        acc = double(ok) / stream_frames;
        macs = ct.macs / double(stream_frames);
      }
      ms = median(pass_ms) / stream_frames;
    };

    // temporal warmup (unsupervised online learning over the stream)
    {
      vector<float> lg(cls);
      for (int s = 0; s < stream.seqs; s++) {
        for (int ni = 0; ni < ec.m.n; ni++)
          if (!ec.gate.prevmap[ni].empty()) std::fill(ec.gate.prevmap[ni].begin(), ec.gate.prevmap[ni].end(), 0);
        for (int fr = 0; fr < stream.frames; fr++) {
          Ctr c1;
          forward(ec, stream.px.data() + ((size_t(s) * stream.frames + fr) * 32 * 32 * 3), M_LEARN, 1, c1, lg.data(), nullptr, 0);
          for (int ni = 0; ni < ec.m.n; ni++)
            if (!ec.gate.prevmap[ni].empty()) {
              const auto& am = ec.gate.actmap[ni];
              auto& pm = ec.gate.prevmap[ni];
              for (size_t j = 0; j < am.size(); j++) pm[j] = (uint8_t)((am[j] == 1 ? 1 : 0) | ((am[j] == 2) ? 2 : 0));
            }
        }
      }
    }
    time_stream(e0, M_DENSE, 0, max(1, reps_stream / 2), stream_acc, stream_ms, stream_macs);
    time_stream(ec, M_TEMPORAL, 1, reps_stream, stream_acc4, stream_ms4, stream_macs4);
    (void)run_stream;
  }
  printf("  \"stream\": {\"frames\": %d, \"seqs\": %d, \"frames_per_seq\": %d, \"dense\": {\"accuracy\": %.6f, \"ms_per_frame\": %.4f, \"macs_per_frame\": %.0f}, "
         "\"M4: cm-v4-temporal\": {\"accuracy\": %.6f, \"ms_per_frame\": %.4f, \"macs_per_frame\": %.0f, \"mac_elimination\": %.6f, \"speedup\": %.4f}}\n",
         stream_frames, stream.seqs, stream.frames, stream_acc, stream_ms, stream_macs,
         stream_acc4, stream_ms4, stream_macs4,
         1.0 - stream_macs4 / max(1.0, stream_macs), stream_ms / max(1e-9, stream_ms4));

  printf("}\n");

  if (!jp.empty()) {
    // re-emit the same JSON into the file via a re-run of the printing would be wasteful;
    // instead the caller captures stdout.
  }
  if (verify) {
    fprintf(stderr, "dense acc %.4f macs %.0f ms %.3f\n", m0.acc, dense_macs, m0.ms);
    for (size_t si = 0; si < specs.size(); si++)
      fprintf(stderr, "%-28s acc %.4f rmse %.2e feat %.2e agree %.4f macs %.0f ms %.3f\n",
              specs[si].name, res[si].acc, res[si].rmse, res[si].feat, res[si].agree, res[si].macs, res[si].ms);
  }
  return 0;
}
