// Fused zero-copy SwiGLU coupled skipping, CUDA (RTX 5070) — Phase 4 / Task 1.
//
// Same contract as swiglu_avx512.cpp: gate dense -> one compacted block list ->
// up_proj active rows -> h = g*u -> down_proj active columns.  No weight copy:
// up rows and down columns are read in place from the existing buffers, and the
// idx[] array is produced once per token and shared by both projections.
//
// Kernels are deliberately simple (thread = one output element for the GEMVs,
// one thread block per token for selection) because at tau>0 the win comes from
// reading fewer weight bytes, not from tensor-core throughput.
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <math_constants.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef LANE
#define LANE 32
#endif

__device__ __forceinline__ float silu_f(float v) { return v / (1.f + __expf(-v)); }

__device__ __forceinline__ float warp_reduce(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// warp-per-row GEMV: 32 lanes read 32 consecutive floats (fully coalesced) and
// reduce with shuffles.  A thread-per-row version is 5-10x slower because every
// lane strides D floats.
__global__ void gemv_warp_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                 float* __restrict__ y, int nrows, int D, int with_silu) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= nrows) return;
    const float* w = W + (size_t)warp * D;
    float acc = 0.f;
    for (int d = lane; d < D; d += 32) acc += x[d] * w[d];
    acc = warp_reduce(acc);
    if (lane == 0) y[warp] = with_silu ? silu_f(acc) : acc;
}

// up over the active rows: same warp-per-row shape, row index read from idx.
__global__ void up_warp_kernel(const float* __restrict__ W, const float* __restrict__ x,
                               float* __restrict__ u, int D, const int* __restrict__ idx,
                               const int* __restrict__ kp, int max_blocks) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int K = kp[0];
    const int j = warp;
    if (j >= K * LANE) return;
    const int r = idx[j >> 5] * LANE + (j & 31);
    const float* w = W + (size_t)r * D;
    float acc = 0.f;
    for (int d = lane; d < D; d += 32) acc += x[d] * w[d];
    acc = warp_reduce(acc);
    if (lane == 0) u[j] = acc;
    (void)max_blocks;
}

// y[r] = dot(x, W + r*D)  (optionally SiLU)
__global__ void gemv_rows_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                 float* __restrict__ y, int nrows, int D, int with_silu) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nrows) return;
    const float* w = W + (size_t)r * D;
    float acc = 0.f;
    for (int d = 0; d < D; ++d) acc += x[d] * w[d];
    y[r] = with_silu ? silu_f(acc) : acc;
}


// ---- float4 + 2 rows per warp ---------------------------------------------
// Each lane loads 4 consecutive floats of x once and reuses them for two output
// rows, so the x traffic is halved and the weight loads are 16-byte vectorised.
__device__ __forceinline__ float dot4(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__global__ void gemv2_warp_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                  float* __restrict__ y, int nrows, int D, int with_silu) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int r0 = warp * 2;
    if (r0 >= nrows) return;
    const int r1 = (r0 + 1 < nrows) ? r0 + 1 : r0;
    const float* w0 = W + (size_t)r0 * D;
    const float* w1 = W + (size_t)r1 * D;
    float acc0 = 0.f, acc1 = 0.f;
    for (int d = lane * 4; d + 3 < D; d += 128) {
        const float4 xv = *reinterpret_cast<const float4*>(x + d);
        acc0 += dot4(xv, *reinterpret_cast<const float4*>(w0 + d));
        acc1 += dot4(xv, *reinterpret_cast<const float4*>(w1 + d));
    }
    acc0 = warp_reduce(acc0);
    acc1 = warp_reduce(acc1);
    if (lane == 0) {
        y[r0] = with_silu ? silu_f(acc0) : acc0;
        if (r0 + 1 < nrows) y[r1] = with_silu ? silu_f(acc1) : acc1;
    }
}

__global__ void up2_warp_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                float* __restrict__ u, int D, const int* __restrict__ idx,
                                const int* __restrict__ kp) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int j0 = warp * 2;
    if (j0 >= kp[0] * LANE) return;
    const int j1 = j0 + 1;
    const int r0 = idx[j0 >> 5] * LANE + (j0 & 31);
    const int r1 = (j1 < kp[0] * LANE) ? idx[j1 >> 5] * LANE + (j1 & 31) : r0;
    const float* w0 = W + (size_t)r0 * D;
    const float* w1 = W + (size_t)r1 * D;
    float acc0 = 0.f, acc1 = 0.f;
    for (int d = lane * 4; d + 3 < D; d += 128) {
        const float4 xv = *reinterpret_cast<const float4*>(x + d);
        acc0 += dot4(xv, *reinterpret_cast<const float4*>(w0 + d));
        acc1 += dot4(xv, *reinterpret_cast<const float4*>(w1 + d));
    }
    acc0 = warp_reduce(acc0);
    acc1 = warp_reduce(acc1);
    if (lane == 0) {
        u[j0] = acc0;
        if (j1 < kp[0] * LANE) u[j1] = acc1;
    }
}

// down: two output rows per warp, both reusing the same h block
__global__ void down2_warp_kernel(const float* __restrict__ W, const float* __restrict__ h,
                                  float* __restrict__ out, int H, const int* __restrict__ idx,
                                  const int* __restrict__ kp, int O) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int o0 = warp * 2;
    if (o0 >= O) return;
    const int o1 = (o0 + 1 < O) ? o0 + 1 : o0;
    const float* wr0 = W + (size_t)o0 * H;
    const float* wr1 = W + (size_t)o1 * H;
    const int K = kp[0];
    float acc0 = 0.f, acc1 = 0.f;
    for (int k = 0; k < K; ++k) {
        const float hv = h[k * LANE + lane];
        const int base = idx[k] * LANE + lane;
        acc0 += hv * wr0[base];
        acc1 += hv * wr1[base];
    }
    acc0 = warp_reduce(acc0);
    acc1 = warp_reduce(acc1);
    if (lane == 0) {
        out[o0] = acc0;
        if (o0 + 1 < O) out[o1] = acc1;
    }
}


// ---- 4 rows per warp ------------------------------------------------------
// One x load serves four weight rows: better MLP (memory-level parallelism) than
// the 2-row variant, which is what the GEMV needs on a bandwidth-bound card.
__global__ void gemv4_warp_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                  float* __restrict__ y, int nrows, int D, int with_silu) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int r = warp * 4;
    if (r >= nrows) return;
    const int n = min(4, nrows - r);
    const float* w0 = W + (size_t)r * D;
    const float* w1 = w0 + D;
    const float* w2 = w0 + 2 * D;
    const float* w3 = w0 + 3 * D;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    for (int d = lane * 4; d + 3 < D; d += 128) {
        const float4 xv = *reinterpret_cast<const float4*>(x + d);
        a0 += dot4(xv, *reinterpret_cast<const float4*>(w0 + d));
        if (n > 1) a1 += dot4(xv, *reinterpret_cast<const float4*>(w1 + d));
        if (n > 2) a2 += dot4(xv, *reinterpret_cast<const float4*>(w2 + d));
        if (n > 3) a3 += dot4(xv, *reinterpret_cast<const float4*>(w3 + d));
    }
    a0 = warp_reduce(a0);
    if (n > 1) a1 = warp_reduce(a1);
    if (n > 2) a2 = warp_reduce(a2);
    if (n > 3) a3 = warp_reduce(a3);
    if (lane == 0) {
        y[r] = with_silu ? silu_f(a0) : a0;
        if (n > 1) y[r + 1] = with_silu ? silu_f(a1) : a1;
        if (n > 2) y[r + 2] = with_silu ? silu_f(a2) : a2;
        if (n > 3) y[r + 3] = with_silu ? silu_f(a3) : a3;
    }
}

__global__ void up4_warp_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                float* __restrict__ u, int D, const int* __restrict__ idx,
                                const int* __restrict__ kp) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int J = kp[0] * LANE;
    const int j0 = warp * 4;
    if (j0 >= J) return;
    const int n = min(4, J - j0);
    int rows[4];
    for (int q = 0; q < 4; ++q) rows[q] = (q < n) ? idx[(j0 + q) >> 5] * LANE + ((j0 + q) & 31) : 0;
    const float* w[4];
    for (int q = 0; q < 4; ++q) w[q] = W + (size_t)rows[q] * D;
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    for (int d = lane * 4; d + 3 < D; d += 128) {
        const float4 xv = *reinterpret_cast<const float4*>(x + d);
        for (int q = 0; q < n; ++q) a[q] += dot4(xv, *reinterpret_cast<const float4*>(w[q] + d));
    }
    for (int q = 0; q < n; ++q) a[q] = warp_reduce(a[q]);
    if (lane == 0) for (int q = 0; q < n; ++q) u[j0 + q] = a[q];
}

__global__ void down4_warp_kernel(const float* __restrict__ W, const float* __restrict__ h,
                                  float* __restrict__ out, int H, const int* __restrict__ idx,
                                  const int* __restrict__ kp, int O) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int r = warp * 4;
    if (r >= O) return;
    const int n = min(4, O - r);
    const float* w0 = W + (size_t)r * H;
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    const int K = kp[0];
    for (int k = 0; k < K; ++k) {
        const float hv = h[k * LANE + lane];
        const int base = idx[k] * LANE + lane;
        for (int q = 0; q < n; ++q) a[q] += hv * w0[(size_t)q * H + base];
    }
    for (int q = 0; q < n; ++q) a[q] = warp_reduce(a[q]);
    if (lane == 0) for (int q = 0; q < n; ++q) out[r + q] = a[q];
}

// per-block L2 energy from g, then a serial top-K selection (nb <= 288).
__global__ void compact_kernel(const float* __restrict__ g, int* __restrict__ idx,
                               int* __restrict__ kout, int nb, float rate,
                               const float* __restrict__ wb2) {
    extern __shared__ float e[];
    for (int b = threadIdx.x; b < nb; b += blockDim.x) {
        float s = 0.f;
        for (int l = 0; l < LANE; ++l) { float v = g[b * LANE + l]; s += v * v; }
        // offline importance: score_b = w_b * ||g_b||  <=>  energy *= w_b^2
        e[b] = wb2 ? s * wb2[b] : s;
    }
    __syncthreads();
    int n_keep = (int)lroundf(nb * (1.0f - rate));
    if (n_keep < 1) n_keep = 1;
    if (n_keep > nb) n_keep = nb;
    unsigned char* used = (unsigned char*)(e + nb);
    for (int b = threadIdx.x; b < nb; b += blockDim.x) used[b] = 0;
    __syncthreads();
    // block-parallel top-K: K <= 288 passes of a max-reduce; the serial version
    // (one thread scanning nb blocks K times) cost ~300 us per layer.
    __shared__ float red[256];
    for (int k = 0; k < n_keep; ++k) {
        float mine = -CUDART_INF_F;
        for (int b = threadIdx.x; b < nb; b += blockDim.x)
            if (!used[b] && e[b] > mine) mine = e[b];
        red[threadIdx.x] = mine;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (threadIdx.x < s)
                red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
            __syncthreads();
        }
        const float mx = red[0];
        __syncthreads();
        for (int b = threadIdx.x; b < nb; b += blockDim.x)
            if (!used[b] && e[b] == mx) { used[b] = 1; idx[k] = b; }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int a = 1; a < n_keep; ++a) {
            int v = idx[a], j = a - 1;
            while (j >= 0 && idx[j] > v) { idx[j + 1] = idx[j]; --j; }
            idx[j + 1] = v;
        }
        kout[0] = n_keep;
    }
}

// u[k*LANE+l] = (W_up row idx[k]*LANE+l) . x   (active rows only)
__global__ void up_active_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                 float* __restrict__ u, int D, const int* __restrict__ idx,
                                 int K) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= K * LANE) return;
    const int r = idx[j / LANE] * LANE + (j % LANE);
    const float* w = W + (size_t)r * D;
    float acc = 0.f;
    for (int d = 0; d < D; ++d) acc += x[d] * w[d];
    u[j] = acc;
}

__global__ void mul_gate_kernel(float* __restrict__ u, const float* __restrict__ g,
                                const int* __restrict__ idx, const int* __restrict__ kp,
                                int kcap, float alpha) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= kp[0] * LANE || j >= kcap * LANE) return;
    const int r = idx[j / LANE] * LANE + (j % LANE);
    u[j] *= g[r] * alpha;
}

// out[o] = sum_{k<K} sum_{l<LANE} h[k*LANE+l] * W_down[o, idx[k]*LANE+l]
__global__ void down_warp_kernel(const float* __restrict__ W, const float* __restrict__ h,
                                 float* __restrict__ out, int H, const int* __restrict__ idx,
                                 const int* __restrict__ kp, int O) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= O) return;
    const int K = kp[0];
    const float* wr = W + (size_t)warp * H;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += h[k * LANE + lane] * wr[idx[k] * LANE + lane];
    }
    acc = warp_reduce(acc);
    if (lane == 0) out[warp] = acc;
}

// Host entry: returns K (active blocks); fills `out`.
int swiglu_coupled(torch::Tensor x, torch::Tensor Wg, torch::Tensor Wu, torch::Tensor Wd,
                   torch::Tensor g_buf, torch::Tensor idx_buf, torch::Tensor k_buf,
                   torch::Tensor h_buf, torch::Tensor out, double rate,
                   c10::optional<torch::Tensor> wb2_opt) {
    const float* wb2 = nullptr;
    if (wb2_opt.has_value() && wb2_opt->defined())
        wb2 = wb2_opt->data_ptr<float>();
    const int D = Wg.size(1), H = Wg.size(0), O = Wd.size(0);
    const int nb = H / LANE;
    const float* xp = x.data_ptr<float>();
    const float* gwp = Wg.data_ptr<float>();
    const float* wup = Wu.data_ptr<float>();
    const float* wdp = Wd.data_ptr<float>();
    float* gp = g_buf.data_ptr<float>();
    int* ip = idx_buf.data_ptr<int>();
    int* kp = k_buf.data_ptr<int>();
    float* hp = h_buf.data_ptr<float>();
    float* op = out.data_ptr<float>();
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();

    // no host sync: every kernel is sized for the worst case and reads K itself
    // warps = ceil(rows/2), 8 warps per 256-thread block
    gemv2_warp_kernel<<<((H / 2) * 32 + 255) / 256, 256, 0, st>>>(gwp, xp, gp, H, D, 1);
    compact_kernel<<<1, 256, nb * sizeof(float) + nb, st>>>(gp, ip, kp, nb, (float)rate, wb2);
    up2_warp_kernel<<<((H / 2) * 32 + 255) / 256, 256, 0, st>>>(wup, xp, hp, D, ip, kp);
    mul_gate_kernel<<<(H * 32 + 255) / 256, 256, 0, st>>>(hp, gp, ip, kp, H, 1.0f);
    down2_warp_kernel<<<((O / 2) * 32 + 255) / 256, 256, 0, st>>>(wdp, hp, op, H, ip, kp, O);
    int K = 0;
    cudaMemcpyAsync(&K, kp, sizeof(int), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    return K;
}


// Same pipeline without the per-layer device->host copy of K: used by the
// low-overhead dispatch path, where the caller never needs K on the host.
void swiglu_coupled_nosync(torch::Tensor x, torch::Tensor Wg, torch::Tensor Wu,
                           torch::Tensor Wd, torch::Tensor g_buf,
                           torch::Tensor idx_buf, torch::Tensor k_buf,
                           torch::Tensor h_buf, torch::Tensor out, double rate,
                           c10::optional<torch::Tensor> wb2_opt) {
    const float* wb2 = nullptr;
    if (wb2_opt.has_value() && wb2_opt->defined())
        wb2 = wb2_opt->data_ptr<float>();
    const int D = Wg.size(1), H = Wg.size(0), O = Wd.size(0);
    const int nb = H / LANE;
    cudaStream_t st = c10::cuda::getCurrentCUDAStream();
    gemv2_warp_kernel<<<((H / 2) * 32 + 255) / 256, 256, 0, st>>>(
        Wg.data_ptr<float>(), x.data_ptr<float>(), g_buf.data_ptr<float>(), H, D, 1);
    compact_kernel<<<1, 256, nb * sizeof(float) + nb, st>>>(
        g_buf.data_ptr<float>(), idx_buf.data_ptr<int>(), k_buf.data_ptr<int>(), nb,
        (float)rate, wb2);
    up2_warp_kernel<<<((H / 2) * 32 + 255) / 256, 256, 0, st>>>(
        Wu.data_ptr<float>(), x.data_ptr<float>(), h_buf.data_ptr<float>(), D,
        idx_buf.data_ptr<int>(), k_buf.data_ptr<int>());
    mul_gate_kernel<<<(H * 32 + 255) / 256, 256, 0, st>>>(
        h_buf.data_ptr<float>(), g_buf.data_ptr<float>(), idx_buf.data_ptr<int>(),
        k_buf.data_ptr<int>(), H, 1.0f);
    down2_warp_kernel<<<((O / 2) * 32 + 255) / 256, 256, 0, st>>>(
        Wd.data_ptr<float>(), h_buf.data_ptr<float>(), out.data_ptr<float>(), H,
        idx_buf.data_ptr<int>(), k_buf.data_ptr<int>(), O);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_coupled", &swiglu_coupled, "fused zero-copy SwiGLU coupled skip");
    m.def("swiglu_coupled_nosync", &swiglu_coupled_nosync, "same, without the K sync");
}
