// 32-lane structured block-skipping down_proj kernel (CM-Transform P1).
// out[t, r] = sum_{b : mask[t,b]} sum_{l<32} A[t, 32b+l] * W[r, 32b+l]
// One warp per output row; each lane owns one lane of the 32-lane block, so
// every 32-float block of W is fetched with a single coalesced 128 B request.
// Skipped blocks are never fetched -> traffic scales with (1 - skip rate).
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#define LANE 32
#define WARPS 8
#define THREADS (WARPS * LANE)

__global__ void down_skip_kernel(const float* __restrict__ A,
                                 const float* __restrict__ W,
                                 const unsigned char* __restrict__ M,
                                 float* __restrict__ Out,
                                 int T, int I, int O) {
    extern __shared__ float as[];
    const int t = blockIdx.y;
    const int nb = I / LANE;
    for (int i = threadIdx.x; i < I; i += blockDim.x) as[i] = A[(long)t * I + i];
    __syncthreads();

    const int warp = threadIdx.x / LANE;
    const int lane = threadIdx.x % LANE;
    const int r = blockIdx.x * WARPS + warp;
    if (r >= O) return;
    const unsigned char* m = M + (long)t * nb;
    const float* w = W + (long)r * I + lane;

    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        if (m[b]) acc = fmaf(as[b * LANE + lane], __ldg(w + b * LANE), acc);
    }
    // warp reduction over the 32 lanes
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) Out[(long)t * O + r] = acc;
}

// Same math, but the active 32-lane block list is pre-compacted per token so
// dead blocks cost neither a loop iteration nor a warps-wide branch.
__global__ void down_skip_list_kernel(const float* __restrict__ A,
                                      const float* __restrict__ W,
                                      const int* __restrict__ IDX,
                                      float* __restrict__ Out,
                                      int T, int I, int O, int K) {
    extern __shared__ float as[];
    const int t = blockIdx.y;
    for (int i = threadIdx.x; i < I; i += blockDim.x) as[i] = A[(long)t * I + i];
    __syncthreads();

    const int warp = threadIdx.x / LANE;
    const int lane = threadIdx.x % LANE;
    const int r = blockIdx.x * WARPS + warp;
    if (r >= O) return;
    const int* idx = IDX + (long)t * K;
    const float* w = W + (long)r * I + lane;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const int b = __ldg(idx + k);
        if (b < 0) break;
        acc = fmaf(as[b * LANE + lane], __ldg(w + b * LANE), acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) Out[(long)t * O + r] = acc;
}

torch::Tensor down_skip_list(torch::Tensor A, torch::Tensor W, torch::Tensor Idx) {
    TORCH_CHECK(A.is_cuda() && W.is_cuda() && Idx.is_cuda(), "tensors must be CUDA");
    A = A.contiguous(); W = W.contiguous(); Idx = Idx.contiguous().to(torch::kInt32);
    const int T = A.size(0), I = A.size(1), K = Idx.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + WARPS - 1) / WARPS, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    down_skip_list_kernel<<<grid, THREADS, I * sizeof(float), stream>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), Idx.data_ptr<int>(),
        Out.data_ptr<float>(), T, I, (int)O, K);
    return Out;
}

// Latency-tolerant variant: 4 independent load/FMA slots per iteration so the
// 32-lane tiles are fetched with memory-level parallelism instead of serialising
// on one accumulator.
__global__ void down_skip4_kernel(const float* __restrict__ A,
                                  const float* __restrict__ W,
                                  const unsigned char* __restrict__ M,
                                  float* __restrict__ Out,
                                  int T, int I, int O) {
    const int t = blockIdx.y;
    const int nb = I / LANE;
    const int warp = threadIdx.x / LANE;
    const int lane = threadIdx.x % LANE;
    const int r = blockIdx.x * WARPS + warp;
    if (r >= O) return;
    const unsigned char* m = M + (long)t * nb;
    const float* w = W + (long)r * I + lane;
    const float* a_row = A + (long)t * I + lane;   // re-read from L1: 196 B/token

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    int b = 0;
    for (; b + 4 <= nb; b += 4) {
        const float a0 = __ldg(a_row + (b + 0) * LANE), a1 = __ldg(a_row + (b + 1) * LANE);
        const float a2 = __ldg(a_row + (b + 2) * LANE), a3 = __ldg(a_row + (b + 3) * LANE);
        const float w0 = m[b + 0] ? __ldg(w + (b + 0) * LANE) : 0.f;
        const float w1 = m[b + 1] ? __ldg(w + (b + 1) * LANE) : 0.f;
        const float w2 = m[b + 2] ? __ldg(w + (b + 2) * LANE) : 0.f;
        const float w3 = m[b + 3] ? __ldg(w + (b + 3) * LANE) : 0.f;
        acc0 = fmaf(a0, w0, acc0); acc1 = fmaf(a1, w1, acc1);
        acc2 = fmaf(a2, w2, acc2); acc3 = fmaf(a3, w3, acc3);
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (; b < nb; ++b) if (m[b]) acc = fmaf(__ldg(a_row + b * LANE), __ldg(w + b * LANE), acc);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) Out[(long)t * O + r] = acc;
}

// 8-way unrolled, shared-memory A staging + 8 independent accumulators.
// Threads = ROWS_PER_BLOCK output rows; each block stages the activation row
// once (I floats) and streams only the blocks its rows actually need.
#define T5 128
__global__ void down_skip_u8_kernel(const float* __restrict__ A,
                                    const float* __restrict__ W,
                                    const unsigned char* __restrict__ M,
                                    float* __restrict__ Out,
                                    int T, int I, int O) {
    extern __shared__ float as5[];
    const int t = blockIdx.y;
    const int nb = I / LANE;
    for (int i = threadIdx.x; i < I; i += blockDim.x) as5[i] = A[(long)t * I + i];
    __syncthreads();
    const int r = blockIdx.x * T5 + threadIdx.x;
    if (r >= O) return;
    const unsigned char* m = M + (long)t * nb;
    const float* w = W + (long)r * I;
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;
    int b = 0;
    for (; b + 8 <= nb; b += 8) {
        float wv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) wv[j] = m[b + j] ? __ldg(w + (b + j) * LANE + threadIdx.x % LANE) : 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = fmaf(as5[(b + j) * LANE + threadIdx.x % LANE], wv[j], acc[j]);
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) s += acc[j];
    for (; b < nb; ++b) if (m[b]) s = fmaf(as5[b * LANE + threadIdx.x % LANE], __ldg(w + b * LANE + threadIdx.x % LANE), s);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    if ((threadIdx.x % LANE) == 0) Out[(long)t * O + r] = s;
}

// v6: one output row per thread, float4 over the 32 lanes (2 ZMM worth), A row
// staged once in shared memory. Skipped blocks cost a single byte read.
#define T6 128
__global__ void down_skip_v6_kernel(const float* __restrict__ A,
                                    const float* __restrict__ W,
                                    const unsigned char* __restrict__ M,
                                    float* __restrict__ Out,
                                    int T, int I, int O) {
    extern __shared__ float as6[];
    const int t = blockIdx.y;
    const int nb = I / LANE;
    for (int i = threadIdx.x; i < I; i += blockDim.x) as6[i] = A[(long)t * I + i];
    __syncthreads();
    const int r = blockIdx.x * T6 + threadIdx.x;
    if (r >= O) return;
    const unsigned char* m = M + (long)t * nb;
    const float4* w4 = (const float4*)(W + (long)r * I);
    const float4* a4 = (const float4*)as6;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int b = 0; b < nb; ++b) {
        if (!m[b]) continue;
        const int o = b * (LANE / 4);
#pragma unroll
        for (int g = 0; g < LANE / 4; ++g) {
            const float4 av = a4[o + g];
            const float4 wv = __ldg(&w4[o + g]);
            acc.x = fmaf(av.x, wv.x, acc.x);
            acc.y = fmaf(av.y, wv.y, acc.y);
            acc.z = fmaf(av.z, wv.z, acc.z);
            acc.w = fmaf(av.w, wv.w, acc.w);
        }
    }
    Out[(long)t * O + r] = (acc.x + acc.y) + (acc.z + acc.w);
}

torch::Tensor down_skip_v6(torch::Tensor A, torch::Tensor W, torch::Tensor M) {
    A = A.contiguous(); W = W.contiguous(); M = M.contiguous().to(torch::kUInt8);
    const int T = A.size(0), I = A.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + T6 - 1) / T6, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    down_skip_v6_kernel<<<grid, T6, I * sizeof(float), stream>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
        Out.data_ptr<float>(), T, I, (int)O);
    return Out;
}

double bench_down_skip_v6(torch::Tensor A, torch::Tensor W, torch::Tensor M, int iters) {
    A = A.contiguous(); W = W.contiguous(); M = M.contiguous().to(torch::kUInt8);
    const int T = A.size(0), I = A.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + T6 - 1) / T6, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int i = 0; i < 20; ++i)
        down_skip_v6_kernel<<<grid, T6, I * sizeof(float), stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
            Out.data_ptr<float>(), T, I, (int)O);
    cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
    cudaEventRecord(s0, stream);
    for (int i = 0; i < iters; ++i)
        down_skip_v6_kernel<<<grid, T6, I * sizeof(float), stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
            Out.data_ptr<float>(), T, I, (int)O);
    cudaEventRecord(s1, stream); cudaEventSynchronize(s1);
    float ms = 0.f; cudaEventElapsedTime(&ms, s0, s1);
    cudaEventDestroy(s0); cudaEventDestroy(s1);
    return ms / iters;
}

torch::Tensor down_skip_u8(torch::Tensor A, torch::Tensor W, torch::Tensor M) {
    A = A.contiguous(); W = W.contiguous(); M = M.contiguous().to(torch::kUInt8);
    const int T = A.size(0), I = A.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + T5 - 1) / T5, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    down_skip_u8_kernel<<<grid, T5, I * sizeof(float), stream>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
        Out.data_ptr<float>(), T, I, (int)O);
    return Out;
}

double bench_down_skip_u8(torch::Tensor A, torch::Tensor W, torch::Tensor M, int iters) {
    A = A.contiguous(); W = W.contiguous(); M = M.contiguous().to(torch::kUInt8);
    const int T = A.size(0), I = A.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + T5 - 1) / T5, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int i = 0; i < 20; ++i)
        down_skip_u8_kernel<<<grid, T5, I * sizeof(float), stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
            Out.data_ptr<float>(), T, I, (int)O);
    cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
    cudaEventRecord(s0, stream);
    for (int i = 0; i < iters; ++i)
        down_skip_u8_kernel<<<grid, T5, I * sizeof(float), stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
            Out.data_ptr<float>(), T, I, (int)O);
    cudaEventRecord(s1, stream); cudaEventSynchronize(s1);
    float ms = 0.f; cudaEventElapsedTime(&ms, s0, s1);
    cudaEventDestroy(s0); cudaEventDestroy(s1);
    return ms / iters;
}

torch::Tensor down_skip4(torch::Tensor A, torch::Tensor W, torch::Tensor M) {
    A = A.contiguous(); W = W.contiguous(); M = M.contiguous().to(torch::kUInt8);
    const int T = A.size(0), I = A.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + WARPS - 1) / WARPS, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    down_skip4_kernel<<<grid, THREADS, 0, stream>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
        Out.data_ptr<float>(), T, I, (int)O);
    return Out;
}

double bench_down_skip4(torch::Tensor A, torch::Tensor W, torch::Tensor M, int iters) {
    A = A.contiguous(); W = W.contiguous(); M = M.contiguous().to(torch::kUInt8);
    const int T = A.size(0), I = A.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + WARPS - 1) / WARPS, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int i = 0; i < 20; ++i)
        down_skip4_kernel<<<grid, THREADS, 0, stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
            Out.data_ptr<float>(), T, I, (int)O);
    cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
    cudaEventRecord(s0, stream);
    for (int i = 0; i < iters; ++i)
        down_skip4_kernel<<<grid, THREADS, 0, stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
            Out.data_ptr<float>(), T, I, (int)O);
    cudaEventRecord(s1, stream); cudaEventSynchronize(s1);
    float ms = 0.f; cudaEventElapsedTime(&ms, s0, s1);
    cudaEventDestroy(s0); cudaEventDestroy(s1);
    return ms / iters;
}

torch::Tensor down_skip(torch::Tensor A, torch::Tensor W, torch::Tensor M) {
    TORCH_CHECK(A.is_cuda() && W.is_cuda() && M.is_cuda(), "tensors must be CUDA");
    A = A.contiguous(); W = W.contiguous(); M = M.contiguous().to(torch::kUInt8);
    const int T = A.size(0), I = A.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + WARPS - 1) / WARPS, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    down_skip_kernel<<<grid, THREADS, I * sizeof(float), stream>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
        Out.data_ptr<float>(), T, I, (int)O);
    return Out;
}

// In-process timing (events recorded inside C++), so the number reported is
// GPU kernel time and not Python/torch dispatch overhead.
double bench_down_skip(torch::Tensor A, torch::Tensor W, torch::Tensor M, int iters) {
    A = A.contiguous(); W = W.contiguous(); M = M.contiguous().to(torch::kUInt8);
    const int T = A.size(0), I = A.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + WARPS - 1) / WARPS, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int i = 0; i < 20; ++i)
        down_skip_kernel<<<grid, THREADS, I * sizeof(float), stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
            Out.data_ptr<float>(), T, I, (int)O);
    cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
    cudaEventRecord(s0, stream);
    for (int i = 0; i < iters; ++i)
        down_skip_kernel<<<grid, THREADS, I * sizeof(float), stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), M.data_ptr<unsigned char>(),
            Out.data_ptr<float>(), T, I, (int)O);
    cudaEventRecord(s1, stream);
    cudaEventSynchronize(s1);
    float ms = 0.f; cudaEventElapsedTime(&ms, s0, s1);
    cudaEventDestroy(s0); cudaEventDestroy(s1);
    return ms / iters;
}

double bench_down_skip_list(torch::Tensor A, torch::Tensor W, torch::Tensor Idx, int iters) {
    A = A.contiguous(); W = W.contiguous(); Idx = Idx.contiguous().to(torch::kInt32);
    const int T = A.size(0), I = A.size(1), K = Idx.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + WARPS - 1) / WARPS, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int i = 0; i < 20; ++i)
        down_skip_list_kernel<<<grid, THREADS, I * sizeof(float), stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), Idx.data_ptr<int>(),
            Out.data_ptr<float>(), T, I, (int)O, K);
    cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
    cudaEventRecord(s0, stream);
    for (int i = 0; i < iters; ++i)
        down_skip_list_kernel<<<grid, THREADS, I * sizeof(float), stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), Idx.data_ptr<int>(),
            Out.data_ptr<float>(), T, I, (int)O, K);
    cudaEventRecord(s1, stream);
    cudaEventSynchronize(s1);
    float ms = 0.f; cudaEventElapsedTime(&ms, s0, s1);
    cudaEventDestroy(s0); cudaEventDestroy(s1);
    return ms / iters;
}

// v7: warp-per-row (the 760 GFLOP/s dense-optimal layout of v4) but driven by a
// compacted active-block index list instead of a byte mask.  Every warp loops
// only over K active 32-lane blocks, 4 deep for memory-level parallelism, so
// the kernel time scales with K/nb = (1 - skip rate) instead of paying a branch
// per block.
__global__ void down_skip4_list_kernel(const float* __restrict__ A,
                                       const float* __restrict__ W,
                                       const int* __restrict__ IDX,
                                       float* __restrict__ Out,
                                       int T, int I, int O, int K) {
    const int t = blockIdx.y;
    const int warp = threadIdx.x / LANE;
    const int lane = threadIdx.x % LANE;
    const int r = blockIdx.x * WARPS + warp;
    if (r >= O) return;
    const int* idx = IDX + (long)t * K;
    const float* w = W + (long)r * I + lane;
    const float* a_row = A + (long)t * I + lane;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        const int b0 = __ldg(idx + k + 0), b1 = __ldg(idx + k + 1);
        const int b2 = __ldg(idx + k + 2), b3 = __ldg(idx + k + 3);
        const float a0 = __ldg(a_row + b0 * LANE), a1 = __ldg(a_row + b1 * LANE);
        const float a2 = __ldg(a_row + b2 * LANE), a3 = __ldg(a_row + b3 * LANE);
        acc0 = fmaf(a0, __ldg(w + b0 * LANE), acc0);
        acc1 = fmaf(a1, __ldg(w + b1 * LANE), acc1);
        acc2 = fmaf(a2, __ldg(w + b2 * LANE), acc2);
        acc3 = fmaf(a3, __ldg(w + b3 * LANE), acc3);
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (; k < K; ++k) {
        const int b = __ldg(idx + k);
        acc = fmaf(__ldg(a_row + b * LANE), __ldg(w + b * LANE), acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) Out[(long)t * O + r] = acc;
}

torch::Tensor down_skip4_list(torch::Tensor A, torch::Tensor W, torch::Tensor Idx) {
    A = A.contiguous(); W = W.contiguous(); Idx = Idx.contiguous().to(torch::kInt32);
    const int T = A.size(0), I = A.size(1), K = Idx.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + WARPS - 1) / WARPS, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    down_skip4_list_kernel<<<grid, THREADS, 0, stream>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), Idx.data_ptr<int>(),
        Out.data_ptr<float>(), T, I, (int)O, K);
    return Out;
}

double bench_down_skip4_list(torch::Tensor A, torch::Tensor W, torch::Tensor Idx, int iters) {
    A = A.contiguous(); W = W.contiguous(); Idx = Idx.contiguous().to(torch::kInt32);
    const int T = A.size(0), I = A.size(1), K = Idx.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    dim3 grid((O + WARPS - 1) / WARPS, T);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (int i = 0; i < 20; ++i)
        down_skip4_list_kernel<<<grid, THREADS, 0, stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), Idx.data_ptr<int>(),
            Out.data_ptr<float>(), T, I, (int)O, K);
    cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
    cudaEventRecord(s0, stream);
    for (int i = 0; i < iters; ++i)
        down_skip4_list_kernel<<<grid, THREADS, 0, stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), Idx.data_ptr<int>(),
            Out.data_ptr<float>(), T, I, (int)O, K);
    cudaEventRecord(s1, stream); cudaEventSynchronize(s1);
    float ms = 0.f; cudaEventElapsedTime(&ms, s0, s1);
    cudaEventDestroy(s0); cudaEventDestroy(s1);
    return ms / iters;
}


// ---------------------------------------------------------------------------
// v8: device-side index compaction (no host sync, no Python-side index build)
// ---------------------------------------------------------------------------
__global__ void compact_kernel(const unsigned char* __restrict__ M,
                               int* __restrict__ IDX, int* __restrict__ NACT,
                               int B) {
    const int t = blockIdx.x;
    const int lane = threadIdx.x;
    const unsigned char* m = M + (long)t * B;
    int* out = IDX + (long)t * B;
    int base = 0;
    for (int s = 0; s < B; s += LANE) {
        const int b = s + lane;
        const bool k = (b < B) && (m[b] != 0);
        const unsigned ball = __ballot_sync(0xffffffffu, k);
        const int pre = __popc(ball & ((1u << lane) - 1u));
        if (k) out[base + pre] = b;
        base += __popc(ball);
    }
    if (lane == 0) NACT[t] = base;
}

__global__ void down_skip4_var_kernel(const float* __restrict__ A,
                                      const float* __restrict__ W,
                                      const int* __restrict__ IDX,
                                      const int* __restrict__ NACT,
                                      float* __restrict__ Out,
                                      int T, int I, int O, int Kmax) {
    const int t = blockIdx.y;
    const int warp = threadIdx.x / LANE;
    const int lane = threadIdx.x % LANE;
    const int r = blockIdx.x * WARPS + warp;
    if (r >= O) return;
    const int K = __ldg(NACT + t);
    const int* idx = IDX + (long)t * Kmax;
    const float* w = W + (long)r * I + lane;
    const float* a_row = A + (long)t * I + lane;

    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        const int b0 = __ldg(idx + k + 0), b1 = __ldg(idx + k + 1);
        const int b2 = __ldg(idx + k + 2), b3 = __ldg(idx + k + 3);
        acc0 = fmaf(__ldg(a_row + b0 * LANE), __ldg(w + b0 * LANE), acc0);
        acc1 = fmaf(__ldg(a_row + b1 * LANE), __ldg(w + b1 * LANE), acc1);
        acc2 = fmaf(__ldg(a_row + b2 * LANE), __ldg(w + b2 * LANE), acc2);
        acc3 = fmaf(__ldg(a_row + b3 * LANE), __ldg(w + b3 * LANE), acc3);
    }
    float acc = (acc0 + acc1) + (acc2 + acc3);
    for (; k < K; ++k) {
        const int b = __ldg(idx + k);
        acc = fmaf(__ldg(a_row + b * LANE), __ldg(w + b * LANE), acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) Out[(long)t * O + r] = acc;
}

torch::Tensor down_skip4_auto(torch::Tensor A, torch::Tensor W, torch::Tensor M) {
    A = A.contiguous(); W = W.contiguous();
    M = M.contiguous().to(torch::kUInt8);
    const int T = A.size(0), I = A.size(1), B = M.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    auto Idx = torch::empty({T, B}, A.options().dtype(torch::kInt32));
    auto Nact = torch::empty({T}, A.options().dtype(torch::kInt32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    compact_kernel<<<T, LANE, 0, stream>>>(M.data_ptr<unsigned char>(),
                                           Idx.data_ptr<int>(), Nact.data_ptr<int>(), B);
    dim3 grid((O + WARPS - 1) / WARPS, T);
    down_skip4_var_kernel<<<grid, THREADS, 0, stream>>>(
        A.data_ptr<float>(), W.data_ptr<float>(), Idx.data_ptr<int>(),
        Nact.data_ptr<int>(), Out.data_ptr<float>(), T, I, (int)O, B);
    return Out;
}

double bench_down_skip4_auto(torch::Tensor A, torch::Tensor W, torch::Tensor M, int iters) {
    A = A.contiguous(); W = W.contiguous();
    M = M.contiguous().to(torch::kUInt8);
    const int T = A.size(0), I = A.size(1), B = M.size(1);
    const long O = W.size(0);
    auto Out = torch::empty({T, O}, A.options());
    auto Idx = torch::empty({T, B}, A.options().dtype(torch::kInt32));
    auto Nact = torch::empty({T}, A.options().dtype(torch::kInt32));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((O + WARPS - 1) / WARPS, T);
    auto body = [&]() {
        compact_kernel<<<T, LANE, 0, stream>>>(M.data_ptr<unsigned char>(),
                                               Idx.data_ptr<int>(), Nact.data_ptr<int>(), B);
        down_skip4_var_kernel<<<grid, THREADS, 0, stream>>>(
            A.data_ptr<float>(), W.data_ptr<float>(), Idx.data_ptr<int>(),
            Nact.data_ptr<int>(), Out.data_ptr<float>(), T, I, (int)O, B);
    };
    for (int i = 0; i < 20; ++i) body();
    cudaEvent_t s0, s1; cudaEventCreate(&s0); cudaEventCreate(&s1);
    cudaEventRecord(s0, stream);
    for (int i = 0; i < iters; ++i) body();
    cudaEventRecord(s1, stream); cudaEventSynchronize(s1);
    float ms = 0.f; cudaEventElapsedTime(&ms, s0, s1);
    cudaEventDestroy(s0); cudaEventDestroy(s1);
    return ms / iters;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("down_skip", &down_skip, "32-lane block-skipping down_proj (CUDA)");
    m.def("down_skip_list", &down_skip_list, "32-lane block-skipping down_proj, compacted block list (CUDA)");
    m.def("down_skip4", &down_skip4, "latency-tolerant 32-lane block-skip down_proj (CUDA)");
    m.def("down_skip_u8", &down_skip_u8, "u8-unrolled block-skip down_proj (CUDA)");
    m.def("down_skip_v6", &down_skip_v6, "float4 row-per-thread block-skip down_proj (CUDA)");
    m.def("bench_down_skip_v6", &bench_down_skip_v6, "GPU-time benchmark of down_skip_v6");
    m.def("bench_down_skip_u8", &bench_down_skip_u8, "GPU-time benchmark of down_skip_u8");
    m.def("bench_down_skip4", &bench_down_skip4, "GPU-time benchmark of down_skip4");
    m.def("bench_down_skip", &bench_down_skip, "GPU-time benchmark of down_skip");
    m.def("down_skip4_list", &down_skip4_list, "warp-per-row down_proj on a compacted active-block list");
    m.def("bench_down_skip4_list", &bench_down_skip4_list, "GPU-time benchmark of down_skip4_list");
    m.def("bench_down_skip_list", &bench_down_skip_list, "GPU-time benchmark of down_skip_list");
    m.def("down_skip4_auto", &down_skip4_auto, "block-skip down_proj with device-side compaction");
    m.def("bench_down_skip4_auto", &bench_down_skip4_auto, "GPU-time benchmark of down_skip4_auto");
}
