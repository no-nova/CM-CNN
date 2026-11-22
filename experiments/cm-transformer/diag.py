import ctypes, time, numpy as np, torch
from p1_kernels_v2 import build_avx
lib = build_avx()
P = ctypes.POINTER(ctypes.c_float); U = ctypes.POINTER(ctypes.c_ubyte)
I, O = 3584, 1024
torch.manual_seed(0)
W = torch.randn(O, I); A = torch.randn(1, I)
An, Wn = A.numpy(), W.numpy(); o = np.zeros((1, O), np.float32)
M = np.ones((1, I//32), np.uint8)
args = (An.ctypes.data_as(P), Wn.ctypes.data_as(P), M.ctypes.data_as(U), o.ctypes.data_as(P), 1, I, O)
lib.down_skip_avx512_auto2(*args)
ref = (A @ W.t()).numpy()[0]
print("K=all  max|err|", np.abs(o[0]-ref).max(), "ref|max|", np.abs(ref).max())
for rate in (0.2, 0.6):
    nk = int(round((1-rate)*112)); e = A[0].reshape(112,32).pow(2).sum(-1)
    idx = e.argsort(descending=True)[:nk].numpy()
    M[:] = 0; M[0, idx] = 1
    o[:] = 0
    lib.down_skip_avx512_auto2(*args)
    mask = np.zeros(I, np.float32); 
    for b in idx: mask[b*32:(b+1)*32] = 1
    ref2 = ((A*mask) @ W.t()).numpy()[0]
    print(f"rate {rate} K={nk} max|err| {np.abs(o[0]-ref2).max():.3e}")
    # timings
    for th in (1, 8, 32):
        import os
        lib_omp = lib
        ts=[]
        for _ in range(20):
            t0=time.perf_counter(); lib_omp.down_skip_avx512_auto2(*args); ts.append((time.perf_counter()-t0)*1e3)
        print(f"   rate {rate} threads_env={os.environ.get('OMP_NUM_THREADS')} ms {min(ts):.4f}")
        break
