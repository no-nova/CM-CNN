"""CPU: token-parallel (auto) vs output-row-parallel (auto2) at T=1 and T=64."""
import json, time, pathlib
import numpy as np, torch
from p1_kernels_v2 import build_avx

HERE = pathlib.Path(__file__).parent
OUT = HERE / "results/p1_cpu_auto2.json"
I, O = 3584, 1024


def cpu_bench(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


def main():
    torch.set_num_threads(16)
    lib = build_avx()
    torch.manual_seed(0)
    W = torch.randn(O, I, dtype=torch.float32)
    out = {"I": I, "O": O, "threads": 32, "rows": []}
    for T in (1, 64):
        A = torch.randn(T, I, dtype=torch.float32)
        e = A.reshape(T, I // 32, 32).pow(2).sum(-1)
        An, Wn = A.numpy(), W.numpy()
        o = np.zeros((T, O), dtype=np.float32)
        M = np.ones((T, I // 32), dtype=np.uint8)
        for rate in (0.0, 0.2, 0.4, 0.6, 0.75):
            if rate:
                nkeep = max(1, int(round((1 - rate) * (I // 32))))
                idx = e.argsort(-1)[:, :nkeep]
                M[:] = 0
                for t in range(T):
                    M[t, idx[t].numpy()] = 1
            r = {"T": T, "skip": rate}
            r["dense_ms"] = cpu_bench(lambda: torch.mm(A, W.t()))
            r["auto_ms"] = cpu_bench(lambda: lib.down_skip_avx512_auto(
                An.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_float)),
                Wn.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_float)),
                M.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_ubyte)),
                o.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_float)),
                T, I, O))
            r["auto2_ms"] = cpu_bench(lambda: lib.down_skip_avx512_auto2(
                An.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_float)),
                Wn.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_float)),
                M.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_ubyte)),
                o.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_float)),
                T, I, O))
            out["rows"].append(r)
            print(f"T={T:3d} skip={rate:.2f} dense {r['dense_ms']:.3f} "
                  f"auto {r['auto_ms']:.3f} auto2 {r['auto2_ms']:.3f} "
                  f"({r['dense_ms'] / r['auto2_ms']:.2f}x vs dense)", flush=True)
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
