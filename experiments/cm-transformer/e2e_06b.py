"""End-to-end decode for Qwen3-0.6B with the CUDA/AVX auto-compaction kernel."""
import argparse, json, pathlib, time
import torch
from p1_v2 import HFP1, build_params, calibrate, CALIB_PROMPTS, log
from p1_kernels_v2 import build_cuda, build_avx, KernelDown

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=str(HERE.parents[1] / "models/Qwen3-0.6B"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-new", type=int, default=20)
    ap.add_argument("--rates", default="0.2,0.4,0.6,0.75")
    ap.add_argument("--out", default="results/p1_kernels_06b.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    gpu = build_cuda() if args.device == "cuda" else None
    cpu = build_avx() if args.device != "cuda" else None
    p1 = HFP1(args.model_dir, device=args.device, dtype=torch.float32)
    ids = torch.tensor(p1.tok("The history of computing begins with",
                              add_special_tokens=False)["input_ids"],
                       device=p1.device)
    scale = {i: p1.down_w(i).detach().float().norm(dim=0) for i in range(p1.L)}
    params = build_params(p1, cache=HERE / "results/p1v2_qwen3-0.6b_params.pt")

    def timed(rate, kernel):
        p1._cm_kernel = kernel
        if kernel is not None:
            for l in p1.layers:
                l.mlp._cm_kernel = kernel
        th = (calibrate(p1, params, rate, CALIB_PROMPTS[:2], variant="natural")
              if rate else None)
        p1.set_cfg(None, rate=rate, scale=scale, theta=th)
        cur, lat = list(ids.tolist()), []
        for k in range(args.n_new):
            t0 = time.perf_counter()
            with torch.no_grad():
                o = p1.model(input_ids=torch.tensor([cur], device=p1.device),
                             use_cache=False).logits[0, -1].float()
            lat.append((time.perf_counter() - t0) * 1e3)
            cur.append(int(o.argmax()))
            cur = cur[-64:]
        p1._cm_kernel = None
        for l in p1.layers:
            l.mlp._cm_kernel = None
        return sum(lat[3:]) / max(len(lat) - 3, 1)

    timed(0.0, None)                       # warm-up
    dense = timed(0.0, None)
    kd0 = KernelDown(gpu_ext=gpu, cpu_lib=cpu)
    kern0 = timed(0.0, kd0)
    out = {"dense_ms_per_token": dense, "kernel_no_skip_ms": kern0, "rows": []}
    log(f"dense {dense:.1f} ms/tok")
    for r in [float(x) for x in args.rates.split(",")]:
        kd = KernelDown(gpu_ext=gpu, cpu_lib=cpu)
        t = timed(r, kd)
        out["rows"].append({"rate": r, "ms_per_token": t,
                            "speedup_vs_dense": dense / t,
                            "speedup_vs_kernel": kern0 / t,
                            "kept": kd.skip_rate()})
        log(f"tau={r:.2f} {t:.1f} ms/tok ({dense / t:.2f}x vs dense) "
            f"kept={kd.skip_rate():.3f}")
    (HERE / args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
