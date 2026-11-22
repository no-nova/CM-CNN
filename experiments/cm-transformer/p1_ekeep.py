"""Retained output-block energy for the HF backbone (Qwen3-0.6B path).

p1_v2.retained_energy used to return None for HFP1, which is why the 0.6B
E_keep column was empty; this script fills it for every (variant, rate) cell.
"""
import argparse, json, pathlib
import torch
from p1_v2 import (HFP1, RefP1, build_params, make_cfg, calibrate,
                   retained_energy, CALIB_PROMPTS, log)

HERE = pathlib.Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-0.6b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--rates", default="0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--out", default="results/p1v2_ekeep.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dtype = getattr(torch, args.dtype)
    if args.model == "qwen3.5-0.8b":
        p1 = RefP1(device=args.device, dtype=dtype)
    else:
        p1 = HFP1(str(HERE.parents[1] / "models/Qwen3-0.6B"),
                  device=args.device, dtype=dtype)
    params = build_params(p1, cache=HERE / f"results/p1v2_{args.model}_params.pt")
    path = HERE / args.out
    out = json.loads(path.read_text()) if path.exists() else {}
    rec = out.setdefault(args.model, {})
    for name in ("ours_adaptive", "ours_uniform", "natural_uniform",
                 "random_uniform"):
        variant, gate = name.split("_")
        cfg = make_cfg(p1, params, variant, rate=0.0, theta=None)
        for rate in [float(r) for r in args.rates.split(",")]:
            th = calibrate(p1, params, rate, CALIB_PROMPTS[:4],
                           override=cfg["override"], variant=variant)
            e = retained_energy(p1, params, variant, rate=rate,
                                theta=th if gate == "adaptive" else None,
                                prompts=CALIB_PROMPTS[:4],
                                override=cfg["override"])
            rec.setdefault(name, {})[f"{rate:.2f}"] = round(float(e), 5)
            log(f"{name} tau={rate:.2f} Ekeep={e:.4f}")
            path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
