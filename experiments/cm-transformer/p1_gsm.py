"""GSM8K under P1 for the selected (variant, rate) cells, greedy 192 tokens.

Separate from p1_v2.eval_one so a single expensive generation benchmark can be
added to a finished sweep without recomputing PPL/MC for every cell.
"""
import argparse, json, pathlib
import torch
from p1_v2 import (RefP1, HFP1, build_params, make_cfg, calibrate, gsm8k,
                   CALIB_PROMPTS, log)

HERE = pathlib.Path(__file__).parent
MODELS = {"qwen3.5-0.8b": "Qwen3.5-0.8B", "qwen3-0.6b": "Qwen3-0.6B"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--cells", default="ours_adaptive:0.1,ours_uniform:0.1,"
                                        "natural_uniform:0.1,random_uniform:0.1")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results/p1v2_gsm.json")
    args = ap.parse_args()
    tag = args.tag or args.model
    torch.set_grad_enabled(False)
    dtype = getattr(torch, args.dtype)
    if args.model == "qwen3.5-0.8b":
        p1 = RefP1(device=args.device, dtype=dtype)
    else:
        p1 = HFP1(str(HERE.parents[1] / "models/Qwen3-0.6B"),
                  device=args.device, dtype=dtype)
    params = build_params(p1, cache=HERE / f"results/p1v2_{args.model}_params.pt")
    out_path = HERE / args.out
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    rec = out.setdefault(tag, {})
    rec.setdefault("dense", {})["gsm8k"] = gsm8k(p1, limit=args.limit,
                                                 max_new=args.max_new)
    log(f"dense gsm8k {rec['dense']['gsm8k']['acc']:.3f}")
    for cell in args.cells.split(","):
        name, rate = cell.split(":")
        variant, gate = name.split("_")
        rate = float(rate)
        cfg = make_cfg(p1, params, variant, rate=0.0, theta=None)
        th = calibrate(p1, params, rate, CALIB_PROMPTS[:4],
                       override=cfg["override"], variant=variant)
        g = gsm8k(p1, limit=args.limit, max_new=args.max_new,
                  override=cfg["override"],
                  rate=0.0 if gate == "adaptive" else rate,
                  scale=cfg["scale"], theta=th if gate == "adaptive" else None)
        rec.setdefault(name, {})[f"{rate:.2f}"] = g
        log(f"{name} tau={rate:.2f} gsm8k {g['acc']:.3f} ({g['exact']})")
        out_path.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
