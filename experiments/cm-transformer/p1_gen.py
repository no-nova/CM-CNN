"""Generation stability under P1: 50 prompts x 64 greedy tokens.

Dense continuation is the reference; ROUGE-L, first-divergence step and
first-token agreement are measured for each P1 cell.
"""
import argparse, json, pathlib
import torch
from p1_v2 import (RefP1, HFP1, build_params, make_cfg, calibrate, CALIB_PROMPTS, log)
from bench_text import GEN_PROMPTS, compare_generation

HERE = pathlib.Path(__file__).parent


def gen(p1, prompts, n_new, **kw):
    outs, t0 = [], __import__("time").perf_counter()
    for p in prompts:
        ids = torch.tensor(p1.tok(p, add_special_tokens=False)["input_ids"],
                           device=p1.device)
        outs.append({"prompt": p, "ids": p1.generate(ids, n_new, stop_on_answer=False, **kw)})
    dt = __import__("time").perf_counter() - t0
    return {"prompts": outs, "seconds": dt,
            "tokens_per_s": len(prompts) * n_new / max(dt, 1e-6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-0.8b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--cells", default="ours_adaptive:0.1,natural_uniform:0.1")
    ap.add_argument("--out", default="results/p1v2_gen.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dtype = getattr(torch, args.dtype)
    if args.model == "qwen3.5-0.8b":
        p1 = RefP1(device=args.device, dtype=dtype)
    else:
        p1 = HFP1(str(HERE.parents[1] / "models/Qwen3-0.6B"),
                  device=args.device, dtype=dtype)
    params = build_params(p1, cache=HERE / f"results/p1v2_{args.model}_params.pt")
    prompts = GEN_PROMPTS[:args.n_prompts]
    out_path = HERE / args.out
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    rec = out.setdefault(args.model, {})
    base = gen(p1, prompts, args.n_new)
    rec["dense"] = {"tokens_per_s": base["tokens_per_s"]}
    log(f"dense {base['tokens_per_s']:.1f} tok/s")
    for cell in args.cells.split(","):
        name, rate = cell.split(":")
        variant, gate = name.split("_")
        rate = float(rate)
        cfg = make_cfg(p1, params, variant, rate=0.0, theta=None)
        th = calibrate(p1, params, rate, CALIB_PROMPTS[:4],
                       override=cfg["override"], variant=variant)
        kw = dict(override=cfg["override"], scale=cfg["scale"],
                  rate=0.0 if gate == "adaptive" else rate,
                  theta=th if gate == "adaptive" else None)
        other = gen(p1, prompts, args.n_new, **kw)
        cmp_ = compare_generation(base, other)
        cmp_["tokens_per_s"] = other["tokens_per_s"]
        rec[name] = cmp_
        log(f"{name} tau={rate:.2f} ROUGE-L {cmp_['rouge_l']:.3f} "
            f"first_div {cmp_['first_div']:.1f} same_first {cmp_['top1_agreement']:.2f} "
            f"{other['tokens_per_s']:.1f} tok/s")
        out_path.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
