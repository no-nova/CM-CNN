"""Defect-3: SwiGLU-coupled skipping (gate dense, up rows + down cols skipped)."""
import argparse, json, pathlib, time
import torch
from model_ref import permute_mlp, new_state
import p1_v2
from p1_v2 import RefP1, build_params, calibrate, CALIB_PROMPTS, log

HERE = pathlib.Path(__file__).parent


def timed(tb, ids, kw, n=6):
    tb.forward(ids, **kw)                       # warm
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        tb.forward(ids, **kw)
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--rates", default="0.2,0.4,0.6")
    ap.add_argument("--out", default="results/p1_coupled.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dtype = getattr(torch, args.dtype)
    p1 = RefP1(device="cuda", dtype=dtype)
    tb = p1.tb
    params = build_params(p1, cache=HERE / "results/p1v2_qwen3.5-0.8b_params.pt")
    ids = torch.tensor(p1.tok("The history of computing begins with the",
                              add_special_tokens=False)["input_ids"], device="cuda")
    ov = {}
    for i in range(p1.L):
        W = (tb.w[f"{tb.prefix}layers.{i}.mlp.gate_proj.weight"],
             tb.w[f"{tb.prefix}layers.{i}.mlp.up_proj.weight"], p1.down_w(i))
        ov[i] = permute_mlp(W, params["perm"][i].to("cuda"))
    sc = {i: p1.down_w(i).float().norm(dim=0) for i in range(p1.L)}
    ref = tb.forward(ids, mlp_override=None)["logits"][-1].float()
    out = {"dtype": args.dtype, "tokens": len(ids), "rows": [],
           "note": ("prefill latency of one forward over the prompt; 'dense' is "
                    "the unpermuted baseline, 'block' skips only down_proj "
                    "columns, 'coupled' keeps gate dense and skips active rows "
                    "of up_proj plus the matching columns of down_proj")}
    t_dense = timed(tb, ids, dict())
    # permuted, no skip (isomorphism check)
    zp = tb.forward(ids, mlp_override=ov, skip_scale=sc, skip_theta=None)["logits"][-1].float()
    out["isomorphism_max_abs"] = float((zp - ref).abs().max())
    out["dense_ms"] = t_dense
    for rate in [float(r) for r in args.rates.split(",")]:
        th = calibrate(p1, params, rate, CALIB_PROMPTS[:4], override=ov, variant="ours")
        for mode in ("block", "coupled"):
            # The coupled path gates on ||g||, a different random variable than
            # ||act(gate)*up||, so a threshold calibrated for the latter is not
            # transferable; use the top-k rule at the same nominal rate instead.
            kw = (dict(mlp_override=ov, skip_rate=rate, skip_scale=sc, coupled=True)
                  if mode == "coupled" else
                  dict(mlp_override=ov, skip_theta=th, skip_scale=sc))
            o = tb.forward(ids, **kw)["logits"][-1].float()
            t = timed(tb, ids, kw)
            out["rows"].append({"rate": rate, "mode": mode, "ms": t,
                                "speedup": t_dense / t,
                                "max_abs": float((o - ref).abs().max()),
                                "rel_l2": float((o - ref).norm() / ref.norm())})
            log(f"{mode} tau={rate:.2f} {t:.2f} ms ({t_dense / t:.2f}x) "
                f"max|d| {(o - ref).abs().max():.3f}")
    (HERE / args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
