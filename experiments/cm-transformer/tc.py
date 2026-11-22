import torch, time
from model_ref import TextBackbone
from p1_v2 import build_params, CALIB_PROMPTS, log
import p1_v2
torch.set_grad_enabled(False)
p1 = p1_v2.RefP1(device="cuda", dtype=torch.float32)
tb = p1.tb
params = build_params(p1, cache=__import__("pathlib").Path("results/p1v2_qwen3.5-0.8b_params.pt"))
ids = torch.tensor(p1.tok("The history of computing", add_special_tokens=False)["input_ids"], device="cuda")
ov = {}
from model_ref import permute_mlp
for i in range(p1.L):
    W = (p1.tb.w[f"{p1.tb.prefix}layers.{i}.mlp.gate_proj.weight"],
         p1.tb.w[f"{p1.tb.prefix}layers.{i}.mlp.up_proj.weight"], p1.down_w(i))
    ov[i] = permute_mlp(W, params["perm"][i].to("cuda"))
sc = {i: p1.down_w(i).float().norm(dim=0) for i in range(p1.L)}
base = tb.forward(ids, mlp_override=None)["logits"][-1]
for rate in (0.0, 0.2, 0.4, 0.6):
    th = None if rate == 0 else p1_v2.calibrate(p1, params, rate, CALIB_PROMPTS[:4], override=ov, variant="ours")
    o = tb.forward(ids, mlp_override=ov, skip_theta=th, skip_scale=sc, coupled=rate > 0)["logits"][-1]
    d = (o - base).abs().max().item()
    # timing: 8 tokens, recompute (no cache) for simplicity
    t0 = time.perf_counter()
    for _ in range(5):
        tb.forward(ids, mlp_override=ov, skip_theta=th, skip_scale=sc, coupled=rate > 0)
    t1 = (time.perf_counter() - t0) / 5 * 1e3
    print(f"coupled rate={rate:.2f} max|dlogit|={d:.4f} fwd {t1:.2f} ms", flush=True)
