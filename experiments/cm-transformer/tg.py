import torch, pathlib
from model_ref import permute_mlp
import p1_v2
from p1_v2 import RefP1, build_params, calibrate, CALIB_PROMPTS
torch.set_grad_enabled(False)
p1 = RefP1(device="cpu", dtype=torch.float32)
tb = p1.tb
params = build_params(p1, cache=pathlib.Path("results/p1v2_qwen3.5-0.8b_params.pt"))
ids = torch.tensor(p1.tok("The history of computing begins with the invention",
                          add_special_tokens=False)["input_ids"])
ov = {}
for i in range(p1.L):
    W = (tb.w[f"{tb.prefix}layers.{i}.mlp.gate_proj.weight"],
         tb.w[f"{tb.prefix}layers.{i}.mlp.up_proj.weight"], p1.down_w(i))
    ov[i] = permute_mlp(W, params["perm"][i])
sc = {i: p1.down_w(i).norm(dim=0) for i in range(p1.L)}
ref = tb.forward(ids)["logits"][-1]
for rate in (0.2, 0.4):
    th = calibrate(p1, params, rate, CALIB_PROMPTS[:2], override=ov, variant="ours")
    a = tb.forward(ids, mlp_override=ov, skip_theta=th, skip_scale=sc)["logits"][-1]
    b = tb.forward(ids, mlp_override=ov, skip_theta=th, skip_scale=sc, norm_gamma=1.5)["logits"][-1]
    print(f"rate {rate}: |a-ref| {(a-ref).abs().max():.6f}  |b-ref| {(b-ref).abs().max():.6f} "
          f"|b-a| {(b-a).abs().max():.6f}")
