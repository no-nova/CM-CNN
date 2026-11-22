"""Layer-0 stage-by-stage diff: our reference vs official transformers."""
import torch, torch.nn.functional as F
from common import MODEL_DIR, Weights, SAFETENSORS, rms_norm, layer_norm
from model_ref import TextBackbone
from video_data import CALIB_PROMPTS, encode_prompts

dev = "cuda"
torch.set_grad_enabled(False)

from transformers import Qwen3_5ForCausalLM
off = Qwen3_5ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32,
                                         attn_implementation="sdpa").to(dev).eval()
ref = TextBackbone(Weights(torch.float32, dev))
ids = encode_prompts([CALIB_PROMPTS[0]])[0].to(dev)

cap = {}
L0 = off.model.layers[0]
for name, mod in [("attn_in_ln", L0.input_layernorm), ("mixer", L0.linear_attn),
                  ("post_ln", L0.post_attention_layernorm), ("mlp", L0.mlp),
                  ("conv", L0.linear_attn.conv1d)]:
    mod.register_forward_hook(lambda m, i, o, n=name: cap.__setitem__(n, o))

o = off(input_ids=ids.unsqueeze(0), output_hidden_states=True)
emb = off.model.embed_tokens(ids.unsqueeze(0))[0]
print("embed max|Δ|", (emb - ref.w["model.language_model.embed_tokens.weight"][ids]).abs().max().item(),
      "|off|max", emb.abs().max().item())

def stats(n, t):
    t = t.float()
    print(f"  official {n:12s} |max|={t.abs().max():8.4f} mean={t.mean():8.4f} std={t.std():7.4f} shape={tuple(t.shape)}")

stats("emb", emb)
stats("attn_in_ln", cap["attn_in_ln"])
stats("mixer", cap["mixer"])
stats("post_ln", cap["post_ln"])
stats("mlp", cap["mlp"])
# conv hook does not fire (causal_conv1d reference calls F.conv1d)

# reference stages
p = "model.language_model."
h = ref.w[p + "embed_tokens.weight"][ids]
lnw = ref.w[p + "layers.0.input_layernorm.weight"]
h_ln = rms_norm(h, lnw, 1e-6)
print("ref attn_in_ln max|Δ|", (h_ln - cap["attn_in_ln"][0]).abs().max().item(),
      "cos", F.cosine_similarity(h_ln.flatten(), cap["attn_in_ln"][0].flatten(), 0).item())
mx = ref.linear_attention(h_ln, 0)
print("ref mixer    max|Δ|", (mx - cap["mixer"][0]).abs().max().item(),
      "|ref|max", mx.abs().max().item(),
      "cos", F.cosine_similarity(mx.flatten(), cap["mixer"][0].flatten(), 0).item())
h2 = h + mx
h2_ln = rms_norm(h2, ref.w[p + "layers.0.post_attention_layernorm.weight"], 1e-6)
print("ref post_ln  max|Δ|", (h2_ln - cap["post_ln"][0]).abs().max().item())
m = ref.mlp(h2_ln, 0)
print("ref mlp      max|Δ|", (m - cap["mlp"][0]).abs().max().item(),
      "|ref|max", m.abs().max().item())
r_out = ref.forward(ids)["hidden_layers"]
for i in (0, 1, 2, 3, 7, 15, 23):
    a, b = r_out[i], o.hidden_states[i + 1][0]
    print(f"layer{i:2d} max|Δ|={(a-b).abs().max():.4e} cos={F.cosine_similarity(a.flatten(),b.flatten(),0):.6f} |a|max={a.abs().max():.3f} |b|max={b.abs().max():.3f}")

# inside mixer: conv output shape/value
convw = ref.w[p + "layers.0.linear_attn.conv1d.weight"]
print("conv w shape", tuple(convw.shape), "official conv out", tuple(cap["conv"].shape))
