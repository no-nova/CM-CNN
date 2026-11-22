import torch, torch.nn.functional as F
from common import MODEL_DIR, Weights, rms_norm
from model_ref import TextBackbone
from video_data import CALIB_PROMPTS, encode_prompts
dev="cuda"; torch.set_grad_enabled(False)
from transformers import Qwen3_5ForCausalLM
off = Qwen3_5ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
ref = TextBackbone(Weights(torch.float32, dev))
ids = encode_prompts([CALIB_PROMPTS[0]])[0].to(dev)
cap={}
L0=off.model.layers[0]
L0.input_layernorm.register_forward_hook(lambda m,i,o: cap.__setitem__("ln",o))
L0.linear_attn.register_forward_hook(lambda m,i,o: cap.__setitem__("mix",o))
L0.post_attention_layernorm.register_forward_hook(lambda m,i,o: cap.__setitem__("post",o))
L0.mlp.register_forward_hook(lambda m,i,o: cap.__setitem__("mlp",o))
o=off(input_ids=ids.unsqueeze(0), output_hidden_states=True)
p="model.language_model."
h=ref.w[p+"embed_tokens.weight"][ids]
h2=h+cap["mix"][0]
out=o.hidden_states[1][0]
print("official out |max|",out.abs().max().item())
print("official (out-mlp) |max|",(out-cap["mlp"][0]).abs().max().item())
print("ref h2 |max|",h2.abs().max().item())
print("h2 vs out-mlp max|Δ|",(h2-(out-cap["mlp"][0])).abs().max().item())
print("ref out |max|",(h2+cap["mlp"][0]).abs().max().item(),"vs official",out.abs().max().item())
print("mlp cos", F.cosine_similarity(cap['mlp'][0].flatten(),cap['mlp'][0].flatten(),0).item())
print("first row ref h2[:6]",h2[0,:6].tolist())
print("first row off h2[:6]",(out-cap['mlp'][0])[0,:6].tolist())

r = ref.forward(ids, capture={})
print("ref.forward layer0 |max|", r["hidden_layers"][0].abs().max().item())
print("diff vs manual", (r["hidden_layers"][0]-(h2+cap['mlp'][0])).abs().max().item())
print("logits |max|", r["logits"].abs().max().item(), "official", o.logits.abs().max().item())
print("mlp_in diff", (r["capture"] if False else 0))

print("layer_types[:8]", ref.layer_types[:8])
cap2={}
r2 = ref.forward(ids, capture=cap2)
x0 = rms_norm(h + ref.linear_attention(rms_norm(h, ref.w[p+"layers.0.input_layernorm.weight"], 1e-6),0), ref.w[p+"layers.0.post_attention_layernorm.weight"], 1e-6)
print("captured mlp_in[0] vs manual", (cap2["mlp_in"][0]-x0).abs().max().item())
print("captured mlp_in[0] |max|", cap2["mlp_in"][0].abs().max().item())
print("manual x0 |max|", x0.abs().max().item())
