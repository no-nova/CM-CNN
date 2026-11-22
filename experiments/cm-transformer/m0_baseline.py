"""Step 1 (M0): architecture inspection + exact baseline alignment.

Runs the vision encoder and the language backbone on V1-V6 and on deterministic
text prompts, on RTX 5070 (bf16/fp16/fp32) and Zen 5 CPU (fp32), and records
determinism + cross-precision agreement.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import torch

from common import (Weights, MODEL_DIR, SAFETENSORS, EXP_DIR, ROOT, REGIMES,
                    bench, dump_json, load_config, log, set_deterministic,
                    sha256_file, LANE)
from model_ref import VisionEncoder, TextBackbone, VisionCfg, TextCfg
from video_data import (load_all, load_regime, to_pairs, to_single_steps,
                        encode_prompts, CALIB_PROMPTS, EVAL_PROMPTS,
                        token_ids_report)


# --------------------------------------------------------------------------- #
def inspect_architecture() -> dict:
    cfg = load_config()
    store = __import__("common").SafetensorsStore()
    names = store.names()

    def shapes(sub):
        rows = []
        for n in sorted(names):
            if sub in n:
                s, dt, _ = store.info(n)
                rows.append({"name": n, "shape": list(s), "dtype": dt})
        return rows

    def params(rows):
        tot = 0
        for r in rows:
            p = 1
            for d in r["shape"]:
                p *= d
            tot += p
        return tot

    vision_rows = shapes("model.visual.")
    text_rows = [r for r in shapes("model.language_model.")]
    mtp_rows = shapes("mtp.")
    total = sum(params(r) for r in (vision_rows, text_rows, mtp_rows))

    vc, tc = cfg["vision_config"], cfg["text_config"]
    vis_mlp = next(r for r in vision_rows if r["name"].endswith("blocks.0.mlp.linear_fc1.weight"))
    txt_gate = next(r for r in text_rows if r["name"].endswith("layers.0.mlp.gate_proj.weight"))
    lin_qkv = next(r for r in text_rows if "layers.0.linear_attn.in_proj_qkv.weight" in r["name"])

    # analytic MACs, one 224x224 frame pair (196 tokens) through the encoder
    N, D, I = 196, vc["hidden_size"], vc["intermediate_size"]
    per_block = {
        "qkv": N * D * 3 * D,
        "attn_qk": N * N * vc["num_heads"] * (D // vc["num_heads"]),
        "attn_av": N * N * vc["num_heads"] * (D // vc["num_heads"]),
        "proj": N * D * D,
        "fc1": N * D * I,
        "fc2": N * I * D,
    }
    vision_macs = {k: int(v) for k, v in per_block.items()}
    vision_macs["total_per_pair"] = int(sum(per_block.values()))
    vision_macs["total_per_block"] = int(sum(per_block.values()) // vc["depth"])

    # text MACs per token, per full-attention / linear-attention layer
    H = tc["hidden_size"]
    lin = {
        "in_proj_qkv": H * 3 * tc["linear_num_key_heads"] * tc["linear_key_head_dim"],
        "in_proj_z": H * tc["linear_num_value_heads"] * tc["linear_value_head_dim"],
        "out_proj": tc["linear_num_value_heads"] * tc["linear_value_head_dim"] * H,
        "conv1d": 3 * tc["linear_num_key_heads"] * tc["linear_key_head_dim"] * tc["linear_conv_kernel_dim"],
        "delta_rule": 2 * tc["linear_num_key_heads"] * tc["linear_key_head_dim"] * tc["linear_value_head_dim"],
    }
    full = {
        "q_proj": H * tc["num_attention_heads"] * tc["head_dim"] * 2,
        "k_proj": H * tc["num_key_value_heads"] * tc["head_dim"],
        "v_proj": H * tc["num_key_value_heads"] * tc["head_dim"],
        "o_proj": H * tc["num_attention_heads"] * tc["head_dim"],
    }
    ffn = {"gate": H * tc["intermediate_size"], "up": H * tc["intermediate_size"],
           "down": tc["intermediate_size"] * H}
    n_lin = sum(1 for t in tc["layer_types"] if t == "linear_attention")
    n_full = sum(1 for t in tc["layer_types"] if t == "full_attention")

    return {
        "checkpoint": {
            "path": str(SAFETENSORS.relative_to(ROOT)),
            "sha256": sha256_file(SAFETENSORS),
            "bytes": SAFETENSORS.stat().st_size,
            "n_tensors": len(names),
            "total_params": int(total),
            "vision_params": int(params(vision_rows)),
            "language_params": int(params(text_rows)),
            "mtp_params": int(params(mtp_rows)),
        },
        "vision_encoder": {
            "class": cfg["architectures"][0],
            "depth": vc["depth"],
            "hidden_size": vc["hidden_size"],
            "num_heads": vc["num_heads"],
            "head_dim": vc["hidden_size"] // vc["num_heads"],
            "intermediate_size": vc["intermediate_size"],
            "patch_size": vc["patch_size"],
            "temporal_patch_size": vc["temporal_patch_size"],
            "spatial_merge_size": vc["spatial_merge_size"],
            "pos_embed": {"entries": vc["num_position_embeddings"], "base_grid": "48x48",
                          "interpolation": "bilinear (align_corners=True) to the patch grid",
             "type": "learned absolute 48x48", "rope": "axial 2-D RoPE, full head dim, half-split pairs"},
            "activation": vc["hidden_act"],
            "norm": "LayerNorm(affine)",
            "attention": "fused qkv + axial 2-D RoPE, bidirectional within a frame-pair, "
                         "spatial-merge token order",
            "mlp_tensor": vis_mlp,
            "merger": "LayerNorm(768) -> 2x2 concat(3072) -> fc1(3072) -> gelu_tanh -> fc2(1024)",
            "halo_dilation_L": 0,
            "halo_note": "patch_embed.proj is a non-overlapping 16x16x2 patchify: no spatial "
                         "convolution, so a dirty input patch can only dirty its own token; "
                         "pointwise MLP + global attention do not spread the dirty mask.",
            "macs_per_frame_pair_224": vision_macs,
            "tensor_inventory": vision_rows,
        },
        "language_backbone": {
            "num_hidden_layers": tc["num_hidden_layers"],
            "hidden_size": H,
            "intermediate_size": tc["intermediate_size"],
            "vocab_size": tc["vocab_size"],
            "tie_word_embeddings": tc["tie_word_embeddings"],
            "layer_types": tc["layer_types"],
            "n_linear_attention": n_lin,
            "n_full_attention": n_full,
            "full_attention": {
                "num_heads": tc["num_attention_heads"],
                "num_kv_heads": tc["num_key_value_heads"],
                "head_dim": tc["head_dim"],
                "q_proj_out": tc["num_attention_heads"] * tc["head_dim"] * 2,
                "output_gate": tc["attn_output_gate"],
                "q_norm_k_norm": True,
                "rope": {"type": "partial interleaved", "theta": tc["rope_parameters"]["rope_theta"],
                         "rotary_dim": int(tc["head_dim"] * tc["rope_parameters"]["partial_rotary_factor"]),
                         "mrope_sections": tc["rope_parameters"]["mrope_section"],
                         "text_only_note": "all three mRoPE sections share position ids for text"},
            },
            "linear_attention": {
                "num_key_heads": tc["linear_num_key_heads"],
                "num_value_heads": tc["linear_num_value_heads"],
                "key_head_dim": tc["linear_key_head_dim"],
                "value_head_dim": tc["linear_value_head_dim"],
                "conv_kernel_dim": tc["linear_conv_kernel_dim"],
                "state_dtype": tc["mamba_ssm_dtype"],
                "in_proj_qkv": lin_qkv,
                "gate": "g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)",
            },
            "mlp": {"type": "SwiGLU", "activation": tc["hidden_act"],
                    "tensors": ["mlp.gate_proj [3584,1024]", "mlp.up_proj [3584,1024]",
                                "mlp.down_proj [1024,3584]"],
                    "tensor_example": txt_gate, "blocks_32_lane": 3584 // LANE},
            "macs_per_token": {"linear_layer": lin, "full_layer_proj": full, "ffn": ffn,
                               "total_linear_layer": int(sum(lin.values()) + sum(ffn.values())),
                               "total_full_layer_excl_attn_scores": int(sum(full.values()) + sum(ffn.values()))},
            "tensor_inventory": text_rows,
            "mtp_inventory": mtp_rows,
        },
        "notes": [
            "Reference implementation written from config.json + checkpoint shapes; "
            "transformers 4.57.dev (required for qwen3_5) is not installed in this environment.",
            "Gate/up/down projections and vision fc1/fc2 are addressed by exact tensor names.",
        ],
    }


# --------------------------------------------------------------------------- #
def vision_baseline(dtype, device, regimes=None, batch_pairs=True):
    enc = VisionEncoder(Weights(dtype, device))
    out = {"device": str(device), "dtype": str(dtype).replace("torch.", ""), "regimes": {}}
    for r in regimes or REGIMES:
        frames = load_regime(r).to(device)
        pairs = to_pairs(frames).to(dtype)
        with torch.no_grad():
            feats = enc.forward(pairs)
            # per-pair independent forwards must equal the batched forward
            feats_single = torch.cat([enc.forward(pairs[i:i + 1]) for i in range(pairs.shape[0])], 0)
        ms, _ = bench(lambda: enc.forward(pairs), device, 1 if device == "cpu" else 3,
                      3 if device == "cpu" else 10)
        ms_single, _ = bench(lambda: enc.forward(pairs[0:1]), device, 1 if device == "cpu" else 3,
                             3 if device == "cpu" else 10)
        f32 = feats.float()
        rec = {
            "n_pairs": int(pairs.shape[0]),
            "tokens_per_pair": int(feats.shape[0] // pairs.shape[0]),
            "feature_mean": float(f32.mean()),
            "feature_std": float(f32.std()),
            "feature_abs_mean": float(f32.abs().mean()),
            "feature_sha256": __import__("hashlib").sha256(
                f32.cpu().numpy().tobytes()).hexdigest()[:16],
            "batch_vs_single_max_abs": float((feats.float() - feats_single.float()).abs().max()),
            "mean_ms_per_regime": ms,
            "mean_ms_per_pair": ms / pairs.shape[0],
            "single_pair_ms": ms_single,
        }
        out["regimes"][r] = rec
        log(f"  vision {r} {out['dtype']:9s} {ms/pairs.shape[0]:8.3f} ms/pair "
            f"mean={rec['feature_mean']:+.5f} std={rec['feature_std']:.5f}")
    out["weights"] = None
    del enc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def vision_cross_precision(regimes=("V2", "V5")):
    feats = {}
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        enc = VisionEncoder(Weights(dt, "cuda"))
        key = str(dt).replace("torch.", "")
        feats[key] = {}
        for r in regimes:
            pairs = to_pairs(load_regime(r)).to("cuda").to(dt)
            with torch.no_grad():
                feats[key][r] = enc.forward(pairs).float().cpu()
        del enc
        torch.cuda.empty_cache()
    out = {}
    for key in ("bfloat16", "float16"):
        out[key + "_vs_fp32"] = {}
        for r in regimes:
            a, b = feats["float32"][r], feats[key][r]
            out[key + "_vs_fp32"][r] = {
                "max_abs": float((a - b).abs().max()),
                "rel_l2": float((a - b).norm() / a.norm()),
                "cosine": float(torch.nn.functional.cosine_similarity(
                    a.flatten(), b.flatten(), dim=0)),
            }
    return out


def text_baseline(dtype, device, max_len=64):
    tb = TextBackbone(Weights(dtype, device))
    prompts = encode_prompts(CALIB_PROMPTS[:4] + EVAL_PROMPTS[:2], max_len=max_len)
    recs = []
    for p in prompts:
        p = p.to(device)
        with torch.no_grad():
            r1 = tb.forward(p, logits_positions=[-1])["logits"].float()
            r2 = tb.forward(p, logits_positions=[-1])["logits"].float()
        ms, _ = bench(lambda: tb.forward(p, logits_positions=[-1]), device,
                      1 if device == "cpu" else 3, 3 if device == "cpu" else 10)
        recs.append({
            "n_tokens": int(p.numel()),
            "logit_mean": float(r1.mean()), "logit_std": float(r1.std()),
            "argmax": int(r1[-1].argmax()),
            "repeat_max_abs": float((r1 - r2).abs().max()),
            "logits_f32": r1.cpu(),
            "ms": ms,
        })
        log(f"  text {len(p)} tok {str(dtype).replace('torch.',''):9s} "
            f"{ms:8.2f} ms argmax={recs[-1]['argmax']} repeat={recs[-1]['repeat_max_abs']:.2e}")
    del tb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return recs


def compare_logits(a, b):
    import torch.nn.functional as F
    d = (a - b).abs()
    pa, pb = F.softmax(a, -1), F.softmax(b, -1)
    kl = (pa * (pa.clamp_min(1e-12).log() - pb.clamp_min(1e-12).log())).sum(-1)
    return {
        "max_abs": float(d.max()),
        "rmse": float(torch.sqrt(((a - b) ** 2).mean())),
        "top1_agreement": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
        "kl": float(kl.mean()),
    }


def main():
    set_deterministic(0)
    torch.set_num_threads(1)
    result = {"arch": inspect_architecture(), "m0": {}}
    log("arch inspected:", result["arch"]["checkpoint"]["total_params"], "params")

    result["m0"]["text_tokenization"] = token_ids_report()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # --- GPU fp32 / bf16 / fp16 ------------------------------------------- #
    gpu = {}
    if torch.cuda.is_available():
        for dt in (torch.float32, torch.bfloat16, torch.float16):
            log(f"GPU {dt}")
            gpu[str(dt).replace("torch.", "")] = vision_baseline(dt, "cuda")
            if torch.cuda.is_available():
                gpu[str(dt).replace("torch.", "")]["peak_vram_mb"] = \
                    torch.cuda.max_memory_allocated() / 2**20
                torch.cuda.reset_peak_memory_stats()

        # cross-precision agreement of vision features (V2, V5 = the regimes P2 keeps)
        result["m0"]["vision_gpu"] = gpu
        result["m0"]["vision_cross_precision"] = vision_cross_precision()

        # --- text (GPU) --------------------------------------------------- #
        text = {}
        ref_logits = {}
        for dt in (torch.float32, torch.bfloat16, torch.float16):
            log(f"GPU text {dt}")
            recs = text_baseline(dt, "cuda")
            key = str(dt).replace("torch.", "")
            text[key] = [{k: v for k, v in r.items() if k != "logits_f32"} for r in recs]
            ref_logits[key] = [r["logits_f32"] for r in recs]
        for key in ("bfloat16", "float16"):
            text[key + "_vs_fp32"] = [
                compare_logits(a, b) for a, b in zip(ref_logits[key], ref_logits["float32"])]
        result["m0"]["text_gpu"] = text
        del ref_logits
        torch.cuda.empty_cache()

    # --- CPU fp32 ---------------------------------------------------------- #
    log("CPU fp32 vision")
    cpu = {"vision": vision_baseline(torch.float32, "cpu")}
    log("CPU fp32 text")
    cpu["text"] = [{k: v for k, v in r.items() if k != "logits_f32"}
                   for r in text_baseline(torch.float32, "cpu", max_len=32)]
    result["m0"]["cpu"] = cpu

    # --- determinism hashes (fresh process re-run compared separately) ----- #
    result["integrity"] = {
        "codec_diff": __import__("subprocess").run(
            ["git", "diff", "--", "src/lib/codec.ts"], cwd=ROOT,
            capture_output=True, text=True).stdout,
        "weights_unmodified_note": "models/ opened read-only (np.memmap mode='r')",
        "model_mtime": SAFETENSORS.stat().st_mtime,
    }
    dump_json(result, EXP_DIR / "results" / "m0_baseline.json")
    log("wrote results/m0_baseline.json")


if __name__ == "__main__":
    main()
