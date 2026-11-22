"""Reference Qwen3.5-0.8B implementation (vision encoder + text backbone).

Written from config.json and the checkpoint tensor shapes because the installed
PyTorch has no `transformers` and Qwen3.5 requires transformers>=4.57.dev.
The implementation is deliberately explicit (plain functions + weight dict) so
that every projection the mechanisms touch is addressable by name.

Layout assumptions that the checkpoint shapes pin down:
  * patch_embed.proj.weight : [768, 3, 2, 16, 16]  (C, T, H, W) flatten order
  * attn.qkv.weight         : [2304, 768]          -> 3 x 12 heads x 64
  * self_attn.q_proj.weight : [4096, 1024]         -> 8 heads x (256 q | 256 gate)
  * linear_attn.in_proj_qkv : [6144, 1024]         -> q|k|v, 16 heads x 128
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from common import (Weights, load_config, rms_norm, gated_rms_norm, layer_norm,
                    gelu_tanh, silu, LANE, _acc_dtype)


@dataclass
class VisionCfg:
    depth: int = 12
    hidden: int = 768
    heads: int = 12
    inter: int = 3072
    patch: int = 16
    temporal: int = 2
    merge: int = 2
    pos_embed_n: int = 2304
    out_hidden: int = 1024
    eps: float = 1e-6


@dataclass
class TextCfg:
    layers: int = 24
    hidden: int = 1024
    inter: int = 3584
    heads: int = 8
    kv_heads: int = 2
    head_dim: int = 256
    vocab: int = 248320
    eps: float = 1e-6
    rope_theta: float = 1e7
    rotary_frac: float = 0.25
    lin_heads: int = 16
    lin_kdim: int = 128
    lin_vdim: int = 128
    conv_k: int = 4


# --------------------------------------------------------------------------- #
# vision encoder
# --------------------------------------------------------------------------- #
class VisionEncoder:
    def __init__(self, w: Weights, cfg: VisionCfg | None = None):
        self.w = w
        self.cfg = cfg or VisionCfg()
        self.prefix = "model.visual."
        self._pos_cache: dict[tuple[int, int], torch.Tensor] = {}

    # -- helpers ----------------------------------------------------------- #
    def token_grid(self, gh: int, gw: int, device=None):
        """Patch (row, col) for every token, in the official spatial-merge-block order
        (block_row, block_col, in_row, in_col) — NOT raster order."""
        m = self.cfg.merge
        assert gh % m == 0 and gw % m == 0
        rr = torch.arange(gh, device=device).reshape(gh // m, m)
        cc = torch.arange(gw, device=device).reshape(gw // m, m)
        rows = rr[:, None, :, None].expand(-1, gw // m, -1, m).reshape(-1)
        cols = cc[None, :, None, :].expand(gh // m, -1, m, -1).reshape(-1)
        return rows, cols

    def pos_embed(self, gh: int, gw: int, device, dtype):
        key = (gh, gw)
        if key not in self._pos_cache:
            c = self.cfg
            m = c.merge
            base = self.w[self.prefix + "pos_embed.weight"]           # [2304,768]
            side = int(round(c.pos_embed_n ** 0.5))                    # 48
            g = base.reshape(1, side, side, c.hidden).permute(0, 3, 1, 2)
            g = F.interpolate(g.float(), size=(gh, gw), mode="bilinear",
                              align_corners=True)
            g = g.permute(0, 2, 3, 1).reshape(gh, gw, c.hidden)
            # merge-block order, same as the token stream
            g = g.reshape(gh // m, m, gw // m, m, c.hidden).permute(0, 2, 1, 3, 4)
            self._pos_cache[key] = g.reshape(gh * gw, c.hidden)
        return self._pos_cache[key].to(device=device, dtype=dtype)

    def patchify(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [S, T, 3, H, W] -> [S*gh*gw, 3*T*16*16] (T must be 2)."""
        c = self.cfg
        S, T, C, H, W = frames.shape
        assert T == c.temporal and C == 3 and H % c.patch == 0 and W % c.patch == 0
        gh, gw = H // c.patch, W // c.patch
        x = frames.permute(0, 2, 1, 3, 4)                  # S,C,T,H,W
        x = x.reshape(S, C, T, gh, c.patch, gw, c.patch)
        x = x.permute(0, 3, 5, 1, 2, 4, 6)                 # S,gh,gw,C,T,ph,pw
        m = c.merge
        x = x.reshape(S, gh // m, m, gw // m, m, C, T, c.patch, c.patch)
        x = x.permute(0, 1, 3, 2, 4, 5, 6, 7, 8)           # official merge-block order
        return x.reshape(S * gh * gw, C * T * c.patch * c.patch)

    # -- dense forward ----------------------------------------------------- #
    def embed(self, frames: torch.Tensor):
        """frames [S,2,3,H,W] -> patch tokens [S*gh*gw,768] with pos embed."""
        c = self.cfg
        p = self.patchify(frames)
        S, T, C, H, W = frames.shape
        gh, gw = H // c.patch, W // c.patch
        N = gh * gw
        h = F.linear(p, self.w[self.prefix + "patch_embed.proj.weight"].reshape(c.hidden, -1),
                     self.w[self.prefix + "patch_embed.proj.bias"])
        h = h.reshape(S, N, c.hidden) + self.pos_embed(gh, gw, h.device, h.dtype)[None]
        return h.reshape(S * N, c.hidden), (S, gh, gw)

    def vision_rope(self, gh: int, gw: int, device, dtype):
        """Axial 2D RoPE over the full head dim: freqs for (h,w) concatenated twice,
        then the half-split rotation pairs element i with i + head_dim/2."""
        c = self.cfg
        key = ("rope", gh, gw)
        if key not in self._pos_cache:
            hd = c.hidden // c.heads                       # 64
            sd = hd // 2                                   # 32
            acc = _acc_dtype(dtype)
            inv = 1.0 / (10000.0 ** (torch.arange(0, sd, 2, device=device, dtype=acc) / sd))
            rows, cols = self.token_grid(gh, gw, device)
            ang = torch.cat([torch.outer(rows.float(), inv),
                             torch.outer(cols.float(), inv)], dim=-1)   # [N,32]
            self._pos_cache[key] = (torch.cos(ang), torch.sin(ang))
        cos, sin = self._pos_cache[key]
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def block_dense(self, h, i, shape):
        c = self.cfg
        S, gh, gw = shape
        N = gh * gw
        p = f"{self.prefix}blocks.{i}."
        w = self.w
        x = layer_norm(h, w[p + "norm1.weight"], w[p + "norm1.bias"], c.eps)
        qkv = F.linear(x, w[p + "attn.qkv.weight"], w[p + "attn.qkv.bias"])
        q, k, v = qkv.reshape(S * N, 3, c.heads, c.hidden // c.heads).permute(1, 0, 2, 3)
        # axial 2D RoPE on the (h,w) patch grid (full head dim, half-split pairs)
        cos, sin = self.vision_rope(gh, gw, h.device, q.dtype)
        hd = c.hidden // c.heads
        half = hd // 2
        cos_ = cos.repeat(S, 1)[:, None, :]
        sin_ = sin.repeat(S, 1)[:, None, :]

        def rot(t):
            t1, t2 = t[..., :half], t[..., half:]
            return torch.cat([t1 * cos_ - t2 * sin_, t2 * cos_ + t1 * sin_], dim=-1)

        q, k = rot(q), rot(k)
        q = q.reshape(S, N, c.heads, -1).transpose(1, 2)
        k = k.reshape(S, N, c.heads, -1).transpose(1, 2)
        v = v.reshape(S, N, c.heads, -1).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(S * N, c.hidden)
        h = h + F.linear(o, w[p + "attn.proj.weight"], w[p + "attn.proj.bias"])
        x = layer_norm(h, w[p + "norm2.weight"], w[p + "norm2.bias"], c.eps)
        a = gelu_tanh(F.linear(x, w[p + "mlp.linear_fc1.weight"], w[p + "mlp.linear_fc1.bias"]))
        h = h + F.linear(a, w[p + "mlp.linear_fc2.weight"], w[p + "mlp.linear_fc2.bias"])
        return h

    def merger(self, h, shape):
        c = self.cfg
        S, gh, gw = shape
        m = c.merge
        assert gh % m == 0 and gw % m == 0
        p = self.prefix + "merger."
        w = self.w
        # per-token LayerNorm(768) first, then 2x2 spatial concat -> 4*768
        x = layer_norm(h, w[p + "norm.weight"], w[p + "norm.bias"], c.eps)
        # tokens already arrive grouped as consecutive 2x2 blocks -> plain view
        g = x.reshape(S * (gh // m) * (gw // m), m * m * c.hidden)
        x = F.gelu(F.linear(g, w[p + "linear_fc1.weight"], w[p + "linear_fc1.bias"]))
        return F.linear(x, w[p + "linear_fc2.weight"], w[p + "linear_fc2.bias"])

    def forward(self, frames: torch.Tensor, capture: dict | None = None):
        """frames [S, 2, 3, H, W] -> merged features [S*gh/2*gw/2, 1024]."""
        h, shape = self.embed(frames)
        for i in range(self.cfg.depth):
            if capture is not None:
                capture.setdefault("block_in", []).append(h)
            h = self.block_dense(h, i, shape)
        if capture is not None:
            capture["pre_merger"] = h
            capture["shape"] = shape
        return self.merger(h, shape)


# --------------------------------------------------------------------------- #
# text backbone
# --------------------------------------------------------------------------- #
class TextBackbone:
    def __init__(self, w: Weights, cfg: TextCfg | None = None):
        self.w = w
        self.cfg = cfg or TextCfg()
        self.prefix = "model.language_model."
        self.layer_types = load_config()["text_config"]["layer_types"]

    # -- rope -------------------------------------------------------------- #
    def rope(self, q, k, positions=None):
        """Partial RoPE over the first rotary_frac of head_dim.

        HF builds `inv_freq` over `rd` and then `cat(freqs, freqs)`, and `rotate_half`
        splits the rotary block in halves, so element i rotates with element i + rd/2
        (GPT-NeoX layout) — not neighbouring pairs.
        """
        c = self.cfg
        T = q.shape[0]
        rd = int(c.head_dim * c.rotary_frac)                 # 64
        half = rd // 2                                       # 32
        pos = torch.arange(T, device=q.device) if positions is None else positions
        acc = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        inv = 1.0 / (c.rope_theta ** (torch.arange(0, rd, 2, device=q.device,
                                                  dtype=acc) / rd))
        ang = torch.outer(pos.to(acc), inv)                  # [T, half]
        cos = torch.cos(ang)[:, None, :].to(q.dtype)          # [T,1,half]
        sin = torch.sin(ang)[:, None, :].to(q.dtype)

        def rot(x):
            xr, xp = x[..., :rd], x[..., rd:]
            x1, x2 = xr[..., :half], xr[..., half:]
            return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, xp], dim=-1)

        return rot(q), rot(k)

    # -- attention --------------------------------------------------------- #
    def full_attention(self, x, i, causal=True, state=None, pos0=0):
        c = self.cfg
        p = f"{self.prefix}layers.{i}.self_attn."
        w = self.w
        T = x.shape[0]
        qg = F.linear(x, w[p + "q_proj.weight"])
        qg = qg.reshape(T, c.heads, 2 * c.head_dim)
        q, gate = qg.chunk(2, dim=-1)
        k = F.linear(x, w[p + "k_proj.weight"]).reshape(T, c.kv_heads, c.head_dim)
        v = F.linear(x, w[p + "v_proj.weight"]).reshape(T, c.kv_heads, c.head_dim)
        q = rms_norm(q, w[p + "q_norm.weight"], c.eps)
        k = rms_norm(k, w[p + "k_norm.weight"], c.eps)
        q, k = self.rope(q, k, positions=torch.arange(pos0, pos0 + T, device=x.device))
        k = k.repeat_interleave(c.heads // c.kv_heads, dim=1)
        v = v.repeat_interleave(c.heads // c.kv_heads, dim=1)
        q = q.transpose(0, 1)                                # [H,T,D]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal and T > 1)
        if state is not None:
            kc, vc = state["kv"].get(i, (None, None))
            if kc is not None:
                k = torch.cat([kc, k], dim=1)
                v = torch.cat([vc, v], dim=1)
                causal = False          # queries are the new tail only
            state["kv"][i] = (k, v)
        q = q[:, -T:]
        o = F.scaled_dot_product_attention(q, k, v,
                                           is_causal=causal and T > 1)
        o = o.transpose(0, 1).reshape(T, c.heads * c.head_dim)
        o = o * torch.sigmoid(gate.reshape(T, c.heads * c.head_dim))
        return F.linear(o, w[p + "o_proj.weight"])

    def linear_attention(self, x, i, state=None):
        """Gated delta rule (Qwen3-Next linear attention), exact recurrent form.

        `state` (optional dict) carries {"conv": {i: [k-1, C]}, "S": {i: [H,Dk,Dv]}}
        so the same routine serves prefill and single-token incremental decode.
        """
        c = self.cfg
        p = f"{self.prefix}layers.{i}.linear_attn."
        w = self.w
        T = x.shape[0]
        H, Dk, Dv = c.lin_heads, c.lin_kdim, c.lin_vdim

        qkv = F.linear(x, w[p + "in_proj_qkv.weight"])
        # causal depthwise conv, kernel 4, groups = 3*H*D
        cw = w[p + "conv1d.weight"].reshape(-1, c.conv_k)
        tail = None if state is None else state.setdefault("conv", {}).get(i)
        inp = qkv if tail is None else torch.cat([tail, qkv], 0)
        if state is not None:
            state["conv"][i] = inp[-(c.conv_k - 1):].clone()
        qkv = inp.transpose(0, 1)[None]                        # [1,C,T']
        qkv = F.pad(qkv, (c.conv_k - 1, 0))
        qkv = F.conv1d(qkv, cw[:, None, :], groups=qkv.shape[1]).transpose(1, 2)[0]
        qkv = silu(qkv)[-T:]
        if tail is not None:
            qkv = qkv[-T:]
        q, k, v = qkv.split(H * Dk, dim=-1)
        q = q.reshape(T, H, Dk)
        k = k.reshape(T, H, Dk)
        v = v.reshape(T, H, Dv)
        z = F.linear(x, w[p + "in_proj_z.weight"]).reshape(T, H, Dv)
        b = F.linear(x, w[p + "in_proj_b.weight"])
        a = F.linear(x, w[p + "in_proj_a.weight"])
        A = torch.exp(w[p + "A_log"].float())
        g = -A * F.softplus(a.float() + w[p + "dt_bias"].float())   # [T,H] <= 0
        beta = torch.sigmoid(b.float())

        q = F.normalize(q.float(), dim=-1, eps=1e-6)
        k = F.normalize(k.float(), dim=-1, eps=1e-6)
        v = v.float()
        scale = Dk ** -0.5
        if state is None or i not in state.get("S", {}):
            S = torch.zeros(H, Dk, Dv, device=x.device, dtype=torch.float32)
        else:
            S = state["S"][i]
        if state is not None:
            state.setdefault("S", {})[i] = S
        outs = []
        for t in range(T):
            S = S * torch.exp(g[t])[:, None, None]
            v_new = v[t] - torch.einsum("hkv,hk->hv", S, k[t])
            S = S + beta[t][:, None, None] * k[t].unsqueeze(-1) * v_new.unsqueeze(-2)
            o = torch.einsum("hkv,hk->hv", S, q[t]) * scale
            outs.append(o)
        o = torch.stack(outs, 0)                               # [T,H,Dv]
        if state is not None:
            state["S"][i] = S
        o = gated_rms_norm(o, z, w[p + "norm.weight"], c.eps)
        return F.linear(o.to(x.dtype).reshape(T, H * Dv), w[p + "out_proj.weight"])

    # -- ffn --------------------------------------------------------------- #
    def mlp(self, x, i, W=None, keep=None, skip_mask=None, skip_rate=None,
            energy_out=None, skip_scale=None, skip_theta=None, norm_gamma=None,
            coupled=False, skip_wb=None):
        p = f"{self.prefix}layers.{i}.mlp."
        w = self.w
        if W is None:
            W = (w[p + "gate_proj.weight"], w[p + "up_proj.weight"], w[p + "down_proj.weight"])
        if coupled:
            # Defect-3: SwiGLU-coupled skipping.  gate is always dense (it is
            # what tells us which 32-lane blocks matter); then *only* the active
            # rows of up_proj and the active columns of down_proj are touched.
            g = silu(F.linear(x, W[0]))
            sc = None if skip_scale is None else skip_scale[i]
            e = block_energy(g if sc is None else g * sc)
            if skip_wb is not None:
                e = e * skip_wb[i]
            if energy_out is not None:
                energy_out[i] = e
            sta = getattr(self, "static_keep", None)
            if sta is not None:
                kp = sta[i].to(g.device).expand(g.shape[0], -1)
            elif skip_theta is not None:
                th = skip_theta[i] if isinstance(skip_theta, dict) else skip_theta
                kp = e >= th
            else:
                kp = keep_mask_from_energy(e, skip_rate)
            T = g.shape[0]
            if T == 1 and kp.dim() == 2 and bool(kp[0].any()):
                # kp indexes 32-lane blocks; expand to lane indices before the
                # gather (using block ids directly selects lanes 0..K-1).
                blk = torch.nonzero(kp[0], as_tuple=False).flatten()
                idx = (blk[:, None] * LANE +
                       torch.arange(LANE, device=blk.device)).flatten()
                u = F.linear(x, W[1].index_select(0, idx))       # K rows of up
                gg = g.index_select(-1, idx)
                a = gg * u
                if norm_gamma:
                    n_keep = a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    n_full = (g * F.linear(x, W[1])).norm(dim=-1, keepdim=True)
                    a = a * (n_full / n_keep).clamp(max=float(norm_gamma))
                return F.linear(a, W[2].index_select(1, idx))    # K cols of down
            a = g * F.linear(x, W[1])
            if kp is not None:
                a = a * kp.repeat_interleave(LANE, dim=-1).to(a.dtype)
            if norm_gamma and kp is not None:
                n_keep = a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                n_full = (g * F.linear(x, W[1])).norm(dim=-1, keepdim=True)
                a = a * (n_full / n_keep).clamp(max=float(norm_gamma))
            return F.linear(a, W[2])
        a = silu(F.linear(x, W[0])) * F.linear(x, W[1])
        keep = None
        if skip_mask is not None:
            keep = skip_mask
        elif getattr(self, "static_keep", None) is not None:
            keep = self.static_keep[i].to(a.device).expand(a.shape[0], -1)
        elif skip_theta is not None:
            # importance-weighted (output-norm) block energy vs a calibrated
            # threshold: skip every 32-lane block whose ||s_B||_2 < theta.
            sc = None if skip_scale is None else skip_scale[i]
            e = block_energy(a if sc is None else a * sc)
            if energy_out is not None:
                energy_out[i] = e
            th = skip_theta[i] if isinstance(skip_theta, dict) else skip_theta
            keep = e >= th
        elif skip_rate:
            sc = None if skip_scale is None else skip_scale[i]
            e = block_energy(a if sc is None else a * sc)
            if energy_out is not None:
                energy_out[i] = e
            keep = keep_mask_from_energy(e, skip_rate)
        if keep is not None:
            # Defect-4 fix: zeroing blocks shrinks ||a||_2, and over 24-28 layers
            # that drift compounds.  Rescale the surviving vector to the norm of
            # the untruncated one: alpha = ||a|| / ||a_keep||, capped at
            # norm_gamma (a full rescale over-corrects when the dropped blocks
            # carry direction, not just length).
            n_full = a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            a = a * keep.repeat_interleave(LANE, dim=-1).to(a.dtype)
            if norm_gamma:
                n_keep = a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                alpha = (n_full / n_keep).clamp(max=float(norm_gamma))
                a = a * alpha
            if getattr(self, "down_fn", None) is not None:
                return self.down_fn(a, W[2], keep)
        return F.linear(a, W[2])

    # -- forward ----------------------------------------------------------- #
    def forward(self, ids=None, capture=None, mlp_override=None, mlp_mask=None,
                skip_rate=None, energy_out=None, logits_positions=None, state=None,
                skip_scale=None, skip_theta=None, norm_gamma=None,
                coupled=False, pos0=0, embeds=None):
        """ids [T] or [1,T] (or pre-built `embeds` [T,H]) -> dict(logits, hidden, ...)."""
        c = self.cfg
        if embeds is not None:
            h = embeds
        else:
            if ids.dim() == 1:
                ids = ids[None]
            h = F.embedding(ids, self.w[self.prefix + "embed_tokens.weight"])[0]
        hidden_layers = []
        for i in range(c.layers):
            # pre-norm on the attention branch (Qwen3.5DecoderLayer.input_layernorm)
            xa = rms_norm(h, self.w[f"{self.prefix}layers.{i}.input_layernorm.weight"], c.eps)
            if self.layer_types[i] == "full_attention":
                h = h + self.full_attention(xa, i, state=state, pos0=pos0)
            else:
                h = h + self.linear_attention(xa, i, state=state)
            x = rms_norm(h, self.w[f"{self.prefix}layers.{i}.post_attention_layernorm.weight"], c.eps)
            if capture is not None:
                capture.setdefault("mlp_in", {})[i] = x
            W = None if mlp_override is None else mlp_override.get(i)
            keep = None if mlp_mask is None else mlp_mask.get(i)
            h = h + self.mlp(x, i, W=W, skip_mask=keep,
                             skip_rate=skip_rate, energy_out=energy_out,
                             skip_scale=skip_scale, skip_theta=skip_theta,
                             norm_gamma=norm_gamma, coupled=coupled)
            hidden_layers.append(h)
            if capture is not None:
                capture.setdefault("mlp_out", {})[i] = h
        h = rms_norm(h, self.w[self.prefix + "norm.weight"], c.eps)
        if logits_positions is None:
            logits = F.linear(h, self.w[self.prefix + "embed_tokens.weight"])
        else:
            logits = F.linear(h[logits_positions], self.w[self.prefix + "embed_tokens.weight"])
        return {"logits": logits, "hidden": h, "hidden_layers": hidden_layers}

    def attn(self, h, i):
        if self.layer_types[i] == "full_attention":
            return self.full_attention(h, i)
        return self.linear_attention(h, i)


# --------------------------------------------------------------------------- #
# 32-lane block utilities (shared by P1 and vision P1+P2)
# --------------------------------------------------------------------------- #
def new_state() -> dict:
    return {"kv": {}, "conv": {}, "S": {}}


def block_energy(a: torch.Tensor, lane: int = LANE) -> torch.Tensor:
    """a [...,I] -> per-token per-block L2 energy [..., I//lane]."""
    lead, I = a.shape[:-1], a.shape[-1]
    return a.reshape(*lead, I // lane, lane).float().pow(2).sum(-1)


def keep_mask_from_energy(e: torch.Tensor, rate: float) -> torch.Tensor:
    """Keep the highest-energy blocks per token so that ~`rate` of blocks are skipped."""
    lead, B = e.shape[:-1], e.shape[-1]
    n_keep = max(1, int(round(B * (1.0 - rate))))
    flat = e.reshape(-1, B)
    idx = flat.argsort(dim=-1, descending=True)[:, :n_keep]
    m = torch.zeros_like(flat, dtype=torch.bool)
    m.scatter_(1, idx, True)
    return m.reshape(*lead, B)


def apply_block_mask(a: torch.Tensor, keep: torch.Tensor, lane: int = LANE) -> torch.Tensor:
    m = keep.repeat_interleave(lane, dim=-1)
    return a * m.to(a.dtype)


def permute_mlp(W, perm: torch.Tensor):
    """Permute SwiGLU FFN weights so that new lane order == perm (new->old)."""
    g, u, d = W
    return (g[perm].contiguous(), u[perm].contiguous(), d[:, perm].contiguous())


def identity_perm(I: int, device="cpu") -> torch.Tensor:
    return torch.arange(I, device=device)


def lane_perm_to_blocks(perm: torch.Tensor, lane: int = LANE):
    """Return [B,lane] old-neuron indices grouped by the new contiguous blocks."""
    return perm.reshape(-1, lane)
