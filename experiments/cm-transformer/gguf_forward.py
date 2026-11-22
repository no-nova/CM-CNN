#!/usr/bin/env python3
"""Streaming fp32 forward pass of Qwen3.5-4B straight out of the IQ4_NL GGUF.

Why this exists
---------------
Phases 1-5 evaluated 32-lane block skipping on *synthetic* activation vectors
(and, for the text model, on 0.8B hidden states).  This module produces the real
thing: the hidden states of the actual 4B checkpoint, obtained by dequantising
one weight tensor at a time, using it, and freeing it.  A full fp32 copy of the
model is 16 GB; this box has ~3 GB free, so nothing is ever held whole.

Weight source
-------------
`GGUFReader` (gguf_iq4nl.py) + dequantisers for the four types this file uses
(IQ4_NL 168 tensors, Q8_0 48, Q5_K 32, Q6_K 1, F32 177).  The Q5_K/Q6_K layouts
follow ggml's `dequantize_row_q5_K` / `dequantize_row_q6_K` / `get_scale_min_k4`
exactly (field order from ggml-common.h: q5_K = d,dmin,scales[12],qh[32],qs[128];
q6_K = ql[128],qh[64],scales[16],d).

Two details that the GGUF, not the safetensors, decides:
  * RMSNorm weights in this file are already the *effective* scale (mean ~1.0-1.2,
    not zero-centred), so `rms_norm` multiplies by `w` directly.  model_ref.py
    used `(1 + w)` because the safetensors checkpoint stores zero-centred weights.
  * `ssm_a` is stored as `-A_log.exp()` (all negative), so the decay gate is
    `softplus(a + dt_bias) * ssm_a` with no extra sign flip.

Architecture (read off the GGUF metadata + tensor shapes):
  32 layers, n_embd 2560, n_ff 9216, vocab 248320, eps 1e-6, rope_theta 1e7,
  rope.dimension_count 64 (partial NeoX over the first 64 of 256 head dims),
  16 q heads / 4 kv heads, head_dim 256, output gate on q (attn_q is 2*hd*nh),
  full_attention_interval 4 -> layers 3,7,...,31 are GQA attention, the other 24
  are gated-DeltaNet linear attention with n_k_heads 16, n_v_heads 32 (ssm_dt_rank),
  head dim 128, conv k=4, and a per-v-head gated RMSNorm (ssm_norm, 128).
  No output.weight: logits are the tied embedding matrix (token_embd, Q6_K).

Usage
-----
  python gguf_forward.py --self-test           # dequant structural checks
  python gguf_forward.py --ppl --tokens 512    # sanity PPL + " Paris" check
  python gguf_forward.py --capture             # write results/p6_4b_acts.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gguf_iq4nl import (GGUFReader, IQ4_NL_BLOCK_BYTES, KVALUES_IQ4NL,
                        dequant_iq4nl_np)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GGUF = ROOT / "models" / "Qwen3.5-4B-IQ4_NL.gguf"
ACTS_NPZ = HERE / "results" / "p6_4b_acts.npz"
ACTS_JSON = HERE / "results" / "p6_4b_acts.json"

QK_K = 256
LANE = 32

# --------------------------------------------------------------------------- #
# dequantisers (ggml layouts)
# --------------------------------------------------------------------------- #


def _fp16(raw: np.ndarray) -> np.ndarray:
    """(n,2) uint8 -> (n,) float32."""
    return np.ascontiguousarray(raw).view(np.float16).astype(np.float32).reshape(-1)


def dequant_q8_0(raw: np.ndarray) -> np.ndarray:
    """(nb,34) uint8 -> (nb,32) float32.  x = d * int8(q)."""
    nb = raw.shape[0]
    d = _fp16(raw[:, :2])
    q = raw[:, 2:34].view(np.int8).astype(np.float32)
    return q * d[:, None]


def dequant_q6_k(raw: np.ndarray) -> np.ndarray:
    """(nb,210) uint8 -> (nb,256) float32.

    y[l+ 0] = d*sc[is+0]*q1  q1 = lo4(ql[l])    | hi2(qh[l]>>0)
    y[l+32] = d*sc[is+2]*q2  q2 = lo4(ql[l+32]) | hi2(qh[l]>>2)
    y[l+64] = d*sc[is+4]*q3  q3 = hi4(ql[l])    | hi2(qh[l]>>4)
    y[l+96] = d*sc[is+6]*q4  q4 = hi4(ql[l+32]) | hi2(qh[l]>>6)
    with qx = val-32 and is = l//16; the second 128-element half uses
    ql[64:], qh[32:], sc[8:].
    """
    nb = raw.shape[0]
    d = _fp16(raw[:, 208:210])
    ql = raw[:, 0:128].reshape(nb, 2, 64)
    qh = raw[:, 128:192].reshape(nb, 2, 32)
    sc = raw[:, 192:208].view(np.int8).astype(np.float32).reshape(nb, 2, 8)
    out = np.empty((nb, 2, 128), dtype=np.float32)
    for c in range(2):
        lo1 = ql[:, c, 0:32].astype(np.int32)
        lo2 = ql[:, c, 32:64].astype(np.int32)
        hh = qh[:, c, :].astype(np.int32)
        s = sc[:, c, :]                                    # (nb,8)
        part = (
            ((lo1 & 0xF) | (((hh >> 0) & 3) << 4)) - 32,
            ((lo2 & 0xF) | (((hh >> 2) & 3) << 4)) - 32,
            ((lo1 >> 4) | (((hh >> 4) & 3) << 4)) - 32,
            ((lo2 >> 4) | (((hh >> 6) & 3) << 4)) - 32,
        )
        # sub-block scale index: is+0/+2/+4/+6, is = l//16 in {0,1}
        for k in range(4):
            sk = np.repeat(s[:, 2 * k:2 * k + 2], 16, axis=1)      # (nb,32)
            out[:, c, k * 32:(k + 1) * 32] = part[k] * sk
    out = out.reshape(nb, 256) * d[:, None]
    return out


def _scale_min_k4(scales: np.ndarray):
    """ggml get_scale_min_k4 for j=0..7.  scales (nb,12) uint8 -> 8x (nb,) vectors."""
    s = scales.astype(np.int32)
    d = np.empty((scales.shape[0], 8), np.float32)
    m = np.empty((scales.shape[0], 8), np.float32)
    for j in range(8):
        if j < 4:
            d[:, j] = s[:, j] & 63
            m[:, j] = s[:, j + 4] & 63
        else:
            d[:, j] = ((s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)).astype(np.float32)
            m[:, j] = ((s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)).astype(np.float32)
    return d, m


def dequant_q5_k(raw: np.ndarray) -> np.ndarray:
    """(nb,176) uint8 -> (nb,256) float32.

    elements 64j+l      : d1*((ql[l]&0xF) + 16*(qh[l]&u1)) - m1
    elements 64j+32+l   : d2*((ql[l]>>4)  + 16*(qh[l]&u2)) - m2
    u1 = 1<<2j, u2 = 2<<2j, ql = qs[32j:], (d1,m1)=scales[2j], (d2,m2)=scales[2j+1]
    """
    nb = raw.shape[0]
    d = _fp16(raw[:, 0:2])
    dmin = _fp16(raw[:, 2:4])
    sd, sm = _scale_min_k4(raw[:, 4:16])
    qh = raw[:, 16:48].astype(np.int32)
    qs = raw[:, 48:176]
    out = np.empty((nb, 256), dtype=np.float32)
    for j in range(4):
        ql = qs[:, 32 * j:32 * j + 32].astype(np.int32)
        u1, u2 = 1 << (2 * j), 2 << (2 * j)
        d1 = (d * sd[:, 2 * j])[:, None]
        m1 = (dmin * sm[:, 2 * j])[:, None]
        d2 = (d * sd[:, 2 * j + 1])[:, None]
        m2 = (dmin * sm[:, 2 * j + 1])[:, None]
        out[:, 64 * j:64 * j + 32] = d1 * ((ql & 0xF) + (qh & u1) // u1 * 16) - m1
        out[:, 64 * j + 32:64 * j + 64] = d2 * ((ql >> 4) + (qh & u2) // u2 * 16) - m2
    return out


def dequant_f32(raw: np.ndarray) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(raw).tobytes(), dtype=np.float32).copy()


def dequant_tensor(raw: np.ndarray, type_name: str) -> np.ndarray:
    """raw (nblocks*block_bytes,) or (rows, row_bytes) uint8 -> flat fp32."""
    bb = BLOCK_BYTES[type_name]
    if bb > 1:
        raw = np.ascontiguousarray(raw).reshape(-1, bb)
    if type_name == "IQ4_NL":
        return dequant_iq4nl_np(raw).reshape(-1)
    if type_name == "Q8_0":
        return dequant_q8_0(raw).reshape(-1)
    if type_name == "Q6_K":
        return dequant_q6_k(raw).reshape(-1)
    if type_name == "Q5_K":
        return dequant_q5_k(raw).reshape(-1)
    if type_name == "F32":
        return dequant_f32(raw.reshape(-1))
    if type_name == "F16":
        return np.ascontiguousarray(raw.reshape(-1, 2)).view(np.float16).astype(np.float32).reshape(-1)
    raise ValueError(f"unsupported ggml type {type_name}")


BLOCK_ELEMS = {"IQ4_NL": 32, "Q8_0": 32, "Q6_K": 256, "Q5_K": 256, "F32": 1, "F16": 1}
BLOCK_BYTES = {"IQ4_NL": 18, "Q8_0": 34, "Q6_K": 210, "Q5_K": 176, "F32": 4, "F16": 2}


# --------------------------------------------------------------------------- #
# streaming weight source
# --------------------------------------------------------------------------- #
class TensorSource:
    """Reads + dequantises individual GGUF tensors on demand; caller frees them."""

    def __init__(self, path: Path = GGUF, verbose: bool = False):
        self.path = str(path)
        self.r = GGUFReader(self.path)
        self.f = open(self.path, "rb")
        self.tmap = {t["name"]: t for t in self.r.tensors}
        self.verbose = verbose
        self.bytes_read = 0
        self.perf: dict[str, float] = {"dequant_s": 0.0, "read_s": 0.0}

    # -- raw I/O -----------------------------------------------------------
    def _raw(self, t: dict, nbytes: int | None = None) -> np.ndarray:
        lay = self.r.tensor_layout(t)
        n = lay["expected_bytes"] if nbytes is None else nbytes
        t0 = time.perf_counter()
        self.f.seek(self.r.data_start + t["offset"])
        b = self.f.read(n)
        self.perf["read_s"] += time.perf_counter() - t0
        self.bytes_read += n
        return np.frombuffer(b, dtype=np.uint8)

    def shape(self, name: str) -> tuple[int, ...]:
        """GGUF dims are [ne0=in, ne1=out, ...]; returned as torch (out, in)."""
        d = self.tmap[name]["dims"]
        return tuple(int(x) for x in reversed(d))

    def has(self, name: str) -> bool:
        return name in self.tmap

    # -- full tensor -------------------------------------------------------
    def get(self, name: str) -> torch.Tensor:
        """Dequantise a whole tensor -> torch fp32 of shape (out, in)."""
        t = self.tmap[name]
        be, bb = BLOCK_ELEMS[t["type"]], BLOCK_BYTES[t["type"]]
        n_elems = int(np.prod(t["dims"]))
        raw = self._raw(t)
        t0 = time.perf_counter()
        arr = dequant_tensor(raw, t["type"])
        self.perf["dequant_s"] += time.perf_counter() - t0
        assert arr.size == n_elems, (name, arr.size, n_elems)
        return torch.from_numpy(arr.reshape(self.shape(name)).copy())

    # -- rows only (token_embd is 521 MB; never dequantise it whole) --------
    def rows(self, name: str, ids) -> torch.Tensor:
        """Dequantise only the given rows of a 2-D tensor -> (len(ids), ne0) fp32."""
        t = self.tmap[name]
        be, bb = BLOCK_ELEMS[t["type"]], BLOCK_BYTES[t["type"]]
        ne0 = int(t["dims"][0])
        assert ne0 % be == 0
        row_bytes = ne0 // be * bb
        ids = np.asarray(ids, dtype=np.int64)
        out_bytes = np.empty((ids.size, row_bytes), dtype=np.uint8)
        t0 = time.perf_counter()
        for i, rid in enumerate(ids):
            self.f.seek(self.r.data_start + t["offset"] + int(rid) * row_bytes)
            out_bytes[i] = np.frombuffer(self.f.read(row_bytes), dtype=np.uint8)
        self.perf["read_s"] += time.perf_counter() - t0
        self.bytes_read += ids.size * row_bytes
        t0 = time.perf_counter()
        arr = dequant_tensor(out_bytes, t["type"]).reshape(ids.size, ne0)
        self.perf["dequant_s"] += time.perf_counter() - t0
        return torch.from_numpy(arr.copy())

    def close(self):
        self.f.close()
        self.r.close()


# --------------------------------------------------------------------------- #
# config from GGUF metadata
# --------------------------------------------------------------------------- #
def read_cfg(r: GGUFReader) -> dict:
    kv = r.kv
    arch = kv["general.architecture"]
    g = lambda k, d=None: kv.get(f"{arch}.{k}", d)
    n_layer = int(g("block_count"))
    interval = int(g("full_attention_interval", 4))
    cfg = dict(
        arch=arch,
        layers=n_layer,
        hidden=int(g("embedding_length")),
        inter=int(g("feed_forward_length")),
        heads=int(g("attention.head_count")),
        kv_heads=int(g("attention.head_count_kv")),
        head_dim=int(g("attention.key_length")),
        v_head_dim=int(g("attention.value_length")),
        vocab=int(kv["tokenizer.ggml.tokens"].__len__()),
        eps=float(g("attention.layer_norm_rms_epsilon")),
        rope_theta=float(g("rope.freq_base")),
        rope_dim=int(g("rope.dimension_count")),
        # linear (gated DeltaNet) attention
        lin_k_heads=int(g("ssm.group_count")),
        lin_v_heads=int(g("ssm.time_step_rank")),
        lin_k_dim=int(g("ssm.state_size")),
        lin_v_dim=int(g("ssm.inner_size")) // int(g("ssm.time_step_rank")),
        conv_k=int(g("ssm.conv_kernel")),
    )
    # llama.cpp: is_recr(i) = (i+1) % full_attn_interval != 0
    cfg["layer_types"] = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                          for i in range(n_layer)]
    return cfg


# --------------------------------------------------------------------------- #
# math (copied from model_ref.py, adapted to GGUF weight conventions)
# --------------------------------------------------------------------------- #
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """GGUF stores the effective scale, so multiply by w (no zero-centring)."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()


def gated_rms_norm(x: torch.Tensor, gate: torch.Tensor, w: torch.Tensor,
                   eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y * w.float() * F.silu(gate.float())


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True).clamp_min(eps * eps))


@torch.no_grad()
def build_model(src: TensorSource, cfg: dict) -> dict:
    """Load the small tensors that are reused (norms, ssm scalars) once."""
    W = {}
    for L in range(cfg["layers"]):
        p = f"blk.{L}."
        W[p + "attn_norm"] = src.get(p + "attn_norm.weight")[0]
        W[p + "post_norm"] = src.get(p + "post_attention_norm.weight")[0]
        if cfg["layer_types"][L] == "full_attention":
            W[p + "q_norm"] = src.get(p + "attn_q_norm.weight")[0]
            W[p + "k_norm"] = src.get(p + "attn_k_norm.weight")[0]
        else:
            W[p + "ssm_a"] = src.get(p + "ssm_a")
            W[p + "ssm_dt"] = src.get(p + "ssm_dt.bias")
            W[p + "ssm_norm"] = src.get(p + "ssm_norm.weight")[0]
            W[p + "ssm_conv"] = src.get(p + "ssm_conv1d.weight")
    W["output_norm"] = src.get("output_norm.weight")[0]
    return W


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forward(ids, src: TensorSource, cfg: dict, W: dict | None = None,
            capture: bool = False, skip_tau: float | None = None,
            layers: int | None = None, log=print) -> dict:
    """ids [T] -> dict(hidden [T,H], acts [n,T,H] fp16 if capture, skip_stats)."""
    W = W if W is not None else build_model(src, cfg)
    c = cfg
    H, eps = c["hidden"], c["eps"]
    nL = c["layers"] if layers is None else min(layers, c["layers"])
    T = int(np.asarray(ids).size)
    h = src.rows("token_embd.weight", ids)
    acts = np.zeros((nL, T, H), dtype=np.float16) if capture else None
    nb = c["inter"] // LANE
    skip_stats = []
    n_skip = 0 if skip_tau is None else max(1, int(round(nb * (1.0 - skip_tau))))
    t_start = time.perf_counter()

    for L in range(nL):
        p = f"blk.{L}."
        xa = rms_norm(h, W[p + "attn_norm"], eps)
        if c["layer_types"][L] == "full_attention":
            h = h + _full_attn(xa, src, cfg, W, p, L)
        else:
            h = h + _linear_attn(xa, src, cfg, W, p)
        x = rms_norm(h, W[p + "post_norm"], eps)
        if capture:
            acts[L] = x.numpy().astype(np.float16)

        # ---- SwiGLU FFN, streamed one tensor at a time --------------------
        Wg = src.get(p + "ffn_gate.weight")
        g = F.linear(x, Wg)
        del Wg
        g = F.silu(g)
        Wu = src.get(p + "ffn_up.weight")
        a = g * F.linear(x, Wu)
        del g, Wu
        if skip_tau is not None:
            e = a.reshape(T, nb, LANE).float().pow(2).sum(-1)          # [T,nb]
            idx = e.argsort(dim=-1, descending=True)[:, :n_skip]
            keep = torch.zeros_like(e, dtype=torch.bool)
            keep.scatter_(1, idx, True)
            frac = float(keep.float().mean())
            skip_stats.append({"layer": L, "kept_blocks": int(n_skip),
                               "kept_frac": frac})
            a = a * keep.repeat_interleave(LANE, dim=-1).to(a.dtype)
        Wd = src.get(p + "ffn_down.weight")
        h = h + F.linear(a, Wd)
        del Wd, a
        if L % 8 == 0 or L == nL - 1:
            log(f"  [fwd] layer {L:2d}/{nL} {c['layer_types'][L][:4]} "
                f"|h| {h.float().norm(dim=-1).mean():.3f} "
                f"{time.perf_counter() - t_start:5.1f}s")

    h = rms_norm(h, W["output_norm"], eps)
    return {"hidden": h, "acts": acts, "skip_stats": skip_stats,
            "seconds": time.perf_counter() - t_start, "layers": nL}


@torch.no_grad()
def _full_attn(x, src: TensorSource, cfg: dict, W: dict, p: str, L: int):
    """GQA with output gate and partial NeoX RoPE (same math as model_ref)."""
    c = cfg
    T = x.shape[0]
    nh, nkv, hd = c["heads"], c["kv_heads"], c["head_dim"]
    Wq = src.get(p + "attn_q.weight")
    qg = F.linear(x, Wq).reshape(T, nh, 2 * hd)
    del Wq
    q, gate = qg.chunk(2, dim=-1)
    Wk = src.get(p + "attn_k.weight")
    k = F.linear(x, Wk).reshape(T, nkv, hd)
    del Wk
    Wv = src.get(p + "attn_v.weight")
    v = F.linear(x, Wv).reshape(T, nkv, hd)
    del Wv
    q = rms_norm(q, W[p + "q_norm"], c["eps"])
    k = rms_norm(k, W[p + "k_norm"], c["eps"])
    q, k = _rope(q, k, c)
    k = k.repeat_interleave(nh // nkv, dim=1)
    v = v.repeat_interleave(nh // nkv, dim=1)
    o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1),
                                      v.transpose(0, 1), is_causal=T > 1)
    o = o.transpose(0, 1).reshape(T, nh * hd)
    o = o * torch.sigmoid(gate.reshape(T, nh * hd))
    Wo = src.get(p + "attn_output.weight")
    out = F.linear(o, Wo)
    del Wo
    return out


def _rope(q, k, cfg):
    """Partial NeoX half-split RoPE over the first rope_dim of each head."""
    T = q.shape[0]
    rd = cfg["rope_dim"]
    half = rd // 2
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    ang = torch.outer(torch.arange(T, dtype=torch.float32), inv)
    cos = torch.cos(ang)[:, None, :]
    sin = torch.sin(ang)[:, None, :]

    def rot(x):
        xr, xp = x[..., :rd], x[..., rd:]
        x1, x2 = xr[..., :half], xr[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, xp], dim=-1)

    return rot(q), rot(k)


@torch.no_grad()
def _linear_attn(x, src: TensorSource, cfg: dict, W: dict, p: str):
    """Gated DeltaNet: causal depthwise conv -> silu -> delta rule -> gated norm."""
    c = cfg
    T = x.shape[0]
    Hk, Hv, Dk, Dv = c["lin_k_heads"], c["lin_v_heads"], c["lin_k_dim"], c["lin_v_dim"]
    eps = c["eps"]

    Wqkv = src.get(p + "attn_qkv.weight")
    qkv = F.linear(x, Wqkv)
    del Wqkv
    # causal depthwise conv, kernel 4, groups = all channels
    cw = W[p + "ssm_conv"].reshape(-1, c["conv_k"])
    z_ = F.pad(qkv.transpose(0, 1)[None], (c["conv_k"] - 1, 0))
    qkv = F.conv1d(z_, cw[:, None, :], groups=qkv.shape[1]).transpose(1, 2)[0]
    qkv = F.silu(qkv)

    q = qkv[:, :Hk * Dk].reshape(T, Hk, Dk)
    k = qkv[:, Hk * Dk:2 * Hk * Dk].reshape(T, Hk, Dk)
    v = qkv[:, 2 * Hk * Dk:].reshape(T, Hv, Dv)

    Wz = src.get(p + "attn_gate.weight")
    z = F.linear(x, Wz).reshape(T, Hv, Dv)
    del Wz
    Wb = src.get(p + "ssm_beta.weight")
    b = F.linear(x, Wb)
    del Wb
    Wa = src.get(p + "ssm_alpha.weight")
    a = F.linear(x, Wa)
    del Wa

    g = F.softplus(a + W[p + "ssm_dt"]) * W[p + "ssm_a"]      # [T,Hv] <= 0
    beta = torch.sigmoid(b)

    q = l2norm(q)
    k = l2norm(k)
    rep = Hv // Hk
    q = q.repeat_interleave(rep, dim=1)
    k = k.repeat_interleave(rep, dim=1)

    scale = Dk ** -0.5
    S = torch.zeros(Hv, Dk, Dv, dtype=torch.float32)
    outs = torch.empty(T, Hv, Dv, dtype=torch.float32)
    for t in range(T):
        S = S * torch.exp(g[t])[:, None, None]
        v_new = v[t] - torch.einsum("hkv,hk->hv", S, k[t])
        S = S + beta[t][:, None, None] * k[t].unsqueeze(-1) * v_new.unsqueeze(-2)
        outs[t] = torch.einsum("hkv,hk->hv", S, q[t])
    o = gated_rms_norm(outs * scale, z, W[p + "ssm_norm"], eps)
    Wo = src.get(p + "ssm_out.weight")
    out = F.linear(o.reshape(T, Hv * Dv), Wo)
    del Wo
    return out


# --------------------------------------------------------------------------- #
# logits (tied embeddings, streamed over the 248k vocab)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def ppl_and_argmax(h: torch.Tensor, ids, src: TensorSource, cfg: dict,
                   chunk: int = 8192, log=print) -> dict:
    """Cross-entropy of ids[t+1] given h[t] without materialising [T,248320].

    Streams the tied embedding matrix in vocab chunks, keeping a running
    log-sum-exp per position.  Target logits are gathered by row instead.
    """
    V = cfg["vocab"]
    T = h.shape[0]
    ids = np.asarray(ids, dtype=np.int64)
    m = torch.full((T,), -torch.inf)
    s = torch.zeros(T)
    amax_val = torch.full((T,), -torch.inf)
    amax_id = torch.zeros(T, dtype=torch.long)
    t0 = time.perf_counter()
    for v0 in range(0, V, chunk):
        n = min(chunk, V - v0)
        Wv = src.rows("token_embd.weight", np.arange(v0, v0 + n))
        lg = h @ Wv.T                                       # [T,n]
        del Wv
        cm = lg.max(dim=-1)
        m_new = torch.maximum(m, cm.values)
        s = s * torch.exp(m - m_new) + torch.exp(lg - m_new[:, None]).sum(-1)
        m = m_new
        cmv, cmi = lg.max(dim=-1)
        upd = cmv > amax_val
        amax_val = torch.where(upd, cmv, amax_val)
        amax_id = torch.where(upd, cmi + v0, amax_id)
        if (v0 // chunk) % 8 == 0:
            log(f"  [logits] {v0 + n}/{V} {time.perf_counter() - t0:.1f}s")
    # target logits: gather only the rows we need
    E = src.rows("token_embd.weight", ids[1:])
    tgt = (h[:-1] * E).sum(-1)
    nll = (m[:-1] + torch.log(s[:-1]) - tgt)
    return {"nll": nll, "tokens": T - 1,
            "argmax_id": amax_id.tolist(), "argmax_val": amax_val.tolist(),
            "ppl": float(torch.exp(nll.mean())), "seconds": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_tokenizer():
    """The 4B GGUF ships BPE vocab+merges but not tokenizer.json; the Qwen3.5
    0.8B tokenizer.json is byte-identical over the 248070 shared ids (checked
    against tokenizer.ggml.tokens), and the 4B GGUF carries the same pre-tokenizer
    (tokenizer.ggml.pre = qwen35)."""
    import sys
    sys.path.insert(0, str(HERE))
    from common import Tokenizer
    return Tokenizer(ROOT / "models" / "Qwen3.5-0.8B")


def wikitext_ids(n_tokens: int, seed: int = 1234, offset_max: int | None = None):
    """Deterministic (seeded) contiguous slice of tokenised WikiText-2 test."""
    tok = build_tokenizer()
    rows = [json.loads(l) for l in (ROOT / "data/benchmarks/wikitext2_test.jsonl").open()]
    text = "\n\n".join(r["text"] for r in rows[:200])
    ids = tok(text, add_special_tokens=False)["input_ids"]
    rng = np.random.default_rng(seed)
    hi = offset_max if offset_max is not None else max(1, len(ids) // 2)
    off = int(rng.integers(0, hi))
    ids = ids[off:off + n_tokens]
    return ids, {"source": "data/benchmarks/wikitext2_test.jsonl(rows 0:200)",
                 "seed": seed, "offset": off, "n_tokens": len(ids)}


# --------------------------------------------------------------------------- #
# self-test: structural dequant checks against the ggml layout
# --------------------------------------------------------------------------- #
def self_test(src: TensorSource, log=print) -> dict:
    out = {}
    # Q6_K: every value must be d*sc*q with q an integer in [-32,31]
    raw = src._raw(src.tmap["token_embd.weight"], 210 * 400)
    nb = raw.size // 210
    raw = raw[:nb * 210].reshape(nb, 210)
    d = _fp16(raw[:, 208:210])
    sc = raw[:, 192:208].view(np.int8).astype(np.float32)
    y = dequant_q6_k(raw)
    worst, ok = 0.0, True
    for c in range(2):
        for k in range(4):
            # element group (c,k) -> sub-block order in the layout
            seg = y[:, c * 128 + k * 32:c * 128 + (k + 1) * 32]
            s = np.repeat(sc[:, c * 8 + 2 * k:c * 8 + 2 * k + 2], 16, axis=1)
            r = seg / (d[:, None] * s)
            worst = max(worst, float(np.abs(r - np.round(r)).max()))
            ok &= bool(np.all((np.round(r) >= -32) & (np.round(r) <= 31)))
    out["q6_k"] = {"blocks": int(nb), "max_dev_from_integer": worst,
                   "q_in_range": ok}
    # Q5_K: (y + m)/(d*sc) must be an integer in [0,31]
    raw = src._raw(src.tmap["blk.3.attn_v.weight"], 176 * 400)
    nb = raw.size // 176
    raw = raw[:nb * 176].reshape(nb, 176)
    y = dequant_q5_k(raw)
    d = _fp16(raw[:, 0:2])
    dmin = _fp16(raw[:, 2:4])
    sd, sm = _scale_min_k4(raw[:, 4:16])
    worst, ok = 0.0, True
    for j in range(4):
        for half in range(2):
            seg = y[:, 64 * j + 32 * half:64 * j + 32 * (half + 1)]
            dd = (d * sd[:, 2 * j + half])[:, None]
            mm = (dmin * sm[:, 2 * j + half])[:, None]
            r = (seg + mm) / dd
            worst = max(worst, float(np.abs(r - np.round(r)).max()))
            ok &= bool(np.all((np.round(r) >= 0) & (np.round(r) <= 31)))
    out["q5_k"] = {"blocks": int(nb), "max_dev_from_integer": worst,
                   "q_in_range": ok}
    # Q8_0: y/d must be an exact integer in [-127,127]
    raw = src._raw(src.tmap["blk.0.ssm_alpha.weight"], 34 * 200)
    nb = raw.size // 34
    raw = raw[:nb * 34].reshape(nb, 34)
    y = dequant_q8_0(raw)
    r = y / _fp16(raw[:, :2])[:, None]
    out["q8_0"] = {"blocks": int(nb), "max_dev_from_integer":
                   float(np.abs(r - np.round(r)).max()),
                   "q_in_range": bool(np.all(np.abs(np.round(r)) <= 127))}
    # IQ4_NL: already verified vs numpy in gguf_iq4nl.py; recheck the table here
    raw = src._raw(src.tmap["blk.0.ffn_gate.weight"], 18 * 200)
    nb = raw.size // 18
    y = dequant_iq4nl_np(raw[:nb * 18].reshape(nb, 18))
    d = _fp16(raw[:nb * 18].reshape(nb, 18)[:, :2])
    ratio = y / d[:, None]
    kv = np.asarray(KVALUES_IQ4NL, np.float32)
    dist = np.abs(ratio[..., None] - kv[None, None, :]).min(-1)
    out["iq4_nl"] = {"blocks": int(nb), "n_vals": int(y.size),
                     "max_dist_to_table": float(dist.max())}
    log(f"[self-test] {json.dumps(out)}")
    return out


def sha256_file(p: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ppl", action="store_true")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", default=str(ACTS_JSON))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    src = TensorSource()
    cfg = read_cfg(src.r)
    print(f"[cfg] {json.dumps({k: v for k, v in cfg.items() if k != 'layer_types'})}")
    print(f"[cfg] full_attention layers: "
          f"{[i for i, t in enumerate(cfg['layer_types']) if t == 'full_attention']}")
    res: dict = {"model": str(GGUF), "cfg": cfg, "seed": a.seed,
                 "threads": a.threads}
    if a.self_test:
        res["self_test"] = self_test(src)

    if a.ppl or a.capture:
        ids, meta = wikitext_ids(a.tokens + 1, a.seed)
        res["data"] = meta
        res["token_ids"] = [int(x) for x in ids]
        # --- "The capital of France is" --------------------------------
        tok = build_tokenizer()
        pr = tok("The capital of France is", add_special_tokens=False)["input_ids"]
        t0 = time.perf_counter()
        out = forward(pr, src, cfg, capture=False, layers=a.layers)
        pr_sec = time.perf_counter() - t0
        r = ppl_and_argmax(out["hidden"], np.asarray(pr) + [0], src, cfg,
                           chunk=a.chunk)
        top = r["argmax_id"][-1]
        res["prompt_check"] = {
            "prompt": "The capital of France is", "ids": pr,
            "argmax_id": int(top), "argmax_token": tok.decode([int(top)]),
            "ok_paris": tok.decode([int(top)]).strip().lower() == "paris",
            "top1_logit": float(r["argmax_val"][-1]), "seconds": pr_sec}
        print(f"[prompt] next token = {tok.decode([int(top)])!r} (id {top})")

        # --- WikiText-2 forward ---------------------------------------
        t0 = time.perf_counter()
        out = forward(ids, src, cfg, capture=a.capture, layers=a.layers)
        fwd_s = time.perf_counter() - t0
        finite = bool(torch.isfinite(out["hidden"]).all())
        res["forward"] = {"tokens": len(ids), "seconds": fwd_s,
                          "finite_hidden": finite,
                          "hidden_absmax": float(out["hidden"].abs().max()),
                          "src_perf": src.perf, "bytes_read": src.bytes_read}
        print(f"[fwd] {len(ids)} tokens in {fwd_s:.1f}s "
              f"(dequant {src.perf['dequant_s']:.1f}s read {src.perf['read_s']:.1f}s)")
        if a.ppl:
            r = ppl_and_argmax(out["hidden"], ids, src, cfg, chunk=a.chunk)
            res["ppl"] = {"ppl": r["ppl"], "tokens": r["tokens"],
                          "seconds": r["seconds"],
                          "logits_finite": bool(np.isfinite(r["nll"].numpy()).all()),
                          "nll_per_token": float(r["nll"].mean())}
            print(f"[ppl] {r['ppl']:.3f} over {r['tokens']} tokens")
        if a.capture and out["acts"] is not None:
            acts = out["acts"]
            ACTS_NPZ.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ACTS_NPZ, acts=acts,
                                ids=np.asarray(ids, dtype=np.int32))
            sha = sha256_file(ACTS_NPZ)
            side = {"npz": str(ACTS_NPZ), "sha256": sha, "shape": list(acts.shape),
                    "dtype": str(acts.dtype), "token_ids": [int(x) for x in ids],
                    "data": meta, "seed": a.seed}
            ACTS_JSON.write_text(json.dumps(side, indent=2))
            res["acts"] = {"npz": str(ACTS_NPZ), "sha256": sha,
                           "shape": list(acts.shape), "mb": ACTS_NPZ.stat().st_size / 1e6}
            print(f"[acts] {acts.shape} -> {ACTS_NPZ} ({ACTS_NPZ.stat().st_size/1e6:.1f} MB) sha {sha[:16]}")
    src.close()
    if len(res) > 4:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
