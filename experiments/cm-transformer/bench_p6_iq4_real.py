"""Phase-7 / Task 1: Block-Major IQ4_NL measured on real text activations.

Real data path
--------------
token ids  : WikiText-2 test text encoded with the Qwen3.5 tokenizer shipped in
             this repository (same vocabulary the 4B GGUF uses).
layer input: the checkpoint's own `token_embd` row for that token, RMSNormed
             (`blk.0.attn_norm`) = the exact layer-0 input of the model.
per layer  : activations are propagated through the real 4B FFN weights as a
             residual stream, x <- x + ffn(x), using the real IQ4_NL bytes of
             `ffn_gate`/`ffn_up`/`ffn_down` of that layer.  Attention is not
             applied (the GGUF attention path is not implemented in this
             harness), so the activations are "real text inputs propagated by
             the real FFN weights", not a full forward pass.  Stated as such in
             the report.

Per layer it reports, for tau in {0.2,0.4,0.6,0.8}:
  * E_captured  - share of the dense FFN output energy kept,
  * rel_l2      - ||y - y_keep|| / ||y||,
  * cos         - cosine(y, y_keep),
  * bytes / ms  - block-major bytes touched and wall-clock time for the whole
                  gate+up+down FFN when only the kept blocks are read.

Progress is printed after every layer (the harness watchdog kills silent runs).
Budget: ~10-20 s for 32 layers at N=32 tokens.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench_p5_iq4 import GGUF, IQ4_NL_BLOCK_BYTES, Tensors, f32p, i32p, load_lib, u8p

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results" / "p6_iq4_real.json"
D_MODEL, H_INTER, NB, BLK = 2560, 9216, 288, IQ4_NL_BLOCK_BYTES


def q6k_rows(buf: np.ndarray, d_row: int, rows=None) -> np.ndarray:
    """Decode Q6_K rows (210-byte blocks of 256 weights) to float32.

    Layout per block: uint8 ql[128]; uint8 qh[64]; int8 scales[16]; fp16 d.
    """
    n_rows = buf.size // ((d_row // 256) * 210)
    rows = np.arange(n_rows) if rows is None else np.asarray(rows)
    nblk = d_row // 256
    out = np.empty((rows.size, d_row), dtype=np.float32)
    blk = buf.reshape(n_rows, nblk, 210)
    for i, rid in enumerate(rows):
        for b in range(nblk):
            q = blk[rid, b]
            ql = q[0:128].astype(np.int32)
            qh = q[128:192].astype(np.int32)
            sc = q[192:208].view(np.int8).astype(np.float32)
            d = np.frombuffer(q[208:210].tobytes(), dtype=np.float16)[0].astype(np.float32)
            y = np.empty(256, dtype=np.float32)
            for n in range(2):
                o, oh = n * 64, n * 32           # ql += 64, qh += 32 per 128 weights
                y0 = n * 128
                l = np.arange(32)
                q1 = ((ql[o + l] & 0xF) | (((qh[oh + l] >> 0) & 3) << 4)) - 32
                q2 = ((ql[o + l + 32] & 0xF) | (((qh[oh + l] >> 2) & 3) << 4)) - 32
                q3 = ((ql[o + l] >> 4) | (((qh[oh + l] >> 4) & 3) << 4)) - 32
                q4 = ((ql[o + l + 32] >> 4) | (((qh[oh + l] >> 6) & 3) << 4)) - 32
                s8 = sc[n * 8: n * 8 + 8]
                y[y0 + l] = d * s8[0] * q1
                y[y0 + l + 32] = d * s8[2] * q2
                y[y0 + l + 64] = d * s8[4] * q3
                y[y0 + l + 96] = d * s8[6] * q4
            out[i, b * 256:(b + 1) * 256] = y
    return out


def silu(v):
    return v / (1.0 + np.exp(-np.clip(v, -60.0, 60.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--taus", default="0.2,0.4,0.6,0.8")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",")]
    t0 = time.perf_counter()

    lib = load_lib()
    tt = Tensors(GGUF)
    r = tt.r
    print(f"[gguf] {Path(GGUF).name}: {r.kv['qwen35.block_count']} layers, "
          f"d_model {r.kv['qwen35.embedding_length']}, "
          f"d_ff {r.kv['qwen35.feed_forward_length']}", flush=True)

    # ---- real token ids ---------------------------------------------------- #
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / "models" / "Qwen3.5-0.8B" / "tokenizer.json"))
    text = "".join(json.loads(l)["text"] for l in
                   open(ROOT / "data" / "benchmarks" / "wikitext2_test.jsonl"))
    ids = tok.encode(text).ids[: a.tokens]
    print(f"[data] {len(ids)} real WikiText-2 token ids", flush=True)

    # ---- real layer-0 input ------------------------------------------------ #
    fh = open(GGUF, "rb")
    w = tt.tmap["token_embd.weight"]
    d_row = int(w["dims"][0])          # ggml order: ne0 = d_model, ne1 = vocab
    rbs = d_row // 256 * 210

    def rd(off, n):
        fh.seek(r.data_start + off)
        return fh.read(n)

    if w["type"] in (0, "F32"):
        X = np.stack([np.frombuffer(rd(w["offset"] + i * d_row * 4, d_row * 4),
                                    dtype=np.float32) for i in ids])
    elif w["type"] in (14, "Q6_K"):
        X = np.stack([q6k_rows(np.frombuffer(rd(w["offset"] + i * rbs, rbs),
                                             dtype=np.uint8), d_row)[0] for i in ids])
    else:
        raise SystemExit(f"token_embd type {w['type']} not supported")
    nw = tt.tmap["blk.0.attn_norm.weight"]
    nr = np.frombuffer(rd(nw["offset"], d_row * 4), dtype=np.float32)
    X = X / np.sqrt((X ** 2).mean(-1, keepdims=True) + 1e-6) * (1.0 + nr)
    X = X.astype(np.float32)
    print(f"[act] layer-0 input: |x| mean {np.abs(X).mean():.4f} "
          f"max {np.abs(X).max():.3f}  (N={X.shape[0]})", flush=True)

    # ---- per-layer buffers ------------------------------------------------- #
    g = np.zeros(H_INTER, np.float32)
    u = np.zeros(H_INTER, np.float32)
    act = np.zeros(H_INTER, np.float32)
    y = np.zeros(D_MODEL, np.float32)
    yk = np.zeros(D_MODEL, np.float32)
    idx = np.zeros(NB, np.int32)
    kp = np.zeros(1, np.int32)
    all_idx = np.arange(NB, dtype=np.int32)
    ptr = {k: v.ctypes.data_as(p) for k, v, p in (
        ("g", g, f32p), ("u", u, f32p), ("act", act, f32p), ("y", y, f32p),
        ("yk", yk, f32p), ("idx", idx, i32p), ("kp", kp, i32p),
        ("all", all_idx, i32p))}

    rows, keep_share = [], {t: [] for t in taus}
    tot_dense_ms = tot_keep_ms = {t: 0.0 for t in taus}
    for L in range(a.layers):
        Graw, _ = tt.raw(f"blk.{L}.ffn_gate.weight")
        Uraw, _ = tt.raw(f"blk.{L}.ffn_up.weight")
        Draw, _ = tt.raw(f"blk.{L}.ffn_down.weight")
        Dbm = np.empty(NB * D_MODEL * BLK, dtype=np.uint8)
        lib.repack_blockmajor(Draw.ctypes.data_as(u8p), Dbm.ctypes.data_as(u8p),
                              D_MODEL, NB, BLK)
        Gp, Up, Dp = (Graw.ctypes.data_as(u8p), Uraw.ctypes.data_as(u8p),
                      Dbm.ctypes.data_as(u8p))

        x = X.mean(0).astype(np.float32)              # representative token
        t1 = time.perf_counter()
        lib.up_rows_blockmajor(Gp, x.ctypes.data_as(f32p), ptr["g"], H_INTER, NB,
                               D_MODEL, ptr["all"], NB, a.threads)
        lib.up_rows_blockmajor(Up, x.ctypes.data_as(f32p), ptr["u"], H_INTER, NB,
                               D_MODEL, ptr["all"], NB, a.threads)
        np.multiply(silu(g), u, out=act)
        lib.gemv_blockmajor8_mt(Dp, ptr["act"], ptr["y"], D_MODEL, NB, BLK,
                                ptr["all"], NB, a.threads)
        dense_ms = (time.perf_counter() - t1) * 1e3

        e = (act.reshape(NB, 32) ** 2).sum(1)
        order = np.argsort(-e, kind="stable")
        rec = {"layer": L, "dense_ms": dense_ms, "tau": {}}
        for tau in taus:
            K = max(1, int(round(NB * (1.0 - tau))))
            idx[:K] = np.sort(order[:K])
            t3 = time.perf_counter()
            lib.up_rows_blockmajor(Gp, x.ctypes.data_as(f32p), ptr["g"], H_INTER,
                                   NB, D_MODEL, ptr["all"], NB, a.threads)
            lib.up_rows_blockmajor(Up, x.ctypes.data_as(f32p), ptr["u"], H_INTER,
                                   NB, D_MODEL, ptr["idx"], K, a.threads)
            # the down kernel indexes activations by *block position*
            # (a + idx[k]*32), so scatter the packed up-rows back through idx.
            sel = (idx[:K, None] * 32 + np.arange(32)[None, :]).ravel()
            act.fill(0.0)
            act[sel] = silu(g[sel]) * u[: K * 32]
            t2 = time.perf_counter()
            lib.gemv_blockmajor8_mt(Dp, ptr["act"], ptr["yk"], D_MODEL, NB,
                                    BLK, ptr["idx"], K, a.threads)
            keep_ms = (time.perf_counter() - t3) * 1e3
            dy = y - yk
            rel = float(np.linalg.norm(dy) / max(np.linalg.norm(y), 1e-12))
            cos = float(np.dot(y, yk) / max(np.linalg.norm(y) * np.linalg.norm(yk), 1e-12))
            ecap = 1.0 - float(np.dot(dy, dy) / max(np.dot(y, y), 1e-12))
            rec["tau"][f"{tau:.2f}"] = {"kept_blocks": K, "rel_l2": rel, "cos": cos,
                                        "e_captured": ecap, "ffn_ms": keep_ms,
                                        "ffn_speedup": dense_ms / max(keep_ms, 1e-9),
                                        "bytes_mb": (K * 32 * (D_MODEL // 32) * BLK
                                        + NB * 32 * (D_MODEL // 32) * BLK) / 1e6}
            keep_share[tau].append(1.0 - tau)
        rows.append(rec)
        el = time.perf_counter() - t0
        print(f"[layer {L + 1}/{a.layers}] |x| {np.linalg.norm(x):.2f} "
              f"|y| {np.linalg.norm(y):.3f} dense {dense_ms:.2f} ms "
              f"rel_l2@0.2 {rec['tau']['0.20']['rel_l2']:.3f} "
              f"elapsed {el:.1f}s", flush=True)

    def agg(key):
        return float(np.mean([r_["tau"][f"{t:.2f}"][key] for r_ in rows]))

    summary = {f"{t:.2f}": {k: float(np.mean([r_["tau"][f"{t:.2f}"][k] for r_ in rows]))
                            for k in ("rel_l2", "cos", "e_captured", "ffn_ms",
                                      "bytes_mb", "kept_blocks")}
               for t in taus}
    out = {"model": "Qwen3.5-4B-IQ4_NL.gguf", "tokens": int(X.shape[0]),
           "layers": len(rows), "threads": a.threads,
           "activations": "real WikiText-2 token embeddings, RMSNormed, propagated "
                          "through the real FFN residual stream (no attention)",
           "dense_ms_mean": float(np.mean([r_["dense_ms"] for r_ in rows])),
           "summary": summary, "per_layer": rows,
           "wall_s": time.perf_counter() - t0}
    Path(a.out).parent.mkdir(exist_ok=True, parents=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[done] {out['wall_s']:.1f}s -> {a.out}", flush=True)
    for t in taus:
        s = summary[f"{t:.2f}"]
        print(f"  tau {t:.2f}: E_cap {s['e_captured']:.4f} relL2 {s['rel_l2']:.3f} "
              f"cos {s['cos']:.4f} ffn {s['ffn_ms']:.2f} ms "
              f"({out['dense_ms_mean'] / max(s['ffn_ms'], 1e-9):.2f}x) "
              f"{s['bytes_mb']:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
