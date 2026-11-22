"""Phase-7 / Task 2: Whisper encoder FFN-only cost (fc1+fc2), dense vs 32-lane skip.

Isolating the FFN from the 192.8 ms encoder of whisper-base (of which the FFN is
12.3 ms) answers the question the end-to-end table could not: did the gated path
lose time in the filter, or in the operator?

Activations: the real input of encoder layer 0 for 16 LibriSpeech utterances
(Whisper pads every clip to 1500 frames, so the shape is 1500x512 per utterance),
propagated through the twelve FFN blocks of the checkpoint as a residual stream
`x <- x + fc2(gelu(fc1(x)))`.  Attention is not re-applied, which keeps the
measurement focused on the operator while the statistics stay those of real
speech features.

Timing: torch BLAS (16 threads) for the dense FFN and for the in-PyTorch masked
FFN (top-k 32-lane blocks per token), plus the AVX-512 compacted-list kernel on
one `fc2` to show the C++ path without any Python-side mask.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT / "models" / "whisper-base"
AUDIO = ROOT / "data" / "p6_whisper_audio"
OUT = HERE / "results" / "p6_whisper.json"
SO = HERE / "results" / "build" / "lanemlp_avx512_v2.so"
LANE = 32

os.environ.setdefault("OMP_WAIT_POLICY", "passive")
torch.set_num_threads(16)


def load_kernel():
    if not SO.exists():
        return None
    lib = ctypes.CDLL(str(SO))
    fp = ctypes.POINTER(ctypes.c_float)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    lib.down_skip_avx512_auto2.argtypes = [fp, fp, u8, fp, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int]
    return lib


def read_wav(p: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(p), "-f", "f32le",
                          "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def main():
    from transformers import WhisperFeatureExtractor, WhisperModel

    m = WhisperModel.from_pretrained(str(MODEL)).eval()
    enc = m.encoder
    fe = WhisperFeatureExtractor.from_pretrained(str(MODEL))
    wavs = sorted(AUDIO.rglob("*.flac"))[:16]
    if not wavs:
        raise SystemExit(f"no audio under {AUDIO}")

    # real layer-0 inputs
    got = []
    h = enc.layers[0].register_forward_hook(
        lambda mod, inp, out: got.append(inp[0].detach()))
    with torch.no_grad():
        for p in wavs:
            feats = fe(read_wav(p), sampling_rate=16000, return_tensors="pt").input_features
            enc(feats)
    h.remove()
    X0 = torch.cat([g.reshape(-1, g.shape[-1]) for g in got], 0)[:4096]
    print(f"[capture] {len(wavs)} utts -> {tuple(X0.shape)} real layer-0 inputs",
          flush=True)

    blocks = [(b.fc1.weight, b.fc1.bias, b.fc2.weight, b.fc2.bias)
              for b in enc.layers]
    lib = load_kernel()
    rows, last = {}, None
    for tau in (0.0, 0.2, 0.4, 0.6):
        t_dense = t_mask = 0.0
        x = X0.clone()
        with torch.no_grad():
            for (w1, b1, w2, b2) in blocks:
                t0 = time.perf_counter()
                hd = torch.nn.functional.gelu(torch.nn.functional.linear(x, w1, b1))
                yd = torch.nn.functional.linear(hd, w2, b2)
                t_dense += time.perf_counter() - t0

                t0 = time.perf_counter()
                hk = torch.nn.functional.gelu(torch.nn.functional.linear(x, w1, b1))
                B = hk.shape[-1] // LANE
                e = (hk.reshape(-1, B, LANE) ** 2).sum(-1)
                k = max(1, int(round(B * (1 - tau))))
                thr = e.kthvalue(B - k + 1, dim=-1, keepdim=True).values
                hk = hk * (e >= thr).repeat_interleave(LANE, -1).to(hk.dtype)
                yk = torch.nn.functional.linear(hk, w2, b2)
                t_mask += time.perf_counter() - t0
                x = x + (yd if tau == 0.0 else yk)
        rel = float((yd - yk).norm() / yd.norm())
        rows[f"{tau:.2f}"] = {"dense_ms": t_dense * 1e3, "mask_ms": t_mask * 1e3,
                              "rel_l2": rel, "speedup": t_dense / max(t_mask, 1e-9),
                              "tokens": int(X0.shape[0])}
        print(f"[tau {tau:.2f}] ffn-only dense {t_dense*1e3:7.1f} ms  "
              f"masked {t_mask*1e3:7.1f} ms  relL2 {rel:.4f}  "
              f"{t_dense/max(t_mask,1e-9):.2f}x", flush=True)

    kern = {}
    if lib is not None:
        w1, b1, w2, b2 = blocks[-1]
        with torch.no_grad():
            x1 = X0[:1]
            h1 = torch.nn.functional.gelu(torch.nn.functional.linear(x1, w1, b1))[0]
            ref = torch.nn.functional.linear(h1, w2).numpy()
            e = (h1.reshape(-1, LANE) ** 2).sum(-1)
            hn = h1.numpy()
            wn = w2.detach().numpy()
            out = np.zeros(wn.shape[0], np.float32)
            fp = ctypes.POINTER(ctypes.c_float)
            for tau in (0.0, 0.2, 0.4, 0.6):
                k = e.numel() if tau == 0 else max(1, int(round(e.numel() * (1 - tau))))
                keep = torch.zeros(1, e.numel(), dtype=torch.bool)
                keep[0, torch.topk(e, k).indices] = True
                mask = keep.numpy().astype(np.uint8)
                reps = 20
                t0 = time.perf_counter()
                for _ in range(reps):
                    lib.down_skip_avx512_auto2(hn.ctypes.data_as(fp),
                                               wn.ctypes.data_as(fp),
                                               mask.ctypes.data_as(
                                                   ctypes.POINTER(ctypes.c_uint8)),
                                               out.ctypes.data_as(fp), 1,
                                               wn.shape[1], wn.shape[0], 16)
                ms = (time.perf_counter() - t0) / reps * 1e3
                t0 = time.perf_counter()
                for _ in range(reps):
                    torch.nn.functional.linear(h1, w2)
                tms = (time.perf_counter() - t0) / reps * 1e3
                kern[f"{tau:.2f}"] = {"torch_ms": tms, "kernel_ms": ms,
                                      "speedup": tms / max(ms, 1e-9),
                                      "max_abs_err": float(np.abs(out - ref).max())}
                print(f"[fc2 kernel tau {tau:.2f}] torch {tms:.4f} ms  "
                      f"kernel {ms:.4f} ms  {tms/max(ms,1e-9):.2f}x  "
                      f"max|d| {kern[f'{tau:.2f}']['max_abs_err']:.2e}", flush=True)

    doc = json.loads(OUT.read_text()) if OUT.exists() else {}
    doc["ffn_only"] = {
        "utts": len(wavs), "threads": 16, "d_model": 512, "d_ff": 2048,
        "note": "12 encoder layers, fc1+fc2 only, real layer-0 speech features "
                "propagated as a residual stream (attention excluded)",
        "rows": rows, "fc2_kernel": kern}
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"[done] merged into {OUT}", flush=True)


if __name__ == "__main__":
    main()
