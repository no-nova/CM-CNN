"""Assemble the unified end-to-end table (Phase 9, Q1/Q7/Q8) from measured files.

Path A (per-layer Python decision + masked GEMM) was measured in `p8_system.json`,
path B (fused extension, pre-allocated buffers) in `p1_kernels_v2.json`, and the
dense reference in both. This script only *adds* the one missing number, the cost
of the decision path itself, measured directly on the calibration activations; it
does not re-time the model.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
os.environ.setdefault("OMP_NUM_THREADS", "16")

import torch  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "p9", EXP / "scripts" / "p9_round2_closure.py")
p9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p9)

RES = EXP / "results"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"prompt_tokens": 128, "generated": 32, "seed": 1234, "paths": {},
           "amortization": {}, "assembled_from": ["p8_system.json", "p1_kernels_v2.json"],
           "notes": {"pathA": "per-layer python decision, torch mask, masked GEMM (p8_system)",
                     "pathB": "fused CUDA extension, buffers preallocated (p1_kernels_v2)",
                     "pathC": "AVX-512 row-parallel, 16 threads (p1_kernels_v2)"}}

    prev = json.loads((RES / "p9_e2e.json").read_text()) if (RES / "p9_e2e.json").exists() else {}
    if prev.get("dispatch"):
        disp = prev["dispatch"]
        print("reusing dispatch:", disp, flush=True)
    else:
        runner = p9.P6.make_runner("qwen3.5-0.8b", dev, torch.float32)
        acts = p9.capture(runner, p9.calib_texts())
        perm = p9.perms_for(runner, acts, kinds=("ours",))["ours"]
        disp = {f"{r:.2f}": round(p9.dispatch_cost_ms(runner.tb, acts, perm, r), 3)
                for r in (0.20, 0.40)}
        print("dispatch ms/token:", disp, flush=True)

    sysj = json.loads((RES / "p8_system.json").read_text())["ladder"]["qwen3.5-0.8b"]
    kern = json.loads((RES / "p1_kernels_v2.json").read_text())
    g = kern["e2e_gpu"]

    def row(ms, tok_s=None):
        return {"decode_ms_per_token": round(ms, 3),
                "tokens_per_s": round(tok_s if tok_s else 1000.0 / ms, 1)}

    paths = {"qwen3.5-0.8b": {
        "layers": 24,
        "dense": row(g["dense"]["decode_ms_per_tok"], g["dense"]["decode_tok_s"]),
        "dispatch_ms_per_token": disp["0.20"],
        "pathA_tau0.20": row(sysj["tau0.20"]["end_to_end_ms"]),
        "pathA_tau0.40": row(sysj["tau0.40"]["end_to_end_ms"]),
        "pathB_tau0.20": row(g["skip_0.20"]["decode_ms_per_tok"]),
        "pathB_tau0.40": row(g["skip_0.40"]["decode_ms_per_tok"]),
        "pathB_tau0.00": row(g.get("kernel_dense", {}).get("decode_ms_per_tok",
                                                          g["dense"]["decode_ms_per_tok"])),
    }}
    for k in ("skip_0.60", "skip_0.75"):
        if k in g:
            paths["qwen3.5-0.8b"][f"pathB_tau{k.split('_')[1]}"] = row(g[k]["decode_ms_per_tok"])
    cpu_file = RES / "p1_kernels_v2cpu.json"
    c = (json.loads(cpu_file.read_text())["e2e_cpu"] if cpu_file.exists()
         else kern["e2e_cpu"])
    paths["qwen3.5-0.8b"]["pathC_dense_ms_per_token"] = round(c["dense"]["decode_ms_per_tok"], 2)
    for k in ("skip_0.20", "skip_0.40", "skip_0.60", "skip_0.75"):
        if k in c:
            paths["qwen3.5-0.8b"][f"pathC_tau{k.split('_')[1]}"] = row(c[k]["decode_ms_per_tok"])

    out["paths"] = paths
    out["dispatch"] = disp
    (RES / "p9_e2e.json").write_text(json.dumps(out, indent=1))
    print("wrote", RES / "p9_e2e.json", flush=True)


if __name__ == "__main__":
    main()
