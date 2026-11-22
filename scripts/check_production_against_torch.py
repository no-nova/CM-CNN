"""Independent end-to-end check: C++ AVX-512 engine vs the PyTorch model it was exported from.

For every exported model in experiments/cm-cnn-production/export_report.json this script
  1. runs the committed C++ binary in M0 (dense) mode over the test set and dumps its logits,
  2. runs the official torchvision model (same weights, same ImageNet normalisation) over the
     exact same pixels,
  3. reports max |Δlogit|, mean |Δlogit| and the top-1 agreement.

This is the check that catches kernel-level bugs the NumPy exporter check cannot see (e.g. a
5x5/7x7 kernel that is only addressed correctly for k = 3). It only reads files, it never
touches weights, and it writes experiments/cm-cnn-production/torch_agreement.json.
"""
import json, struct, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "experiments" / "cm-cnn-production"
BIN = DIR / "cm_production_avx512"
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def load_cm_images(path):
    raw = Path(path).read_bytes()
    assert raw[:8] == b"CMRGB001", f"bad magic in {path}"
    cnt, h, w, c = struct.unpack_from("<4I", raw, 8)
    off = 32 + cnt
    while off % 4:
        off += 1
    px = np.frombuffer(raw[off:off + cnt * h * w * c * 4], "<f4").reshape(cnt, h, w, c)
    return px.astype(np.float32)


def torch_model(name):
    from torchvision import models as M
    table = {
        "resnet18": (M.resnet18, M.ResNet18_Weights.DEFAULT),
        "resnet50": (M.resnet50, M.ResNet50_Weights.DEFAULT),
        "mobilenet_v2": (M.mobilenet_v2, M.MobileNet_V2_Weights.DEFAULT),
        "mobilenetv3_small": (M.mobilenet_v3_small, M.MobileNet_V3_Small_Weights.DEFAULT),
        "shufflenetv2_x0_5": (M.shufflenet_v2_x0_5, M.ShuffleNet_V2_X0_5_Weights.DEFAULT),
        # the exported 320 model is the MobileNetV3-Large conv trunk, not the SSDLite detector
        "mbv3_large_320": (M.mobilenet_v3_large, M.MobileNet_V3_Large_Weights.DEFAULT),
    }
    fn, w = table[name]
    return fn(weights=w).eval()


def main():
    rep = json.loads((DIR / "export_report.json").read_text())
    out = {}
    for name in sorted(rep):
        size = rep[name]["size"]
        test = DIR / "dataset" / f"test_{size}.bin"
        X = load_cm_images(test)[:32].transpose(0, 3, 1, 2)      # NHWC -> NCHW
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            dpath = tf.name
        subprocess.run([str(BIN), "--model", name, "--weights", str(DIR / f"{name}_{size}.bin"),
                        "--test", str(test), "--reps", "1", "--dump-logits", dpath],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raw = Path(dpath).read_bytes()
        n, cls = struct.unpack_from("<2i", raw, 0)
        cpp = np.frombuffer(raw[8:], "<f4").reshape(n, cls).astype(np.float64)
        with torch.no_grad():
            ref = torch_model(name)(torch.from_numpy((X - MEAN) / STD)).numpy().astype(np.float64)
        k = min(len(cpp), len(ref))
        d = np.abs(cpp[:k] - ref[:k])
        agree = float((cpp[:k].argmax(1) == ref[:k].argmax(1)).mean())
        out[name] = dict(images=int(k), max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
                         top1_agreement=agree, source=f"dataset/test_{size}.bin[0:{k}]",
                         engine="cm_production_avx512 M0 (dense), AVX-512, 1 thread")
        print(f"{name}: n={k} max|dlogit|={d.max():.3e} mean={d.mean():.3e} top1={agree:.4f}")
        assert agree == 1.0
        Path(dpath).unlink()
    p = DIR / "torch_agreement.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    sys.exit(main())
