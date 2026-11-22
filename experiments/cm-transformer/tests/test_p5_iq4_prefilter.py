"""Phase 5: IQ4_NL gate pre-filter and block-major kernel correctness.

Skipped when the GGUF model or the compiled shared object is missing, so the
suite still runs on a machine without the 2.6 GB quantised checkpoint.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[1]
GGUF = ROOT / "models" / "Qwen3.5-4B-IQ4_NL.gguf"
LIB = HERE / "results" / "build" / "iq4nl_bm_v2.so"

pytestmark = pytest.mark.skipif(
    not (GGUF.exists() and LIB.exists()),
    reason="needs models/Qwen3.5-4B-IQ4_NL.gguf and results/build/iq4nl_bm_v2.so",
)

NB, BLK, H, M = 288, 18, 9216, 2560
KTAB = np.array([-127, -104, -83, -65, -49, -35, -22, -10,
                 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


@pytest.fixture(scope="module")
def env():
    import sys
    sys.path.insert(0, str(HERE))
    import bench_p5_iq4 as B
    lib = B.load_lib()
    tt = B.Tensors(str(GGUF))
    raw, _ = tt.raw("blk.0.ffn_gate.weight")
    gate = np.ascontiguousarray(raw)          # 32 consecutive rows are contiguous
    x = (np.random.RandomState(3).randn(2560) * 0.02).astype(np.float32)
    g = np.zeros(NB * 32, np.float32)
    idx = np.zeros(NB, np.int32)
    kp = np.zeros(1, np.int32)
    return dict(B=B, lib=lib, gate=gate, x=x, g=g, idx=idx, kp=kp)


def _decode_row(raw, x, row_start):
    """Reference decode of one block-major row: 80 blocks of 18 bytes."""
    acc = 0.0
    for k in range(80):
        b = raw[row_start + k * 18: row_start + (k + 1) * 18]
        d = float(np.frombuffer(b[0:2].tobytes(), dtype=np.float16)[0])
        nib = np.frombuffer(b[2:].tobytes(), dtype=np.uint8)
        vals = np.concatenate([KTAB[nib & 0xF], KTAB[nib >> 4]]) * d
        acc += float(np.dot(vals, x[k * 32:(k + 1) * 32]))
    return acc


def test_prefilter_matches_reference_decode(env):
    B, lib, gate, x = env["B"], env["lib"], env["gate"], env["x"]
    g, idx, kp = env["g"], env["idx"], env["kp"]
    lib.gate_prefilter_blockmajor(
        gate.ctypes.data_as(B.u8p), x.ctypes.data_as(B.f32p),
        g.ctypes.data_as(B.f32p), idx.ctypes.data_as(B.i32p),
        kp.ctypes.data_as(B.i32p), NB, 2560, 32, 0.0, None, 4)
    assert int(kp[0]) == NB
    for row in (0, 37, 9215):
        ref = _decode_row(gate, x, row * 80 * 18)
        assert abs(ref - g[row]) <= 1e-5 * max(1.0, abs(ref)), (row, ref, g[row])


def test_prefilter_keeps_requested_number_of_sorted_unique_blocks(env):
    B, lib, gate, x = env["B"], env["lib"], env["gate"], env["x"]
    g, idx, kp = env["g"], env["idx"], env["kp"]
    for rate in (0.2, 0.4, 0.6):
        lib.gate_prefilter_blockmajor(
            gate.ctypes.data_as(B.u8p), x.ctypes.data_as(B.f32p),
            g.ctypes.data_as(B.f32p), idx.ctypes.data_as(B.i32p),
            kp.ctypes.data_as(B.i32p), NB, 2560, 2, rate, None, 4)
        K = int(kp[0])
        assert K == round(NB * (1 - rate))
        kept = idx[:K]
        assert np.all(np.diff(kept) > 0), "idx must be ascending for the up/down kernels"
        assert kept.min() >= 0 and kept.max() < NB


def test_prefilter_energy_ordering_beats_random_selection(env):
    """Partial-row ranking must beat a random equal-size selection.

    Honest scale of the effect on a synthetic activation (near-uniform block
    energies): r=2 keeps 1.05x, r=8 1.07x, r=16 1.10x of the energy that random
    selection keeps.  The r=2 test therefore only asserts "not worse than
    random", while r=16 must be clearly better.
    """
    B, lib, gate, x = env["B"], env["lib"], env["gate"], env["x"]
    g, idx, kp = env["g"], env["idx"], env["kp"]
    lib.gate_prefilter_blockmajor(
        gate.ctypes.data_as(B.u8p), x.ctypes.data_as(B.f32p),
        g.ctypes.data_as(B.f32p), idx.ctypes.data_as(B.i32p),
        kp.ctypes.data_as(B.i32p), NB, 2560, 32, 0.0, None, 4)   # dense energies
    full = (g.reshape(NB, 32) ** 2).sum(1)
    lib.gate_prefilter_blockmajor(
        gate.ctypes.data_as(B.u8p), x.ctypes.data_as(B.f32p),
        g.ctypes.data_as(B.f32p), idx.ctypes.data_as(B.i32p),
        kp.ctypes.data_as(B.i32p), NB, 2560, 2, 0.4, None, 4)
    kept = idx[:int(kp[0])]
    rng = np.random.RandomState(0)
    rand = rng.choice(NB, len(kept), replace=False)
    assert full[kept].sum() >= 0.99 * full[rand].sum()

    lib.gate_prefilter_blockmajor(
        gate.ctypes.data_as(B.u8p), x.ctypes.data_as(B.f32p),
        g.ctypes.data_as(B.f32p), idx.ctypes.data_as(B.i32p),
        kp.ctypes.data_as(B.i32p), NB, 2560, 16, 0.4, None, 4)
    kept16 = idx[:int(kp[0])]
    assert full[kept16].sum() > 1.05 * full[rand].sum()
