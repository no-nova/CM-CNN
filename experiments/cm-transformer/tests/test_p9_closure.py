"""Phase 9 (round-2 review closure) invariants.

These tests are cheap and do not need a GPU: they check the arithmetic helpers,
the leak check, the cache-line census and the consistency of whatever result files
are present. The heavy experiments live in ``scripts/p9_round2_closure.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "p9", EXP / "scripts" / "p9_round2_closure.py")
p9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p9)

RES = EXP / "results"

PAPER_TYP = EXP.parents[1] / "paper2_bundle" / "CM_Transform_Paper2.typ"
PAPER_SKIP = pytest.mark.skipif(not PAPER_TYP.exists(),
                                reason="manuscript source not present")


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- #
def test_bootstrap_brackets_the_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 200)
    ci = p9.boot(x)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert abs(ci["mean"] - x.mean()) < 5e-4


def test_paired_interval_is_centred_on_the_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.1, 100)
    b = a + 0.5
    ci = p9.paired(a, b)
    assert abs(abs(ci["mean"]) - 0.5) < 5e-4   # paired() returns a - b
    assert ci["lo"] <= ci["mean"] <= ci["hi"]


def test_leak_check_accepts_disjoint_streams(monkeypatch):
    import torch
    monkeypatch.setattr(p9, "CAL_ROWS", (0, 120))
    monkeypatch.setattr(p9, "EVAL_ROWS", (300, 420))
    cal = torch.arange(1000)
    ev = torch.arange(5000, 6000)
    out = p9.leak_check(cal, ev, probe=64)
    assert out["row_ranges_disjoint"] is True
    assert out["cal_sha256_64tok_probe_hits"] == 0


def test_leak_check_flags_overlap(monkeypatch):
    import torch
    monkeypatch.setattr(p9, "CAL_ROWS", (0, 120))
    monkeypatch.setattr(p9, "EVAL_ROWS", (100, 220))     # overlapping row ranges
    with pytest.raises(AssertionError):
        p9.leak_check(torch.arange(1000), torch.arange(500, 1500), probe=64)


def test_line_census_deduplicates():
    """A 40-byte object at offset 0 touches one line, not two."""
    assert p9.exact_lines([0], 40) == 1
    assert p9.exact_lines([0, 40], 40) == 2
    # two blocks inside one 64-byte line share it
    assert p9.exact_lines([0, 18], 18) == 1
    assert p9.exact_lines([0, 18, 36, 54], 18) == 2


def test_block_energy_matches_direct_sum():
    import torch
    g = torch.Generator().manual_seed(2)
    a = torch.randn(3, 96, generator=g)
    e = p9.block_energy(a, 32)
    direct = (a.reshape(3, 3, 32) ** 2).sum(-1)
    assert torch.allclose(e, direct)


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,key", [("p9_reorder16_08b.json", "models"),
                                      ("p9_calib2.json", "models"),
                                      ("p9_e2e.json", "paths"),
                                      ("p9_corr.json", "models"),
                                      ("p9_cacheline.json", "rows")])
def test_result_files_are_well_formed(name, key):
    d = load(name)
    if d is None:
        pytest.skip(f"{name} not produced yet")
    assert key in d
    assert d[key]
    if key == "models" and name.startswith("p9_reorder16"):
        assert "strategies" in next(iter(d["models"].values()))


def test_ordering_gap_and_gini_are_consistent():
    d = load("p9_reorder16_08b.json")
    if d is None:
        pytest.skip("ordering sweep not produced yet")
    m = d["models"]["qwen3.5-0.8b"]
    st = m["strategies"]
    # the co-activation permutation must not lose to the random order
    for rate, cell in st["random"].items():
        assert st["ours"][rate]["delta_ppl"] <= cell["delta_ppl"] + 1e-6
    # and it must produce the highest within-block coherence (Gini of block energy)
    gini_ours = np.mean([v["gini_mean"] for v in st["ours"].values()])
    gini_random = np.mean([v["gini_mean"] for v in st["random"].values()])
    assert gini_ours > gini_random


def test_cacheline_rows_are_monotone_in_tau():
    d = load("p9_cacheline.json")
    if d is None:
        pytest.skip("census not produced yet")
    rows = sorted(d["rows"], key=lambda r: r["tau"])
    lines = [r["block_major_lines"] for r in rows]
    assert all(x >= y for x, y in zip(lines, lines[1:]))


def test_whisper_closure_uses_at_least_50_utterances():
    """The run selects 100 LibriSpeech utterances; 60 decode successfully."""
    d = load("p9_whisper100.json")
    if d is None:
        pytest.skip("whisper closure not produced yet")
    ns = [r.get("n_utterances", 0) for r in d.get("rows", [])]
    assert ns and max(ns) >= 50


def test_manuscript_keeps_the_squared_norm():
    typ = EXP.parents[1] / "paper2_bundle" / "CM_Transform_Paper2.typ"
    if not typ.exists():
        pytest.skip("manuscript not present")
    text = typ.read_text()
    # Eq. (2) must be a squared norm on the left-hand side
    assert "‖Delta y‖_2^2 = sum_(i,j in S)" in text
    for banned in ("honest", "honestly", "rigorous", "elegant", "impressive", "remarkably"):
        assert banned not in text.lower(), banned


def test_eq2_left_hand_side_is_a_squared_norm():
    """Eq. (2) must state ||Δy||_2^2 = Σ a_i a_j <W_i, W_j> (squared, not plain norm)."""
    from pathlib import Path
    typ = Path(__file__).resolve().parents[3] / "paper2_bundle" / "CM_Transform_Paper2.typ"
    if not typ.exists():
        pytest.skip("manuscript not present")
    src = typ.read_text()
    line = next(l for l in src.splitlines() if "Delta y" in l and "sum_(i,j in S)" in l)
    assert "‖Delta y‖_2^2" in line, line
    assert "chevron.l W_(d,i)" in line
    # the expectation identity must use the same squared quantity
    exp_line = next(l for l in src.splitlines()
                    if "bb(E)" in l and "Delta y" in l and "C_(i j)" in l)
    assert "‖Delta y‖_2^2" in exp_line


def test_simd_width_builds_agree():
    """Scalar / AVX2 / AVX-512 builds must decode identically."""
    import ctypes
    import numpy as np
    from pathlib import Path

    build = Path(__file__).resolve().parent.parent / "results" / "build"
    libs = {w: build / f"iq4nl_width_{w}.so" for w in ("scalar", "avx2", "avx512")}
    if not all(p.exists() for p in libs.values()) or not (build / "iq4nl_bm_v2.so").exists():
        import pytest
        pytest.skip("width kernels not built")
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int)
    rng = np.random.default_rng(0)
    M, nb = 64, 8
    raw = rng.integers(0, 256, size=M * nb * 18, dtype=np.uint8)
    raw[0::18] = 0x00
    raw[1::18] = 0x3c                      # fp16 d = 1.0
    rec = np.zeros_like(raw)
    bm = ctypes.CDLL(str(build / "iq4nl_bm_v2.so"))
    bm.repack_blockmajor.argtypes = [u8p, u8p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    bm.repack_blockmajor(raw.ctypes.data_as(u8p), rec.ctypes.data_as(u8p), M, nb, 18)
    x = rng.standard_normal(nb * 32).astype(np.float32)
    idx = np.arange(nb, dtype=np.int32)
    sums = {}
    for name, path in libs.items():
        lib = ctypes.CDLL(str(path))
        lib.gemv_bm.argtypes = [u8p, f32p, f32p, ctypes.c_int, ctypes.c_int, i32p, ctypes.c_int]
        out = np.zeros(M, dtype=np.float32)
        lib.gemv_bm(rec.ctypes.data_as(u8p), x.ctypes.data_as(f32p), out.ctypes.data_as(f32p),
                    M, nb, idx.ctypes.data_as(i32p), nb)
        sums[name] = out
    ref = sums["scalar"]
    for name, v in sums.items():
        assert np.abs(v - ref).max() < 2e-3, (name, float(np.abs(v - ref).max()))


# ---------------------------------------------------------------- paper source
@PAPER_SKIP
def test_paper_has_no_stale_block_overlap_string():
    """The superseded 'block overlap 0.000' sentence must not survive in the source."""
    src = PAPER_TYP.read_text()
    assert "wiki|zh: 0.000" not in src
    assert "wiki|zh:** 0.000" not in src


@PAPER_SKIP
def test_paper_conclusion_follows_all_experiment_sections():
    """Section 1..N are numbered in source order; Conclusion must come after every
    results/analysis section, with only unnumbered back matter after it."""
    import re as _re
    src = PAPER_TYP.read_text()
    heads = []
    for line in src.splitlines():
        m = _re.match(r"^(=+)\s+([A-Z][^\[\]]{2,80})$", line)
        if m:
            heads.append((len(m.group(1)), m.group(2).strip()))
    # replicate typst's numbering: level-1 counters
    counters, numbered = [0, 0, 0], []
    for lvl, title in heads:
        counters[lvl - 1] += 1
        for k in range(lvl, 3):
            counters[k] = 0
        numbered.append((".".join(str(c) for c in counters[:lvl]), title))
    concl = [n for n, t in numbered if t == "Conclusion"]
    assert concl, "no Conclusion heading"
    cnum = tuple(int(x) for x in concl[0].split("."))
    exp_words = ("Results", "Ablations", "Kernel", "Analysis", "Where the speedup",
                 "Negative results")
    for num, title in numbered:
        if any(title.startswith(w) for w in exp_words):
            assert tuple(int(x) for x in num.split(".")) < cnum, (num, title)
    # nothing numbered may follow Conclusion
    seen = False
    for num, title in numbered:
        if title == "Conclusion":
            seen = True
            continue
        if seen:
            raise AssertionError(f"numbered section after Conclusion: {num} {title}")
