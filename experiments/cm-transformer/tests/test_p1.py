"""P1 tests: 32-lane co-activation permutation, exact isomorphism, kernels."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

from common import RESULTS_DIR, LANE  # noqa: E402
from model_ref import block_energy, silu  # noqa: E402
from p1_ffn import cluster_blocks, grouping_stats, make_override  # noqa: E402


def synthetic_profiles(groups: int = 8, lane: int = LANE, tokens: int = 200, seed: int = 0):
    """`groups` disjoint 32-neuron cohorts that fire together, plus noise."""
    g = torch.Generator().manual_seed(seed)
    I = groups * lane
    firing = torch.rand(tokens, groups, generator=g) < 0.25      # cohort active or not
    A = (firing.repeat_interleave(lane, dim=1) *
         torch.rand(tokens, I, generator=g).pow(3) * 10.0)
    return A[:, torch.randperm(I, generator=g)]                  # scramble the native order


def test_cluster_blocks_returns_valid_partition():
    A = synthetic_profiles()
    perm, meta = cluster_blocks(A)
    I = A.shape[1]
    assert perm.numel() == I
    assert torch.equal(perm.sort().values, torch.arange(I))       # a permutation
    blocks = perm.reshape(-1, LANE)
    assert blocks.shape == (I // LANE, LANE)
    e = meta["energy"]
    blk = e[blocks].mean(-1)
    assert torch.all(blk[:-1] >= blk[1:] - 1e-6)                   # high energy first


def test_clustering_beats_natural_on_cohort_data():
    A = synthetic_profiles()
    perm, _ = cluster_blocks(A)
    prof = A.t().abs()
    prof = prof / prof.norm(dim=1, keepdim=True).clamp_min(1e-8)
    c = grouping_stats(perm, prof)
    n = grouping_stats(torch.arange(A.shape[1]), prof)
    assert 0.0 <= n <= c <= 1.0
    assert c > n + 0.05


def test_permuted_mlp_is_exactly_isomorphic():
    from model_ref import permute_mlp
    torch.manual_seed(0)
    I, D = 96, 24
    g, u, d = torch.randn(I, D), torch.randn(I, D), torch.randn(D, I)
    bg, bu = torch.randn(I), torch.randn(I)
    x = torch.randn(7, D)
    perm = torch.randperm(I)
    ref = (silu(x @ g.T + bg) * (x @ u.T + bu)) @ d.T
    gp, up, dp = permute_mlp((g, u, d), perm)
    out = (silu(x @ gp.T + bg[perm]) * (x @ up.T + bu[perm])) @ dp.T
    assert (ref - out).abs().max() < 1e-5 * ref.abs().max()


def test_block_skipping_is_lossless_when_blocks_are_empty():
    a = torch.zeros(2, 64)
    a[:, :32] = torch.randn(2, 32)
    e = block_energy(a)
    keep = torch.zeros_like(e, dtype=torch.bool)
    keep[:, 0] = True
    masked = a * keep.repeat_interleave(LANE, dim=-1).to(a.dtype)
    assert torch.equal(masked, a)


@pytest.mark.skipif(not Path("/usr/bin/g++").exists() and not Path("/usr/bin/c++").exists(),
                    reason="no C++ compiler")
def test_avx512_kernel_matches_reference(tmp_path):
    src = EXP / "lanemlp_avx512.cpp"
    so = tmp_path / "liblanemlp.so"
    subprocess.run(["g++", "-O2", "-march=native", "-fPIC", "-shared", "-fno-math-errno",
                    str(src), "-o", str(so)], check=True, capture_output=True)
    import ctypes
    handle = ctypes.CDLL(str(so))
    fn = handle.down_skip_avx512
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3
    T, I, O = 3, 64, 8
    g = torch.Generator().manual_seed(0)
    A = torch.randn(T, I, generator=g)
    W = torch.randn(O, I, generator=g)          # kernel reads W row-major as [O, I]
    # the kernel's mask is per 32-lane block: shape [T, I/32]
    keep = (torch.rand(T, I // LANE, generator=g) > 0.5).to(torch.uint8).contiguous()
    ref = (A * keep.repeat_interleave(LANE, dim=-1).float()) @ W.T
    out = torch.zeros(T, O)
    fn(A.data_ptr(), W.data_ptr(), keep.data_ptr(), out.data_ptr(), T, I, O)
    assert (out - ref).abs().max() < 1e-4 * ref.abs().max()


# --------------------------------------------------------- recorded results
def _load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        pytest.skip(f"{name} not produced yet")
    return json.loads(p.read_text())


def test_recorded_isomorphism_is_at_rounding_floor():
    d = _load("p1_cuda_float32.json")
    iso = d["isomorphism_tau0"]["clustered"]
    assert iso["top1_agreement"] == 1.0
    assert iso["max_abs_dlogit"] <= 1e-3
    assert iso["logitRmse"] <= 1e-4
    assert iso["kl"] <= 1e-6


def test_recorded_skip_sweep_is_monotone_and_clustered_is_not_better():
    """Recorded evidence: at a matched block-skip rate the clustered grouping is
    *not* more accurate than natural neuron order — natural wins on ΔPPL.
    The permutation's value is hardware alignment, not fidelity."""
    f = RESULTS_DIR / "capability_qwen3.5_cuda.json"
    if not f.exists():
        pytest.skip("no capability results")
    rows = json.loads(f.read_text())["rows"]
    by = {(r["rate"], r["variant"]): r for r in rows if "delta_ppl" in r}
    rates = sorted({r for r, _ in by if r > 0})
    for var in ("clustered", "natural", "random"):
        ppls = [by[(r, var)]["delta_ppl"] for r in rates]
        assert ppls == sorted(ppls), (var, ppls)      # monotone degradation
    for r in rates:
        assert by[(r, "natural")]["delta_ppl"] <= by[(r, "clustered")]["delta_ppl"] + 1e-9, r


def test_recorded_kernels_speed_up_with_block_skipping():
    d = _load("p1_kernels.json")
    avx = d["avx512"]["sweep"]
    sp = [r["speedup_vs_dense"] for r in avx]
    assert avx[0]["rate"] == 0.0 and sp[0] == 1.0
    assert sp[-1] > 1.5 and sp[-1] == max(sp)
    cuda = d["cuda"]["v6"]
    csp = [r["speedup_vs_dense"] for r in cuda]
    assert csp[-1] > 1.5 and csp[-1] == max(csp)
    assert d["avx512"]["correctness_layer0_rate50"]["max_abs"] < 1e-6
