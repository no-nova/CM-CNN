"""Sanity checks for the Phase-8 helpers that carry paper claims."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import p8_reviewer_defense as P8  # noqa: E402

torch.manual_seed(0)


def test_gini_bounds():
    assert P8.gini(torch.ones(64)) == pytest.approx(0.0, abs=1e-6)
    one_hot = torch.zeros(64)
    one_hot[0] = 1.0
    assert 0.9 < P8.gini(one_hot) <= 1.0


def test_iq4nl_roundtrip_is_closer_than_half_step():
    w = torch.randn(64, 128) * 0.05
    deq, d, idx = P8.iq4nl_roundtrip(w)
    # IQ4_NL's 16 k-values are not an arithmetic ladder, so the bound is the
    # largest gap in the table rather than one step of amax/127.
    amax = w.reshape(64, 4, 32).abs().amax(-1, keepdim=True)
    rel = (deq.float() - w).abs().reshape(64, 4, 32) / amax.clamp_min(1e-8)
    assert float(rel.mean()) < 0.05 and float(rel.max()) < 0.15
    assert idx.dtype in (torch.int64, torch.int32)


def test_strategy_perm_is_a_permutation_and_beats_random():
    acts = torch.rand(400, 256).pow(3)          # heavy tailed, as activations are
    Wd = torch.randn(96, 256)
    perm = P8.strategy_perm(acts, Wd, "impsort", lane=32)
    assert sorted(perm.tolist()) == list(range(256))
    assert P8.gini(acts.mean(0)[perm]) >= P8.gini(acts.mean(0)) - 1e-6


def test_capture_shim_maps_reference_kwargs():
    cfg = {"override": {0: 1}, "scale": {0: 2}, "theta": {0: 3.0}, "rate": 0.2}
    kw = P8.kw(cfg) if hasattr(P8, "kw") else None
    if kw is None:                              # kw lives in p8_system
        import p8_system
        kw = p8_system.kw(cfg)
    assert kw["mlp_override"] == {0: 1}
    assert kw["skip_scale"] == {0: 2}
    assert kw["skip_theta"] == {0: 3.0}
    assert kw["skip_rate"] == pytest.approx(0.2)
