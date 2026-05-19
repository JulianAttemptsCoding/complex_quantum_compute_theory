"""Perturbative leakage: L ~ eps^2."""

import numpy as np

import samei


def test_perturbative_slope_two():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    rng = np.random.default_rng(0)
    G0 = samei.skew_commuting_with_M(M, rng=rng)
    B = samei.skew_cross_sector(M, rng=rng)
    # commutation check
    assert np.allclose(G0 @ M - M @ G0, 0.0, atol=1e-12)
    # cross-sector check
    Pp, Pm = samei.projectors(M)
    assert np.linalg.norm(Pm @ B @ Pp) > 1e-6
    psi0 = samei.leakage_perturbative_curve  # type: ignore  # silence linter
    # psi0 in H_+
    n = M.shape[0]
    rng2 = np.random.default_rng(2)
    v = rng2.standard_normal(n)
    v = Pp @ v
    v = v / np.linalg.norm(v)
    out = samei.leakage_perturbative_curve(M, G0, B, v, t=1.0)
    assert 1.95 <= out["slope"] <= 2.05


def test_G0_skew_and_B_skew():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    rng = np.random.default_rng(0)
    G0 = samei.skew_commuting_with_M(M, rng=rng)
    B = samei.skew_cross_sector(M, rng=rng)
    assert np.allclose(G0, -G0.T, atol=1e-12)
    assert np.allclose(B, -B.T, atol=1e-12)
