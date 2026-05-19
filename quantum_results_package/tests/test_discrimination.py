"""Sector-parity discrimination: optimum and block-diagonal baseline."""

import numpy as np

import samei


def test_optimum_formula_and_saturator():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    for eps in np.linspace(0.0, 1.0, 51):
        p_formula = samei.optimal_psucc_at_budget(eps)
        rho_sat = samei.saturating_state(M, eps)
        p_num = samei.sector_parity_psucc(rho_sat, M)
        assert abs(p_formula - p_num) <= 1e-12


def test_block_diagonal_psucc_one_half():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    rng = np.random.default_rng(0)
    for lam in np.linspace(0.0, 1.0, 11):
        rho = samei.block_diagonal_state(M, lam, rng=rng)
        p = samei.sector_parity_psucc(rho, M)
        assert abs(p - 0.5) <= 1e-10
