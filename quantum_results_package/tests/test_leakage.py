"""Leakage and coherent leakage: envelope, block-diagonal, pure states."""

import numpy as np
import pytest

import samei


def test_envelope_random_states():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    rng = np.random.default_rng(0)
    n = M.shape[0]
    for _ in range(200):
        rank = int(rng.integers(1, n + 1))
        rho = samei.linear_algebra.random_real_density(n, rank=rank, rng=rng)
        L = samei.leakage(rho, M)
        C = samei.coherent_leakage(rho, M)
        assert C <= np.sqrt(max(L * (1.0 - L), 0.0)) + 1e-10


def test_pure_state_saturates_envelope():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    rng = np.random.default_rng(0)
    for lam in np.linspace(0.0, 1.0, 21):
        rho, _, _ = samei.coherent_pure_state(M, lam, rng=rng)
        L = samei.leakage(rho, M)
        C = samei.coherent_leakage(rho, M)
        assert abs(L - lam) <= 1e-12
        assert abs(C - np.sqrt(lam * (1 - lam))) <= 1e-12


def test_block_diagonal_has_zero_C():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    rng = np.random.default_rng(0)
    for lam in np.linspace(0.0, 1.0, 11):
        rho = samei.block_diagonal_state(M, lam, rng=rng)
        L = samei.leakage(rho, M)
        C = samei.coherent_leakage(rho, M)
        assert C <= 1e-12
        assert abs(L - lam) <= 1e-12
