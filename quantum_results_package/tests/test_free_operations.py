"""Free operations: Kraus criterion, Phi^dagger(P_-) <= P_-, violator example."""

import numpy as np
import pytest

import samei


def test_random_sector_block_channel_is_free():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    rng = np.random.default_rng(0)
    K = samei.random_sector_block_channel(M, n_kraus=4, rng=rng)
    assert samei.is_trace_preserving(K, atol=1e-10)
    assert samei.kraus_no_leakage(K, M, atol=1e-10)
    assert samei.is_leakage_nonincreasing(K, M, atol=1e-9)
    n = M.shape[0]
    max_inc = 0.0
    for _ in range(20):
        rho = samei.linear_algebra.random_real_density(n, rng=rng)
        L0 = samei.leakage(rho, M)
        L1 = samei.leakage(samei.apply_kraus(rho, K), M)
        max_inc = max(max_inc, L1 - L0)
    assert max_inc <= 1e-10


def test_violator_increases_leakage():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    K, rho = samei.violating_channel_example(M)
    assert samei.is_trace_preserving(K, atol=1e-10)
    L0 = samei.leakage(rho, M)
    L1 = samei.leakage(samei.apply_kraus(rho, K), M)
    assert L1 > L0 + 0.9
