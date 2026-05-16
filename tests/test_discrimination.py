"""Theorem 13: sector-parity discrimination."""

import numpy as np
import pytest

import samei_leakage as sl


def _M(dA=2, dB=2):
    return sl.sector_operators(sl.canonical_J(dA), sl.canonical_J(dB))[2]


@pytest.mark.parametrize(
    "epsilon", [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
)
def test_saturating_state_achieves_optimum(epsilon):
    M = _M()
    rho = sl.saturating_state(M, epsilon)
    L = sl.leakage(rho, M)
    assert L <= min(epsilon, 0.5) + 1e-10
    p_num = sl.sector_parity_psucc(rho, M)
    p_closed = sl.optimal_psucc_at_budget(epsilon)
    assert abs(p_num - p_closed) < 1e-10


def test_block_diagonal_probes_give_no_advantage():
    """For rho block-diagonal on H_+ + H_-, p_succ = 1/2 regardless of leakage."""
    M = _M()
    Pp, Pm = sl.projectors(M)
    rng = np.random.default_rng(0)
    for _ in range(5):
        rho_pp = sl.random_real_density(M.shape[0], rng=rng)
        rho_pp = Pp @ rho_pp @ Pp
        rho_mm = sl.random_real_density(M.shape[0], rng=rng)
        rho_mm = Pm @ rho_mm @ Pm
        # mix sectors with some weight
        w = 0.3
        rho = w * rho_mm / max(np.trace(rho_mm).real, 1e-15) + (1 - w) * rho_pp / max(np.trace(rho_pp).real, 1e-15)
        # symmetrize and renormalize
        rho = 0.5 * (rho + rho.T)
        rho = rho / np.trace(rho).real
        L = sl.leakage(rho, M)
        p_num = sl.sector_parity_psucc(rho, M)
        assert abs(p_num - 0.5) < 1e-10
        assert 0.0 <= L <= 1.0 + 1e-10


def test_no_state_exceeds_optimum():
    """No real density operator can exceed p_succ^{(L)} = 1/2 + sqrt(L(1-L))."""
    M = _M()
    rng = np.random.default_rng(1)
    for _ in range(30):
        rho = sl.random_real_density(M.shape[0], rank=rng.integers(1, 17), rng=rng)
        L = sl.leakage(rho, M)
        p_num = sl.sector_parity_psucc(rho, M)
        bound = 0.5 + np.sqrt(L * max(0.0, 1.0 - L))
        assert p_num <= bound + 1e-10
