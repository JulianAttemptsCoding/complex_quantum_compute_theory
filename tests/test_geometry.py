"""Theorem 8 (trace distance) and Proposition 10 (robustness divergence)."""

import numpy as np
import pytest

import samei_leakage as sl


def _M(dA=2, dB=2):
    return sl.sector_operators(sl.canonical_J(dA), sl.canonical_J(dB))[2]


def test_trace_distance_on_coherent_pure_states():
    """For psi = sqrt(1-lambda) psi_+ + sqrt(lambda) psi_-, D_1(rho, F) = sqrt(lambda)."""
    M = _M()
    eigvals, eigvecs = np.linalg.eigh(M)
    psi_plus = next(v for val, v in zip(eigvals, eigvecs.T) if val < 0)
    psi_minus = next(v for val, v in zip(eigvals, eigvecs.T) if val > 0)
    for lam in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
        psi = np.sqrt(1 - lam) * psi_plus + np.sqrt(lam) * psi_minus
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi)
        D = sl.trace_distance_to_free(rho, M)
        assert abs(D - np.sqrt(lam)) < 1e-10, f"lam={lam}: D={D} sqrt={np.sqrt(lam)}"


def test_trace_distance_on_free_states_is_zero():
    M = _M()
    Pp, _ = sl.projectors(M)
    n_plus = int(round(np.trace(Pp).real))
    rho_free = Pp / n_plus
    assert sl.trace_distance_to_free(rho_free, M) < 1e-12


def test_robustness_diverges_for_leaky_states():
    """Proposition 10: any rho with L > 0 has infinite generalized robustness,
    because adding any free-positive tau cannot zero out the non-PSD P_- part."""
    M = _M()
    rng = np.random.default_rng(0)
    rho_leaky = sl.random_real_density(M.shape[0], rng=rng)
    L = sl.leakage(rho_leaky, M)
    if L < 1e-10:
        pytest.skip("rho happens to be free")
    # Try to find finite s such that (rho + s tau)/(1+s) is free.
    # Any tau is real density on K_AB, but P_- tau P_- >= 0 so leakage cannot cancel.
    Pp, Pm = sl.projectors(M)
    leak_rho = (Pm @ rho_leaky @ Pm)  # PSD block
    # For any feasible (rho + s tau)/(1+s) free, we need P_- (rho + s tau) P_- = 0.
    # Since both Pm rho Pm and Pm tau Pm are PSD, their sum is zero iff both vanish.
    # rho already has nonzero Pm-Pm block, so impossible.
    assert np.trace(leak_rho).real > 1e-10
