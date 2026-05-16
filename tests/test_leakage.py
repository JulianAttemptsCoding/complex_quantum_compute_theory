"""Theorems 4 and 5: leakage basic properties and coherent bound."""

import numpy as np
import pytest

import samei_leakage as sl


def _M(dA=1, dB=1):
    return sl.sector_operators(sl.canonical_J(dA), sl.canonical_J(dB))[2]


def test_leakage_range_and_alternative_formula():
    rng = np.random.default_rng(0)
    M = _M(2, 2)
    for _ in range(20):
        rho = sl.random_real_density(M.shape[0], rng=rng)
        L = sl.leakage(rho, M)
        L_alt = 0.5 * (1.0 + np.trace(M @ rho).real)
        assert -1e-12 <= L <= 1.0 + 1e-12
        assert abs(L - L_alt) < 1e-12


def test_free_face_membership():
    """rho with L=0 supported in H_+; rho with L=1 supported in H_-."""
    JA = sl.canonical_J(2)
    JB = sl.canonical_J(2)
    _, _, M = sl.sector_operators(JA, JB)
    Pp, Pm = sl.projectors(M)
    # state supported in H_+: rho = P_+ / dim H_+
    n_plus = int(round(np.trace(Pp).real))
    rho_plus = Pp / n_plus
    assert sl.leakage(rho_plus, M) < 1e-12
    # state supported in H_-
    n_minus = int(round(np.trace(Pm).real))
    rho_minus = Pm / n_minus
    assert abs(sl.leakage(rho_minus, M) - 1.0) < 1e-12


def test_leakage_is_affine():
    rng = np.random.default_rng(1)
    M = _M(2, 2)
    rho1 = sl.random_real_density(M.shape[0], rng=rng)
    rho2 = sl.random_real_density(M.shape[0], rng=rng)
    for p in [0.0, 0.2, 0.5, 0.7, 1.0]:
        rho_mix = p * rho1 + (1 - p) * rho2
        lhs = sl.leakage(rho_mix, M)
        rhs = p * sl.leakage(rho1, M) + (1 - p) * sl.leakage(rho2, M)
        assert abs(lhs - rhs) < 1e-12


def test_coherent_leakage_bound():
    rng = np.random.default_rng(2)
    M = _M(2, 2)
    for _ in range(50):
        rho = sl.random_real_density(M.shape[0], rank=rng.integers(1, 17), rng=rng)
        L = sl.leakage(rho, M)
        C = sl.coherent_leakage(rho, M)
        bound = np.sqrt(L * max(0.0, 1.0 - L))
        assert C <= bound + 1e-10, f"violated: L={L} C={C} bound={bound}"


def test_coherent_leakage_pure_saturation():
    """Coherent pure state |psi> = sqrt(1-lambda) psi_+ + sqrt(lambda) psi_-
    saturates the bound."""
    M = _M(2, 2)
    Pp, Pm = sl.projectors(M)
    eigvals, eigvecs = np.linalg.eigh(M)
    psi_plus = None
    psi_minus = None
    for val, vec in zip(eigvals, eigvecs.T):
        if val < 0 and psi_plus is None:
            psi_plus = vec
        elif val > 0 and psi_minus is None:
            psi_minus = vec
    for lam in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        psi = np.sqrt(1 - lam) * psi_plus + np.sqrt(lam) * psi_minus
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi)
        L = sl.leakage(rho, M)
        C = sl.coherent_leakage(rho, M)
        # clip negative floating-point residual before sqrt
        bound = np.sqrt(max(0.0, L * (1.0 - L)))
        assert abs(L - lam) < 1e-10
        assert abs(C - bound) < 1e-10
