"""Theorems 9, 10, 11 and Corollary 12: Hamiltonian and Lindblad dynamics."""

import numpy as np
import pytest
from scipy.linalg import expm

import samei_leakage as sl


def _setup_global_J(dA=2, dB=2):
    JA = sl.canonical_J(dA)
    JB = sl.canonical_J(dB)
    _, _, M = sl.sector_operators(JA, JB)
    # global complex structure that respects same-i composition: J = J_A on factor A
    # tensor I, summed... but the simplest global J for tests is the alignment-
    # commuting choice J = A_op = J_A x I (any choice that commutes with M works).
    A_op, B_op, _ = sl.sector_operators(JA, JB)
    return JA, JB, M, A_op


def test_hamiltonian_conservation_when_G_commutes_with_M():
    """[G, M] = 0 implies L(rho(t)) = L(rho(0)) for all t."""
    JA, JB, M, J_global = _setup_global_J(2, 2)
    n = M.shape[0]
    # Build H_R commuting with J_global so that G = -J_global H_R; require [G, M] = 0.
    # Easiest construction: pick H_R = a P_+ + b P_- with real a, b. Then both
    # [H_R, M] = 0 and [G, M] = 0 (since J_global commutes with itself and with M).
    Pp, Pm = sl.projectors(M)
    H_R = 1.7 * Pp + 0.3 * Pm  # diagonal in M-eigenbasis -> commutes with M
    G = sl.schrodinger_generator(H_R, J_global)
    assert np.allclose(G @ M - M @ G, 0.0, atol=1e-12)
    rng = np.random.default_rng(0)
    rho = sl.random_real_density(n, rng=rng)
    L0 = sl.leakage(rho, M)
    for t in [0.1, 0.5, 1.3, 2.7]:
        rho_t = sl.evolve_hamiltonian(rho, G, t)
        Lt = sl.leakage(rho_t, M)
        assert abs(Lt - L0) < 1e-10, f"t={t}: drift {abs(Lt - L0)}"


def test_perturbative_leakage_is_O_eps_squared():
    """Theorem 10: starting at a free state, leakage grows as eps^2 in the
    perturbation strength."""
    JA, JB, M, J_global = _setup_global_J(2, 2)
    n = M.shape[0]
    Pp, Pm = sl.projectors(M)
    # G_0: a sector-preserving generator. Take G_0 = -J_global H_R with H_R block-
    # diagonal in M-eigenbasis (commutes with M).
    H_R0 = 1.0 * Pp + 0.5 * Pm
    G0 = sl.schrodinger_generator(H_R0, J_global)
    # B: a real antisymmetric matrix coupling H_+ <-> H_-.
    rng = np.random.default_rng(0)
    Braw = rng.standard_normal((n, n))
    Bskew = (Braw - Braw.T) / 2
    # ensure mixing across sectors (it generically will)
    psi0 = np.linalg.eigh(M)[1][:, 0]  # an H_+ eigenvector (M-eigenvalue -1)
    rho0 = np.outer(psi0, psi0)
    assert sl.leakage(rho0, M) < 1e-10
    t = 0.5
    # Fit L(eps) ~ A eps^2 over a small grid
    eps_grid = np.array([0.005, 0.01, 0.02, 0.04, 0.08])
    L_vals = []
    for eps in eps_grid:
        G_eps = G0 + eps * Bskew
        rho_t = sl.evolve_hamiltonian(rho0, G_eps, t)
        L_vals.append(sl.leakage(rho_t, M))
    L_vals = np.array(L_vals)
    # Log-log slope should be ~2
    slope, intercept = np.polyfit(np.log(eps_grid), np.log(L_vals + 1e-20), 1)
    assert abs(slope - 2.0) < 0.1, f"perturbative slope {slope} != 2"


def test_lindblad_conservation_when_all_commute_with_M():
    """Corollary 12: [G, M] = 0 and [R_l, M] = 0 for all l implies L conserved."""
    JA, JB, M, J_global = _setup_global_J(1, 1)  # smaller for speed
    n = M.shape[0]
    Pp, Pm = sl.projectors(M)
    H_R = 0.7 * Pp - 0.4 * Pm
    G = sl.schrodinger_generator(H_R, J_global)
    # jump operators commuting with M: take any operator block-diagonal in M-eigenbasis
    R1 = 0.3 * (Pp - Pm)  # commutes with M
    R2 = 0.2 * Pp  # commutes with M
    jumps = [R1, R2]
    # Verify commutation explicitly
    for R in jumps:
        assert np.allclose(R @ M - M @ R, 0.0, atol=1e-12)
    rng = np.random.default_rng(1)
    rho = sl.random_real_density(n, rng=rng)
    L0 = sl.leakage(rho, M)
    rho_t = sl.integrate_lindblad(rho, G, jumps, t=0.5, n_steps=500)
    Lt = sl.leakage(rho_t, M)
    assert abs(Lt - L0) < 1e-4  # Euler integrator accuracy


def test_lindblad_rate_formula():
    """dL/dt|_{t=0} = Tr(L^dagger(P_-) rho_0)."""
    JA, JB, M, J_global = _setup_global_J(1, 1)
    n = M.shape[0]
    Pp, Pm = sl.projectors(M)
    H_R = 0.5 * Pp + 0.2 * Pm
    G = sl.schrodinger_generator(H_R, J_global)
    rng = np.random.default_rng(2)
    # generic jump operator NOT commuting with M (introduces leakage rate)
    R_raw = rng.standard_normal((n, n))
    jumps = [0.3 * R_raw]
    rho = sl.random_real_density(n, rng=rng)
    rate_analytic = np.trace(sl.lindblad_generator_dual_on(Pm, G, jumps) @ rho).real
    # numerical derivative
    dt = 1e-5
    rho_dt = sl.integrate_lindblad(rho, G, jumps, t=dt, n_steps=10)
    L0 = sl.leakage(rho, M)
    Ldt = sl.leakage(rho_dt, M)
    rate_numerical = (Ldt - L0) / dt
    assert abs(rate_analytic - rate_numerical) < 5e-3
