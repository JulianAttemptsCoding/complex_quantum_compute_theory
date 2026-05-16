"""Propositions 14 and 15 plus imaginarity inequivalence."""

import numpy as np
import pytest

import samei_leakage as sl


def test_cfr_leakage_matches_closed_form_exact():
    JA = sl.canonical_J(1)
    JB = sl.canonical_J(1)
    _, _, M = sl.sector_operators(JA, JB)
    for alpha in np.linspace(-1.0, 1.0, 21):
        rho = sl.cfr_state(alpha)
        L = sl.leakage(rho, M)
        L_closed = sl.cfr_leakage_closed_form(alpha)
        assert abs(L - L_closed) < 1e-12, f"alpha={alpha}: {L} vs {L_closed}"


def test_cfr_endpoints_are_pure_sector_projectors():
    """alpha = +1: rho = (1/2) P_+. alpha = -1: rho = (1/2) P_-."""
    JA = sl.canonical_J(1)
    JB = sl.canonical_J(1)
    _, _, M = sl.sector_operators(JA, JB)
    Pp, Pm = sl.projectors(M)
    rho_plus = sl.cfr_state(1.0)
    rho_minus = sl.cfr_state(-1.0)
    assert np.allclose(rho_plus, 0.5 * Pp, atol=1e-12)
    assert np.allclose(rho_minus, 0.5 * Pm, atol=1e-12)


def test_bob_node_product_obstruction_is_exactly_one_half():
    """Proposition 15: any product source gives L_B = 1/2."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        dA = int(rng.integers(1, 4))
        dB = int(rng.integers(1, 4))
        rho1 = sl.random_real_density(2 * dA, rng=rng)
        rho2 = sl.random_real_density(2 * dB, rng=rng)
        # canonical local J's for each factor
        JA = sl.canonical_J(dA)
        JB = sl.canonical_J(dB)
        L = sl.bob_node_leakage_product(rho1, rho2, JA, JB)
        assert abs(L - 0.5) < 1e-12


def test_imaginarity_example_A():
    """Entrywise real state with L > 0 in the canonical basis."""
    rho, M = sl.imaginarity_inequivalence_A()
    # entrywise real
    assert np.allclose(rho.imag, 0.0, atol=1e-12)
    # unit trace, PSD
    assert abs(np.trace(rho).real - 1.0) < 1e-12
    eigs = np.linalg.eigvalsh(rho)
    assert eigs.min() > -1e-12
    # nontrivial leakage
    L = sl.leakage(rho, M)
    assert L > 0.1
    # Hickey-Gour imaginarity in the canonical basis is zero by construction
    # (the off-diagonal imaginary part vanishes since entries are entirely real)
    assert np.allclose(rho - rho.T, 0.0, atol=1e-12)


def test_imaginarity_example_B():
    """Product of complex states with nonzero entrywise imaginarity, but
    realification has L = 0."""
    rho_real, M, rho_complex = sl.imaginarity_inequivalence_B()
    # complex state has nonzero imaginary entries
    assert np.linalg.norm(rho_complex.imag) > 0.1
    # realification: L should equal 0 (or very close)
    L = sl.leakage(rho_real, M)
    assert L < 1e-10, f"L={L} should be 0"
