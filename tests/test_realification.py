"""Theorem 1: realification preserves quantum statistics."""

import numpy as np
import pytest

import samei_leakage as sl


def random_complex_hermitian(d: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    return (X + X.conj().T) / 2


def random_complex_density(d: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    A = X @ X.conj().T
    return A / np.trace(A).real


def test_realify_operator_is_ring_homomorphism():
    """R(XY) = R(X) R(Y) for any complex matrices X, Y."""
    rng = np.random.default_rng(0)
    for d in (1, 2, 3, 5):
        X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        Y = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        lhs = sl.realify_operator(X @ Y)
        rhs = sl.realify_operator(X) @ sl.realify_operator(Y)
        assert np.allclose(lhs, rhs, atol=1e-12), f"d={d}"


def test_realify_state_unit_trace():
    """rho_R = (1/2) R(rho_C) has unit real trace when rho_C has unit complex trace."""
    rng = np.random.default_rng(1)
    for d in (1, 2, 3, 5):
        rho_C = random_complex_density(d, rng)
        rho_R = sl.realify_state(rho_C)
        assert abs(np.trace(rho_R).real - 1.0) < 1e-12
        # rho_R is real symmetric PSD
        assert np.allclose(rho_R.imag, 0.0, atol=1e-12)
        assert np.allclose(rho_R, rho_R.T, atol=1e-12)
        eigs = np.linalg.eigvalsh(rho_R)
        assert eigs.min() > -1e-12


def test_realify_born_rule():
    """Tr_R(E_R rho_R) = Tr_C(E_C rho_C)."""
    rng = np.random.default_rng(2)
    for d in (1, 2, 3, 5):
        rho_C = random_complex_density(d, rng)
        E_C = random_complex_hermitian(d, rng)
        rho_R = sl.realify_state(rho_C)
        E_R = sl.realify_operator(E_C)
        lhs = np.trace(E_R @ rho_R).real
        rhs = np.trace(E_C @ rho_C).real
        assert abs(lhs - rhs) < 1e-12, f"d={d}: lhs={lhs} rhs={rhs}"


def test_canonical_J_properties():
    """J^2 = -I and J^T = -J."""
    for d in (1, 2, 3, 5, 8):
        J = sl.canonical_J(d)
        assert sl.is_orthogonal_complex_structure(J)
        n = 2 * d
        assert J.shape == (n, n)
        assert np.allclose(J @ J, -np.eye(n))
        assert np.allclose(J.T, -J)


def test_complex_unitary_realifies_to_orthogonal():
    """A complex unitary U becomes a real orthogonal matrix commuting with J."""
    rng = np.random.default_rng(3)
    for d in (2, 3, 5):
        X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        # QR for unitary
        Q, _ = np.linalg.qr(X)
        UR = sl.realify_operator(Q)
        n = 2 * d
        assert np.allclose(UR @ UR.T, np.eye(n), atol=1e-10)
        J = sl.canonical_J(d)
        assert np.allclose(UR @ J, J @ UR, atol=1e-10)
