"""Theorems 2 and 3: same-i quotient and sector decomposition."""

import numpy as np
import pytest

import samei_leakage as sl


@pytest.mark.parametrize("dA, dB", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)])
def test_M_is_real_symmetric_involution(dA, dB):
    JA = sl.canonical_J(dA)
    JB = sl.canonical_J(dB)
    _, _, M = sl.sector_operators(JA, JB)
    n = 4 * dA * dB
    assert M.shape == (n, n)
    assert np.allclose(M.T, M, atol=1e-12)
    assert np.allclose(M @ M, np.eye(n), atol=1e-12)


@pytest.mark.parametrize("dA, dB", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)])
def test_projector_completeness_and_orthogonality(dA, dB):
    JA = sl.canonical_J(dA)
    JB = sl.canonical_J(dB)
    _, _, M = sl.sector_operators(JA, JB)
    Pp, Pm = sl.projectors(M)
    n = 4 * dA * dB
    assert np.allclose(Pp + Pm, np.eye(n), atol=1e-12)
    assert np.allclose(Pp @ Pp, Pp, atol=1e-12)
    assert np.allclose(Pm @ Pm, Pm, atol=1e-12)
    assert np.allclose(Pp @ Pm, 0.0, atol=1e-12)
    assert np.allclose(Pp.T, Pp, atol=1e-12)
    assert np.allclose(Pm.T, Pm, atol=1e-12)


@pytest.mark.parametrize("dA, dB", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)])
def test_sector_dimensions(dA, dB):
    """dim H_+ = dim H_- = 2 dA dB (half the full real tensor space)."""
    JA = sl.canonical_J(dA)
    JB = sl.canonical_J(dB)
    _, _, M = sl.sector_operators(JA, JB)
    Pp, Pm = sl.projectors(M)
    dim_plus = int(round(np.trace(Pp).real))
    dim_minus = int(round(np.trace(Pm).real))
    assert dim_plus == 2 * dA * dB
    assert dim_minus == 2 * dA * dB


@pytest.mark.parametrize("dA, dB", [(1, 1), (2, 2), (2, 3)])
def test_aligned_sector_carries_complex_structure(dA, dB):
    """On H_+, A_op = B_op =: J_AB and J_AB^2 = -I_{H_+}."""
    JA = sl.canonical_J(dA)
    JB = sl.canonical_J(dB)
    A_op, B_op, M = sl.sector_operators(JA, JB)
    Pp, _ = sl.projectors(M)
    # restricted operators agree on H_+
    A_on_plus = Pp @ A_op @ Pp
    B_on_plus = Pp @ B_op @ Pp
    assert np.allclose(A_on_plus, B_on_plus, atol=1e-12)
    # J_AB^2 = -P_+ as an operator on the full space restricted to H_+
    J_AB = sl.induced_complex_structure(JA, JB)
    # (J_AB)^2 should equal -P_+ (since it kills H_-)
    assert np.allclose(J_AB @ J_AB, -Pp, atol=1e-12)


def test_M_eigenvalues_are_pm_one():
    """Spectrum of M is {+1, -1} with equal multiplicities."""
    JA = sl.canonical_J(2)
    JB = sl.canonical_J(3)
    _, _, M = sl.sector_operators(JA, JB)
    eigs = np.linalg.eigvalsh(M)
    # rounded to nearest +/- 1
    rounded = np.round(eigs).astype(int)
    n_plus_one = int((rounded == 1).sum())
    n_minus_one = int((rounded == -1).sum())
    assert n_plus_one + n_minus_one == M.shape[0]
    assert n_plus_one == n_minus_one
    assert np.allclose(np.abs(eigs), 1.0, atol=1e-10)
