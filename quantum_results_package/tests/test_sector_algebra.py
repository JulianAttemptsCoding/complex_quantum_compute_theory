"""Sector projector algebra and rank gates."""

import numpy as np
import pytest

import samei


@pytest.mark.parametrize("dA, dB", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)])
def test_projector_algebra(dA, dB):
    JA = samei.canonical_J(dA)
    JB = samei.canonical_J(dB)
    A, B, M = samei.sector_operators(JA, JB)
    n = M.shape[0]
    I = np.eye(n)
    Pp, Pm = samei.projectors(M)
    assert np.allclose(Pp @ Pp, Pp, atol=1e-12)
    assert np.allclose(Pm @ Pm, Pm, atol=1e-12)
    assert np.allclose(Pp @ Pm, 0.0, atol=1e-12)
    assert np.allclose(Pp + Pm, I, atol=1e-12)
    assert np.allclose(M @ Pp, -Pp, atol=1e-12)
    assert np.allclose(M @ Pm, Pm, atol=1e-12)
    assert int(round(np.trace(Pp))) == 2 * dA * dB
    assert int(round(np.trace(Pm))) == 2 * dA * dB


def test_M_symmetric_involution_and_signs():
    JA = samei.canonical_J(2)
    JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    n = M.shape[0]
    assert np.allclose(M, M.T, atol=1e-12)
    assert np.allclose(M @ M, np.eye(n), atol=1e-12)
    eigs = np.linalg.eigvalsh(M)
    assert np.allclose(np.sort(eigs)[: n // 2], -np.ones(n // 2))
    assert np.allclose(np.sort(eigs)[n // 2 :], np.ones(n // 2))
