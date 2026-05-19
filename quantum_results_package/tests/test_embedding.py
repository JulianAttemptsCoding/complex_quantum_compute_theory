"""Same-i embedding iota: balanced, lands in H_+, isometric, J_AB intertwining."""

import numpy as np
import pytest

import samei


@pytest.mark.parametrize("dA, dB", [(1, 1), (2, 2), (2, 3)])
def test_iota_balanced_in_Hplus_and_intertwines(dA, dB):
    rng = np.random.default_rng((dA, dB))
    JA = samei.canonical_J(dA)
    JB = samei.canonical_J(dB)
    A_op, B_op, M = samei.sector_operators(JA, JB)
    Pp, _ = samei.projectors(M)
    JAB = samei.induced_complex_structure(JA, JB)
    for _ in range(20):
        x = rng.standard_normal(2 * dA)
        y = rng.standard_normal(2 * dB)
        v = samei.iota_simple(x, y, JA, JB)
        # in H_+
        assert np.linalg.norm(Pp @ v - v) <= 1e-12
        # balanced
        v1 = samei.iota_simple(JA @ x, y, JA, JB)
        v2 = samei.iota_simple(x, JB @ y, JA, JB)
        assert np.linalg.norm(v1 - v2) <= 1e-12
        # J_AB iota(x, y) = iota(JA x, y)
        assert np.linalg.norm(JAB @ v - v1) <= 1e-12


def test_iota_isometric_for_orthonormal_inputs():
    dA, dB = 2, 2
    JA = samei.canonical_J(dA)
    JB = samei.canonical_J(dB)
    A_op, B_op, M = samei.sector_operators(JA, JB)
    # build orthonormal pairs (e_j, e_k) in canonical basis; for these,
    # ||iota(e_j, e_k)||^2 = 1 (since x.x = y.y = 1 and (JA x).(JB y) = 0)
    for j in range(2 * dA):
        for k in range(2 * dB):
            x = np.zeros(2 * dA); x[j] = 1.0
            y = np.zeros(2 * dB); y[k] = 1.0
            v = samei.iota_simple(x, y, JA, JB)
            assert abs(np.dot(v, v) - 1.0) <= 1e-12
