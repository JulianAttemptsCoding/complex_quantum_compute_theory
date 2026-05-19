"""Realification preserves Born statistics."""

import numpy as np
import pytest

import samei
from samei.realification import R, realify_state, realify_effect, born_check


def test_R_is_ring_homomorphism():
    rng = np.random.default_rng(0)
    for d in (2, 3, 4):
        X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        Y = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        assert np.allclose(R(X @ Y), R(X) @ R(Y), atol=1e-12)


@pytest.mark.parametrize("d", [2, 3, 4])
def test_born_check(d):
    err = born_check(d, n_trials=50, rng=np.random.default_rng(d))
    assert err <= 1e-12
