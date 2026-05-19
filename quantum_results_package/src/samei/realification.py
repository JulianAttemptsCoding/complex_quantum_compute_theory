"""Realification map R: complex (d x d) matrices into real (2d x 2d).

R(X) = [[Re X, -Im X], [Im X, Re X]] is a unital ring homomorphism.
The state convention is rho_R = (1/2) R(rho_C); the effect convention is
E_R = R(E_C). With these, Tr_R(E_R rho_R) = Tr_C(E_C rho_C).
"""

from __future__ import annotations

import numpy as np

from .linear_algebra import random_complex_density, random_effect_complex

__all__ = ["R", "realify_state", "realify_effect", "born_check"]


def R(X_complex: np.ndarray) -> np.ndarray:
    """Realify a complex matrix to a real (2d x 2d) block matrix.

    Layout matches the canonical complex structure J = [[0, -I], [I, 0]]:
    R(X)(a + J b) = (Re X a - Im X b) + J (Im X a + Re X b).
    """
    X = np.asarray(X_complex)
    A = X.real
    B = X.imag
    return np.block([[A, -B], [B, A]])


def realify_state(rho_complex: np.ndarray) -> np.ndarray:
    """rho_R = (1/2) R(rho_C). Yields Tr_R(rho_R) = 1 when Tr_C(rho_C) = 1."""
    return 0.5 * R(rho_complex)


def realify_effect(E_complex: np.ndarray) -> np.ndarray:
    """E_R = R(E_C)."""
    return R(E_complex)


def born_check(
    d: int,
    n_trials: int = 200,
    rng: np.random.Generator | None = None,
) -> float:
    """Return max |Tr_R(E_R rho_R) - Tr_C(E_C rho_C)| over n_trials random samples.

    Uses random complex densities (full rank) and random PSD effects in [0, I].
    """
    if rng is None:
        rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(n_trials):
        rho_C = random_complex_density(d, rng=rng)
        E_C = random_effect_complex(d, rng=rng)
        p_C = np.trace(E_C @ rho_C).real
        rho_R = realify_state(rho_C)
        E_R = realify_effect(E_C)
        p_R = np.trace(E_R @ rho_R)
        max_err = max(max_err, abs(p_R - p_C))
    return float(max_err)
