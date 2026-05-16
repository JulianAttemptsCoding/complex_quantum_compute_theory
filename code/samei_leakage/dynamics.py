"""Hamiltonian and Lindblad dynamics on the real Hilbert space.

Implements Section 6 of the manuscript.

A real Hamiltonian is a real symmetric matrix H_R that commutes with the
global complex structure J (so it corresponds to a complex Hermitian operator).
The Schrodinger generator is G = -J H_R; it is skew-symmetric, hence e^{Gt}
is orthogonal.

A real Lindblad generator has the form
    L(rho) = G rho + rho G^T + sum_l (R_l rho R_l^T - (1/2){R_l^T R_l, rho}).
Its dual is
    L^dagger(X) = [X, G] + sum_l (R_l^T X R_l - (1/2){R_l^T R_l, X}).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = [
    "schrodinger_generator",
    "evolve_hamiltonian",
    "lindblad_generator_dual_on",
    "integrate_lindblad",
]


def schrodinger_generator(H_R: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Return G = -J H_R. Assumes [H_R, J] = 0 (i.e. H_R is complex-Hermitian).
    G is skew-symmetric so e^{Gt} is orthogonal."""
    return -J @ H_R


def evolve_hamiltonian(rho: np.ndarray, G: np.ndarray, t: float) -> np.ndarray:
    """rho(t) = O(t) rho O(t)^T with O(t) = e^{G t}."""
    O = expm(G * t)
    return O @ rho @ O.T


def lindblad_generator_dual_on(
    X: np.ndarray,
    G: np.ndarray,
    jump_ops: list[np.ndarray],
) -> np.ndarray:
    """L^dagger(X) for the dual generator, evaluated at operator X."""
    out = X @ G - G @ X  # [X, G]
    for R in jump_ops:
        RtR = R.T @ R
        out = out + R.T @ X @ R - 0.5 * (RtR @ X + X @ RtR)
    return out


def integrate_lindblad(
    rho0: np.ndarray,
    G: np.ndarray,
    jump_ops: list[np.ndarray],
    t: float,
    n_steps: int = 4000,
) -> np.ndarray:
    """Simple Euler integrator (good enough for short-time tests; not for
    production simulation). Returns rho(t)."""
    rho = rho0.copy()
    dt = t / n_steps
    for _ in range(n_steps):
        drho = G @ rho + rho @ G.T
        for R in jump_ops:
            RtR = R.T @ R
            drho = drho + R @ rho @ R.T - 0.5 * (RtR @ rho + rho @ RtR)
        rho = rho + dt * drho
    return rho
