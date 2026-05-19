"""Moment-matrix soundness check.

For a finite word set W = {u_1, ..., u_N} of real operators on R^n and a
density rho, the moment matrix is

    Gamma[i, j] = Tr( u_i^T u_j  rho ).

Equivalently, with V the column-stack of u_i rho^{1/2}, the matrix is
Gamma = V^T V which is PSD by construction. Numerical eigenvalue floor
should be near zero.
"""

from __future__ import annotations

import numpy as np

from .linear_algebra import random_real_density
from .sectors import projectors

__all__ = ["build_moment_matrix", "default_word_set"]


def build_moment_matrix(words: list[np.ndarray], rho: np.ndarray) -> np.ndarray:
    """Gamma[i, j] = Tr(u_i^T u_j rho)."""
    N = len(words)
    Gamma = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            Gamma[i, j] = float(np.trace(words[i].T @ words[j] @ rho))
    return 0.5 * (Gamma + Gamma.T)


def default_word_set(
    J_A: np.ndarray,
    J_B: np.ndarray,
    n_extra: int = 4,
    seed: int | None = 0,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """A small finite real word set built from algebraic generators."""
    from .sectors import sector_operators

    A_op, B_op, M = sector_operators(J_A, J_B)
    n = M.shape[0]
    I = np.eye(n)
    P_plus, P_minus = projectors(M)
    words = [I, M, P_plus, P_minus, A_op, B_op]
    if rng is None:
        rng = np.random.default_rng(seed)
    for _ in range(n_extra):
        X = rng.standard_normal((n, n))
        words.append(0.5 * (X + X.T))
    return words
