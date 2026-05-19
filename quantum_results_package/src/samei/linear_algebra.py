"""Linear-algebra utilities used across the samei package.

All matrices are real numpy arrays unless explicitly typed as complex.
Conventions are documented in ``samei/__init__.py``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "canonical_J",
    "trace_norm",
    "random_real_density",
    "random_complex_density",
    "random_effect_complex",
    "orthonormal_basis_for_projector",
    "projectors_from_involution",
    "is_symmetric_involution",
]


def canonical_J(d: int) -> np.ndarray:
    """Canonical orthogonal complex structure on R^{2d}.

    J_d = [[0, -I_d], [I_d, 0]]; satisfies J^2 = -I and J^T = -J.
    """
    if d < 1:
        raise ValueError("d must be a positive integer")
    Id = np.eye(d)
    Z = np.zeros((d, d))
    return np.block([[Z, -Id], [Id, Z]])


def trace_norm(X: np.ndarray) -> float:
    """Schatten-1 norm (sum of singular values)."""
    return float(np.linalg.svd(np.asarray(X), compute_uv=False).sum())


def is_symmetric_involution(M: np.ndarray, atol: float = 1e-10) -> bool:
    n = M.shape[0]
    return bool(
        np.allclose(M, M.T, atol=atol)
        and np.allclose(M @ M, np.eye(n), atol=atol)
    )


def projectors_from_involution(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (P_+, P_-) = ((I - M)/2, (I + M)/2) for symmetric involution M."""
    if not is_symmetric_involution(M):
        raise ValueError("M must be a real symmetric involution")
    n = M.shape[0]
    I = np.eye(n)
    return 0.5 * (I - M), 0.5 * (I + M)


def random_real_density(
    n: int,
    rank: int | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random real density on R^n: rho = G G^T / Tr(G G^T) with G ~ N(0, 1)^{n x r}."""
    if rng is None:
        rng = np.random.default_rng(seed)
    r = n if rank is None else int(rank)
    if r < 1:
        raise ValueError("rank must be >= 1")
    G = rng.standard_normal((n, r))
    A = G @ G.T
    return A / np.trace(A)


def random_complex_density(
    d: int,
    rank: int | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random complex density on C^d: rho = G G^*/ Tr(G G^*)."""
    if rng is None:
        rng = np.random.default_rng(seed)
    r = d if rank is None else int(rank)
    G = rng.standard_normal((d, r)) + 1j * rng.standard_normal((d, r))
    A = G @ G.conj().T
    return A / np.trace(A).real


def random_effect_complex(
    d: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random PSD effect 0 <= E <= I on C^d. Uses E = X X^* / (lambda_max + eps)."""
    if rng is None:
        rng = np.random.default_rng(seed)
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    A = X @ X.conj().T
    A = 0.5 * (A + A.conj().T)
    lam = np.linalg.eigvalsh(A).max()
    return A / (lam + 1e-12)


def orthonormal_basis_for_projector(P: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """Return columns of U: an orthonormal basis of range(P).

    P must be a real symmetric projector (P^2 = P, P^T = P).
    """
    P_sym = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    cols = [eigvecs[:, i] for i, v in enumerate(eigvals) if v > 0.5]
    if not cols:
        return np.zeros((P.shape[0], 0))
    U = np.stack(cols, axis=1)
    return U
