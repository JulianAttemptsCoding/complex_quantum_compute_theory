"""Population leakage L and coherent leakage C, plus state factories.

    L(rho) = Tr(P_- rho)
    C(rho) = || P_+ rho P_- ||_1
    C(rho) <= sqrt(L (1 - L))            (envelope bound)
"""

from __future__ import annotations

import numpy as np

from .linear_algebra import trace_norm, random_real_density
from .sectors import projectors

__all__ = [
    "leakage",
    "coherent_leakage",
    "coherent_envelope_violation",
    "coherent_pure_state",
    "block_diagonal_state",
]


def leakage(rho: np.ndarray, M: np.ndarray) -> float:
    """L(rho) = Tr(P_- rho)."""
    _, P_minus = projectors(M)
    return float(np.trace(P_minus @ rho).real)


def coherent_leakage(rho: np.ndarray, M: np.ndarray) -> float:
    """C(rho) = || P_+ rho P_- ||_1 (Schatten-1 norm of the off-diagonal block)."""
    P_plus, P_minus = projectors(M)
    return trace_norm(P_plus @ rho @ P_minus)


def coherent_envelope_violation(rho: np.ndarray, M: np.ndarray) -> float:
    """C(rho) - sqrt(L(1-L)); should be <= 0 for valid real densities."""
    L = leakage(rho, M)
    C = coherent_leakage(rho, M)
    L = max(0.0, min(1.0, L))
    return C - np.sqrt(L * (1.0 - L))


def _unit_vec_in_range(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a unit vector uniformly in range(P) for a real symmetric projector P."""
    n = P.shape[0]
    v = rng.standard_normal(n)
    v = P @ v
    nrm = np.linalg.norm(v)
    if nrm < 1e-14:
        eigvals, eigvecs = np.linalg.eigh(0.5 * (P + P.T))
        for val, vec in zip(eigvals, eigvecs.T):
            if val > 0.5:
                return vec / np.linalg.norm(vec)
        raise RuntimeError("projector has empty range")
    return v / nrm


def coherent_pure_state(
    M: np.ndarray,
    lam: float,
    rng: np.random.Generator | None = None,
    psi_plus: np.ndarray | None = None,
    psi_minus: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a coherent pure state psi = sqrt(1-lam) psi_+ + sqrt(lam) psi_-.

    Returns (rho, psi_plus, psi_minus). Density rho = psi psi^T has
    L(rho) = lam and C(rho) = sqrt(lam (1 - lam)).
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")
    if rng is None:
        rng = np.random.default_rng(0)
    P_plus, P_minus = projectors(M)
    if psi_plus is None:
        psi_plus = _unit_vec_in_range(P_plus, rng)
    if psi_minus is None:
        psi_minus = _unit_vec_in_range(P_minus, rng)
    psi = np.sqrt(1.0 - lam) * psi_plus + np.sqrt(lam) * psi_minus
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi)
    return rho, psi_plus, psi_minus


def block_diagonal_state(
    M: np.ndarray,
    lam: float,
    rng: np.random.Generator | None = None,
    rank_plus: int | None = None,
    rank_minus: int | None = None,
) -> np.ndarray:
    """Return (1 - lam) rho_+ + lam rho_- with rho_pm supported on H_pm.

    By construction L = lam, C = 0.
    """
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0, 1]")
    if rng is None:
        rng = np.random.default_rng(0)
    P_plus, P_minus = projectors(M)
    n = M.shape[0]

    def _density_on(P: np.ndarray, rank: int | None) -> np.ndarray:
        rho = random_real_density(n, rank=rank, rng=rng)
        rho = P @ rho @ P
        tr = np.trace(rho)
        if abs(tr) < 1e-14:
            eigvals, eigvecs = np.linalg.eigh(0.5 * (P + P.T))
            v = None
            for val, vec in zip(eigvals, eigvecs.T):
                if val > 0.5:
                    v = vec
                    break
            return np.outer(v, v)
        return rho / tr

    rho_p = _density_on(P_plus, rank_plus)
    rho_m = _density_on(P_minus, rank_minus)
    return (1.0 - lam) * rho_p + lam * rho_m
