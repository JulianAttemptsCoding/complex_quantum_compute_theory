"""Sector-parity discrimination primitive.

Channels:
    N_0(rho) = rho
    N_1(rho) = M rho M
Equal-prior Holevo-Helstrom optimum with leakage budget L(rho) <= eps is

    p_succ^{(eps)} = 1/2 + sqrt(m (1 - m)),   m = min(eps, 1/2).

Block-diagonal probes (rho commutes with M) yield p_succ = 1/2 exactly.
"""

from __future__ import annotations

import numpy as np

from .leakage import coherent_leakage
from .linear_algebra import trace_norm

__all__ = [
    "optimal_psucc_at_budget",
    "sector_parity_psucc",
    "saturating_state",
]


def optimal_psucc_at_budget(epsilon: float) -> float:
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    m = min(float(epsilon), 0.5)
    return 0.5 + np.sqrt(m * (1.0 - m))


def sector_parity_psucc(rho: np.ndarray, M: np.ndarray) -> float:
    """For a probe rho, p_succ = 1/2 + (1/4) || rho - M rho M ||_1
    = 1/2 + || P_+ rho P_- ||_1.
    """
    diff = rho - M @ rho @ M
    return 0.5 + 0.25 * trace_norm(diff)


def saturating_state(M: np.ndarray, epsilon: float) -> np.ndarray:
    """Coherent pure cross-sector probe with L = min(eps, 1/2) saturating the
    budget-constrained optimum.
    """
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    eigvals, eigvecs = np.linalg.eigh(0.5 * (M + M.T))
    psi_plus = None
    psi_minus = None
    for val, vec in zip(eigvals, eigvecs.T):
        if val < 0 and psi_plus is None:
            psi_plus = vec / np.linalg.norm(vec)
        elif val > 0 and psi_minus is None:
            psi_minus = vec / np.linalg.norm(vec)
        if psi_plus is not None and psi_minus is not None:
            break
    if psi_plus is None or psi_minus is None:
        raise RuntimeError("M lacks both +/-1 eigenvalues")
    m = min(float(epsilon), 0.5)
    psi = np.sqrt(1.0 - m) * psi_plus + np.sqrt(m) * psi_minus
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi)
