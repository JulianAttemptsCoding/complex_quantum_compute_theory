"""Sector-parity discrimination (Section 7).

Channels:
    N_0(rho) = rho,
    N_1(rho) = M rho M.

Equal-prior Holevo-Helstrom optimal success probability with leakage budget
L(rho) <= eps is
    p_succ^{(eps)} = 1/2 + sqrt(m(1-m)),  m = min(eps, 1/2).
"""

from __future__ import annotations

import numpy as np

from .core import projectors
from .leakage import leakage, coherent_leakage

__all__ = [
    "sector_parity_psucc",
    "optimal_psucc_at_budget",
    "saturating_state",
]


def sector_parity_psucc(rho: np.ndarray, M: np.ndarray) -> float:
    """Probe-specific success probability:
        p_succ(rho) = 1/2 + ||P_+ rho P_-||_1.
    Derived from ||rho - M rho M||_1 = 4 ||P_+ rho P_-||_1 and Holevo-Helstrom.
    """
    return 0.5 + coherent_leakage(rho, M)


def optimal_psucc_at_budget(epsilon: float) -> float:
    """Closed-form optimum p_succ^{(eps)} = 1/2 + sqrt(m(1-m)), m = min(eps, 1/2)."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    m = min(epsilon, 0.5)
    return 0.5 + np.sqrt(m * (1.0 - m))


def saturating_state(M: np.ndarray, epsilon: float) -> np.ndarray:
    """A pure coherent cross-sector state with leakage L = min(eps, 1/2) that
    saturates the budget-constrained discrimination optimum.

    Picks unit vectors psi_+ in H_+ and psi_- in H_- and forms
        psi = sqrt(1-m) psi_+ + sqrt(m) psi_-,    m = min(eps, 1/2).
    """
    P_plus, P_minus = projectors(M)
    n = M.shape[0]
    # Eigenvectors of M in numerically clean basis
    eigvals, eigvecs = np.linalg.eigh(M)
    psi_plus = None
    psi_minus = None
    for val, vec in zip(eigvals, eigvecs.T):
        if val < 0 and psi_plus is None:
            psi_plus = vec
        elif val > 0 and psi_minus is None:
            psi_minus = vec
        if psi_plus is not None and psi_minus is not None:
            break
    if psi_plus is None or psi_minus is None:
        raise RuntimeError("M does not have both +1 and -1 eigenvalues")
    m = min(epsilon, 0.5)
    psi = np.sqrt(1.0 - m) * psi_plus + np.sqrt(m) * psi_minus
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi)
