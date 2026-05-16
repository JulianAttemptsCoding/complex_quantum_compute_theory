"""Real CPTP channels, leakage monotonicity tests, Kraus characterization.

Implements Theorems 6, 7 of the manuscript.

A real CP map is given by Kraus operators {K_k}_k on R^n. Trace preservation
is sum_k K_k^T K_k = I. The channel acts as Phi(rho) = sum_k K_k rho K_k^T.
The dual is Phi^dagger(X) = sum_k K_k^T X K_k.
"""

from __future__ import annotations

import numpy as np

from .core import projectors

__all__ = [
    "apply_kraus",
    "kraus_dual",
    "is_trace_preserving",
    "is_leakage_nonincreasing",
    "kraus_no_leakage",
    "is_strongly_sector_preserving",
]


def apply_kraus(rho: np.ndarray, kraus_list: list[np.ndarray]) -> np.ndarray:
    """Phi(rho) = sum_k K_k rho K_k^T."""
    return sum(K @ rho @ K.T for K in kraus_list)


def kraus_dual(X: np.ndarray, kraus_list: list[np.ndarray]) -> np.ndarray:
    """Phi^dagger(X) = sum_k K_k^T X K_k."""
    return sum(K.T @ X @ K for K in kraus_list)


def is_trace_preserving(kraus_list: list[np.ndarray], atol: float = 1e-10) -> bool:
    n = kraus_list[0].shape[1]
    S = sum(K.T @ K for K in kraus_list)
    return bool(np.allclose(S, np.eye(n), atol=atol))


def _psd(X: np.ndarray, atol: float = 1e-9) -> bool:
    Xs = 0.5 * (X + X.T)
    eigs = np.linalg.eigvalsh(Xs)
    return bool(eigs.min() >= -atol)


def is_leakage_nonincreasing(
    kraus_list: list[np.ndarray], M: np.ndarray, atol: float = 1e-9
) -> bool:
    """Test Theorem 6: Phi is leakage-nonincreasing for all rho iff
    Phi^dagger(P_-) <= P_-, i.e. P_- - Phi^dagger(P_-) is PSD.
    """
    _, P_minus = projectors(M)
    diff = P_minus - kraus_dual(P_minus, kraus_list)
    return _psd(diff, atol=atol)


def kraus_no_leakage(
    kraus_list: list[np.ndarray], M: np.ndarray, atol: float = 1e-10
) -> bool:
    """Theorem 7: P_- K_k P_+ = 0 for every k.

    This is equivalent to Phi^dagger(P_-) <= P_- under trace preservation.
    """
    P_plus, P_minus = projectors(M)
    for K in kraus_list:
        if not np.allclose(P_minus @ K @ P_plus, 0.0, atol=atol):
            return False
    return True


def is_strongly_sector_preserving(
    kraus_list: list[np.ndarray], M: np.ndarray, atol: float = 1e-10
) -> bool:
    """Strongly sector-preserving channel: [K_k, M] = 0 for every k.

    By the remarks following Theorem 7, this implies exact leakage conservation
    L(Phi(rho)) = L(rho).
    """
    for K in kraus_list:
        if not np.allclose(K @ M - M @ K, 0.0, atol=atol):
            return False
    return True
