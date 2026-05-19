"""Real CPTP channels: free operations and Kraus criterion.

A real CP map is given by Kraus operators {K_k} on R^n:
    Phi(rho)        = sum_k K_k rho K_k^T
    Phi^dagger(X)   = sum_k K_k^T X K_k
Trace preservation: sum_k K_k^T K_k = I.
Free (leakage-nonincreasing) iff P_- K_k P_+ = 0 for all k iff
Phi^dagger(P_-) <= P_-.
"""

from __future__ import annotations

import numpy as np

from .sectors import projectors

__all__ = [
    "apply_kraus",
    "kraus_dual",
    "is_trace_preserving",
    "is_leakage_nonincreasing",
    "kraus_no_leakage",
    "random_sector_block_channel",
    "violating_channel_example",
    "compression_residual",
]


def apply_kraus(rho: np.ndarray, kraus_list: list[np.ndarray]) -> np.ndarray:
    return sum(K @ rho @ K.T for K in kraus_list)


def kraus_dual(X: np.ndarray, kraus_list: list[np.ndarray]) -> np.ndarray:
    return sum(K.T @ X @ K for K in kraus_list)


def is_trace_preserving(
    kraus_list: list[np.ndarray], atol: float = 1e-10
) -> bool:
    n = kraus_list[0].shape[1]
    S = sum(K.T @ K for K in kraus_list)
    return bool(np.allclose(S, np.eye(n), atol=atol))


def is_leakage_nonincreasing(
    kraus_list: list[np.ndarray], M: np.ndarray, atol: float = 1e-9
) -> bool:
    _, P_minus = projectors(M)
    diff = P_minus - kraus_dual(P_minus, kraus_list)
    eigs = np.linalg.eigvalsh(0.5 * (diff + diff.T))
    return bool(eigs.min() >= -atol)


def kraus_no_leakage(
    kraus_list: list[np.ndarray], M: np.ndarray, atol: float = 1e-10
) -> bool:
    """Free criterion: P_- K_k P_+ = 0 for every k."""
    P_plus, P_minus = projectors(M)
    return all(
        np.allclose(P_minus @ K @ P_plus, 0.0, atol=atol) for K in kraus_list
    )


def compression_residual(
    kraus_list: list[np.ndarray], M: np.ndarray
) -> float:
    """max_k || P_- K_k P_+ ||_F. Zero iff channel is free."""
    P_plus, P_minus = projectors(M)
    return max(
        float(np.linalg.norm(P_minus @ K @ P_plus, "fro")) for K in kraus_list
    )


def random_sector_block_channel(
    M: np.ndarray,
    n_kraus: int = 3,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Generate random real Kraus operators with P_- K_k P_+ = 0 and TP.

    Construction: each K_k = U_+ A_k U_+^T + U_- B_k U_+^T_offdiag (zero) + ...
    Concretely we take K_k = P_+ X_k P_+ + P_- Y_k P_+ * 0 + P_- Z_k P_-
    so that P_- K_k P_+ = 0. Then we normalize sum_k K_k^T K_k = I via
    Cholesky.
    """
    n = M.shape[0]
    if rng is None:
        rng = np.random.default_rng(seed)
    P_plus, P_minus = projectors(M)
    raw = []
    for _ in range(n_kraus):
        Xp = rng.standard_normal((n, n))
        Xm = rng.standard_normal((n, n))
        K = P_plus @ Xp @ P_plus + P_minus @ Xm @ P_minus
        raw.append(K)
    S = sum(K.T @ K for K in raw)
    S = 0.5 * (S + S.T) + 1e-12 * np.eye(n)
    eigvals, eigvecs = np.linalg.eigh(S)
    S_inv_half = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return [K @ S_inv_half for K in raw]


def violating_channel_example(
    M: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (kraus_list, witness_rho) for a channel that *increases* leakage.

    Build a single-Kraus orthogonal map V that swaps a unit vector
    psi_+ in H_+ with a unit vector psi_- in H_-. Then for rho = |psi_+><psi_+|,
    L(Phi(rho)) = 1 > L(rho) = 0.
    """
    n = M.shape[0]
    P_plus, P_minus = projectors(M)
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
    V = np.eye(n) - np.outer(psi_plus, psi_plus) - np.outer(psi_minus, psi_minus)
    V = V + np.outer(psi_minus, psi_plus) + np.outer(psi_plus, psi_minus)
    rho = np.outer(psi_plus, psi_plus)
    return [V], rho
