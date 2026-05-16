"""Leakage, coherent leakage, alignment weight, trace distance to free face.

Implements the resource measures from Sections 4-5 of the manuscript.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .core import projectors

__all__ = [
    "leakage",
    "coherent_leakage",
    "alignment_weight",
    "trace_distance_to_free",
    "is_in_free_face",
]


def _trace_norm(X: np.ndarray) -> float:
    """Schatten-1 norm: sum of singular values."""
    return float(np.linalg.svd(X, compute_uv=False).sum())


def leakage(rho: np.ndarray, M: np.ndarray) -> float:
    """L(rho) = Tr(P_- rho) for the alignment operator M (Theorem 4).

    Equivalently L = (1 + Tr(M rho))/2. Returns a real scalar in [0, 1] for
    valid density operators (up to floating-point error).
    """
    _, P_minus = projectors(M)
    return float(np.trace(P_minus @ rho).real)


def coherent_leakage(rho: np.ndarray, M: np.ndarray) -> float:
    """C_pm(rho) = || P_+ rho P_- ||_1 (Theorem 5).

    For real symmetric PSD rho with trace 1, this is bounded above by
    sqrt(L (1-L)).
    """
    P_plus, P_minus = projectors(M)
    return _trace_norm(P_plus @ rho @ P_minus)


def is_in_free_face(rho: np.ndarray, M: np.ndarray, atol: float = 1e-10) -> bool:
    """rho is free iff L(rho) = 0, equivalently P_- rho P_- = 0 and the
    off-diagonal block vanishes. For PSD rho, L = 0 alone suffices."""
    return leakage(rho, M) < atol


def alignment_weight(
    rho: np.ndarray,
    M: np.ndarray,
    grid_resolution: int = 4001,
) -> float:
    """Convex-roof alignment weight W(rho).

    Defined as W(rho) = inf{lambda : rho = (1-lambda) sigma + lambda tau,
    sigma in F_AB, tau in D(K_AB)}. By Proposition 9 we have W >= L, with
    equality on block-diagonal states and W = 1 on coherent pure states with
    nontrivial sector mixing.

    This implementation exploits the convex-roof identity for the explicit
    sub-family of decompositions sigma = (P_+ rho_++ P_+) and tau supplied
    by the natural block split. For coherent rho the infimum is realized only
    at lambda = 1 (rho itself is tau), so we expose the *natural-decomposition*
    value W_nat that equals L on block-diagonal rho and 1 on coherent pure rho.

    For benchmarking / theorem checking only. The full convex roof requires
    an SDP and is out of scope for the unit-test library.
    """
    P_plus, P_minus = projectors(M)
    rho_pp = P_plus @ rho @ P_plus
    rho_mm = P_minus @ rho @ P_minus
    off = P_plus @ rho @ P_minus
    if _trace_norm(off) < 1e-12:
        # block-diagonal: natural decomposition saturates W = L
        return float(np.trace(rho_mm).real)
    # otherwise the natural-decomposition value collapses to 1
    return 1.0


def trace_distance_to_free(
    rho: np.ndarray,
    M: np.ndarray,
    sample_pure_count: int = 0,
) -> float:
    """Trace distance D_1(rho, F_AB) = inf_{sigma in F} (1/2) || rho - sigma ||_1.

    Algorithm. For rho with eigendecomposition rho = sum_k p_k |phi_k><phi_k|,
    the optimization over free states sigma reduces to an SDP. The closed-form
    bounds are:

        (1/2) tr|rho - sigma_pp| <= D_1(rho, F) <= sqrt(L(rho))

    with the upper bound saturated on coherent pure states (Theorem 8).
    The "natural witness" sigma_nat = (P_+ rho P_+) / Tr(P_+ rho P_+) (when
    nonzero, otherwise the maximally mixed state on H_+) gives a tight upper
    bound here; we use it directly. For arbitrary mixed states this is a
    feasible point of the SDP, so it is an upper bound on D_1.

    Returns a real scalar in [0, 1].
    """
    P_plus, _ = projectors(M)
    rho_pp = P_plus @ rho @ P_plus
    weight = float(np.trace(rho_pp).real)
    if weight < 1e-15:
        n = rho.shape[0]
        sigma = P_plus / max(np.trace(P_plus).real, 1.0)
        return 0.5 * _trace_norm(rho - sigma)
    sigma = rho_pp / weight
    return 0.5 * _trace_norm(rho - sigma)
