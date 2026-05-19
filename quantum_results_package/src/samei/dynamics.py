"""Perturbative leakage generation.

Set-up:
    psi_0 in H_+ (unit vector).
    G_0    real skew, [G_0, M] = 0   (sector-preserving generator).
    B      real skew, with cross-sector blocks  P_- B P_+ != 0.

Closed form to leading order:
    psi(eps, t) = exp((G_0 + eps B) t) psi_0
    a(t) = integral_0^t  P_-  exp(G_0 (t - s))  B  exp(G_0 s)  psi_0  ds
    L(psi(eps, t)) = eps^2 || a(t) ||^2 + O(eps^3).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm

from .sectors import projectors

__all__ = [
    "skew_commuting_with_M",
    "skew_cross_sector",
    "leakage_perturbative_curve",
    "duhamel_leading_coefficient",
]


def _skew(rng: np.random.Generator, n: int) -> np.ndarray:
    X = rng.standard_normal((n, n))
    return (X - X.T) / 2.0


def skew_commuting_with_M(
    M: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build G_0^T = -G_0 with [G_0, M] = 0.

    Take G_0 = P_+ S_+ P_+ + P_- S_- P_- with S_pm skew.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = M.shape[0]
    P_plus, P_minus = projectors(M)
    Sp = _skew(rng, n)
    Sm = _skew(rng, n)
    G0 = P_plus @ Sp @ P_plus + P_minus @ Sm @ P_minus
    G0 = (G0 - G0.T) / 2.0
    return G0


def skew_cross_sector(
    M: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build B^T = -B with P_- B P_+ != 0.

    Take B = (P_+ R P_-) - (P_+ R P_-)^T = P_+ R P_- - P_- R^T P_+.
    """
    if rng is None:
        rng = np.random.default_rng(1)
    n = M.shape[0]
    P_plus, P_minus = projectors(M)
    R = rng.standard_normal((n, n))
    cross = P_plus @ R @ P_minus
    B = cross - cross.T
    return B


def duhamel_leading_coefficient(
    G0: np.ndarray,
    B: np.ndarray,
    psi0: np.ndarray,
    M: np.ndarray,
    t: float,
    n_quad: int = 64,
) -> np.ndarray:
    """Compute a(t) = integral_0^t P_- exp(G0 (t-s)) B exp(G0 s) psi0 ds.

    Uses Gauss-Legendre nodes on [0, t]. G0 is skew so exp(G0 s) is orthogonal.
    """
    _, P_minus = projectors(M)
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    s_nodes = 0.5 * t * (nodes + 1.0)
    s_weights = 0.5 * t * weights
    acc = np.zeros_like(psi0)
    for s, w in zip(s_nodes, s_weights):
        left = expm(G0 * (t - s))
        right = expm(G0 * s)
        acc = acc + w * (P_minus @ left @ B @ right @ psi0)
    return acc


def leakage_perturbative_curve(
    M: np.ndarray,
    G0: np.ndarray,
    B: np.ndarray,
    psi0: np.ndarray,
    t: float = 1.0,
    eps_grid: np.ndarray | None = None,
) -> dict:
    """Compute numerical leakage L vs eps and leading coefficient ||a(t)||^2.

    Returns dict with keys:
        eps, L_numeric, L_leading, slope, a_norm_sq, t
    """
    _, P_minus = projectors(M)
    if eps_grid is None:
        eps_grid = np.logspace(-4, -1, 25)
    a = duhamel_leading_coefficient(G0, B, psi0, M, t)
    a_norm_sq = float(np.dot(a, a))
    L_numeric = []
    for eps in eps_grid:
        psi_e = expm((G0 + eps * B) * t) @ psi0
        L_numeric.append(float(np.dot(psi_e, P_minus @ psi_e)))
    L_numeric = np.array(L_numeric)
    L_leading = (eps_grid ** 2) * a_norm_sq
    half = len(eps_grid) // 2
    x = np.log10(eps_grid[:half])
    y = np.log10(np.maximum(L_numeric[:half], 1e-300))
    slope, _ = np.polyfit(x, y, 1)
    return {
        "eps": eps_grid,
        "L_numeric": L_numeric,
        "L_leading": L_leading,
        "slope": float(slope),
        "a_norm_sq": a_norm_sq,
        "t": float(t),
    }
