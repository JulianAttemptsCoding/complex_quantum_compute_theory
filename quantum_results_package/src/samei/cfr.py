"""Caves-Fuchs-Rungta two-rebit family.

Local real dimension 2 with J = [[0, 1], [-1, 0]].

    M           = J tensor J
    rho_CFR(a)  = (1/4) (I - a M)
    L(rho_CFR)  = (1 - a) / 2
    C(rho_CFR)  = 0  (block diagonal in H_+ ⊕ H_-)
    rho_CFR(1)  = P_+ / 2
    rho_CFR(-1) = P_- / 2
"""

from __future__ import annotations

import numpy as np

from .sectors import projectors

__all__ = ["cfr_J", "cfr_M", "cfr_projectors", "cfr_state", "cfr_leakage_closed_form"]


def cfr_J() -> np.ndarray:
    return np.array([[0.0, 1.0], [-1.0, 0.0]])


def cfr_M() -> np.ndarray:
    J = cfr_J()
    return np.kron(J, J)


def cfr_projectors() -> tuple[np.ndarray, np.ndarray]:
    return projectors(cfr_M())


def cfr_state(alpha: float) -> np.ndarray:
    """rho_CFR(alpha) = (1/4) (I_4 - alpha M)."""
    if not (-1.0 - 1e-12 <= alpha <= 1.0 + 1e-12):
        raise ValueError("alpha must be in [-1, 1]")
    return 0.25 * (np.eye(4) - alpha * cfr_M())


def cfr_leakage_closed_form(alpha: float) -> float:
    return (1.0 - float(alpha)) / 2.0
