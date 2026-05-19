"""Sector operators, projectors, same-i embedding.

The aligned sector H_+ is the kernel of (M + I); the leakage sector H_- is the
kernel of (M - I). Concretely:

    A     = J_A tensor I_B
    B     = I_A tensor J_B
    M     = A B = J_A tensor J_B
    P_+   = (I - M) / 2
    P_-   = (I + M) / 2
"""

from __future__ import annotations

import numpy as np

from .linear_algebra import canonical_J, projectors_from_involution

__all__ = [
    "sector_operators",
    "projectors",
    "iota_simple",
    "induced_complex_structure",
]


def sector_operators(
    J_A: np.ndarray, J_B: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (A_op, B_op, M) = (J_A x I, I x J_B, J_A x J_B)."""
    dA = J_A.shape[0]
    dB = J_B.shape[0]
    A_op = np.kron(J_A, np.eye(dB))
    B_op = np.kron(np.eye(dA), J_B)
    M = A_op @ B_op
    return A_op, B_op, M


def projectors(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (P_+, P_-) projecting onto H_+ = ker(M+I), H_- = ker(M-I)."""
    return projectors_from_involution(M)


def iota_simple(
    x_real: np.ndarray,
    y_real: np.ndarray,
    J_A: np.ndarray,
    J_B: np.ndarray,
) -> np.ndarray:
    """Same-i tensor representative iota(x tensor_C y).

    For real vectors x in R^{2 d_A}, y in R^{2 d_B}:

        iota(x tensor_C y) = (1 / sqrt(2)) (x tensor y - (J_A x) tensor (J_B y))

    The image of iota over x, y lies in H_+, and iota is balanced over the
    complex action:

        iota((J_A x) tensor_C y) = iota(x tensor_C (J_B y)).
    """
    Jx = J_A @ x_real
    Jy = J_B @ y_real
    return (np.kron(x_real, y_real) - np.kron(Jx, Jy)) / np.sqrt(2.0)


def induced_complex_structure(J_A: np.ndarray, J_B: np.ndarray) -> np.ndarray:
    """J_AB := P_+ A P_+; acts on H_+ as the induced complex structure.

    On H_+, A_op = B_op, and J_AB^2 = -I on H_+.
    """
    A_op, _, M = sector_operators(J_A, J_B)
    P_plus, _ = projectors(M)
    return P_plus @ A_op @ P_plus
