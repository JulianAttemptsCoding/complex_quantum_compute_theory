"""Realification, complex structures, sector operators.

Implements the algebraic objects of Sections 2-3 of the manuscript:
canonical complex structure J on R^{2d}, the realification map R(X), the
sector operators A = J_A x I, B = I x J_B, alignment M = A B = J_A x J_B,
and the projectors P_pm = (I -/+ M)/2.

All operators are real numpy arrays. The complex Hilbert space H_C of complex
dimension d is represented as R^{2d} with J playing the role of i.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "canonical_J",
    "realify_operator",
    "realify_state",
    "sector_operators",
    "projectors",
    "induced_complex_structure",
    "is_orthogonal_complex_structure",
]


def canonical_J(d: int) -> np.ndarray:
    """Return the canonical complex structure on R^{2d}.

    J_d = [[0, -I_d], [I_d, 0]], a real (2d x 2d) matrix satisfying
    J^2 = -I and J^T = -J.

    Parameters
    ----------
    d : int
        Complex dimension. The resulting matrix has real dimension 2d.
    """
    if d < 1:
        raise ValueError("d must be a positive integer")
    Id = np.eye(d)
    Z = np.zeros((d, d))
    return np.block([[Z, -Id], [Id, Z]])


def is_orthogonal_complex_structure(J: np.ndarray, atol: float = 1e-12) -> bool:
    """Check that J^2 = -I and J^T = -J to tolerance atol."""
    n = J.shape[0]
    if J.shape != (n, n):
        return False
    return (
        np.allclose(J @ J, -np.eye(n), atol=atol)
        and np.allclose(J.T, -J, atol=atol)
    )


def realify_operator(X_complex: np.ndarray) -> np.ndarray:
    """Realify a complex (d x d) matrix X = A + iB to the real (2d x 2d)
    block [[A, -B], [B, A]].

    R is a unital *ring homomorphism*: R(XY) = R(X) R(Y). Hermiticity becomes
    real symmetry, complex unitarity becomes orthogonality commuting with J.
    """
    X = np.asarray(X_complex)
    A = X.real
    B = X.imag
    return np.block([[A, -B], [B, A]])


def realify_state(rho_complex: np.ndarray) -> np.ndarray:
    """Realify a complex density operator with the 1/2 prefactor.

    rho_R := (1/2) R(rho_C). This is exactly the factor that yields
    Tr_R(rho_R) = 1 from Tr_C(rho_C) = 1, and Tr_R(E_R rho_R) = Tr_C(E_C rho_C)
    for any effect E (Theorem 1).
    """
    return 0.5 * realify_operator(rho_complex)


def sector_operators(J_A: np.ndarray, J_B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (A_op, B_op, M) where A_op = J_A x I, B_op = I x J_B, M = A_op B_op.

    These act on the real tensor product space R^{2 d_A} x R^{2 d_B} of real
    dimension 4 d_A d_B.
    """
    dA = J_A.shape[0]
    dB = J_B.shape[0]
    A_op = np.kron(J_A, np.eye(dB))
    B_op = np.kron(np.eye(dA), J_B)
    M = A_op @ B_op  # equals np.kron(J_A, J_B)
    return A_op, B_op, M


def projectors(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (P_plus, P_minus) where P_plus = (I - M)/2 projects onto H_+ = ker(M+I),
    P_minus = (I + M)/2 projects onto H_- = ker(M-I).

    Sanity-checks that M is a real symmetric involution within tolerance.
    """
    n = M.shape[0]
    I = np.eye(n)
    if not np.allclose(M @ M, I, atol=1e-10):
        raise ValueError("M is not an involution within tolerance")
    if not np.allclose(M.T, M, atol=1e-10):
        raise ValueError("M is not symmetric within tolerance")
    P_plus = 0.5 * (I - M)
    P_minus = 0.5 * (I + M)
    return P_plus, P_minus


def induced_complex_structure(J_A: np.ndarray, J_B: np.ndarray) -> np.ndarray:
    """Return the operator J_AB := A_op restricted to H_+ as a full-space operator.

    By Theorem 3, on the aligned sector H_+ we have A_op = B_op and
    J_AB^2 = -I_{H_+}. As an operator on the full space, J_AB := P_+ A_op P_+.
    """
    A_op, _, M = sector_operators(J_A, J_B)
    P_plus, _ = projectors(M)
    return P_plus @ A_op @ P_plus
