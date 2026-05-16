"""Worked examples (Sections 8-10 of the manuscript).

- Caves-Fuchs-Rungta two-rebit family (Proposition 14).
- Bob-node obstruction for product bilocal sources (Proposition 15).
- Imaginarity inequivalence examples A and B (Section 8).
"""

from __future__ import annotations

import numpy as np

from .core import canonical_J, sector_operators, projectors, realify_state
from .leakage import leakage

__all__ = [
    "cfr_state",
    "cfr_leakage_closed_form",
    "bob_node_leakage_product",
    "imaginarity_inequivalence_A",
    "imaginarity_inequivalence_B",
    "haar_real_state",
    "random_real_density",
]


# --------------------------------------------------------------------- CFR --


# Standard 2x2 Paulis. sigma_y has complex entries, but sigma_y x sigma_y is
# real (two purely imaginary factors yield a real product).
_SIGMA_0 = np.array([[1.0, 0.0], [0.0, 1.0]])
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
_SIGMA_Y_OTIMES_SIGMA_Y = np.real(np.kron(_SIGMA_Y, _SIGMA_Y))


def cfr_state(alpha: float) -> np.ndarray:
    """rho_CFR(alpha) = (1/4)(sigma_0 x sigma_0 + alpha sigma_y x sigma_y).

    Returns a 4x4 real symmetric PSD density matrix for alpha in [-1, 1].
    """
    if not (-1.0 - 1e-12 <= alpha <= 1.0 + 1e-12):
        raise ValueError("alpha must be in [-1, 1]")
    return 0.25 * (np.kron(_SIGMA_0, _SIGMA_0) + alpha * _SIGMA_Y_OTIMES_SIGMA_Y)


def cfr_leakage_closed_form(alpha: float) -> float:
    """L(rho_CFR(alpha)) = (1 - alpha) / 2 (Proposition 14)."""
    return (1.0 - alpha) / 2.0


# ----------------------------------------------------------- Bob node ------


def bob_node_leakage_product(
    rho_B1: np.ndarray,
    rho_B2: np.ndarray,
    J_B1: np.ndarray | None = None,
    J_B2: np.ndarray | None = None,
) -> float:
    """L_B = Tr(P_- (rho_B1 x rho_B2)) for product bilocal sources.

    By Proposition 15, this equals 1/2 exactly for any product source, since
    Tr(J rho) = 0 for real symmetric rho and real antisymmetric J.
    """
    dA = rho_B1.shape[0] // 2
    dB = rho_B2.shape[0] // 2
    if J_B1 is None:
        J_B1 = canonical_J(dA)
    if J_B2 is None:
        J_B2 = canonical_J(dB)
    _, _, M = sector_operators(J_B1, J_B2)
    rho = np.kron(rho_B1, rho_B2)
    return leakage(rho, M)


# -------------------------------------------------------- Imaginarity ------


def imaginarity_inequivalence_A() -> tuple[np.ndarray, np.ndarray]:
    """Example A (Section 8): an entrywise-real density on K_AB (with d_A=d_B=1,
    so the realified bipartite real space has real dim 4) with L = 1/2 but zero
    entrywise imaginarity in the canonical basis.

    Construction: in the eigenbasis of M, pick orthonormal entrywise-real
    eigenvectors psi_+ in H_+ and psi_- in H_-. Form psi = (psi_+ + psi_-)/sqrt(2).
    Then rho = |psi><psi| has all real entries (by construction) and L = 1/2.

    Returns (rho, M) where M = J_A x J_B with J_A = J_B = canonical_J(1).
    """
    JA = canonical_J(1)
    JB = canonical_J(1)
    _, _, M = sector_operators(JA, JB)
    eigvals, eigvecs = np.linalg.eigh(M)
    psi_plus = None
    psi_minus = None
    for val, vec in zip(eigvals, eigvecs.T):
        v = np.real(vec)
        v = v / np.linalg.norm(v)
        if val < 0 and psi_plus is None:
            psi_plus = v
        elif val > 0 and psi_minus is None:
            psi_minus = v
    psi = (psi_plus + psi_minus) / np.sqrt(2.0)
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi)
    return rho, M


def imaginarity_inequivalence_B() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Example B (Section 8): a bipartite state whose *complex* representation
    has nonzero entrywise imaginarity (nonzero sigma_y component) but whose
    realification sits entirely in H_+, so L = 0.

    Take rho_C = (1/2)(I + sigma_y) on a single complex qubit (a pure state of
    sigma_y eigenvalue +1; its 2x2 matrix has a -i/2 entry at position [0,1]).
    Then rho_AB^C = rho_C x rho_C is a complex 2-qubit pure state with nonzero
    imaginary entries in the computational basis.

    Realifying *each factor* with the canonical J = canonical_J(1) and taking
    the real tensor product of the two realified single-qubit states gives a
    rho_real on K_AB (real dim 4 x 4 = 16). On the same-i sector decomposition
    (M = J_A x J_B with the *bipartite* J_A, J_B acting on each realified
    qubit) we get L = 0.

    Returns (rho_real, M, rho_complex) for inspection.
    """
    sigma_y_complex = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    I_complex = np.eye(2)
    rho_complex_single = 0.5 * (I_complex + sigma_y_complex)
    # realify the single-system complex state
    rho_real_single = realify_state(rho_complex_single)  # 4x4 real
    # take real tensor product of the two systems
    rho_real = np.kron(rho_real_single, rho_real_single)  # 16x16
    # The local complex structure on each realified factor is canonical_J(2)
    JA = canonical_J(2)
    JB = canonical_J(2)
    _, _, M = sector_operators(JA, JB)
    # Full complex state for reference
    rho_complex = np.kron(rho_complex_single, rho_complex_single)
    return rho_real, M, rho_complex


# ----------------------------------------------------- Random states -------


def random_real_density(n: int, rank: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random real density operator on R^n: rho = X X^T / Tr(X X^T) with
    X drawn from a (n x rank) Ginibre-type real Gaussian."""
    if rng is None:
        rng = np.random.default_rng()
    r = rank if rank is not None else n
    X = rng.standard_normal((n, r))
    A = X @ X.T
    return A / np.trace(A)


def haar_real_state(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random Haar-uniform real pure state |psi><psi| on R^n."""
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(n)
    v = v / np.linalg.norm(v)
    return np.outer(v, v)
