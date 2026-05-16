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
    "embed_complex_vector_to_H_plus",
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


def embed_complex_vector_to_H_plus(
    psi_C: np.ndarray,
    d_A: int,
    d_B: int,
) -> np.ndarray:
    """Canonical same-i embedding iota: (H_A ⊗_C H_B)_R → H_+ ⊂ K_AB.

    With canonical local complex structures J_A = canonical_J(d_A),
    J_B = canonical_J(d_B), the embedding is

        iota(|j>_C ⊗_C |k>_C) = (1/√2)(r_j ⊗ r_k - (J_A r_j) ⊗ (J_B r_k)),
        iota(i |j>_C ⊗_C |k>_C) = (1/√2)((J_A r_j) ⊗ r_k + r_j ⊗ (J_B r_k)),

    where r_j is the j-th canonical real basis vector of H_{A,R} = R^{2 d_A}
    (j = 0, ..., d_A - 1, representing |j>_C in the complex basis), and
    J_A r_j = e_{d_A + j} for the canonical J_A.

    Verified properties:
      - iota is a real-linear isometry onto H_+.
      - Pulled-back J_AB = A_op acts as multiplication by i in the complex basis.

    Parameters
    ----------
    psi_C : complex array of length d_A * d_B
        Complex amplitudes c_{jk} indexed by j * d_B + k.
    d_A, d_B : int
        Complex dimensions of the two factors.

    Returns
    -------
    psi_R : real array of length 4 * d_A * d_B
        Image in K_AB, supported in H_+.
    """
    psi_C = np.asarray(psi_C).reshape(-1)
    if psi_C.size != d_A * d_B:
        raise ValueError("psi_C must have length d_A * d_B")
    nA = 2 * d_A
    nB = 2 * d_B
    psi_R = np.zeros(nA * nB)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    for j in range(d_A):
        for k in range(d_B):
            c = complex(psi_C[j * d_B + k])
            a, b = c.real, c.imag
            r_j = np.zeros(nA); r_j[j] = 1.0
            Jr_j = np.zeros(nA); Jr_j[d_A + j] = 1.0
            r_k = np.zeros(nB); r_k[k] = 1.0
            Jr_k = np.zeros(nB); Jr_k[d_B + k] = 1.0
            real_part = np.kron(r_j, r_k) - np.kron(Jr_j, Jr_k)
            imag_part = np.kron(Jr_j, r_k) + np.kron(r_j, Jr_k)
            psi_R += inv_sqrt2 * (a * real_part + b * imag_part)
    return psi_R


def imaginarity_inequivalence_B() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Example B (Section 8): a bipartite *complex* state with nonzero entrywise
    imaginarity whose realification under the canonical same-i embedding lies
    entirely in H_+, so L = 0.

    Construction. Take the complex pure state
        |psi>_C = (|00> + i |11>) / sqrt(2)
    on C^2 ⊗_C C^2. Its density matrix rho_C has off-diagonal entries
        rho_C[0, 3] = -i/2,   rho_C[3, 0] = +i/2
    in the computational basis (clearly nonzero imaginary entries).

    Embed |psi>_C via the canonical isometry iota into H_+ ⊂ K_AB (real dim 16).
    The resulting real pure state has L = 0 by construction.

    Returns (rho_real, M, rho_complex).
    """
    d_A, d_B = 2, 2
    # Complex bipartite state vector with imaginary components.
    psi_C = np.zeros(d_A * d_B, dtype=complex)
    psi_C[0] = 1.0 / np.sqrt(2.0)            # |00>
    psi_C[d_A * d_B - 1] = 1.0j / np.sqrt(2.0)  # i |11>
    rho_complex = np.outer(psi_C, psi_C.conj())
    # Canonical embedding into H_+.
    psi_R = embed_complex_vector_to_H_plus(psi_C, d_A, d_B)
    rho_real = np.outer(psi_R, psi_R)
    JA = canonical_J(d_A)
    JB = canonical_J(d_B)
    _, _, M = sector_operators(JA, JB)
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
