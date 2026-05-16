"""Theorems 6 and 7: leakage monotonicity and Kraus characterization."""

import numpy as np
import pytest

import samei_leakage as sl


def _setup(dA=1, dB=1):
    JA = sl.canonical_J(dA)
    JB = sl.canonical_J(dB)
    _, _, M = sl.sector_operators(JA, JB)
    return M


def test_identity_channel_conserves_leakage():
    M = _setup(2, 2)
    n = M.shape[0]
    kraus = [np.eye(n)]
    assert sl.is_trace_preserving(kraus)
    assert sl.is_leakage_nonincreasing(kraus, M)
    assert sl.kraus_no_leakage(kraus, M)


def test_projector_to_free_face_is_leakage_decreasing():
    """The channel rho -> (P_+ rho P_+ + P_- rho P_-) (sector dephasing) cannot
    increase leakage; in fact it preserves it because it's strongly sector-
    preserving on populations. We use a Kraus form K_0 = P_+, K_1 = P_-."""
    M = _setup(2, 2)
    Pp, Pm = sl.projectors(M)
    kraus = [Pp, Pm]
    assert sl.is_trace_preserving(kraus)
    assert sl.kraus_no_leakage(kraus, M)
    assert sl.is_leakage_nonincreasing(kraus, M)


def test_sector_flip_channel_violates_no_leakage():
    """The channel rho -> M rho M = (P_- - P_+) rho (P_- - P_+) commutes with
    M, hence is *strongly sector-preserving*, hence leakage-conserving, but
    its Kraus operator does not satisfy P_- K P_+ = 0 unless we check that
    M commutes with M (trivially). This channel preserves L exactly."""
    M = _setup(2, 2)
    kraus = [M]
    assert sl.is_trace_preserving(kraus)
    assert sl.is_strongly_sector_preserving(kraus, M)
    rng = np.random.default_rng(0)
    for _ in range(5):
        rho = sl.random_real_density(M.shape[0], rng=rng)
        L_before = sl.leakage(rho, M)
        rho_after = sl.apply_kraus(rho, kraus)
        L_after = sl.leakage(rho_after, M)
        assert abs(L_before - L_after) < 1e-12


def test_explicit_leakage_increasing_channel_is_detected():
    """The channel that swaps H_+ and H_- (a partial isometry mapping H_+ -> H_-)
    *increases* leakage on H_+ states. We construct it by taking K = swap on the
    M-eigenbasis, and verify both the dual condition and the Kraus condition fail."""
    M = _setup(1, 2)
    n = M.shape[0]
    eigvals, eigvecs = np.linalg.eigh(M)
    # find indices of -1 and +1 blocks
    idx_plus = np.where(eigvals < 0)[0]   # H_+ has eigenvalue -1 of M
    idx_minus = np.where(eigvals > 0)[0]  # H_- has eigenvalue +1 of M
    # build K = swap H_+ <-> H_- in the eigenbasis
    K_eig = np.zeros((n, n))
    for i, j in zip(idx_plus, idx_minus):
        K_eig[j, i] = 1.0  # |H_-_j><H_+_i|
        K_eig[i, j] = 1.0  # |H_+_i><H_-_j|
    K = eigvecs @ K_eig @ eigvecs.T
    kraus = [K]
    assert sl.is_trace_preserving(kraus)
    # The swap is NOT leakage-nonincreasing on H_+ states
    assert not sl.kraus_no_leakage(kraus, M)
    assert not sl.is_leakage_nonincreasing(kraus, M)
    # explicit verification: a free state becomes maximally leaky
    Pp, _ = sl.projectors(M)
    rho_free = Pp / int(round(np.trace(Pp).real))
    rho_after = sl.apply_kraus(rho_free, kraus)
    assert sl.leakage(rho_free, M) < 1e-12
    assert sl.leakage(rho_after, M) > 1.0 - 1e-10


def test_two_kraus_block_diagonal_implies_monotonicity():
    """A two-Kraus channel built from random sector-block-diagonal orthogonals
    (so each K_k commutes with M, hence P_- K_k P_+ = 0) is trace-preserving,
    no-leakage, and leakage-nonincreasing on every state."""
    M = _setup(2, 2)
    n = M.shape[0]
    Pp, Pm = sl.projectors(M)
    rng = np.random.default_rng(7)
    n_plus = int(round(np.trace(Pp).real))
    n_minus = int(round(np.trace(Pm).real))
    eigvals, eigvecs = np.linalg.eigh(M)
    idx_plus = eigvals < 0
    idx_minus = eigvals > 0
    U_plus = eigvecs[:, idx_plus]
    U_minus = eigvecs[:, idx_minus]

    def random_block_diagonal_orthogonal():
        Q_p, _ = np.linalg.qr(rng.standard_normal((n_plus, n_plus)))
        Q_m, _ = np.linalg.qr(rng.standard_normal((n_minus, n_minus)))
        return U_plus @ Q_p @ U_plus.T + U_minus @ Q_m @ U_minus.T

    K_a = random_block_diagonal_orthogonal() / np.sqrt(2.0)
    K_b = random_block_diagonal_orthogonal() / np.sqrt(2.0)
    kraus = [K_a, K_b]
    assert sl.is_trace_preserving(kraus, atol=1e-10)
    assert sl.kraus_no_leakage(kraus, M)
    assert sl.is_strongly_sector_preserving(kraus, M)
    assert sl.is_leakage_nonincreasing(kraus, M)
    # Explicit leakage conservation on random states
    for _ in range(5):
        rho = sl.random_real_density(n, rng=rng)
        L_before = sl.leakage(rho, M)
        rho_after = sl.apply_kraus(rho, kraus)
        L_after = sl.leakage(rho_after, M)
        assert abs(L_before - L_after) < 1e-10
