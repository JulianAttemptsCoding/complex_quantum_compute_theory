"""Theorem 16: moment-matrix soundness."""

import numpy as np
import pytest

import samei_leakage as sl


def test_moment_matrix_is_psd():
    """For any word list and any real state rho, Gamma_{u,v} = Tr(u^T v rho) is PSD."""
    rng = np.random.default_rng(0)
    JA = sl.canonical_J(2)
    JB = sl.canonical_J(2)
    A_op, B_op, M = sl.sector_operators(JA, JB)
    Pp, Pm = sl.projectors(M)
    n = M.shape[0]
    # word list of small operators
    words = [np.eye(n), A_op, B_op, M, Pp, Pm]
    rho = sl.random_real_density(n, rng=rng)
    G = np.zeros((len(words), len(words)))
    for i, u in enumerate(words):
        for j, v in enumerate(words):
            G[i, j] = np.trace(u.T @ v @ rho).real
    Gsym = 0.5 * (G + G.T)
    eigs = np.linalg.eigvalsh(Gsym)
    assert eigs.min() > -1e-9, f"min eig {eigs.min()}"


def test_moment_matrix_psd_across_many_states():
    rng = np.random.default_rng(1)
    JA = sl.canonical_J(1)
    JB = sl.canonical_J(2)
    A_op, B_op, M = sl.sector_operators(JA, JB)
    n = M.shape[0]
    Pp, Pm = sl.projectors(M)
    # extra random words
    rand_words = [rng.standard_normal((n, n)) for _ in range(3)]
    words = [np.eye(n), A_op, B_op, M, Pp, Pm] + rand_words
    for _ in range(10):
        rho = sl.random_real_density(n, rng=rng)
        G = np.zeros((len(words), len(words)))
        for i, u in enumerate(words):
            for j, v in enumerate(words):
                G[i, j] = np.trace(u.T @ v @ rho).real
        Gsym = 0.5 * (G + G.T)
        eigs = np.linalg.eigvalsh(Gsym)
        assert eigs.min() > -1e-8, f"min eig {eigs.min()}"
