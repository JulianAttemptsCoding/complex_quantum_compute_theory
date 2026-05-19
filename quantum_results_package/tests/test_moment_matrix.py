"""Moment matrix Gamma[i,j] = Tr(u_i^T u_j rho) is PSD by construction."""

import numpy as np

import samei
from samei.moment_matrix import build_moment_matrix, default_word_set
from samei.linear_algebra import random_real_density


def test_moment_matrix_psd_on_random_rho():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    n = M.shape[0]
    rng = np.random.default_rng(0)
    words = default_word_set(JA, JB, n_extra=4, rng=rng)
    min_eig = 0.0
    for _ in range(20):
        rho = random_real_density(n, rng=rng)
        G = build_moment_matrix(words, rho)
        e = float(np.linalg.eigvalsh(0.5 * (G + G.T)).min())
        min_eig = min(min_eig, e)
    assert min_eig >= -1e-10
