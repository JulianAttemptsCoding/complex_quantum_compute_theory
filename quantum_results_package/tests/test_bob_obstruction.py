"""Bob marginal obstruction: for any product source rho_B1 x rho_B2 on K_B1,B2,
L = Tr(P_- (rho_B1 x rho_B2)) = 1/2 exactly.
"""

import numpy as np

import samei


def test_bob_marginal_leakage_one_half():
    dB1, dB2 = 1, 1  # smallest non-trivial case
    JB1 = samei.canonical_J(dB1)
    JB2 = samei.canonical_J(dB2)
    _, _, M = samei.sector_operators(JB1, JB2)
    rng = np.random.default_rng(0)
    max_dev = 0.0
    n1 = 2 * dB1
    n2 = 2 * dB2
    for _ in range(200):
        rho1 = samei.linear_algebra.random_real_density(n1, rng=rng)
        rho2 = samei.linear_algebra.random_real_density(n2, rng=rng)
        rho = np.kron(rho1, rho2)
        L = samei.leakage(rho, M)
        max_dev = max(max_dev, abs(L - 0.5))
    assert max_dev <= 1e-12
