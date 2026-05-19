"""Trace distance to free face for coherent pure cross-sector states.

Closed-form: D_1(|psi><psi|, F) = sqrt(L) for psi = sqrt(1-L) psi_+ + sqrt(L) psi_-
with psi_+ in H_+, psi_- in H_-.
"""

import numpy as np

import samei
from samei.linear_algebra import trace_norm


def test_coherent_pure_trace_distance_sqrtL():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    Pp, _ = samei.projectors(M)
    rng = np.random.default_rng(0)
    for lam in np.linspace(0.0, 1.0, 21):
        rho, _, _ = samei.coherent_pure_state(M, lam, rng=rng)
        # witness: project onto H_+ and normalize
        w_tr = float(np.trace(Pp @ rho @ Pp))
        if w_tr > 1e-15:
            sigma = (Pp @ rho @ Pp) / w_tr
        else:
            sigma = Pp / max(float(np.trace(Pp)), 1.0)
        D_witness = 0.5 * trace_norm(rho - sigma)
        # closed form upper bound is sqrt(lam); witness saturates it for pure
        # coherent states because the off-block is rank-1.
        assert D_witness <= np.sqrt(lam) + 1e-10
        assert D_witness >= np.sqrt(lam) - 1e-10
