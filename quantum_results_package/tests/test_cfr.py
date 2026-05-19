"""CFR two-rebit family: closed-form leakage and zero coherent leakage."""

import numpy as np

import samei


def test_cfr_endpoints_and_formula():
    M = samei.cfr_M()
    Pp, Pm = samei.projectors(M)
    assert np.allclose(samei.cfr_state(1.0), Pp / 2.0, atol=1e-12)
    assert np.allclose(samei.cfr_state(-1.0), Pm / 2.0, atol=1e-12)
    for a in np.linspace(-1.0, 1.0, 201):
        rho = samei.cfr_state(a)
        L = samei.leakage(rho, M)
        C = samei.coherent_leakage(rho, M)
        assert abs(L - (1.0 - a) / 2.0) <= 1e-12
        assert C <= 1e-12
