"""Generate all figures for the manuscript.

Outputs PDFs into the figures/ directory. Each figure has a self-contained
data-generation routine using the samei_leakage library.

Usage:
    python figures/make_figures.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

import samei_leakage as sl

FIGDIR = os.path.dirname(os.path.abspath(__file__))

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 150,
})


def fig_discrimination(ax=None):
    """Sector-parity discrimination optimum vs leakage budget."""
    JA = sl.canonical_J(2)
    JB = sl.canonical_J(2)
    _, _, M = sl.sector_operators(JA, JB)
    eps_grid = np.linspace(0.0, 1.0, 51)
    # closed-form curve
    p_closed = np.array([sl.optimal_psucc_at_budget(e) for e in eps_grid])
    # numerical at saturating state
    p_num = np.array([sl.sector_parity_psucc(sl.saturating_state(M, e), M)
                      for e in eps_grid])
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(eps_grid, p_closed, "-", color="C0", lw=1.4,
            label=r"$\frac{1}{2}+\sqrt{m(1-m)},\ m=\min(\epsilon,\frac{1}{2})$")
    ax.plot(eps_grid, p_num, "o", color="C3", ms=3.0,
            label="numerical saturating probe")
    ax.axvline(0.5, color="gray", ls=":", lw=0.7)
    ax.set_xlabel(r"leakage budget $\epsilon$")
    ax.set_ylabel(r"$p_\mathrm{succ}^{(\epsilon)}$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.45, 1.04)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Sector-parity discrimination")
    return ax


def fig_cfr(ax=None):
    """CFR leakage L = (1-alpha)/2."""
    JA = sl.canonical_J(1)
    JB = sl.canonical_J(1)
    _, _, M = sl.sector_operators(JA, JB)
    a_grid = np.linspace(-1.0, 1.0, 41)
    L_num = np.array([sl.leakage(sl.cfr_state(a), M) for a in a_grid])
    L_closed = np.array([sl.cfr_leakage_closed_form(a) for a in a_grid])
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(a_grid, L_closed, "-", color="C0", lw=1.4, label=r"$L=(1-\alpha)/2$")
    ax.plot(a_grid, L_num, "o", color="C3", ms=3.0, label="numerical")
    ax.axhline(0.0, color="gray", ls=":", lw=0.7)
    ax.axhline(1.0, color="gray", ls=":", lw=0.7)
    ax.set_xlabel(r"CFR parameter $\alpha$")
    ax.set_ylabel(r"$L(\rho_{\mathrm{CFR}}(\alpha))$")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Caves--Fuchs--Rungta leakage")
    return ax


def fig_perturbative(ax=None):
    """Perturbative leakage scales as eps^2 (Theorem 10)."""
    rng = np.random.default_rng(0)
    JA = sl.canonical_J(2)
    JB = sl.canonical_J(2)
    A_op, B_op, M = sl.sector_operators(JA, JB)
    n = M.shape[0]
    Pp, Pm = sl.projectors(M)
    # G0 block-diagonal in M-eigenbasis -> [G0, M] = 0
    H_R0 = 1.0 * Pp + 0.5 * Pm
    G0 = sl.schrodinger_generator(H_R0, A_op)
    Braw = rng.standard_normal((n, n))
    Bskew = (Braw - Braw.T) / 2
    psi0 = np.linalg.eigh(M)[1][:, 0]  # H_+ state
    rho0 = np.outer(psi0, psi0)
    t = 0.5
    eps_grid = np.logspace(-3, -0.5, 12)
    L_vals = []
    for eps in eps_grid:
        G_eps = G0 + eps * Bskew
        rho_t = sl.evolve_hamiltonian(rho0, G_eps, t)
        L_vals.append(sl.leakage(rho_t, M))
    L_vals = np.array(L_vals)
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.loglog(eps_grid, L_vals, "o", color="C3", ms=4.0, label="numerical")
    # slope-2 reference, anchored at the smallest eps point
    ref_y = L_vals[0] * (eps_grid / eps_grid[0]) ** 2
    ax.loglog(eps_grid, ref_y, "-", color="C0", lw=1.4, label=r"$\propto \epsilon^2$")
    ax.set_xlabel(r"perturbation strength $\epsilon$")
    ax.set_ylabel(r"$L(\,|\psi(t)\rangle\langle\psi(t)|\,)$")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Perturbative leakage scaling")
    return ax


def fig_coherent_bound(ax=None):
    """Coherent leakage C vs population leakage L, with the sqrt(L(1-L)) bound
    and the family of pure-state saturators."""
    JA = sl.canonical_J(2)
    JB = sl.canonical_J(2)
    _, _, M = sl.sector_operators(JA, JB)
    rng = np.random.default_rng(0)
    n = M.shape[0]
    # Scatter: many random states (mixed and pure) in (L, C) plane
    Ls, Cs = [], []
    for _ in range(400):
        rank = int(rng.integers(1, n + 1))
        rho = sl.random_real_density(n, rank=rank, rng=rng)
        Ls.append(sl.leakage(rho, M))
        Cs.append(sl.coherent_leakage(rho, M))
    Ls = np.array(Ls)
    Cs = np.array(Cs)
    # Saturating curve (pure coherent cross-sector)
    lam = np.linspace(0, 1, 100)
    sat = np.sqrt(lam * (1 - lam))
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(lam, sat, "-", color="C0", lw=1.4, label=r"$\sqrt{L(1-L)}$")
    ax.plot(Ls, Cs, ".", color="C3", ms=2.0, alpha=0.5, label="random states")
    ax.set_xlabel(r"population leakage $L$")
    ax.set_ylabel(r"coherent leakage $\|P_+\rho P_-\|_1$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.55)
    ax.legend(loc="lower center", frameon=False)
    ax.set_title("Coherent leakage bound (Thm 5)")
    return ax


def main():
    for name, fn in [
        ("fig_discrimination", fig_discrimination),
        ("fig_cfr", fig_cfr),
        ("fig_perturbative", fig_perturbative),
        ("fig_coherent_bound", fig_coherent_bound),
    ]:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        fn(ax)
        fig.tight_layout()
        out_pdf = os.path.join(FIGDIR, f"{name}.pdf")
        out_png = os.path.join(FIGDIR, f"{name}.png")
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close(fig)
        print(f"wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
