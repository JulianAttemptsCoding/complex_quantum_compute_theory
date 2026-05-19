"""Generate all figures (PDF + SVG) and CSV data tables.

Figures:
    fig1_sector_diagram.{pdf,svg}                 (matplotlib vector diagram)
    fig2_cfr_leakage.{pdf,svg}                    (CFR closed-form curve)
    fig3_coherent_leakage_bound.{pdf,svg}         (envelope + random + pure + BD)
    fig4_sector_parity_discrimination.{pdf,svg}   (optimum + saturators + baseline)
    fig5_perturbative_leakage.{pdf,svg}           (eps^2 sanity check)

Data CSVs:
    fig2_cfr_leakage.csv
    fig3_coherent_leakage_random.csv
    fig3_coherent_leakage_curves.csv
    fig4_discrimination.csv
    fig5_perturbative_scaling.csv
    sanity_check_summary.csv  (written by export_results_tex.py)

Output dirs are package-relative: figures/, data/.
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import samei  # noqa: E402

FIG_DIR = os.path.join(ROOT, "figures")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 20260517

mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 150,
})


def _save(fig, name):
    pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    svg = os.path.join(FIG_DIR, f"{name}.svg")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {svg}")


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"wrote {path}")


# --------------------------------------------------------------------- fig1 --


def fig1_sector_diagram():
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_axis_off()

    box_kw = dict(edgecolor="black", facecolor="#f4f4f4", linewidth=1.0)
    ax.add_patch(Rectangle((0.2, 3.3), 2.4, 1.2, **box_kw))
    ax.text(1.4, 3.9, r"$H_{A,\mathbb{R}},\ J_A$", ha="center", va="center", fontsize=10)
    ax.add_patch(Rectangle((0.2, 0.4), 2.4, 1.2, **box_kw))
    ax.text(1.4, 1.0, r"$H_{B,\mathbb{R}},\ J_B$", ha="center", va="center", fontsize=10)

    ax.add_patch(Rectangle((3.6, 1.6), 2.6, 1.8,
                           edgecolor="black", facecolor="#eef3fb", linewidth=1.0))
    ax.text(4.9, 2.5,
            r"$K_{AB}=H_{A,\mathbb{R}}\otimes_{\mathbb{R}}H_{B,\mathbb{R}}$",
            ha="center", va="center", fontsize=10)

    for (x0, y0) in [(2.6, 3.9), (2.6, 1.0)]:
        ax.add_patch(FancyArrowPatch((x0, y0), (3.6, 2.5),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="black", linewidth=0.8))

    ax.add_patch(Rectangle((7.0, 2.7), 2.8, 1.4,
                           edgecolor="black", facecolor="#e8f3ea", linewidth=1.0))
    ax.text(8.4, 3.7, r"$H_+ = \ker(M+I)$", ha="center", va="center", fontsize=10)
    ax.text(8.4, 3.25, r"$M = -I,\ P_+=(I-M)/2$", ha="center", va="center", fontsize=8.5)
    ax.text(8.4, 2.92, r"$(H_{A,\mathbb{C}}\otimes_{\mathbb{C}}H_{B,\mathbb{C}})_{\mathbb{R}}\cong H_+$",
            ha="center", va="center", fontsize=8.5)

    ax.add_patch(Rectangle((7.0, 0.9), 2.8, 1.4,
                           edgecolor="black", facecolor="#fdecec", linewidth=1.0))
    ax.text(8.4, 1.9, r"$H_- = \ker(M-I)$", ha="center", va="center", fontsize=10)
    ax.text(8.4, 1.45, r"$M = +I,\ P_-=(I+M)/2$", ha="center", va="center", fontsize=8.5)
    ax.text(8.4, 1.12, r"leakage sector", ha="center", va="center", fontsize=8.5,
            style="italic")

    ax.add_patch(FancyArrowPatch((6.2, 2.7), (7.0, 3.3),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color="black", linewidth=0.8))
    ax.add_patch(FancyArrowPatch((6.2, 2.3), (7.0, 1.7),
                                 arrowstyle="-|>", mutation_scale=10,
                                 color="black", linewidth=0.8))
    ax.text(6.6, 3.05, r"$P_+$", fontsize=9, ha="left", va="center")
    ax.text(6.6, 1.95, r"$P_-$", fontsize=9, ha="left", va="center")

    ax.text(4.9, 1.25, r"$M=J_A\otimes J_B,\quad K_{AB}=H_+\oplus H_-$",
            ha="center", va="center", fontsize=9)

    _save(fig, "fig1_sector_diagram")


# --------------------------------------------------------------------- fig2 --


def fig2_cfr_leakage():
    M = samei.cfr_M()
    Pp, Pm = samei.cfr_projectors()
    alphas = np.linspace(-1.0, 1.0, 201)
    L_form = (1.0 - alphas) / 2.0
    L_num = np.array([samei.leakage(samei.cfr_state(a), M) for a in alphas])
    C_num = np.array([samei.coherent_leakage(samei.cfr_state(a), M) for a in alphas])
    err = np.abs(L_form - L_num)
    rows = list(zip(alphas, L_form, L_num, C_num, err))
    _write_csv(os.path.join(DATA_DIR, "fig2_cfr_leakage.csv"),
               ["alpha", "L_formula", "L_numeric", "C_numeric", "abs_error_L"],
               rows)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(alphas, L_form, "-", color="#1f3b73", lw=1.6,
            label=r"$L=(1-\alpha)/2$")
    ax.axhline(0.0, color="gray", lw=0.7, ls=":")
    ax.axhline(1.0, color="gray", lw=0.7, ls=":")
    ax.annotate(r"$\rho_{\rm CFR}(1)=P_+/2,\ L=0$",
                xy=(1.0, 0.0), xytext=(-0.3, 0.16),
                fontsize=7.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.7))
    ax.annotate(r"$\rho_{\rm CFR}(-1)=P_-/2,\ L=1$",
                xy=(-1.0, 1.0), xytext=(-0.95, 0.75),
                fontsize=7.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.7))
    ax.set_xlabel(r"CFR parameter $\alpha$")
    ax.set_ylabel(r"$L(\rho_{\rm CFR}(\alpha))$")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    _save(fig, "fig2_cfr_leakage")
    return {
        "max_err_L": float(err.max()),
        "max_C": float(np.abs(C_num).max()),
        "n": len(alphas),
    }


# --------------------------------------------------------------------- fig3 --


def fig3_coherent_leakage_bound():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    Pp, Pm = samei.projectors(M)
    n = M.shape[0]
    rng = np.random.default_rng(SEED)

    # Envelope
    L_grid = np.linspace(0.0, 1.0, 401)
    C_env = np.sqrt(L_grid * (1.0 - L_grid))

    # Coherent pure states (single shared psi_+, psi_- pair, vary lambda)
    rng_pure = np.random.default_rng(SEED + 1)
    psi_plus = rng_pure.standard_normal(n)
    psi_plus = Pp @ psi_plus; psi_plus /= np.linalg.norm(psi_plus)
    psi_minus = rng_pure.standard_normal(n)
    psi_minus = Pm @ psi_minus; psi_minus /= np.linalg.norm(psi_minus)
    lam_pure = np.linspace(0.0, 1.0, 41)
    pure_L, pure_C = [], []
    pure_C_form = []
    for lam in lam_pure:
        rho, _, _ = samei.coherent_pure_state(M, lam, rng=rng_pure,
                                              psi_plus=psi_plus,
                                              psi_minus=psi_minus)
        pure_L.append(samei.leakage(rho, M))
        pure_C.append(samei.coherent_leakage(rho, M))
        pure_C_form.append(np.sqrt(lam * (1.0 - lam)))
    pure_L = np.array(pure_L); pure_C = np.array(pure_C)
    pure_C_form = np.array(pure_C_form)
    max_pure_err = float(np.max(np.abs(pure_C - pure_C_form)))

    # Block-diagonal states
    lam_bd = np.linspace(0.0, 1.0, 41)
    bd_L, bd_C = [], []
    for lam in lam_bd:
        rho = samei.block_diagonal_state(M, lam, rng=rng)
        bd_L.append(samei.leakage(rho, M))
        bd_C.append(samei.coherent_leakage(rho, M))
    bd_L = np.array(bd_L); bd_C = np.array(bd_C)
    max_bd_C = float(np.abs(bd_C).max())

    # Random states across ranks
    ranks = [1, 2, 4, 8, 16]
    states_per_rank = 200
    rand_rows = []
    rand_L_all, rand_C_all = [], []
    state_id = 0
    max_violation = -np.inf
    for r in ranks:
        for _ in range(states_per_rank):
            rho = samei.linear_algebra.random_real_density(n, rank=r, rng=rng)
            L = samei.leakage(rho, M)
            C = samei.coherent_leakage(rho, M)
            env = np.sqrt(max(L * (1 - L), 0.0))
            v = C - env
            max_violation = max(max_violation, v)
            rand_L_all.append(L); rand_C_all.append(C)
            rand_rows.append([state_id, r, L, C, env, v])
            state_id += 1
    rand_L_all = np.array(rand_L_all); rand_C_all = np.array(rand_C_all)

    _write_csv(os.path.join(DATA_DIR, "fig3_coherent_leakage_random.csv"),
               ["state_id", "rank", "L", "C", "C_envelope", "violation"],
               rand_rows)
    curve_rows = []
    for lam, L, C in zip(L_grid, L_grid, C_env):
        curve_rows.append(["envelope", float(lam), float(L), float(C)])
    for lam, L, C in zip(lam_pure, pure_L, pure_C):
        curve_rows.append(["coherent_pure", float(lam), float(L), float(C)])
    for lam, L, C in zip(lam_bd, bd_L, bd_C):
        curve_rows.append(["block_diagonal", float(lam), float(L), float(C)])
    _write_csv(os.path.join(DATA_DIR, "fig3_coherent_leakage_curves.csv"),
               ["kind", "lambda", "L", "C"], curve_rows)

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(L_grid, C_env, "-", color="black", lw=1.4, zorder=4,
            label=r"envelope $C=\sqrt{L(1-L)}$")
    ax.scatter(rand_L_all, rand_C_all, s=4, c="#7791bb", alpha=0.35,
               edgecolors="none", zorder=2, label="random real densities")
    ax.scatter(pure_L, pure_C, s=24, marker="*", c="#c93f3f", zorder=5,
               label="coherent pure states")
    ax.scatter(bd_L, bd_C, s=14, marker="s", c="#3a8e3a", zorder=3,
               label=r"block-diagonal states ($C=0$)")
    ax.set_xlabel(r"population leakage $L(\rho)$")
    ax.set_ylabel(r"coherent leakage $C(\rho)=\|P_+\rho P_-\|_1$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.55)
    ax.grid(alpha=0.15, lw=0.5)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                    ncol=2, frameon=False, fontsize=7.5,
                    handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout()
    _save(fig, "fig3_coherent_leakage_bound")
    return {
        "n_random": len(rand_rows),
        "max_violation": float(max_violation),
        "max_pure_err": max_pure_err,
        "max_bd_C": max_bd_C,
    }


# --------------------------------------------------------------------- fig4 --


def fig4_discrimination():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    eps_grid = np.linspace(0.0, 1.0, 401)
    m = np.minimum(eps_grid, 0.5)
    p_form = 0.5 + np.sqrt(m * (1.0 - m))
    p_num = np.array([
        samei.sector_parity_psucc(samei.saturating_state(M, e), M)
        for e in eps_grid
    ])
    rng = np.random.default_rng(SEED + 2)
    p_bd = np.array([
        samei.sector_parity_psucc(samei.block_diagonal_state(M, min(e, 0.5), rng=rng), M)
        for e in eps_grid
    ])
    err = np.abs(p_form - p_num)
    rows = list(zip(eps_grid, m, p_form, p_num, p_bd, err))
    _write_csv(os.path.join(DATA_DIR, "fig4_discrimination.csv"),
               ["epsilon", "m", "p_formula", "p_numeric", "p_block_diag", "abs_error"],
               rows)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot(eps_grid, p_form, "-", color="#1f3b73", lw=1.5,
            label=r"$p_{\rm succ}^{(\epsilon)}=\frac{1}{2}+\sqrt{m(1-m)}$")
    sub = slice(None, None, 20)
    ax.plot(eps_grid[sub], p_num[sub], "o", color="#c93f3f", ms=3.4,
            mfc="none", mew=1.0, label="saturating probes")
    ax.axhline(0.5, color="#3a8e3a", lw=1.0, ls="--",
               label=r"block-diagonal: $p_{\rm succ}=\frac{1}{2}$")
    ax.axvline(0.5, color="gray", lw=0.8, ls=":", label=r"$\epsilon=1/2$")
    ax.set_xlabel(r"leakage budget $\epsilon$")
    ax.set_ylabel(r"$p_{\rm succ}^{(\epsilon)}$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.46, 1.04)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig4_sector_parity_discrimination")
    return {"n": len(eps_grid), "max_err": float(err.max())}


# --------------------------------------------------------------------- fig5 --


def fig5_perturbative():
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    Pp, Pm = samei.projectors(M)
    rng = np.random.default_rng(SEED + 3)
    G0 = samei.skew_commuting_with_M(M, rng=rng)
    B = samei.skew_cross_sector(M, rng=rng)
    n = M.shape[0]
    v = rng.standard_normal(n)
    v = Pp @ v; v = v / np.linalg.norm(v)
    t = 1.0
    eps_grid = np.logspace(-4.0, -1.0, 25)
    out = samei.leakage_perturbative_curve(M, G0, B, v, t=t, eps_grid=eps_grid)
    L_num = out["L_numeric"]
    L_lead = out["L_leading"]
    rel = np.abs(L_num - L_lead) / np.maximum(L_lead, 1e-300)
    rows = list(zip(eps_grid, L_num, L_lead, rel))
    _write_csv(os.path.join(DATA_DIR, "fig5_perturbative_scaling.csv"),
               ["epsilon", "L_numeric", "L_leading", "relative_error"], rows)

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.loglog(eps_grid, L_lead, "-", color="#1f3b73", lw=1.4,
              label=r"leading order $\epsilon^2\|a(t)\|^2$")
    ax.loglog(eps_grid, L_num, "o", color="#c93f3f", ms=3.6, mfc="none",
              mew=1.0, label="numerical $L$")
    ax.set_xlabel(r"perturbation strength $\epsilon$")
    ax.set_ylabel(r"$L(e^{(G_0+\epsilon B)t}\psi_0)$")
    ax.text(0.05, 0.92, fr"fitted slope $\approx {out['slope']:.3f}$",
            transform=ax.transAxes, fontsize=8)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig5_perturbative_leakage")
    small = len(eps_grid) // 2
    return {
        "slope": float(out["slope"]),
        "a_norm_sq": float(out["a_norm_sq"]),
        "t": float(out["t"]),
        "max_rel_err_small": float(rel[:small].max()),
        "n": len(eps_grid),
    }


def main():
    fig1_sector_diagram()
    r2 = fig2_cfr_leakage()
    r3 = fig3_coherent_leakage_bound()
    r4 = fig4_discrimination()
    r5 = fig5_perturbative()
    print()
    print("Figure gate summary:")
    print(f"  fig2  CFR max L err  : {r2['max_err_L']:.3e}")
    print(f"  fig2  CFR max C      : {r2['max_C']:.3e}")
    print(f"  fig3  envelope viol  : {r3['max_violation']:.3e} (n={r3['n_random']})")
    print(f"  fig3  pure curve err : {r3['max_pure_err']:.3e}")
    print(f"  fig3  block-diag C   : {r3['max_bd_C']:.3e}")
    print(f"  fig4  disc max err   : {r4['max_err']:.3e}")
    print(f"  fig5  slope          : {r5['slope']:.4f}")
    print(f"  fig5  ||a(t)||^2     : {r5['a_norm_sq']:.4e}")
    print(f"  fig5  small-eps rel  : {r5['max_rel_err_small']:.3e}")
    return r2, r3, r4, r5


if __name__ == "__main__":
    main()
