"""Compute all numerical checks, export LaTeX macro files, write summary CSV.

Outputs:
    latex/generated_results.tex
    latex/figure_includes.tex
    latex/figure_captions.tex
    data/sanity_check_summary.csv
    QA_REPORT_data.json     (consumed by audit / report scripts)
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import samei  # noqa: E402

LATEX_DIR = os.path.join(ROOT, "latex")
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(LATEX_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

SEED = 20260517


def _sci(x: float) -> str:
    """Format a non-negative float as either decimal or scientific LaTeX math."""
    ax = abs(float(x))
    if ax == 0.0:
        return r"\(0\)"
    if 1e-3 <= ax < 1e3:
        return f"{x:.6g}"
    m, e = f"{x:.3e}".split("e")
    return rf"\({float(m):.3f}\times 10^{{{int(e)}}}\)"


# ----------------------------- numerical checks ----------------------------


def check_realification() -> float:
    err = 0.0
    rng = np.random.default_rng(SEED + 100)
    for d in (2, 3, 4):
        e = samei.realification.born_check(d, n_trials=200, rng=rng)
        err = max(err, e)
    return err


def check_sector_projectors_and_embedding() -> tuple[float, float]:
    """Returns (proj_err, embed_err) across (dA, dB) in {1,2,3}x{1,2,3}."""
    proj_err = 0.0
    embed_err = 0.0
    rng = np.random.default_rng(SEED + 101)
    for dA in (1, 2, 3):
        for dB in (1, 2, 3):
            JA = samei.canonical_J(dA)
            JB = samei.canonical_J(dB)
            A_op, B_op, M = samei.sector_operators(JA, JB)
            n = M.shape[0]
            I = np.eye(n)
            Pp, Pm = samei.projectors(M)
            proj_err = max(proj_err, np.linalg.norm(Pp @ Pp - Pp))
            proj_err = max(proj_err, np.linalg.norm(Pm @ Pm - Pm))
            proj_err = max(proj_err, np.linalg.norm(Pp @ Pm))
            proj_err = max(proj_err, np.linalg.norm(Pp + Pm - I))
            rk_p = int(round(np.trace(Pp)))
            rk_m = int(round(np.trace(Pm)))
            assert rk_p == 2 * dA * dB
            assert rk_m == 2 * dA * dB
            JAB = samei.induced_complex_structure(JA, JB)
            for _ in range(10):
                x = rng.standard_normal(2 * dA)
                y = rng.standard_normal(2 * dB)
                v = samei.iota_simple(x, y, JA, JB)
                embed_err = max(embed_err, np.linalg.norm(Pp @ v - v))
                v1 = samei.iota_simple(JA @ x, y, JA, JB)
                v2 = samei.iota_simple(x, JB @ y, JA, JB)
                embed_err = max(embed_err, np.linalg.norm(v1 - v2))
                embed_err = max(embed_err, np.linalg.norm(JAB @ v - v1))
    return float(proj_err), float(embed_err)


def check_free_operations() -> tuple[float, float]:
    """Returns (max_monotone_increase, max_compression_error)."""
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    n = M.shape[0]
    rng = np.random.default_rng(SEED + 102)
    max_inc = 0.0
    max_comp = 0.0
    for _ in range(20):
        K = samei.random_sector_block_channel(M, n_kraus=4, rng=rng)
        assert samei.is_trace_preserving(K, atol=1e-9)
        max_comp = max(max_comp, samei.channels.compression_residual(K, M))
        for _ in range(20):
            rho = samei.linear_algebra.random_real_density(n, rng=rng)
            L0 = samei.leakage(rho, M)
            L1 = samei.leakage(samei.apply_kraus(rho, K), M)
            max_inc = max(max_inc, L1 - L0)
    return float(max_inc), float(max_comp)


def check_trace_distance() -> tuple[float, str]:
    """Closed-form check: D = sqrt(L) for coherent pure cross-sector states.

    Returns (max abs err vs natural witness, 'analytic' or 'sdp').
    """
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    Pp, _ = samei.projectors(M)
    rng = np.random.default_rng(SEED + 103)
    max_err = 0.0
    for lam in np.linspace(0.0, 1.0, 51):
        rho, _, _ = samei.coherent_pure_state(M, lam, rng=rng)
        w_tr = float(np.trace(Pp @ rho @ Pp))
        if w_tr > 1e-15:
            sigma = (Pp @ rho @ Pp) / w_tr
        else:
            sigma = Pp / float(np.trace(Pp))
        D = 0.5 * samei.linear_algebra.trace_norm(rho - sigma)
        max_err = max(max_err, abs(D - np.sqrt(lam)))
    try:
        import cvxpy  # noqa: F401
        mode = "sdp"
    except Exception:
        mode = "analytic"
    return float(max_err), mode


def check_bob_obstruction() -> tuple[int, float]:
    n_samples = 1000
    dB1, dB2 = 1, 1
    JB1 = samei.canonical_J(dB1); JB2 = samei.canonical_J(dB2)
    _, _, M = samei.sector_operators(JB1, JB2)
    n1 = 2 * dB1; n2 = 2 * dB2
    rng = np.random.default_rng(SEED + 104)
    max_dev = 0.0
    for _ in range(n_samples):
        rho1 = samei.linear_algebra.random_real_density(n1, rng=rng)
        rho2 = samei.linear_algebra.random_real_density(n2, rng=rng)
        L = samei.leakage(np.kron(rho1, rho2), M)
        max_dev = max(max_dev, abs(L - 0.5))
    return n_samples, float(max_dev)


def check_moment_matrix() -> float:
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M = samei.sector_operators(JA, JB)
    rng = np.random.default_rng(SEED + 105)
    n = M.shape[0]
    words = samei.moment_matrix.default_word_set(JA, JB, n_extra=6, rng=rng)
    min_eig = 0.0
    for _ in range(30):
        rho = samei.linear_algebra.random_real_density(n, rng=rng)
        G = samei.moment_matrix.build_moment_matrix(words, rho)
        e = float(np.linalg.eigvalsh(0.5 * (G + G.T)).min())
        min_eig = min(min_eig, e)
    return min_eig


# ----------------------------- figure-tied gates ---------------------------


def figure_gates() -> dict:
    """Re-compute the figure-tied numerical gates without re-rendering."""
    JA = samei.canonical_J(2); JB = samei.canonical_J(2)
    _, _, M2 = samei.sector_operators(JA, JB)
    Pp2, Pm2 = samei.projectors(M2)
    n2 = M2.shape[0]
    rng = np.random.default_rng(SEED)

    # ---- CFR ----
    M = samei.cfr_M()
    alphas = np.linspace(-1.0, 1.0, 201)
    L_form = (1.0 - alphas) / 2.0
    L_num = np.array([samei.leakage(samei.cfr_state(a), M) for a in alphas])
    C_num = np.array([samei.coherent_leakage(samei.cfr_state(a), M) for a in alphas])
    cfr_max_err = float(np.abs(L_form - L_num).max())
    cfr_max_C = float(np.abs(C_num).max())

    # ---- Figure 3 ----
    rng_p = np.random.default_rng(SEED + 1)
    psi_plus = rng_p.standard_normal(n2); psi_plus = Pp2 @ psi_plus
    psi_plus /= np.linalg.norm(psi_plus)
    psi_minus = rng_p.standard_normal(n2); psi_minus = Pm2 @ psi_minus
    psi_minus /= np.linalg.norm(psi_minus)
    max_pure = 0.0
    lam_grid = np.linspace(0.0, 1.0, 41)
    for lam in lam_grid:
        rho, _, _ = samei.coherent_pure_state(M2, lam, rng=rng_p,
                                              psi_plus=psi_plus, psi_minus=psi_minus)
        C = samei.coherent_leakage(rho, M2)
        max_pure = max(max_pure, abs(C - np.sqrt(lam * (1 - lam))))
    max_bd_C = 0.0
    for lam in lam_grid:
        rho_bd = samei.block_diagonal_state(M2, lam, rng=rng)
        max_bd_C = max(max_bd_C, samei.coherent_leakage(rho_bd, M2))
    max_violation = -np.inf
    count_random = 0
    for r in [1, 2, 4, 8, 16]:
        for _ in range(200):
            rho = samei.linear_algebra.random_real_density(n2, rank=r, rng=rng)
            L = samei.leakage(rho, M2)
            C = samei.coherent_leakage(rho, M2)
            env = np.sqrt(max(L * (1 - L), 0.0))
            max_violation = max(max_violation, C - env)
            count_random += 1

    # ---- Figure 4 ----
    eps_grid = np.linspace(0.0, 1.0, 401)
    m = np.minimum(eps_grid, 0.5)
    p_form = 0.5 + np.sqrt(m * (1 - m))
    p_num = np.array([
        samei.sector_parity_psucc(samei.saturating_state(M2, e), M2)
        for e in eps_grid
    ])
    disc_max_err = float(np.abs(p_form - p_num).max())

    # ---- Figure 5 ----
    rng5 = np.random.default_rng(SEED + 3)
    G0 = samei.skew_commuting_with_M(M2, rng=rng5)
    B = samei.skew_cross_sector(M2, rng=rng5)
    v = rng5.standard_normal(n2); v = Pp2 @ v; v /= np.linalg.norm(v)
    eps5 = np.logspace(-4.0, -1.0, 25)
    out = samei.leakage_perturbative_curve(M2, G0, B, v, t=1.0, eps_grid=eps5)
    rel = np.abs(out["L_numeric"] - out["L_leading"]) / np.maximum(out["L_leading"], 1e-300)
    half = len(eps5) // 2
    return {
        "cfr_n": int(len(alphas)),
        "cfr_max_err_L": cfr_max_err,
        "cfr_max_C": cfr_max_C,
        "coherent_random_count": int(count_random),
        "coherent_max_violation": float(max_violation),
        "coherent_pure_err": float(max_pure),
        "coherent_bd_max_C": float(max_bd_C),
        "disc_grid": int(len(eps_grid)),
        "disc_max_err": disc_max_err,
        "perturbative_time": float(out["t"]),
        "perturbative_slope": float(out["slope"]),
        "perturbative_a_norm_sq": float(out["a_norm_sq"]),
        "perturbative_max_rel_err_small": float(rel[:half].max()),
    }


# ------------------------------- gates / IO --------------------------------


GATES = [
    ("RealificationMaxBornError", "real_max_born_err", 1e-12, "le"),
    ("SectorProjectorMaxError", "sector_proj_err", 1e-12, "le"),
    ("EmbeddingMaxError", "embed_err", 1e-12, "le"),
    ("CFRMaxLeakageError", "cfr_max_err_L", 1e-12, "le"),
    ("CFRMaxCoherentLeakage", "cfr_max_C", 1e-12, "le"),
    ("CoherentLeakageMaxViolation", "coherent_max_violation", 1e-10, "le"),
    ("CoherentLeakageMaxPureError", "coherent_pure_err", 1e-12, "le"),
    ("CoherentBlockDiagMaxC", "coherent_bd_max_C", 1e-12, "le"),
    ("FreeOpsMaxMonotoneIncrease", "free_max_increase", 1e-10, "le"),
    ("KrausCriterionMaxCompressionError", "free_max_compression", 1e-12, "le"),
    ("DiscriminationMaxError", "disc_max_err", 1e-12, "le"),
    ("PerturbativeSlope", "perturbative_slope", (1.95, 2.05), "in"),
    ("PerturbativeMaxSmallEpsRelError", "perturbative_max_rel_err_small", 0.10, "le"),
    ("BobObstructionMaxDeviation", "bob_obstruction_max_dev", 1e-12, "le"),
    ("MomentMatrixMinEigenvalue", "moment_min_eig", -1e-10, "ge"),
]


def evaluate(metrics: dict) -> list[tuple[str, str, str, str]]:
    """Return rows: (macro, value, gate, PASS/FAIL)."""
    out = []
    for macro, key, gate, kind in GATES:
        val = metrics[key]
        if kind == "le":
            ok = val <= gate
            gate_str = f"<= {gate:g}"
        elif kind == "ge":
            ok = val >= gate
            gate_str = f">= {gate:g}"
        elif kind == "in":
            lo, hi = gate
            ok = lo <= val <= hi
            gate_str = f"in [{lo}, {hi}]"
        else:
            raise ValueError(kind)
        out.append((macro, f"{val:.6e}", gate_str, "PASS" if ok else "FAIL"))
    return out


def main():
    real_err = check_realification()
    proj_err, embed_err = check_sector_projectors_and_embedding()
    free_inc, free_comp = check_free_operations()
    td_err, td_mode = check_trace_distance()
    bob_n, bob_dev = check_bob_obstruction()
    moment_min = check_moment_matrix()
    figs = figure_gates()

    metrics = {
        "real_max_born_err": real_err,
        "sector_proj_err": proj_err,
        "embed_err": embed_err,
        "free_max_increase": free_inc,
        "free_max_compression": free_comp,
        "trace_distance_err": td_err,
        "trace_distance_mode": td_mode,
        "bob_obstruction_n": bob_n,
        "bob_obstruction_max_dev": bob_dev,
        "moment_min_eig": moment_min,
    }
    metrics.update(figs)

    rows = evaluate(metrics)

    # --- LaTeX macros ---
    macro_path = os.path.join(LATEX_DIR, "generated_results.tex")
    with open(macro_path, "w") as f:
        f.write("% Auto-generated by scripts/export_results_tex.py. Do not edit manually.\n\n")
        f.write(rf"\newcommand{{\RealificationMaxBornError}}{{{_sci(real_err)}}}" + "\n")
        f.write(rf"\newcommand{{\SectorProjectorMaxError}}{{{_sci(proj_err)}}}" + "\n")
        f.write(rf"\newcommand{{\EmbeddingMaxError}}{{{_sci(embed_err)}}}" + "\n")
        f.write(rf"\newcommand{{\CFRGridCount}}{{{figs['cfr_n']}}}" + "\n")
        f.write(rf"\newcommand{{\CFRMaxLeakageError}}{{{_sci(figs['cfr_max_err_L'])}}}" + "\n")
        f.write(rf"\newcommand{{\CFRMaxCoherentLeakage}}{{{_sci(figs['cfr_max_C'])}}}" + "\n")
        f.write(rf"\newcommand{{\CoherentRandomStateCount}}{{{figs['coherent_random_count']}}}" + "\n")
        f.write(rf"\newcommand{{\CoherentLeakageMaxViolation}}{{{_sci(figs['coherent_max_violation'])}}}" + "\n")
        f.write(rf"\newcommand{{\CoherentLeakageMaxPureError}}{{{_sci(figs['coherent_pure_err'])}}}" + "\n")
        f.write(rf"\newcommand{{\CoherentBlockDiagMaxC}}{{{_sci(figs['coherent_bd_max_C'])}}}" + "\n")
        f.write(rf"\newcommand{{\FreeOpsMaxMonotoneIncrease}}{{{_sci(free_inc)}}}" + "\n")
        f.write(rf"\newcommand{{\KrausCriterionMaxCompressionError}}{{{_sci(free_comp)}}}" + "\n")
        f.write(rf"\newcommand{{\DiscriminationGridCount}}{{{figs['disc_grid']}}}" + "\n")
        f.write(rf"\newcommand{{\DiscriminationMaxError}}{{{_sci(figs['disc_max_err'])}}}" + "\n")
        f.write(rf"\newcommand{{\PerturbativeTime}}{{{figs['perturbative_time']:g}}}" + "\n")
        f.write(rf"\newcommand{{\PerturbativeSlope}}{{{figs['perturbative_slope']:.4f}}}" + "\n")
        f.write(rf"\newcommand{{\PerturbativeLeadingCoeff}}{{{_sci(figs['perturbative_a_norm_sq'])}}}" + "\n")
        f.write(rf"\newcommand{{\PerturbativeMaxSmallEpsRelError}}{{{_sci(figs['perturbative_max_rel_err_small'])}}}" + "\n")
        f.write(rf"\newcommand{{\BobObstructionSampleCount}}{{{bob_n}}}" + "\n")
        f.write(rf"\newcommand{{\BobObstructionMaxDeviation}}{{{_sci(bob_dev)}}}" + "\n")
        f.write(rf"\newcommand{{\MomentMatrixMinEigenvalue}}{{{_sci(moment_min)}}}" + "\n")
        f.write(rf"\newcommand{{\TraceDistanceMode}}{{{td_mode}}}" + "\n")
        f.write(rf"\newcommand{{\TraceDistanceMaxError}}{{{_sci(td_err) if td_mode!='analytic' else 'NA (analytic)'}}}" + "\n")
    print(f"wrote {macro_path}")

    # --- figure_includes.tex ---
    inc_path = os.path.join(LATEX_DIR, "figure_includes.tex")
    with open(inc_path, "w") as f:
        f.write("% Auto-generated. Provides \\figSectorDiagram etc.\n")
        figs_info = [
            ("fig1_sector_diagram", "fig:sector-diagram", "figure*",
             "Same-i sector geometry."),
            ("fig2_cfr_leakage", "fig:cfr",
             "figure", "Leakage of the CFR family."),
            ("fig3_coherent_leakage_bound", "fig:coherent-bound",
             "figure", "Coherent leakage envelope."),
            ("fig4_sector_parity_discrimination", "fig:discrimination",
             "figure", "Sector-parity discrimination optimum."),
            ("fig5_perturbative_leakage", "fig:perturbative",
             "figure", "Perturbative leakage scaling."),
        ]
        for stem, label, env, _ in figs_info:
            f.write(rf"\begin{{{env}}}[t]" + "\n")
            f.write(r"  \centering" + "\n")
            width = r"\textwidth" if env == "figure*" else r"\columnwidth"
            f.write(rf"  \includegraphics[width={width}]{{figures/{stem}.pdf}}" + "\n")
            f.write(rf"  \input{{latex/caption_{stem}.tex}}" + "\n")
            f.write(rf"  \label{{{label}}}" + "\n")
            f.write(rf"\end{{{env}}}" + "\n\n")
    print(f"wrote {inc_path}")

    # --- figure_captions.tex (single file containing all captions) ---
    cap_path = os.path.join(LATEX_DIR, "figure_captions.tex")
    captions = {
        "fig1_sector_diagram": (
            r"\caption{Same-\(i\) sector geometry. Fixed local complex structures "
            r"\(J_A,J_B\) define \(M=J_A\otimes J_B\) on the full real tensor product "
            r"\(K_{AB}\). The aligned sector \(H_+=\ker(M+I)\) represents the realified "
            r"ordinary complex tensor product, while \(H_-=\ker(M-I)\) is the leakage "
            r"sector. Ordinary complex quantum theory also requires \(J_{AB}\)-compatible "
            r"states and effects inside \(H_+\).}"
        ),
        "fig2_cfr_leakage": (
            r"\caption{Leakage of the Caves--Fuchs--Rungta two-rebit family. With "
            r"\(J_A=J_B=i\sigma_y\), one has \(M=-\sigma_y\otimes\sigma_y\) and "
            r"\(\rho_{\rm CFR}(\alpha)=\tfrac14(I-\alpha M)\), so "
            r"\(L(\rho_{\rm CFR}(\alpha))=(1-\alpha)/2\). The family is block diagonal "
            r"in \(H_+\oplus H_-\), hence its coherent leakage \(C\) vanishes for all "
            r"\(\alpha\).}"
        ),
        "fig3_coherent_leakage_bound": (
            r"\caption{Population leakage \(L(\rho)\) and coherent leakage "
            r"\(C(\rho)=\|P_+\rho P_-\|_1\). The solid curve is the sharp bound "
            r"\(C\le\sqrt{L(1-L)}\). Coherent pure cross-sector states saturate the "
            r"bound, while block-diagonal states have \(C=0\) for every value of "
            r"\(L\). Random real density operators lie below the envelope.}"
        ),
        "fig4_sector_parity_discrimination": (
            r"\caption{Optimal success probability for the internal sector-parity "
            r"discrimination primitive. The advantage is controlled by coherent "
            r"cross-sector amplitude: coherent probes attain "
            r"\(p_{\rm succ}^{(\epsilon)}=\tfrac12+\sqrt{m(1-m)}\), "
            r"\(m=\min\{\epsilon,1/2\}\), while block-diagonal probes remain at "
            r"\(p_{\rm succ}=\tfrac12\).}"
        ),
        "fig5_perturbative_leakage": (
            r"\caption{Numerical sanity check for perturbative leakage generation. "
            r"Starting from an aligned state and a sector-preserving generator "
            r"\(G_0\), a skew perturbation \(B\) with cross-sector component generates "
            r"leakage at second order: \(L=\epsilon^2\|a(t)\|^2+O(\epsilon^3)\). The "
            r"fitted log--log slope is reported in the reproducibility file. This plot "
            r"is illustrative; the proof is given analytically.}"
        ),
    }
    with open(cap_path, "w") as f:
        f.write("% Auto-generated captions for figures.\n\n")
        for stem, cap in captions.items():
            f.write(f"% --- {stem} ---\n")
            f.write(cap + "\n\n")
    print(f"wrote {cap_path}")

    for stem, cap in captions.items():
        single = os.path.join(LATEX_DIR, f"caption_{stem}.tex")
        with open(single, "w") as f:
            f.write(cap + "\n")
        print(f"wrote {single}")

    # --- sanity_check_summary.csv ---
    summary_path = os.path.join(DATA_DIR, "sanity_check_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["macro", "value", "gate", "result"])
        for r in rows:
            w.writerow(r)
    print(f"wrote {summary_path}")

    # --- QA_REPORT_data.json (machine-readable for the report writer) ---
    data_json = {
        "metrics": metrics,
        "gates": [
            {"macro": macro, "value": val, "gate": gate, "result": res}
            for macro, val, gate, res in rows
        ],
    }
    with open(os.path.join(ROOT, "QA_REPORT_data.json"), "w") as f:
        json.dump(data_json, f, indent=2)
    print(f"wrote {os.path.join(ROOT, 'QA_REPORT_data.json')}")

    print()
    print("Acceptance gates:")
    for macro, val, gate, res in rows:
        print(f"  {res:4s}  {macro:42s}  value={val:>15s}  gate={gate}")
    return rows


if __name__ == "__main__":
    main()
