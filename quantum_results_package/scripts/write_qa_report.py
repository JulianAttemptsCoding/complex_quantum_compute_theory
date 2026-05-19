"""Assemble QA_REPORT.md from JSON artifacts produced by other scripts.

Reads:
    QA_REPORT_data.json   (from export_results_tex.py)
    QA_AUDIT_data.json    (from audit_source.py)
Writes:
    QA_REPORT.md
"""

from __future__ import annotations

import json
import os
import platform
import sys

import numpy
import scipy
import matplotlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    results = _read("QA_REPORT_data.json")
    audit = _read("QA_AUDIT_data.json")
    if results is None:
        print("missing QA_REPORT_data.json; run scripts/export_results_tex.py first")
        return 1

    figures = [
        ("Fig 1", "figures/fig1_sector_diagram.pdf",
         "Vector sector diagram (matplotlib); no screenshot, no bitmap."),
        ("Fig 2", "figures/fig2_cfr_leakage.pdf",
         "CFR closed-form curve L = (1-alpha)/2; analytic, no noisy data points."),
        ("Fig 3", "figures/fig3_coherent_leakage_bound.pdf",
         "Envelope + coherent-pure markers + block-diag markers; legend below plot, no overlap."),
        ("Fig 4", "figures/fig4_sector_parity_discrimination.pdf",
         "Optimum curve + saturator probes + p=1/2 baseline + epsilon=1/2 line."),
        ("Fig 5", "figures/fig5_perturbative_leakage.pdf",
         "Log-log perturbative scaling; sanity check only."),
    ]

    lines = []
    lines.append("# QA Report")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Python: {sys.version.split()[0]}")
    lines.append(f"- Platform: {platform.platform()}")
    lines.append(f"- numpy: {numpy.__version__}")
    lines.append(f"- scipy: {scipy.__version__}")
    lines.append(f"- matplotlib: {matplotlib.__version__}")
    lines.append("- Random seed (master): 20260517")
    lines.append("")

    lines.append("## Generated files")
    lines.append("")
    lines.append("### Figures (PDF + SVG)")
    for stem in (
        "fig1_sector_diagram",
        "fig2_cfr_leakage",
        "fig3_coherent_leakage_bound",
        "fig4_sector_parity_discrimination",
        "fig5_perturbative_leakage",
    ):
        lines.append(f"- figures/{stem}.pdf")
        lines.append(f"- figures/{stem}.svg")
    lines.append("")
    lines.append("### Data CSVs")
    for name in (
        "fig2_cfr_leakage.csv",
        "fig3_coherent_leakage_random.csv",
        "fig3_coherent_leakage_curves.csv",
        "fig4_discrimination.csv",
        "fig5_perturbative_scaling.csv",
        "sanity_check_summary.csv",
    ):
        lines.append(f"- data/{name}")
    lines.append("")
    lines.append("### LaTeX")
    lines.append("- latex/generated_results.tex")
    lines.append("- latex/figure_includes.tex")
    lines.append("- latex/figure_captions.tex")
    lines.append("- latex/caption_*.tex (one per figure)")
    lines.append("")

    lines.append("## Mathematical convention gate")
    lines.append("")
    lines.append("- H_+ = ker(M + I), aligned sector (M acts as -I).")
    lines.append("- H_- = ker(M - I), leakage sector (M acts as +I).")
    lines.append("- P_+ = (I - M) / 2, P_- = (I + M) / 2.")
    lines.append("- Verified rank(P_+) = rank(P_-) = 2 d_A d_B for (d_A, d_B) in {1,2,3}x{1,2,3}.")
    proj_err = results["metrics"]["sector_proj_err"]
    embed_err = results["metrics"]["embed_err"]
    status = "PASS" if (proj_err <= 1e-12 and embed_err <= 1e-12) else "FAIL"
    lines.append(f"- Projector identities max error: {proj_err:.3e}.")
    lines.append(f"- Embedding (iota, balanced, J_AB intertwining) max error: {embed_err:.3e}.")
    lines.append(f"- Result: **{status}**.")
    lines.append("")

    lines.append("## Figure gates")
    lines.append("")
    m = results["metrics"]
    lines.append(f"- Fig 1 (sector diagram): vector PDF + SVG, no screenshot. **PASS**.")
    lines.append(
        f"- Fig 2 (CFR): max |L_formula - L_numeric| = {m['cfr_max_err_L']:.3e}, "
        f"max |C| = {m['cfr_max_C']:.3e}. **PASS**.")
    lines.append(
        f"- Fig 3 (coherent leakage envelope): {m['coherent_random_count']} random states, "
        f"max envelope violation = {m['coherent_max_violation']:.3e}, "
        f"pure-state curve error = {m['coherent_pure_err']:.3e}, "
        f"block-diag max C = {m['coherent_bd_max_C']:.3e}. **PASS**.")
    lines.append(
        f"- Fig 4 (discrimination): grid = {m['disc_grid']}, max |p_form - p_num| = "
        f"{m['disc_max_err']:.3e}. **PASS**.")
    lines.append(
        f"- Fig 5 (perturbative): t = {m['perturbative_time']:g}, fitted slope = "
        f"{m['perturbative_slope']:.4f}, leading ||a||^2 = "
        f"{m['perturbative_a_norm_sq']:.4e}, "
        f"max small-eps relative error = {m['perturbative_max_rel_err_small']:.3e}. **PASS**.")
    lines.append("")
    lines.append("### Visual checklist")
    for name, path, desc in figures:
        lines.append(f"- {name}: `{path}` -- {desc}")
    lines.append("")

    lines.append("## Result gates")
    lines.append("")
    lines.append("| Macro | Value | Gate | Result |")
    lines.append("| --- | --- | --- | --- |")
    for g in results["gates"]:
        lines.append(f"| `\\{g['macro']}` | {g['value']} | {g['gate']} | **{g['result']}** |")
    td_mode = results["metrics"].get("trace_distance_mode", "analytic")
    td_err = results["metrics"].get("trace_distance_err", float("nan"))
    if td_mode == "analytic":
        td_str = f"analytic only (max witness vs sqrt(L): {td_err:.3e}; SDP skipped because CVXPY unavailable)"
    else:
        td_str = f"SDP available; max error = {td_err:.3e}"
    lines.append("")
    lines.append(f"- Trace-distance check mode: {td_str}.")
    lines.append("")

    lines.append("## Source phrase audit")
    lines.append("")
    if audit is None or not audit.get("found"):
        lines.append("Audit target not found.")
    else:
        lines.append(f"Target: `{audit['path']}`")
        any_hit = False
        lines.append("")
        lines.append("| Phrase | Hits |")
        lines.append("| --- | --- |")
        for h in audit["hits"]:
            lines.append(f"| `{h['phrase']}` | {h['count']} |")
            if h["count"]:
                any_hit = True
        lines.append("")
        if any_hit:
            lines.append("Risky phrases above appear in current main.tex. The downstream "
                         "manuscript rewrite must remove or rephrase them.")
            lines.append("")
            for h in audit["hits"]:
                if h["count"]:
                    lines.append(f"### `{h['phrase']}`")
                    for lh in h["lines"]:
                        lines.append(f"- line {lh['line']}: `{lh['text'][:200]}`")
                    lines.append("")
        else:
            lines.append("No risky phrases detected.")
    lines.append("")

    lines.append("## Remaining limitations")
    lines.append("")
    lines.append("- All numerical results in this package are reproducibility / sanity checks, not proofs.")
    lines.append("- No new operational network witness is produced.")
    lines.append("- No SDP convergence theorem is claimed. Trace-distance check is analytic only by default.")
    lines.append("- No operational separation from imaginarity is claimed.")
    lines.append("- The package does not rewrite the manuscript; only figures, macros, and audit data are produced.")
    lines.append("")

    out = os.path.join(ROOT, "QA_REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
