# quantum_results_package

Computational figure + result package for the manuscript
*Shared Complex Structure as a Resource in Real Quantum Composition*.

This package contains **no manuscript prose**. It produces the figures, CSVs,
LaTeX macro file, captions, and QA report that the downstream LaTeX rewrite
depends on.

- Author: Julian Juan
- ORCID: 0009-0003-7234-2245
- Email: juliansjuan08@gmail.com
- Affiliation: Independent researcher

## Conventions (mandatory)

```
A     = J_A tensor I_B
B     = I_A tensor J_B
M     = A B = J_A tensor J_B
H_+   = ker(M + I)     aligned sector,  M = -I
H_-   = ker(M - I)     leakage sector,   M = +I
P_+   = (I - M) / 2
P_-   = (I + M) / 2
L(rho) = Tr(P_- rho)
C(rho) = || P_+ rho P_- ||_1
```

## Reproducibility

```bash
python scripts/make_all.py
pytest -q
```

`make_all.py` runs, in order:

1. `make_figures.py`         (figures and per-figure CSVs)
2. `export_results_tex.py`   (numerical checks + LaTeX macros)
3. `audit_source.py`         (phrase audit on `../paper/main.tex`, if present)
4. `write_qa_report.py`      (assembles `QA_REPORT.md`)
5. `run_tests.py`            (pytest)

Random seed: 20260517.

## Outputs

```
figures/
    fig1_sector_diagram.{pdf,svg}
    fig2_cfr_leakage.{pdf,svg}
    fig3_coherent_leakage_bound.{pdf,svg}
    fig4_sector_parity_discrimination.{pdf,svg}
    fig5_perturbative_leakage.{pdf,svg}

data/
    fig2_cfr_leakage.csv
    fig3_coherent_leakage_random.csv
    fig3_coherent_leakage_curves.csv
    fig4_discrimination.csv
    fig5_perturbative_scaling.csv
    sanity_check_summary.csv

latex/
    generated_results.tex
    figure_includes.tex
    figure_captions.tex
    caption_<figname>.tex

QA_REPORT.md
```

## Limitations

- Numerical results are sanity checks, not proofs.
- No new operational network witness is produced.
- No SDP convergence theorem is claimed.
- No operational separation from imaginarity is claimed.
- The trace-distance check is analytic only by default; an SDP confirmation
  runs only if CVXPY is available.
