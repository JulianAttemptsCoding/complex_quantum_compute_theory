# QA Report

## Environment

- Python: 3.13.1
- Platform: Windows-11-10.0.26200-SP0
- numpy: 2.3.1
- scipy: 1.17.0
- matplotlib: 3.10.3
- Random seed (master): 20260517

## Generated files

### Figures (PDF + SVG)
- figures/fig1_sector_diagram.pdf
- figures/fig1_sector_diagram.svg
- figures/fig2_cfr_leakage.pdf
- figures/fig2_cfr_leakage.svg
- figures/fig3_coherent_leakage_bound.pdf
- figures/fig3_coherent_leakage_bound.svg
- figures/fig4_sector_parity_discrimination.pdf
- figures/fig4_sector_parity_discrimination.svg
- figures/fig5_perturbative_leakage.pdf
- figures/fig5_perturbative_leakage.svg

### Data CSVs
- data/fig2_cfr_leakage.csv
- data/fig3_coherent_leakage_random.csv
- data/fig3_coherent_leakage_curves.csv
- data/fig4_discrimination.csv
- data/fig5_perturbative_scaling.csv
- data/sanity_check_summary.csv

### LaTeX
- latex/generated_results.tex
- latex/figure_includes.tex
- latex/figure_captions.tex
- latex/caption_*.tex (one per figure)

## Mathematical convention gate

- H_+ = ker(M + I), aligned sector (M acts as -I).
- H_- = ker(M - I), leakage sector (M acts as +I).
- P_+ = (I - M) / 2, P_- = (I + M) / 2.
- Verified rank(P_+) = rank(P_-) = 2 d_A d_B for (d_A, d_B) in {1,2,3}x{1,2,3}.
- Projector identities max error: 0.000e+00.
- Embedding (iota, balanced, J_AB intertwining) max error: 0.000e+00.
- Result: **PASS**.

## Figure gates

- Fig 1 (sector diagram): vector PDF + SVG, no screenshot. **PASS**.
- Fig 2 (CFR): max |L_formula - L_numeric| = 0.000e+00, max |C| = 0.000e+00. **PASS**.
- Fig 3 (coherent leakage envelope): 1000 random states, max envelope violation = 3.331e-16, pure-state curve error = 2.776e-16, block-diag max C = 0.000e+00. **PASS**.
- Fig 4 (discrimination): grid = 401, max |p_form - p_num| = 2.220e-16. **PASS**.
- Fig 5 (perturbative): t = 1, fitted slope = 2.0000, leading ||a||^2 = 4.1999e+00, max small-eps relative error = 2.421e-05. **PASS**.

### Visual checklist
- Fig 1: `figures/fig1_sector_diagram.pdf` -- Vector sector diagram (matplotlib); no screenshot, no bitmap.
- Fig 2: `figures/fig2_cfr_leakage.pdf` -- CFR closed-form curve L = (1-alpha)/2; analytic, no noisy data points.
- Fig 3: `figures/fig3_coherent_leakage_bound.pdf` -- Envelope + coherent-pure markers + block-diag markers; legend below plot, no overlap.
- Fig 4: `figures/fig4_sector_parity_discrimination.pdf` -- Optimum curve + saturator probes + p=1/2 baseline + epsilon=1/2 line.
- Fig 5: `figures/fig5_perturbative_leakage.pdf` -- Log-log perturbative scaling; sanity check only.

## Result gates

| Macro | Value | Gate | Result |
| --- | --- | --- | --- |
| `\RealificationMaxBornError` | 1.110223e-16 | <= 1e-12 | **PASS** |
| `\SectorProjectorMaxError` | 0.000000e+00 | <= 1e-12 | **PASS** |
| `\EmbeddingMaxError` | 0.000000e+00 | <= 1e-12 | **PASS** |
| `\CFRMaxLeakageError` | 0.000000e+00 | <= 1e-12 | **PASS** |
| `\CFRMaxCoherentLeakage` | 0.000000e+00 | <= 1e-12 | **PASS** |
| `\CoherentLeakageMaxViolation` | 3.330669e-16 | <= 1e-10 | **PASS** |
| `\CoherentLeakageMaxPureError` | 2.775558e-16 | <= 1e-12 | **PASS** |
| `\CoherentBlockDiagMaxC` | 0.000000e+00 | <= 1e-12 | **PASS** |
| `\FreeOpsMaxMonotoneIncrease` | 0.000000e+00 | <= 1e-10 | **PASS** |
| `\KrausCriterionMaxCompressionError` | 1.529461e-15 | <= 1e-12 | **PASS** |
| `\DiscriminationMaxError` | 2.220446e-16 | <= 1e-12 | **PASS** |
| `\PerturbativeSlope` | 1.999994e+00 | in [1.95, 2.05] | **PASS** |
| `\PerturbativeMaxSmallEpsRelError` | 2.420836e-05 | <= 0.1 | **PASS** |
| `\BobObstructionMaxDeviation` | 2.220446e-16 | <= 1e-12 | **PASS** |
| `\MomentMatrixMinEigenvalue` | -3.496831e-15 | >= -1e-10 | **PASS** |

- Trace-distance check mode: analytic only (max witness vs sqrt(L): 1.221e-15; SDP skipped because CVXPY unavailable).

## Source phrase audit

Target: `C:\Users\Julia\OneDrive\Desktop\quantum i\paper\main.tex`

| Phrase | Hits |
| --- | --- |
| `complete resource theory` | 1 |
| `basis-independent` | 4 |
| `ordinary complex composition is exactly` | 3 |
| `center of the aligned sector` | 2 |
| `requires operational` | 1 |
| `no quantity built linearly` | 1 |
| `J-covariant sources` | 0 |
| `minimum-effort` | 1 |
| `constructed by hand` | 0 |
| `unitarity constraint` | 1 |
| `convex-roof alignment weight` | 2 |
| `verifies every theorem` | 0 |
| `SDP convergence` | 0 |
| `SDP completeness` | 0 |

Risky phrases above appear in current main.tex. The downstream manuscript rewrite must remove or rephrase them.

### `complete resource theory`
- line 78: `\Tr(\Pminus\rho)$ --- and we develop it into a complete resource theory.`

### `basis-independent`
- line 77: `basis-independent monotone --- the \textbf{same-$i$ leakage} $\Lleak(\rho) =`
- line 189: `  same-$i$ leakage is basis-independent --- it depends only on the choice of local`
- line 800: `basis-independent. The two are therefore distinct resources, but the precise nature`
- line 925: `Imaginarity is basis-relative; same-$i$ leakage is basis-independent. Examples~%`

### `ordinary complex composition is exactly`
- line 75: `show that ordinary complex composition is exactly the zero-leakage, aligned-sector`
- line 147: `  ordinary complex composition is exactly the aligned sector $\Hplus$ together`
- line 915: `Ordinary complex composition is exactly the zero-leakage, aligned sector of a`

### `center of the aligned sector`
- line 89: `the center of the aligned sector, demonstrating that same-$i$ leakage and`
- line 445: `$\rho=\tfrac{1}{2}\Pplus$, sitting at the center of the aligned sector. This is the`

### `requires operational`
- line 96: `alignment requires operational, rather than product-state, source independence ---`

### `no quantity built linearly`
- line 896: `\textbf{Negative:} no quantity built linearly from the marginal $\rho_{B_1B_2}$`

### `minimum-effort`
- line 737: `internal operational probe of the resource theory: a ``minimum-effort'' task whose`

### `unitarity constraint`
- line 409: `second condition is a unitarity constraint at fixed sector; once a state sits inside`

### `convex-roof alignment weight`
- line 576: `The convex-roof alignment weight is`
- line 636: `$W(\rho)$ & convex-roof alignment weight & Mixture resource \\`


## Remaining limitations

- All numerical results in this package are reproducibility / sanity checks, not proofs.
- No new operational network witness is produced.
- No SDP convergence theorem is claimed. Trace-distance check is analytic only by default.
- No operational separation from imaginarity is claimed.
- The package does not rewrite the manuscript; only figures, macros, and audit data are produced.
