# Same-$i$ Leakage

A resource theory of shared complex structure in real quantum composition.

This repository contains the companion code and manuscript for the preprint
**"Same-$i$ Leakage: A Resource Theory of Shared Complex Structure in Real Quantum Composition."**

## What is in here

```
paper/        Quantum-targeted manuscript (LaTeX, quantumarticle class)
code/         Python reference implementation of the framework
  samei_leakage/   library: realification, sector ops, leakage, dynamics, examples
tests/        pytest suite verifying every theorem numerically
figures/      figure-generating scripts and PDF outputs
```

## Quick start

Requires Python 3.11+ with `numpy`, `scipy`, `matplotlib`, `pytest`, `cvxpy` (optional, for SDP template).

```bash
pip install -r requirements.txt
pytest -v                       # run the full numerical verification
python figures/make_figures.py  # regenerate figures
```

To build the manuscript (requires a TeX distribution with `quantumarticle`):

```bash
cd paper
latexmk -pdf main.tex
```

## What this code verifies

Every theorem in the manuscript is checked numerically:

- Realification preserves trace and Born rule (Theorem 1).
- Sector decomposition $K_{AB} = H_+ \oplus H_-$ via $M = J_A \otimes J_B$ (Theorem 3).
- Leakage $L(\rho) = \mathrm{Tr}(P_-\rho)$ with $L \in [0,1]$ and affineness (Theorem 4).
- Coherent leakage bound $\lVert P_+\rho P_- \rVert_1 \le \sqrt{L(1-L)}$ (Theorem 5).
- CPTP monotonicity iff $\Phi^\dagger(P_-) \le P_-$, with the Kraus characterization $P_- K_k P_+ = 0$ (Theorems 6, 7).
- Trace distance to free face $= \sqrt{\lambda}$ on coherent pure states (Theorem 8).
- Generalized robustness diverges when $L > 0$ (Proposition 10).
- Hamiltonian conservation when $[G, M] = 0$ and perturbative $O(\epsilon^2)$ scaling (Theorems 9, 10).
- Lindblad rate formula and sufficient conservation condition (Theorem 11, Corollary 12).
- Sector-parity discrimination $p_\mathrm{succ}^{(\epsilon)} = \tfrac{1}{2} + \sqrt{m(1-m)}$ (Theorem 13).
- CFR family closed form $L = (1-\alpha)/2$ (Proposition 14).
- Bob-node obstruction $L_B = 1/2$ for product bilocal sources (Proposition 15).
- Moment-matrix soundness for any real-state realization (Theorem 16).

## Reproducibility

Tests run deterministically (no RNG seeds needed for the algebraic checks; numerical
optimizations use fixed seeds). Tolerances are conservative ($10^{-10}$ on closed-form
identities, $10^{-6}$ on discrimination optimizations).

## License

MIT.
