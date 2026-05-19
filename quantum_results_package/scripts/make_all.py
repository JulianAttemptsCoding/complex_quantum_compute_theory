"""Reproducibility entry point.

Runs:
    1. make_figures.py     -> PDF/SVG figures + per-figure CSV data
    2. export_results_tex.py -> numerical checks + LaTeX macros
    3. audit_source.py     -> grep ../paper/main.tex
    4. run_tests.py        -> pytest

Each step prints its own progress; the final summary lists PASS/FAIL gates.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


def _run(name: str) -> int:
    cmd = [sys.executable, os.path.join(SCRIPTS, name)]
    print("\n" + "=" * 72)
    print(f"$ {' '.join(cmd)}")
    print("=" * 72)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    rc = 0
    for script in (
        "make_figures.py",
        "export_results_tex.py",
        "audit_source.py",
        "write_qa_report.py",
        "run_tests.py",
    ):
        rc |= _run(script)
    print("\n" + "=" * 72)
    print("make_all.py finished. See QA_REPORT.md / latex/generated_results.tex.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
