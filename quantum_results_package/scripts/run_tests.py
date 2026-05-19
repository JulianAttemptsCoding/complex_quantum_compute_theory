"""Run the pytest suite for the package."""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    env = dict(os.environ)
    src = os.path.join(ROOT, "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "pytest", "-q", os.path.join(ROOT, "tests")]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd, env=env, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
