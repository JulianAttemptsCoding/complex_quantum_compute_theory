"""Grep ../paper/main.tex for risky manuscript phrases. Emit JSON + console table.

Looks for the phrases the audit identified as overclaiming or theorem-ledger
style. Used by QA_REPORT.md and the rewrite step downstream.
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.normpath(os.path.join(ROOT, "..", "paper"))
MAIN_TEX = os.path.join(PAPER_DIR, "main.tex")

PHRASES = [
    "complete resource theory",
    "basis-independent",
    "ordinary complex composition is exactly",
    "center of the aligned sector",
    "requires operational",
    "no quantity built linearly",
    "J-covariant sources",
    "minimum-effort",
    "constructed by hand",
    "unitarity constraint",
    "convex-roof alignment weight",
    "verifies every theorem",
    "SDP convergence",
    "SDP completeness",
]


def audit(path: str = MAIN_TEX) -> dict:
    if not os.path.exists(path):
        return {"path": path, "found": False, "hits": []}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    hits = []
    for phrase in PHRASES:
        pat = re.compile(re.escape(phrase), flags=re.IGNORECASE)
        line_hits = []
        for i, line in enumerate(lines, start=1):
            if pat.search(line):
                line_hits.append({"line": i, "text": line.rstrip("\n")})
        hits.append({"phrase": phrase, "count": len(line_hits), "lines": line_hits})
    return {"path": path, "found": True, "hits": hits}


def main():
    out = audit()
    if not out["found"]:
        print(f"audit: main.tex not found at {out['path']} -- skipping")
    else:
        print(f"audit target: {out['path']}")
        for h in out["hits"]:
            mark = "[hit]" if h["count"] else "[ok ]"
            print(f"  {mark} {h['count']:3d}  '{h['phrase']}'")
    dest = os.path.join(ROOT, "QA_AUDIT_data.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {dest}")
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
