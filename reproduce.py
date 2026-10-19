"""Run the full pipeline for this article from the raw CSV.

Usage (from the repository root):
    python reproduce.py

Every generated file goes to outputs/. The raw CSV is expected at
data/greenhouse_dataset.csv (see data/README.md).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DATA = ROOT / "data" / "greenhouse_dataset.csv"

STEPS = [
    ["prep3.py", "{data}"],
    ["eval3.py"],
    ["figs3.py"],
]

def main():
    if not DATA.exists():
        sys.exit(f"Missing {DATA}. Download it first (see data/README.md).")
    OUT.mkdir(exist_ok=True)
    for step in STEPS:
        cmd = [sys.executable, str(ROOT / "src" / step[0])] + [
            str(DATA) if a == "{data}" else a for a in step[1:]
        ]
        print(">>", " ".join(step), flush=True)
        subprocess.run(cmd, cwd=OUT, check=True)
    print("\nDone. Results are in", OUT)

if __name__ == "__main__":
    main()
