"""Terminal wrapper for building the TaS2 initial-state switching ML dataset.

Run from the repository root:

    python src/ml/build_switching_prediction_dataset.py
"""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tase2_phase_switching.ml_switching_dataset import main


if __name__ == "__main__":
    main()
