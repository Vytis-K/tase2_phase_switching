"""Terminal wrapper for training the TaS2 initial-state switching predictor.

Run from the repository root after building the dataset:

    python src/ml/train_switching_predictor.py
"""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tase2_phase_switching.ml_switching_train import main


if __name__ == "__main__":
    main()
