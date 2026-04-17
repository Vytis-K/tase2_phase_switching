from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tase2_phase_switching.desktop_app import main


if __name__ == "__main__":
    raise SystemExit(main())
