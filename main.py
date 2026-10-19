"""Repository entry point; runs without an editable installation."""

import os
import sys
from pathlib import Path

# Set before importing numerical libraries, including in spawned workers.
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from unknown_dataset.runner import main

    raise SystemExit(main())

