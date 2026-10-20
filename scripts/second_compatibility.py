"""Run existing default prediction probes against the second-batch checkpoint.

Uses the unchanged first-batch probe definitions, writes only a new output path,
and compares Git-frozen sources with current sources. This is compatibility
evidence rather than an additional model-quality experiment.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import batch_default_equivalence as verification

verification.CHECKPOINT = 'd770d8bbea25820dd0f4cbd231786e4427c7ca30'

if __name__ == '__main__':
    raise SystemExit(verification.main())
