"""Run unchanged compatibility probes against the approved pre-third-batch checkpoint."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import batch_default_equivalence as verification
verification.CHECKPOINT = 'a49253cbab7579ad407ca0c02d012998b55b918d'
if __name__ == '__main__':
    raise SystemExit(verification.main())
