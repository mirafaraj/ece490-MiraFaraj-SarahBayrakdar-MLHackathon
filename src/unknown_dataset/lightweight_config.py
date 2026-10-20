"""Load router configuration without importing the modeling stack."""
from __future__ import annotations

import json
from pathlib import Path

from .policies import validate_config


def load_router_config(path=None):
    """Preserve the repository/default/partial-override configuration contract."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "router.json"
        if not path.is_file():
            return validate_config()
    with Path(path).open(encoding="utf-8") as handle:
        return validate_config(json.load(handle))
