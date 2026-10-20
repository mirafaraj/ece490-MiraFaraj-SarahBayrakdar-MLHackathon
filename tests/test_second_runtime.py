"""Import isolation and compatible configuration controls (no timing thresholds)."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from unknown_dataset.lightweight_config import load_router_config
from unknown_dataset.routing import load_router_config as public_load
from unknown_dataset.runner import main


def test_lightweight_loader_matches_public_loader(tmp_path):
    assert load_router_config() == public_load()
    path = tmp_path / "router.json"
    path.write_text(json.dumps({"limits": {"final_tree_count": 35}}))
    assert load_router_config(path) == public_load(path)
    assert load_router_config(path)["limits"]["final_tree_count"] == 35


def test_lightweight_loader_rejects_invalid_config(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"unexpected": true}')
    with pytest.raises(ValueError):
        load_router_config(path)
    with pytest.raises(FileNotFoundError):
        load_router_config(tmp_path / "missing.json")


def test_configuration_does_not_import_modeling_stack():
    source = Path(__file__).resolve().parents[1] / "src"
    code = ("import sys; sys.path.insert(0, " + repr(str(source)) + "); "
            "from unknown_dataset.lightweight_config import load_router_config; "
            "load_router_config(); "
            "assert not any(x in sys.modules for x in ('sklearn', 'numpy', 'pandas'))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("strategy", ["baseline", "auto"])
def test_lazy_flag_requires_routed(strategy, tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--input", str(tmp_path), "--output", str(tmp_path / "out"),
              "--strategy", strategy, "--lazy-router-imports"])
    assert error.value.code == 2
