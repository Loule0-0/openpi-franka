from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from openpi.training import config as _config


@pytest.mark.parametrize("unsafe", [".", "..", "../escape", "nested/run", r"nested\run", " run", "run "])
def test_checkpoint_dir_rejects_unsafe_experiment_component(tmp_path: Path, unsafe: str) -> None:
    config = dataclasses.replace(
        _config._CONFIGS_DICT["debug"],  # noqa: SLF001
        checkpoint_base_dir=str(tmp_path),
        exp_name=unsafe,
    )

    with pytest.raises(ValueError, match="single safe path component"):
        _ = config.checkpoint_dir


def test_checkpoint_dir_is_exactly_two_components_below_base(tmp_path: Path) -> None:
    config = dataclasses.replace(
        _config._CONFIGS_DICT["debug"],  # noqa: SLF001
        checkpoint_base_dir=str(tmp_path),
        exp_name="smoke-01",
    )

    assert config.checkpoint_dir == (tmp_path / "debug" / "smoke-01").resolve()


def test_config_name_cannot_escape_asset_or_checkpoint_base(tmp_path: Path) -> None:
    config = dataclasses.replace(
        _config._CONFIGS_DICT["debug"],  # noqa: SLF001
        name="..",
        assets_base_dir=str(tmp_path / "assets"),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        exp_name="run",
    )

    with pytest.raises(ValueError, match="config name"):
        _ = config.assets_dirs
    with pytest.raises(ValueError, match="config name"):
        _ = config.checkpoint_dir
