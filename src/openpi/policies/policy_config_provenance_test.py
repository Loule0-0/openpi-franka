from __future__ import annotations

from types import SimpleNamespace

from franka_runtime import EXPECTED_POLICY_METADATA
from franka_runtime import ProvenanceError
import pytest

from openpi.policies import policy_config


def test_franka_policy_rejects_norm_stats_override_before_checkpoint_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_download(*args, **kwargs):
        raise AssertionError("checkpoint access must not happen after a forbidden override")

    monkeypatch.setattr(policy_config.download, "maybe_download", unexpected_download)
    config = SimpleNamespace(policy_metadata=EXPECTED_POLICY_METADATA.to_mapping())

    with pytest.raises(ProvenanceError, match="norm_stats overrides are forbidden"):
        policy_config.create_trained_policy(
            config,
            "unused-checkpoint",
            repack_transforms=object(),
            norm_stats={},
        )


def test_non_franka_policy_does_not_reject_norm_stats_override(monkeypatch: pytest.MonkeyPatch) -> None:
    class DownloadReachedError(RuntimeError):
        pass

    def stop_at_download(*args, **kwargs):
        raise DownloadReachedError

    monkeypatch.setattr(policy_config.download, "maybe_download", stop_at_download)
    config = SimpleNamespace(policy_metadata=None)

    with pytest.raises(DownloadReachedError):
        policy_config.create_trained_policy(
            config,
            "unused-checkpoint",
            repack_transforms=object(),
            norm_stats={},
        )
