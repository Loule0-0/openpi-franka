from __future__ import annotations

from types import SimpleNamespace

from franka_runtime import EMPTY_SHA256
from franka_runtime import EXPECTED_POLICY_METADATA
from franka_runtime import ProvenanceError
from franka_runtime import SourceFingerprint
from franka_runtime import provenance as _provenance
import numpy as np
import pytest

from openpi.shared import normalize as _normalize
from openpi.training import checkpoints as _checkpoints

SOURCE_FINGERPRINT = SourceFingerprint(source_commit="a" * 40, source_diff_sha256=EMPTY_SHA256)


def _norm_stats(offset: float = 0.0) -> dict[str, _normalize.NormStats]:
    return {
        "state": _normalize.NormStats(
            mean=np.array([offset, 1.0]),
            std=np.array([1.0, 2.0]),
            q01=np.array([-1.0, -2.0]),
            q99=np.array([1.0, 2.0]),
        )
    }


def _train_config(policy_metadata: object = EXPECTED_POLICY_METADATA.to_mapping()) -> SimpleNamespace:
    model = SimpleNamespace(model_type="pi05", action_dim=32, action_horizon=20)
    return SimpleNamespace(name="pi05_franka_jointpos", model=model, policy_metadata=policy_metadata)


def _data_config(norm_stats: object) -> SimpleNamespace:
    return SimpleNamespace(repo_id="local/franka_gello", asset_id="local/franka_gello", norm_stats=norm_stats)


def _write_checkpoint_norm_stats(checkpoint_dir, norm_stats: dict[str, _normalize.NormStats]):
    directory = checkpoint_dir / "assets" / "local" / "franka_gello"
    directory.mkdir(parents=True)
    # Deliberately use CRLF bytes. Provenance binds the exact raw file, while
    # resume compatibility compares its parsed, canonical normalization data.
    serialized = _normalize.serialize_json(norm_stats).replace("\n", "\r\n")
    path = directory / "norm_stats.json"
    path.write_bytes(serialized.encode("utf-8"))
    return path


def test_resume_accepts_exact_current_norm_stats(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    norm_stats = _norm_stats()
    norm_stats_path = _write_checkpoint_norm_stats(tmp_path, norm_stats)
    monkeypatch.setattr(
        _provenance,
        "verify_checkpoint_provenance",
        lambda *args, **kwargs: SimpleNamespace(norm_stats_sha256=_provenance.sha256_file(norm_stats_path)),
    )

    _checkpoints.verify_franka_checkpoint(
        tmp_path,
        train_config=_train_config(),
        data_config=_data_config(norm_stats),
        source_fingerprint=SOURCE_FINGERPRINT,
        dataset_manifest_sha256="b" * 64,
    )


def test_resume_rejects_current_norm_stats_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    norm_stats_path = _write_checkpoint_norm_stats(tmp_path, _norm_stats())
    monkeypatch.setattr(
        _provenance,
        "verify_checkpoint_provenance",
        lambda *args, **kwargs: SimpleNamespace(norm_stats_sha256=_provenance.sha256_file(norm_stats_path)),
    )

    with pytest.raises(ProvenanceError, match="current normalization statistics do not match"):
        _checkpoints.verify_franka_checkpoint(
            tmp_path,
            train_config=_train_config(),
            data_config=_data_config(_norm_stats(offset=0.5)),
            source_fingerprint=SOURCE_FINGERPRINT,
            dataset_manifest_sha256="b" * 64,
        )


def test_resume_rejects_missing_current_norm_stats(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def unexpected_verify(*args, **kwargs):
        raise AssertionError("checkpoint verifier must not run without current normalization statistics")

    monkeypatch.setattr(_provenance, "verify_checkpoint_provenance", unexpected_verify)

    with pytest.raises(ProvenanceError, match="requires current normalization statistics"):
        _checkpoints.verify_franka_checkpoint(
            tmp_path,
            train_config=_train_config(),
            data_config=_data_config(None),
            source_fingerprint=SOURCE_FINGERPRINT,
            dataset_manifest_sha256="b" * 64,
        )


def test_non_franka_resume_remains_a_noop(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    def unexpected_verify(*args, **kwargs):
        raise AssertionError("non-Franka configs must not enter provenance verification")

    monkeypatch.setattr(_provenance, "verify_checkpoint_provenance", unexpected_verify)

    _checkpoints.verify_franka_checkpoint(
        tmp_path,
        train_config=_train_config(policy_metadata=None),
        data_config=_data_config(None),
        source_fingerprint=None,
    )


def test_dataset_payload_is_verified_at_startup_but_can_be_skipped_for_save_immutability(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    calls: list[tuple[object, str, bool]] = []

    def capture(path, *, expected_repo_id: str, verify_content: bool) -> str:
        calls.append((path, expected_repo_id, verify_content))
        return "b" * 64

    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path))
    monkeypatch.setattr(_provenance, "hash_dataset_manifest", capture)
    train_config = _train_config()
    data_config = _data_config(_norm_stats())

    assert _checkpoints.capture_franka_dataset_manifest_sha256(train_config, data_config) == "b" * 64
    assert (
        _checkpoints.capture_franka_dataset_manifest_sha256(train_config, data_config, verify_payload=False) == "b" * 64
    )
    expected_path = tmp_path / "local" / "franka_gello" / "franka_manifest.json"
    assert calls == [
        (expected_path, "local/franka_gello", True),
        (expected_path, "local/franka_gello", False),
    ]


def test_finalize_seals_completed_model_and_norm_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    norm_stats = _norm_stats()
    data_config = _data_config(norm_stats)
    checkpoint_dir = tmp_path / "1000"
    _normalize.save(checkpoint_dir / "assets" / data_config.asset_id, norm_stats)
    params_file = checkpoint_dir / "params" / "weights"
    params_file.parent.mkdir(parents=True)
    params_file.write_bytes(b"completed orbax params")
    monkeypatch.setattr(_provenance, "capture_source_fingerprint", lambda _path: SOURCE_FINGERPRINT)
    payload_verifications = []

    def capture_manifest(*_args, **kwargs):
        payload_verifications.append(kwargs["verify_payload"])
        return "b" * 64

    monkeypatch.setattr(
        _checkpoints,
        "capture_franka_dataset_manifest_sha256",
        capture_manifest,
    )

    _checkpoints.finalize_franka_checkpoint(
        checkpoint_dir,
        train_config=_train_config(),
        data_config=data_config,
        source_fingerprint=SOURCE_FINGERPRINT,
        dataset_manifest_sha256="b" * 64,
    )

    provenance = _provenance.load_checkpoint_provenance(checkpoint_dir)
    norm_stats_path = checkpoint_dir / "assets" / data_config.asset_id / "norm_stats.json"
    assert provenance.norm_stats_sha256 == _provenance.sha256_file(norm_stats_path)
    assert _normalize.serialize_json(_normalize.load(norm_stats_path.parent)) == _normalize.serialize_json(norm_stats)
    assert provenance.model_artifact_format == "orbax-params/v1"
    assert provenance.model_artifacts_sha256 == _provenance.hash_model_artifacts(checkpoint_dir)[1]
    assert payload_verifications == [False]
