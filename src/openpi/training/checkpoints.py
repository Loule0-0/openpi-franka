from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import hashlib
import logging
import os
import pathlib
from typing import Protocol

from etils import epath
from franka_runtime import provenance as _provenance
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    train_config: object,
    source_fingerprint: _provenance.SourceFingerprint | None,
    dataset_manifest_sha256: str | None,
):
    uses_franka_provenance = _provenance.is_franka_policy_metadata(getattr(train_config, "policy_metadata", None))
    has_source = source_fingerprint is not None
    has_dataset = dataset_manifest_sha256 is not None
    if (uses_franka_provenance and not (has_source and has_dataset)) or (
        not uses_franka_provenance and (has_source or has_dataset)
    ):
        raise _provenance.ProvenanceError(
            "Franka checkpoint saves require source and dataset provenance; non-Franka saves must not provide it"
        )

    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)
        if uses_franka_provenance:
            if norm_stats is None:
                raise _provenance.ProvenanceError("Franka checkpoints require normalization statistics")
            current_source = _provenance.capture_source_fingerprint(pathlib.Path(__file__))
            if current_source != source_fingerprint:
                raise _provenance.ProvenanceError("source checkout changed after Franka training started")
            current_dataset_manifest_sha256 = capture_franka_dataset_manifest_sha256(
                train_config, data_config, verify_payload=False
            )
            if current_dataset_manifest_sha256 != dataset_manifest_sha256:
                raise _provenance.ProvenanceError("dataset manifest changed after Franka training started")

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def finalize_franka_checkpoint(
    checkpoint_dir: epath.Path | str,
    *,
    train_config: object,
    data_config: object,
    source_fingerprint: _provenance.SourceFingerprint | None,
    dataset_manifest_sha256: str | None,
) -> None:
    """Seal a completed Orbax step with normalization, model, data, and source digests."""

    if not _provenance.is_franka_policy_metadata(getattr(train_config, "policy_metadata", None)):
        return
    if source_fingerprint is None or dataset_manifest_sha256 is None:
        raise _provenance.ProvenanceError("Franka checkpoint finalization requires source and dataset provenance")
    current_source = _provenance.capture_source_fingerprint(pathlib.Path(__file__))
    if current_source != source_fingerprint:
        raise _provenance.ProvenanceError("source checkout changed after Franka training started")
    current_dataset_manifest_sha256 = capture_franka_dataset_manifest_sha256(
        train_config, data_config, verify_payload=False
    )
    if current_dataset_manifest_sha256 != dataset_manifest_sha256:
        raise _provenance.ProvenanceError("dataset manifest changed after Franka training started")

    root = pathlib.Path(str(checkpoint_dir))
    fields = _franka_provenance_fields(train_config, data_config)
    norm_stats_path = root / "assets" / fields["asset_id"] / "norm_stats.json"
    model_artifact_format, model_artifacts_sha256 = _provenance.hash_model_artifacts(root)
    provenance = _provenance.CheckpointProvenance.create(
        **fields,
        dataset_manifest_sha256=dataset_manifest_sha256,
        norm_stats_sha256=_provenance.sha256_file(norm_stats_path),
        model_artifact_format=model_artifact_format,
        model_artifacts_sha256=model_artifacts_sha256,
        source_fingerprint=source_fingerprint,
    )
    _provenance.write_checkpoint_provenance(root / "assets", provenance)


def capture_franka_source_fingerprint(train_config: object) -> _provenance.SourceFingerprint | None:
    """Capture source only for configs that opt into the Franka wire contract."""

    if not _provenance.is_franka_policy_metadata(getattr(train_config, "policy_metadata", None)):
        return None
    return _provenance.capture_source_fingerprint(pathlib.Path(__file__))


def capture_franka_dataset_manifest_sha256(
    train_config: object,
    data_config: object,
    *,
    verify_payload: bool = True,
) -> str | None:
    """Validate the converter manifest and, at startup, its complete dataset payload."""

    if not _provenance.is_franka_policy_metadata(getattr(train_config, "policy_metadata", None)):
        return None
    repo_id = _provenance.validate_dataset_repo_id(getattr(data_config, "repo_id", None))
    dataset_home = os.environ.get("HF_LEROBOT_HOME")
    if not dataset_home:
        raise _provenance.ProvenanceError("HF_LEROBOT_HOME is required for Franka checkpoint provenance")
    manifest_path = pathlib.Path(dataset_home) / repo_id / "franka_manifest.json"
    return _provenance.hash_dataset_manifest(
        manifest_path,
        expected_repo_id=repo_id,
        verify_content=verify_payload,
    )


def verify_franka_checkpoint(
    checkpoint_dir: epath.Path | str,
    *,
    train_config: object,
    data_config: object,
    source_fingerprint: _provenance.SourceFingerprint | None,
    dataset_manifest_sha256: str | None = None,
) -> None:
    """Verify a Franka checkpoint before resume; leave other configs unchanged."""

    if not _provenance.is_franka_policy_metadata(getattr(train_config, "policy_metadata", None)):
        return
    if source_fingerprint is None:
        raise _provenance.ProvenanceError("Franka checkpoint verification requires a source fingerprint")
    norm_stats = getattr(data_config, "norm_stats", None)
    if norm_stats is None:
        raise _provenance.ProvenanceError("Franka checkpoint resume requires current normalization statistics")
    provenance_fields = _franka_provenance_fields(train_config, data_config)
    _provenance.verify_checkpoint_provenance(
        pathlib.Path(str(checkpoint_dir)),
        **provenance_fields,
        source_fingerprint=source_fingerprint,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    checkpoint_norm_stats = _normalize.load(
        pathlib.Path(str(checkpoint_dir)) / "assets" / str(provenance_fields["asset_id"])
    )
    current_norm_stats_json = _normalize.serialize_json(norm_stats)
    checkpoint_norm_stats_json = _normalize.serialize_json(checkpoint_norm_stats)
    if current_norm_stats_json != checkpoint_norm_stats_json:
        current_norm_stats_sha256 = hashlib.sha256(current_norm_stats_json.encode("utf-8")).hexdigest()
        checkpoint_norm_stats_sha256 = hashlib.sha256(checkpoint_norm_stats_json.encode("utf-8")).hexdigest()
        raise _provenance.ProvenanceError(
            "current normalization statistics do not match checkpoint provenance: "
            f"expected canonical digest {checkpoint_norm_stats_sha256}, got {current_norm_stats_sha256}"
        )


def _franka_provenance_fields(train_config: object, data_config: object) -> dict[str, object]:
    repo_id = _provenance.validate_dataset_repo_id(getattr(data_config, "repo_id", None))
    asset_id = _provenance.validate_asset_id(getattr(data_config, "asset_id", None))
    model = train_config.model
    model_type_value = getattr(model.model_type, "value", model.model_type)
    return {
        "train_config_name": train_config.name,
        "dataset_repo_id": repo_id,
        "asset_id": asset_id,
        "model_type": model_type_value,
        "model_action_dim": model.action_dim,
        "model_action_horizon": model.action_horizon,
        "policy_metadata": train_config.policy_metadata,
    }


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
