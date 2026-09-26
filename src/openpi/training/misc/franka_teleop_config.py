"""Training configs for the Franka Panda joint-position data contract."""

from __future__ import annotations

import os

from franka_runtime import EXPECTED_POLICY_METADATA

import openpi.models.pi0_config as pi0_config
import openpi.policies.franka_policy as franka_policy
import openpi.transforms as _transforms

DATASET_REPO_ID = os.environ.get("FRANKA_DATASET_REPO_ID", "local/franka_gello")
BASE_CHECKPOINT = os.environ.get(
    "FRANKA_BASE_CHECKPOINT",
    "gs://openpi-assets/checkpoints/pi05_base/params",
)
ACTION_HORIZON = EXPECTED_POLICY_METADATA.action_horizon
POLICY_METADATA = EXPECTED_POLICY_METADATA.to_mapping()


def _make_data_config():
    # Import here to avoid a circular import with openpi.training.config.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import SimpleDataConfig

    return SimpleDataConfig(
        repo_id=DATASET_REPO_ID,
        assets=AssetsConfig(asset_id=DATASET_REPO_ID),
        base_config=DataConfig(
            prompt_from_task=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "exterior_image": "observation.images.exterior",
                            "wrist_image": "observation.images.wrist",
                            "state": "observation.state",
                            "actions": "action",
                            "prompt": "prompt",
                        }
                    )
                ]
            ),
            action_sequence_keys=("action",),
        ),
        data_transforms=lambda model: _transforms.Group(
            inputs=[
                franka_policy.FrankaInputs(model_type=model.model_type),
                _transforms.DeltaActions(_transforms.make_bool_mask(7, -1)),
            ],
            outputs=[
                _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                franka_policy.FrankaOutputs(),
            ],
        ),
    )


def get_franka_teleop_configs():
    """Return the deployable full-fine-tuning config with dataset-specific stats."""
    from openpi.training import weight_loaders
    from openpi.training.config import TrainConfig

    model = pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=ACTION_HORIZON)
    common = {
        "model": model,
        "weight_loader": weight_loaders.CheckpointWeightLoader(BASE_CHECKPOINT),
        "num_train_steps": 20_000,
        "batch_size": 32,
        "policy_metadata": POLICY_METADATA,
    }
    return [
        TrainConfig(
            name="pi05_franka_jointpos",
            data=_make_data_config(),
            **common,
        ),
    ]
