"""Input and output transforms for the Franka joint-position policy contract.

The public runtime contract is intentionally small and robot-native:

* ``state``: 7 Panda joint positions in radians followed by gripper closure
  (0=open, 1=closed).
* ``actions``: absolute targets with the same eight dimensions.
* ``exterior_image`` and ``wrist_image``: RGB uint8 images in HWC layout.

The training config converts the seven arm targets to deltas before
normalization and converts them back to absolute targets after inference.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

STATE_DIM = 8
ACTION_DIM = 8


def make_franka_example() -> dict:
    """Return a correctly shaped example observation for smoke tests."""
    return {
        "exterior_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0, 0.0], dtype=np.float32),
        "prompt": "pick up the object",
    }


def _parse_rgb_image(value: np.ndarray, *, key: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{key} must be a 3D RGB image, got shape {image.shape}")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"{key} must have three RGB channels, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise ValueError(f"floating-point {key} must be finite and in [0, 1]")
        image = np.rint(image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        raise ValueError(f"{key} must be uint8 or float in [0, 1], got {image.dtype}")
    return np.ascontiguousarray(image)


@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
    """Map the stable Franka runtime schema into an OpenPI observation."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"state must end in {STATE_DIM} values, got shape {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("state contains NaN or infinity")

        exterior_image = _parse_rgb_image(data["exterior_image"], key="exterior_image")
        wrist_image = _parse_rgb_image(data["wrist_image"], key="wrist_image")

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (exterior_image, wrist_image, np.zeros_like(exterior_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _:
                raise ValueError(f"Franka joint-position policy does not support {self.model_type}")

        result = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != ACTION_DIM or not np.isfinite(actions).all():
                raise ValueError(f"actions must be finite and end in {ACTION_DIM} values, got {actions.shape}")
            result["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            result["prompt"] = prompt
        return result


@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    """Strip OpenPI padding and return the eight physical target values."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"expected action chunk [T, >=8], got {actions.shape}")
        return {"actions": actions[:, :ACTION_DIM]}
