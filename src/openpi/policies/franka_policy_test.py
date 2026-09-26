import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import franka_policy


def test_inputs_preserve_rgb_and_mask_padding_view() -> None:
    exterior = np.zeros((12, 16, 3), dtype=np.uint8)
    exterior[..., 0] = 255
    wrist = np.zeros((12, 16, 3), dtype=np.uint8)
    wrist[..., 1] = 127

    result = franka_policy.FrankaInputs(_model.ModelType.PI05)(
        {
            "exterior_image": exterior,
            "wrist_image": wrist,
            "state": np.arange(8, dtype=np.float32),
            "prompt": "test",
        }
    )

    np.testing.assert_array_equal(result["image"]["base_0_rgb"], exterior)
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], wrist)
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }


def test_inputs_reject_invalid_state_and_image() -> None:
    transform = franka_policy.FrankaInputs(_model.ModelType.PI05)
    valid = {
        "exterior_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
    }

    with pytest.raises(ValueError, match="state contains"):
        transform({**valid, "state": np.array([0, 0, 0, 0, 0, 0, 0, np.nan])})
    with pytest.raises(ValueError, match="three RGB channels"):
        transform({**valid, "wrist_image": np.zeros((8, 8, 4), dtype=np.uint8)})


def test_outputs_strip_model_padding() -> None:
    actions = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    result = franka_policy.FrankaOutputs()({"actions": actions})
    np.testing.assert_array_equal(result["actions"], actions[:, :8])
