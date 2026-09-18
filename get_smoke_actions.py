#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test script: load a LeRobotDataset (with its stats), run a finetuned policy
checkpoint on the first frame's images + state, and save the predicted action
chunk to disk.
"""

from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.random_utils import set_seed  # noqa: E402

try:
    # Only available if you're running this inside lerobot's own test suite.
    from tests.utils import require_cuda  # noqa: E402
except ImportError:  # pragma: no cover - fine when run as a standalone script

    def require_cuda(fn):
        return fn


# --- Edit these three for your setup --------------------------------------
DATASET_REPO_ID = "globcy/apple_on_plate_l1"  # repo_id or local root for LeRobotDataset
POLICY_PATH = Path("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/outputs_real/pi05_smoke/checkpoints/001000/pretrained_model")  # finetuned checkpoint dir
OUTPUT_PATH = Path("outputs/action_chunk.pt")  # where the predicted chunk is saved
# ----------------------------------------------------------------------------

STATE_KEY = "observation.state"
IMAGE_KEYS = ["observation.images.image", "observation.images.image2"]


def _run():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the dataset. `dataset.meta.stats` carries the per-feature
    # normalization stats (mean/std/min/max/q01/q99) shipped with the dataset,
    # which the pre/post-processors use to (un)normalize inputs and actions.
    dataset = LeRobotDataset(DATASET_REPO_ID)
    dataset_stats = dataset.meta.stats

    for key in [*IMAGE_KEYS, STATE_KEY]:
        assert key in dataset.features, f"'{key}' not found in dataset features: {list(dataset.features)}"

    # 2. Load the config from the checkpoint. `PreTrainedConfig.from_pretrained`
    # reads config.json and dispatches to the right policy config subclass
    # (pi0, pi05, act, diffusion, ...) automatically, so you don't need to know
    # the policy type up front. Setting `pretrained_path` tells `make_policy`
    # to load the finetuned weights rather than randomly initializing them.
    config = PreTrainedConfig.from_pretrained(POLICY_PATH)
    config.pretrained_path = str(POLICY_PATH)

    policy = make_policy(config, ds_meta=dataset.meta)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        POLICY_PATH,
        dataset_stats=dataset_stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # 3. Grab the first frame's two images and state.
    frame = dict(dataset[0])
    batch = {
        IMAGE_KEYS[0]: frame[IMAGE_KEYS[0]],
        IMAGE_KEYS[1]: frame[IMAGE_KEYS[1]],
        STATE_KEY: frame[STATE_KEY],
        "task": frame.get("task", ""),
    }
    batch = preprocessor(batch)

    # 4. Run inference. `predict_action_chunk` returns the full predicted
    # horizon, shape (batch, chunk_size, action_dim). `select_action` would
    # instead pop a single action off the policy's internal queue, which is
    # not what we want here.
    with torch.inference_mode():
        if hasattr(policy, "predict_action_chunk"):
            action_chunk = policy.predict_action_chunk(batch)
        else:
            action_chunk = policy.select_action(batch)
        action_chunk = postprocessor(action_chunk)

    print(f"Action chunk shape: {tuple(action_chunk.shape)}")

    # 5. Save the generated action chunk.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(action_chunk.cpu(), OUTPUT_PATH)
    print(f"Saved action chunk to {OUTPUT_PATH.resolve()}")

    return action_chunk


@require_cuda
def test_finetuned_policy_action_chunk():
    action_chunk = _run()
    assert action_chunk is not None
    assert action_chunk.ndim in (2, 3)  # (chunk_size, action_dim) or (batch, chunk_size, action_dim)


if __name__ == "__main__":
    _run()