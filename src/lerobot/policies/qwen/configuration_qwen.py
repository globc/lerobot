#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig


@PreTrainedConfig.register_subclass("qwen")
@dataclass
class QwenConfig(PreTrainedConfig):
    """Configuration for native Qwen policy integration in LeRobot."""
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    past_observations: int = 0 # Don't forget to set message "video"

    def validate_features(self) -> None:
        """Validate and set up Qwen input and output features."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "Qwen policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig()

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig()

    @property
    def observation_delta_indices(self) -> None:
        return list(range(self.past_observations)) if self.past_observations > 0 else None

    @property
    def action_delta_indices(self) -> list[int]:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None