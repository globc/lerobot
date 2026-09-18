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


@PreTrainedConfig.register_subclass("llarva")
@dataclass
class LlarvaConfig(PreTrainedConfig):
    """Configuration for native Llarva policy integration in LeRobot."""
    model_name: str = "globcy/llarva_hf"
    chunk_size: int = 1000
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    dynamic_action_chunking: str = "trace"
    zero_shot: bool = False
    
    def validate_features(self) -> None:
        """Validate and set up Llarva input and output features."""
        image_features = [key for key, feat in self.input_features.items() if feat.type == FeatureType.VISUAL]
        if not image_features:
            raise ValueError(
                "Llarva policy requires at least one visual input feature. "
                "No features of type FeatureType.VISUAL found in input_features."
            )

    optimizer_lr: float = 5e-5
    optimizer_weight_decay: float = 0.0
    scheduler_decay_lr: float = 2.5e-6
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return [-4, -3, -2, -1, 0]

    @property
    def action_delta_indices(self) -> list[int]:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None