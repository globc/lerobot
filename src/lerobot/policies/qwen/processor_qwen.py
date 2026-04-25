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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torchvision.transforms import ToPILImage

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from .configuration_qwen import QwenConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import TransitionKey
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


@dataclass
@ProcessorStepRegistry.register(name="qwen_conversation_template_processor")
class QwenConversationTemplateStep(ComplementaryDataProcessorStep):
    input_features: dict[str, PolicyFeature] | dict[str, dict[str, Any]]

    _image_keys: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # Robust JSON deserialization handling (guard empty maps).
        if self.input_features:
            first_val = next(iter(self.input_features.values()))
            if isinstance(first_val, dict):
                reconstructed = {}
                for key, ft_dict in self.input_features.items():
                    reconstructed[key] = PolicyFeature(
                        type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                    )
                self.input_features = reconstructed

        self._image_keys = [
            key for key, value in self.input_features.items() if value.type == FeatureType.VISUAL
        ]

    def complementary_data(self, complementary_data):
        tasks = complementary_data.get("task")
        if tasks is None:
            raise ValueError("Task is required for QwenConversationTemplateStep.")

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for QwenConversationTemplateStep.")

        # LeRobot visual observations reach in processor as float32 tensors in [0, 1].
        # Convert to uint8 in [0, 255] to meet the input requirement of Qwen2.5-VL-3B-Instruct.
        images = {
            key: observation[key].clamp(0, 1).mul(255.0).round().to(torch.uint8) for key in self._image_keys
        }
        to_pil = ToPILImage()

        messages = []
        for i in range(len(tasks)):
            messages.append([
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": to_pil(images[key][i])} for key in self._image_keys],
                        {"type": "text", "text": f"The above images show the current state of a robot arm trying to complete the following task: {tasks[i]}. Given the current state, decompose the instruction into short subtasks and output only the current subtask to complete."},
                    ],
                }
            ])

        complementary_data["messages"] = messages

        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {
            "input_features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.input_features.items()
            },
        }


@dataclass
@ProcessorStepRegistry.register(name="qwen_processor")
class QwenProcessorStep(ComplementaryDataProcessorStep):
    processor_name: str = "Qwen/Qwen3-VL-8B-Instruct"

    _processor: AutoProcessor | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._processor = AutoProcessor.from_pretrained(
            self.processor_name,
            trust_remote_code=True
        )

    def complementary_data(self, complementary_data):
        messages = complementary_data.pop("messages", None)
        if messages is None:
            raise ValueError("Messages are required for QwenProcessorStep.")

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to("cuda")
        complementary_data["inputs"] = inputs

        return complementary_data

    def get_config(self) -> dict[str, Any]:
        return {
            "processor_name": self.processor_name,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step only converts the messages to the model input format.
        """
        return features

@dataclass
@ProcessorStepRegistry.register(name="qwen_output_processor")
class QwenOutputProcessorStep(ProcessorStep):
    processor_name: str = "Qwen/Qwen3-VL-8B-Instruct"

    _processor: AutoProcessor | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._processor = AutoProcessor.from_pretrained(
            self.processor_name,
            trust_remote_code=True
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()

        # Handle action unnormalization.
        generated_ids_trimmed = new_transition.get(TransitionKey.ACTION)
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        print(f"Model output text: {output_text}")

        new_transition[TransitionKey.ACTION] = output_text

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_qwen_pre_post_processors(
    config: QwenConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build pre/post processor pipelines for Qwen."""

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        QwenConversationTemplateStep(input_features=config.input_features),
        QwenProcessorStep(
            processor_name=config.model_name,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        QwenOutputProcessorStep(
            processor_name=config.model_name,
        ),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=lambda transition: transition[TransitionKey.ACTION],
        ),
    )