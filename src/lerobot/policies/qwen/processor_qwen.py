#!/usr/bin/env python 

# Copyright 2026 The HuggingFace Inc. team. All rights reserved. 
#  
# Licensed under the Apache License, Version 2.0 (the "License"); 
# ... (standard license header) ... 

from __future__ import annotations 

import logging 
from lerobot.utils.utils import init_logging 
from dataclasses import dataclass, field 
from typing import TYPE_CHECKING, Any, cast 
from .utils import to_pil, ImageT 

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
    debug = True

    def __post_init__(self):
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
        init_logging()
        tasks = complementary_data.get("task")
        if tasks is None:
            raise ValueError("Task is required for QwenConversationTemplateStep.")

        observation = self.transition.get(TransitionKey.OBSERVATION, self.transition)
        if self.debug:
            print(f"Processing image keys: {self._image_keys}")
            self.debug = False

        # Dynamically load all image tensors based on the detected keys
        image_tensors = {}
        for key in self._image_keys:
            tensor = observation.get(key)
            if tensor is None:
                raise ValueError(f"Image key '{key}' is missing from the transition observation.")
            image_tensors[key] = tensor

        # Extract subtasks for supervised fine-tuning (if present)
        subtasks = complementary_data.get("subtask") or self.transition.get("subtask")

        messages = []

        for b in range(len(tasks)):
            message_content = []
            
            # Process and append an arbitrary number of images
            for key in self._image_keys:
                img_tensor = image_tensors[key][b]

                # Ensure channel-first format (C, H, W)
                if img_tensor.shape[-1] in (1, 3):
                    img_tensor = img_tensor.permute(2, 0, 1)

                # Convert float to uint8 scale if necessary
                if img_tensor.is_floating_point():
                    img_tensor = img_tensor * 255.0

                # Clamp, round, and convert to PIL Image
                pil_img = to_pil(img_tensor.clamp(0, 255).round().to(torch.uint8).contiguous())
                
                # Qwen-VL expects each image to be a separate dictionary in the content list
                message_content.append({"type": "image", "image": pil_img})

            # Generalized prompt for an arbitrary number of camera views
            prompt_text = (
                f"The above images show multiple camera views of a robot arm trying to complete the task '{tasks[b]}'. "
                "Pay attention to the pose of the robot and the surrounding environment and identify which subtask the robot should perform "
                "and in which direction it should move next. "
                "Output in the format SUBTASK (DIRECTION), where DIRECTION is '[[slightly] forward/backward], [[slightly] right/left], [[slightly] up/down]' ([] = optional). "
                "If the robot is at the boundary of completing a subtask or changing direction, sequence the immediate next steps using 'THEN' e.g. 'SUBTASK 1 (DIRECTION 1) THEN SUBTASK 1 (DIRECTION 2)."
            )
            message_content.append({"type": "text", "text": prompt_text})

            conv = [{"role": "user", "content": message_content}]

            # During training, append the ground truth label as the assistant's response
            if subtasks is not None and b < len(subtasks) and subtasks[b] is not None:
                conv.append({"role": "assistant", "content": str(subtasks[b])})

            messages.append(conv)

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
    max_length: int = 512  # CRITICAL FIX 2: Fixed sequence length required for DataLoader collation
    _processor: AutoProcessor | None = field(default=None, init=False, repr=False) 

    def __post_init__(self): 
        self._processor = AutoProcessor.from_pretrained( 
            self.processor_name, 
            trust_remote_code=True 
        ) 
        # CRITICAL FIX 3: Enforce right-padding so labels are masked correctly from left-to-right
        self._processor.tokenizer.padding_side = "right"

    def complementary_data(self, complementary_data): 
        messages = complementary_data.get("messages", None) 
        if messages is None: 
            raise ValueError("Messages are required for QwenProcessorStep.") 

        is_training = len(messages[0]) > 1 and messages[0][-1]["role"] == "assistant" 
        image_inputs, video_inputs = process_vision_info(messages) 

        if is_training: 
            full_texts = [ 
                self._processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
                for msg in messages 
            ] 
            inputs = self._processor( 
                text=full_texts, 
                images=image_inputs, 
                videos=video_inputs, 
                padding="max_length",  # Enforce uniform shape across unbatched samples
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt", 
            ) 

            prompt_messages = [msg[:-1] for msg in messages] 
            prompt_texts = [ 
                self._processor.apply_chat_template(p_msg, tokenize=False, add_generation_prompt=True) 
                for p_msg in prompt_messages 
            ] 
            prompt_inputs = self._processor( 
                text=prompt_texts, 
                images=image_inputs, 
                videos=video_inputs, 
                padding=True,  # FIX 1: Pad to longest prompt so internal NumPy array conversions succeed
                return_tensors="pt", 
            ) 

            labels = inputs["input_ids"].clone() 
            for i in range(len(messages)): 
                # FIX 2: Sum attention_mask to get the true unpadded prompt token count
                prompt_len = prompt_inputs["attention_mask"][i].sum().item()
                
                # Mask prompt tokens (including visual grid tokens) with -100
                labels[i, :prompt_len] = -100 
                # Mask trailing padding tokens with -100
                labels[i, inputs["attention_mask"][i] == 0] = -100

            inputs["labels"] = labels 
        else: 
            # CRITICAL FIX 1: Autoregressive generation requires left-padding!
            # Temporarily switch padding side so the prompt ends at the very last token.
            self._processor.tokenizer.padding_side = "left"
            
            texts = [ 
                self._processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False) 
                for msg in messages 
            ] 
            inputs = self._processor( 
                text=texts, 
                images=image_inputs, 
                videos=video_inputs, 
                padding=True,  # CRITICAL FIX 2: Pad only to the longest prompt in the batch, NOT max_length
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt", 
            ) 
            
            # Revert back to right-padding in case this step instance is reused for training later
            self._processor.tokenizer.padding_side = "right"

        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()} 
        complementary_data["inputs"] = inputs 
        return complementary_data 

    def get_config(self) -> dict[str, Any]: 
        return { 
            "processor_name": self.processor_name, 
            "max_length": self.max_length,
        } 

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
        QwenProcessorStep(processor_name=config.model_name),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = []

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