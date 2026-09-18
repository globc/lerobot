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
from .configuration_llarva import LlarvaConfig 
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


import numpy as np
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action 
from lerobot.types import TransitionKey 
from lerobot.utils.constants import ( 
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME, 
    POLICY_PREPROCESSOR_DEFAULT_NAME, 
) 

SYSTEM_PROMPT = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "

def process_state(state):
    # state = state + np.random.normal(0, 0.01, state.shape)
    if len(state) == 8: # LIBERO
        gripper_width = state[-2] - state[-1]
        state = state[:-2].tolist() + ([1.0] if gripper_width < 0.05 else [0.0])
    state = np.round(state, 4).tolist()
    return state

@dataclass 
@ProcessorStepRegistry.register(name="llarva_conversation_template_processor") 
class LlarvaConversationTemplateStep(ComplementaryDataProcessorStep):
    input_features: dict[str, PolicyFeature] | dict[str, dict[str, Any]] 

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

    def complementary_data(self, complementary_data): 
        init_logging() 
        tasks = complementary_data.get("task") 
        if tasks is None: 
            raise ValueError("Task is required for LlarvaConversationTemplateStep.")
        states = self.transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)

        if states.ndim == 2:
            states = states.unsqueeze(1)

        messages = [] 

        for b in range(len(tasks)):
            state_buffer = []
            for _ in range(5):
                state_buffer.append(process_state(states[b][0]))

            for state in states[b][-5:]:
                state_buffer.append(process_state(state))
                state_buffer = state_buffer[1:]
            input_text =  f'<image>\nYou are a Franka robot using the end effector control. The task is \"{tasks[b]}\", and the previous five (including current) steps is {state_buffer}, can you predict action of the next 8 step and the trajectories of the end effector?'
            prompt_text = SYSTEM_PROMPT + f"USER: {input_text} ASSISTANT:"
            
            messages.append(prompt_text) 
            
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


def make_llarva_pre_post_processors( 
    config: LlarvaConfig, 
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None, 
) -> tuple[ 
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]], 
    PolicyProcessorPipeline[PolicyAction, PolicyAction], 
]: 
    """Build pre/post processor pipelines for Llarva.""" 

    input_steps: list[ProcessorStep] = [ 
        RenameObservationsProcessorStep(rename_map={}), 
        AddBatchDimensionProcessorStep(), 
        LlarvaConversationTemplateStep(input_features=config.input_features),
        DeviceProcessorStep(device=config.device), 
    ] 

    output_steps: list[ProcessorStep] = [ 
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