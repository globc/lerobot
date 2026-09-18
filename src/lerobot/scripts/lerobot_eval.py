#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Evaluate a policy on an environment by running rollouts and computing metrics.

Requires: pip install 'lerobot[evaluation]' plus the policy extra (e.g. lerobot[pi])
          and the environment extra (e.g. lerobot[pusht]) if evaluating in simulation.

Usage examples:

You want to evaluate a model from the hub (eg: https://huggingface.co/lerobot/diffusion_pusht)
for 10 episodes.

```
lerobot-eval \
    --policy.path=lerobot/diffusion_pusht \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

OR, you want to evaluate a model checkpoint from the LeRobot training script for 10 episodes.
```
lerobot-eval \
    --policy.path=outputs/train/diffusion_pusht/checkpoints/005000/pretrained_model \
    --env.type=pusht \
    --eval.batch_size=10 \
    --eval.n_episodes=10 \
    --policy.use_amp=false \
    --policy.device=cuda
```

Note that in both examples, the repo/folder should contain at least `config.json` and `model.safetensors` files.

You can learn about the CLI options for this script in the `EvalPipelineConfig` in lerobot/configs/eval.py
"""

import concurrent.futures as cf
from itertools import cycle
import json
import logging
import threading
import time
import textwrap
import cv2
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from pprint import pformat
from typing import Any, TypedDict
from collections import deque
import re

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import (
    check_env_attributes_and_types,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.types import PolicyAction
from lerobot.utils.constants import ACTION, DONE, OBS_STR, REWARD
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import (
    init_logging,
    inside_slurm,
)

def is_noop(action, prev_action=None, threshold=1e-4):
    action = action[0] # assert batch_size=1
    if prev_action is None: # or True
        # logging.info("Noop skip prefetch")
        return np.linalg.norm(action[:-1]) < threshold

    prev_action = prev_action[0]
    gripper_action = round(action[-1].item())
    prev_gripper_action = round(prev_action[-1].item())
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action

import re
import logging
import os
from collections import deque
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import einops

# (Assuming other required imports like is_noop, ACTION, OBS_STR, preprocess_observation, etc. are handled globally)

def rollout(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    planner: PreTrainedPolicy | None,
    planner_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None,
    planner_postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None,
    seeds: list[int] | None = None,
    return_observations: bool = False,
    render_callback: Callable[[gym.vector.VectorEnv, Any], None] | None = None,
    force_policy_steps: int = 0,
) -> dict:
    """Run a batched policy rollout once through a batch of environments."""
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    # Helper function to enforce directional ordering in subtasks
    def _reorder_match(match):
        content = match.group(1)
        if "|" in content:
            dirs_part, steps_part = content.split("|", 1)
            dirs_part = dirs_part.strip()
            steps_part = " | " + steps_part.strip()
        else:
            dirs_part = content.strip()
            steps_part = ""
        
        dirs = [d.strip() for d in dirs_part.split(",")]
        
        def get_prio(d):
            d_lower = d.lower()
            if "forward" in d_lower or "backward" in d_lower: return 1
            if "left" in d_lower or "right" in d_lower: return 2
            if "up" in d_lower or "down" in d_lower: return 3
            return 4 # Keeps unrelated matches functionally untouched
            
        return "(" + ", ".join(sorted(dirs, key=get_prio)) + steps_part + ")"

    # Reset the policy and environments.
    policy.reset()
    current_task = [""] * env.num_envs
    observation, info = env.reset(seed=seeds)
    if render_callback is not None:
        render_callback(env, current_task)

    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    observations_planner = None
    action_queue = deque(maxlen=policy.config.n_action_steps)
    subtask_queue = deque(maxlen=2)
    prev_action = None
    policy_subtask_end = False

    # Forcing mechanism trackers
    steps_since_planner = 0
    current_force_target = 0
    last_executed_subtask = ""

    step = 0
    # Keep track of which environments are done.
    done = np.array([False] * env.num_envs)
    max_steps = env.call("_max_episode_steps")[0]
    progbar = trange(
        max_steps,
        desc=f"Running rollout with at most {max_steps} steps",
        disable=inside_slurm(),  # we dont want progress bar when we use slurm, since it clutters the logs
        leave=False,
    )
    check_env_attributes_and_types(env)
    try:
        while not np.all(done) and step < max_steps:
            # Numpy array to tensor and changing dictionary keys to LeRobot policy format.
            observation = preprocess_observation(observation)
            if return_observations:
                all_observations.append(deepcopy(observation))

            # Infer "task" from sub-environments (prefer natural language description).
            try:
                observation["task"] = list(env.call("task_description"))
            except (AttributeError, NotImplementedError):
                try:
                    observation["task"] = list(env.call("task"))
                except (AttributeError, NotImplementedError):
                    observation["task"] = [""] * env.num_envs

            # Apply environment-specific preprocessing
            observation = env_preprocessor(observation)
            if planner is not None:
                if planner.name not in ["llarva"]: # ["qwen", "llarva"]:
                    observations_planner = observation
                elif observations_planner is None:
                    observations_planner = deepcopy(observation)
                    
                    if planner.name == "llarva" and observations_planner["observation.state"].ndim == 2:
                        observations_planner["observation.state"] = observations_planner["observation.state"].unsqueeze(1)
                    
                    # elif planner.name == "qwen" and observations_planner["observation.images.image"].ndim == 4:
                    #     observations_planner["observation.images.image"] = observations_planner["observation.images.image"].unsqueeze(1)
                
                else:
                    if planner.name == "llarva":
                        curr_state = observation["observation.state"]
                        if curr_state.ndim == 2:
                            curr_state = curr_state.unsqueeze(1)
                            
                        observations_planner["observation.state"] = torch.cat(
                            (observations_planner["observation.state"], curr_state), dim=1
                        )
                        observations_planner["observation.images.image"] = observation["observation.images.image"]
                        
                    # elif planner.name == "qwen":
                    #     curr_image = observation["observation.images.image"]
                    #     if curr_image.ndim == 4:
                    #         curr_image = curr_image.unsqueeze(1)
                            
                    #     observations_planner["observation.images.image"] = torch.cat(
                    #         (observations_planner["observation.images.image"], curr_image), dim=1
                    #     )

            while len(action_queue) == 0:
                if len(subtask_queue) == 0:
                    # ---------------------------------------------------------
                    # Check if we should enforce minimum steps before replanning
                    # ---------------------------------------------------------
                    if current_force_target > 0 and steps_since_planner < current_force_target:
                        # Note: We re-queue the RAW subtask so we still have step numbers if needed internally
                        subtask_queue.append(last_executed_subtask)
                        logging.info(f"Forcing policy steps ({steps_since_planner}/{current_force_target}). Requeuing subtask: {last_executed_subtask}")
                    else:
                        if planner is not None:
                            observation_planner = planner_preprocessor(observations_planner)
                            with torch.inference_mode():
                                subtask = planner.select_action(observation_planner)
                                if planner_postprocessor is not None:
                                    subtask = planner_postprocessor(subtask)

                            subtask = subtask[0] if isinstance(subtask, list) else subtask
                            
                            # Apply forcing rules / exceptions
                            has_then = " THEN " in subtask
                            match = re.search(r'\|\s*(\d+)', subtask)
                            
                            if match:
                                parsed_steps = int(match.group(1))
                                chunk_size = getattr(policy.config, "chunk_size", policy.config.n_action_steps)
                                # Exception 2: if planner specifies steps, use min(planner_steps, chunk_size) regardless of "THEN"
                                current_force_target = min(parsed_steps, chunk_size) if force_policy_steps != 0 else 0
                            else:
                                # Exception 1: if "THEN" is present, no forcing
                                current_force_target = 0 if has_then else force_policy_steps

                            steps_since_planner = 0

                            if policy.config.train_then:
                                subtask_queue.append(subtask)
                            else:
                                subtasks = subtask.split(" THEN ")
                                for sub in subtasks:
                                    subtask_queue.append(sub)
                            # subtask_queue.append(subtask)
                            logging.info(f"Planner selected task: {subtask}")
                            
                        elif getattr(policy.config, "hierarchical", False):
                            import matplotlib.pyplot as plt
                            from pathlib import Path
                            
                            img = observation['observation.images.image'][0]
                            img_wrist = observation['observation.images.image2'][0]
                            if isinstance(img, torch.Tensor):
                                img = img.detach().cpu().numpy()
                                img_wrist = img_wrist.detach().cpu().numpy()
                            
                            if img.ndim == 3 and img.shape[0] in (1, 3):
                                img = np.transpose(img, (1, 2, 0))
                                img_wrist = np.transpose(img_wrist, (1, 2, 0))
                            
                            plt.imsave("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/current_observation.png", img)
                            plt.imsave("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/current_observation_wrist.png", img_wrist)
                            
                            user_subtask = input(f"\nEnter subtask for task '{observation['task'][0]}': ")
                            subtask_queue.append(user_subtask)
                            
                            steps_since_planner = 0
                            current_force_target = force_policy_steps
                        else:
                            subtask_queue.append(observation["task"][0])
                            steps_since_planner = 0
                            current_force_target = force_policy_steps

                # Pop raw subtask first
                next_raw_subtask = subtask_queue.popleft()
                
                # Enforce deterministic directional order when chunking strategy dictates
                # if getattr(policy.config, "dynamic_action_chunking", None) == "subtask_move":
                #     next_raw_subtask = re.sub(r'\(([^)]+)\)', _reorder_match, next_raw_subtask)
                
                # Store the raw subtask for the forcing logic to use if it needs to requeue
                last_executed_subtask = next_raw_subtask
                
                planner_steps = policy.config.n_action_steps
                # Let planner predict steps for sequence length using raw subtask string
                match = re.search(r'\|\s*(\d+)', next_raw_subtask)
                if match:
                    planner_steps = int(match.group(1))

                # Clean the string for the policy condition: remove " | <digits>"
                next_clean_subtask = re.sub(r'\s*\|\s*\d+', '', next_raw_subtask)
                
                observation["subtask"] = [next_clean_subtask]
                current_task = [next_clean_subtask]
                observation = preprocessor(observation)

                actions = []
                for _ in range(min(policy.config.n_action_steps, planner_steps)):
                    with torch.inference_mode():
                        action = policy.select_action(observation)
                    _prev_action = actions[-1] if len(actions) > 0 else prev_action

                    if not (policy.config.ilfm and is_noop(action.detach().cpu().numpy(), _prev_action, threshold=1e-10)):
                        action = postprocessor(action)
                    else:
                        logging.info("Skipping action postprocessing due to ILFM noop.")
                        logging.info(f"Action: {action})")

                    action_transition = {ACTION: action}
                    action_transition = env_postprocessor(action_transition)
                    action = action_transition[ACTION]

                    # Convert to CPU / numpy.
                    action_numpy: np.ndarray = action.to("cpu").numpy()
                    assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

                    actions.append(action_numpy)
                
                if policy.config.dynamic_action_chunking:
                    consecutive_noops = 0
                    for i in range(len(actions) - 1, -1, -1):
                        _prev_action = actions[i - 1] if i > 0 else prev_action
                        if is_noop(actions[i], _prev_action, threshold=0.01):
                            consecutive_noops += 1
                        else:
                            policy_subtask_end = consecutive_noops > 1
                            action_queue.extend(actions[:i+1])
                            break
                    if len(action_queue) == 0:
                        policy_subtask_end = True
                    logging.info(f"Consecutive no-ops at the end of the action sequence: {consecutive_noops}.")
                    if not policy_subtask_end:
                        subtask_queue.clear()
                else:
                    action_queue.extend(actions)

            action_numpy = action_queue.popleft()
            
            # Increment tracking steps when action is consumed
            steps_since_planner += 1
            
            prev_action = action_numpy
            # Apply the next action.
            observation, reward, terminated, truncated, info = env.step(action_numpy)
            if render_callback is not None:
                render_callback(env, current_task)

            if "final_info" in info:
                final_info = info["final_info"]
                if not isinstance(final_info, dict):
                    raise RuntimeError(
                        "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                        "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
                    )
                successes = final_info["is_success"].tolist()
            elif "is_success" in info:
                is_success = info["is_success"]
                successes = (
                    is_success.tolist() if hasattr(is_success, "tolist") else [bool(is_success)] * env.num_envs
                )
            else:
                successes = [False] * env.num_envs

            done = terminated | truncated | done
            if step + 1 == max_steps:
                done = np.ones_like(done, dtype=bool)

            all_actions.append(torch.from_numpy(action_numpy))
            all_rewards.append(torch.from_numpy(reward))
            all_dones.append(torch.from_numpy(done))
            all_successes.append(torch.tensor(successes))

            step += 1
            running_success_rate = (
                einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy().mean()
            )
            progbar.set_postfix({"running_success_rate": f"{running_success_rate.item() * 100:.1f}%"})
            progbar.update()
    except Exception as e:
        error_msg = (
            f"Error during rollout at step {step}/{max_steps}.\n"
            f" -> Seeds: {seeds}\n"
            f" -> Task(s): {current_task}\n"
            f" -> Last Subtask: '{last_executed_subtask}'\n"
            f" -> Exception: {e}"
        )
        logging.error(error_msg)

    # Track the final observation.
    if return_observations:
        observation = preprocess_observation(observation)
        all_observations.append(deepcopy(observation))

    ret = {
        ACTION: torch.stack(all_actions, dim=1),
        "reward": torch.stack(all_rewards, dim=1),
        "success": torch.stack(all_successes, dim=1),
        "done": torch.stack(all_dones, dim=1),
    }
    if return_observations:
        stacked_observations = {}
        for key in all_observations[0]:
            stacked_observations[key] = torch.stack([obs[key] for obs in all_observations], dim=1)
        ret[OBS_STR] = stacked_observations

    if hasattr(policy, "use_original_modules"):
        policy.use_original_modules()

    if hasattr(policy, "track_analysis") and policy.track_analysis:
        # Append final success state to easily filter good vs bad rollouts offline
        final_success_array = einops.reduce(torch.stack(all_successes, dim=1), "b n -> b", "any").numpy()
        policy.analysis_data.setdefault("final_success", []).append(final_success_array)

    return ret


def eval_policy(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    planner: PreTrainedPolicy | None,
    planner_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None,
    planner_postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None,
    n_episodes: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    force_policy_steps: int = 0,
) -> dict:
    """
    Args:
        env: The batch of environments.
        policy: The policy.
        n_episodes: The number of episodes to evaluate.
        max_episodes_rendered: Maximum number of episodes to render into videos.
        videos_dir: Where to save rendered videos.
        return_episode_data: Whether to return episode data for online training. Incorporates the data into
            the "episodes" key of the returned dictionary.
        start_seed: The first seed to use for the first individual rollout. For all subsequent rollouts the
            seed is incremented by 1. If not provided, the environments are not manually seeded.
    Returns:
        Dictionary with metrics and data regarding the rollouts.
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        exc = ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )
        try:
            from peft import PeftModel

            if not isinstance(policy, PeftModel):
                raise exc
        except ImportError:
            raise exc from None

    start = time.time()
    policy.eval()

    # Determine how many batched rollouts we need to get n_episodes. Note that if n_episodes is not evenly
    # divisible by env.num_envs we end up discarding some data in the last batch.
    n_batches = n_episodes // env.num_envs + int((n_episodes % env.num_envs) != 0)

    # Keep track of some metrics.
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # for video saving threads
    n_episodes_rendered = 0  # for saving the correct number of videos

    # Callback for visualization.
    def render_frame(env: gym.vector.VectorEnv, tasks: list[str] | None = None):
        # noqa: B023
        if n_episodes_rendered >= max_episodes_rendered:
            return
        n_to_render_now = min(max_episodes_rendered - n_episodes_rendered, env.num_envs)
        
        if tasks is None:
            tasks = [""] * n_to_render_now
        elif isinstance(tasks, str):
            tasks = [tasks] * n_to_render_now

        raw_frames = []
        if isinstance(env, gym.vector.SyncVectorEnv):
            raw_frames = [env.envs[i].render() for i in range(n_to_render_now)]
        elif hasattr(env, "call"):
            # Here we must render all frames and discard any we don't need.
            # Covers AsyncVectorEnv and _LazyAsyncVectorEnv (which wraps one).
            raw_frames = env.call("render")[:n_to_render_now]

        annotated_frames = []
        for i, frame in enumerate(raw_frames):
            frame = np.ascontiguousarray(frame)
            task_str = str(tasks[i]) if i < len(tasks) else ""
            
            if task_str:
                # Check for dynamic action chunking "trace" config 
                if getattr(policy.config, "dynamic_action_chunking", None) == "trace":
                    import ast
                    try:
                        # Parse trace string (e.g. "[[32, 12], [32, 32], ... [64, 23]]")
                        points = ast.literal_eval(task_str)
                        for pt in points:
                            x, y = int(pt[0]), int(pt[1])
                            x = round(x / 256 * 360)
                            y = round(y / 256 * 360)
                            # Draw inner red circle
                            cv2.circle(frame, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
                            # Draw outer black edge for contrast
                            cv2.circle(frame, (x, y), radius=4, color=(0, 0, 0), thickness=1)
                    except (ValueError, SyntaxError) as e:
                        logging.warning(f"Failed to parse trace points: {e}")
                else:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.4
                    thickness = 1
                    # Wrap text to ~35 characters to fit within a 256px wide frame nicely
                    wrapped_text = textwrap.wrap(task_str, width=35)
                    
                    y0, dy = 15, 15
                    for j, line in enumerate(wrapped_text):
                        y = y0 + j * dy
                        # Draw text outline for better contrast over arbitrary backgrounds
                        cv2.putText(frame, line, (5, y), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                        # Draw internal white text
                        cv2.putText(frame, line, (5, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
            annotated_frames.append(frame)

        if annotated_frames:
            ep_frames.append(np.stack(annotated_frames))  # noqa: B023

    if max_episodes_rendered > 0:
        video_paths: list[str] = []

    if return_episode_data:
        episode_data: dict | None = None

    # we dont want progress bar when we use slurm, since it clutters the logs
    progbar = trange(n_batches, desc="Stepping through eval batches", disable=inside_slurm())
    for batch_ix in progbar:
        # Cache frames for rendering videos. Each item will be (b, h, w, c), and the list indexes the rollout
        # step.
        if max_episodes_rendered > 0:
            ep_frames: list[np.ndarray] = []

        if start_seed is None:
            seeds = None
        else:
            seeds = range(
                start_seed + (batch_ix * env.num_envs), start_seed + ((batch_ix + 1) * env.num_envs)
            )
        rollout_data = rollout(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            planner=planner,
            planner_preprocessor=planner_preprocessor,
            planner_postprocessor=planner_postprocessor,
            seeds=list(seeds) if seeds else None,
            return_observations=return_episode_data,
            render_callback=render_frame if max_episodes_rendered > 0 else None,
            force_policy_steps=force_policy_steps,
        )

        # Figure out where in each rollout sequence the first done condition was encountered (results after
        # this won't be included).
        n_steps = rollout_data["done"].shape[1]
        # Note: this relies on a property of argmax: that it returns the first occurrence as a tiebreaker.
        done_indices = torch.argmax(rollout_data["done"].to(int), dim=1)

        # Make a mask with shape (batch, n_steps) to mask out rollout data after the first done
        # (batch-element-wise). Note the `done_indices + 1` to make sure to keep the data from the done step.
        mask = (torch.arange(n_steps) <= einops.repeat(done_indices + 1, "b -> b s", s=n_steps)).int()
        # Extend metrics.
        batch_sum_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "sum")
        sum_rewards.extend(batch_sum_rewards.tolist())
        batch_max_rewards = einops.reduce((rollout_data["reward"] * mask), "b n -> b", "max")
        max_rewards.extend(batch_max_rewards.tolist())
        batch_successes = einops.reduce((rollout_data["success"] * mask), "b n -> b", "any")
        all_successes.extend(batch_successes.tolist())
        if seeds:
            all_seeds.extend(seeds)
        else:
            all_seeds.append(None)

        # FIXME: episode_data is either None or it doesn't exist
        if return_episode_data:
            this_episode_data = _compile_episode_data(
                rollout_data,
                done_indices,
                start_episode_index=batch_ix * env.num_envs,
                start_data_index=(0 if episode_data is None else (episode_data["index"][-1].item() + 1)),
                fps=env.unwrapped.metadata["render_fps"],
            )
            if episode_data is None:
                episode_data = this_episode_data
            else:
                # Some sanity checks to make sure we are correctly compiling the data.
                assert episode_data["episode_index"][-1] + 1 == this_episode_data["episode_index"][0]
                assert episode_data["index"][-1] + 1 == this_episode_data["index"][0]
                # Concatenate the episode data.
                episode_data = {k: torch.cat([episode_data[k], this_episode_data[k]]) for k in episode_data}

        # Maybe render video for visualization.
        if max_episodes_rendered > 0 and len(ep_frames) > 0:
            batch_stacked_frames = np.stack(ep_frames, axis=1)  # (b, t, *)
            for stacked_frames, done_index in zip(
                batch_stacked_frames, done_indices.flatten().tolist(), strict=False
            ):
                if n_episodes_rendered >= max_episodes_rendered:
                    break

                videos_dir.mkdir(parents=True, exist_ok=True)
                video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
                video_paths.append(str(video_path))
                thread = threading.Thread(
                    target=write_video,
                    args=(
                        str(video_path),
                        stacked_frames[: done_index + 1],  # + 1 to capture the last observation
                        env.unwrapped.metadata["render_fps"],
                    ),
                )
                thread.start()
                threads.append(thread)
                n_episodes_rendered += 1

        progbar.set_postfix(
            {"running_success_rate": f"{np.mean(all_successes[:n_episodes]).item() * 100:.1f}%"}
        )


    # Wait till all video rendering threads are done.
    for thread in threads:
        thread.join()

    # Compile eval info.
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(
                    sum_rewards[:n_episodes],
                    max_rewards[:n_episodes],
                    all_successes[:n_episodes],
                    all_seeds[:n_episodes],
                    strict=True,
                )
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards[:n_episodes])),
            "avg_max_reward": float(np.nanmean(max_rewards[:n_episodes])),
            "pc_success": float(np.nanmean(all_successes[:n_episodes]) * 100),
            "eval_s": time.time() - start,
            "eval_ep_s": (time.time() - start) / n_episodes,
        },
    }

    if return_episode_data:
        info["episodes"] = episode_data

    if max_episodes_rendered > 0:
        info["video_paths"] = video_paths

    return info


def _compile_episode_data(
    rollout_data: dict, done_indices: Tensor, start_episode_index: int, start_data_index: int, fps: float
) -> dict:
    """Convenience function for `eval_policy(return_episode_data=True)`

    Compiles all the rollout data into a Hugging Face dataset.

    Similar logic is implemented when datasets are pushed to hub (see: `push_to_hub`).
    """
    ep_dicts = []
    total_frames = 0
    for ep_ix in range(rollout_data[ACTION].shape[0]):
        # + 2 to include the first done frame and the last observation frame.
        num_frames = done_indices[ep_ix].item() + 2
        total_frames += num_frames

        # Here we do `num_frames - 1` as we don't want to include the last observation frame just yet.
        ep_dict = {
            ACTION: rollout_data[ACTION][ep_ix, : num_frames - 1],
            "episode_index": torch.tensor([start_episode_index + ep_ix] * (num_frames - 1)),
            "frame_index": torch.arange(0, num_frames - 1, 1),
            "timestamp": torch.arange(0, num_frames - 1, 1) / fps,
            DONE: rollout_data["done"][ep_ix, : num_frames - 1],
            "next.success": rollout_data["success"][ep_ix, : num_frames - 1],
            REWARD: rollout_data["reward"][ep_ix, : num_frames - 1].type(torch.float32),
        }

        # For the last observation frame, all other keys will just be copy padded.
        for k in ep_dict:
            ep_dict[k] = torch.cat([ep_dict[k], ep_dict[k][-1:]])

        for key in rollout_data[OBS_STR]:
            ep_dict[key] = rollout_data[OBS_STR][key][ep_ix, :num_frames]

        ep_dicts.append(ep_dict)

    data_dict = {}
    for key in ep_dicts[0]:
        data_dict[key] = torch.cat([x[key] for x in ep_dicts])

    data_dict["index"] = torch.arange(start_data_index, start_data_index + total_frames, 1)

    return data_dict


@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    if cfg.repo_id is None:
        logging.info("Making environment.")
        envs = make_env(
            cfg.env,
            n_envs=cfg.eval.batch_size,
            use_async_envs=cfg.eval.use_async_envs,
            trust_remote_code=cfg.trust_remote_code,
        )

        logging.info("Making policy.")

        policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
            rename_map=cfg.rename_map,
        )

        if cfg.planner is not None:
            logging.info("Making planner.")

            planner = make_policy(
                cfg=cfg.planner,
                env_cfg=cfg.env,
                rename_map=cfg.rename_map,
            )
            planner.eval()

    else:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.factory import resolve_delta_timestamps
        ds_meta = LeRobotDatasetMetadata(cfg.repo_id)
        delta_timestamps = {}
        delta_timestamps[ACTION] = [i / ds_meta.fps for i in range(1000)]
        dataset = LeRobotDataset(cfg.repo_id, delta_timestamps=delta_timestamps, dynamic_action_chunking=cfg.sub_key, eval=True, split=cfg.split, n_splits=cfg.n_splits, bottom_up=cfg.bottom_up, chunk_size=cfg.bottom_up_chunk_size)

        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            rename_map=cfg.rename_map,
        )

        if cfg.planner is not None:
            logging.info("Making planner.")

            planner = make_policy(
                cfg=cfg.planner,
                ds_meta=dataset.meta,
                rename_map=cfg.rename_map,
            )
            planner.eval()

    policy.eval()

    # The inference device is automatically set to match the detected hardware, overriding any previous device settings from training to ensure compatibility.
    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    if cfg.policy.type == "pi05":
        preprocessor_overrides["pi05_prepare_state_tokenizer_processor_step"] = {
            "hierarchical": policy.config.hierarchical,
            "include_task": policy.config.include_task,
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides=preprocessor_overrides,
    )

    if cfg.planner is not None:
        preprocessor_overrides = {
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        planner_preprocessor, planner_postprocessor = make_pre_post_processors(
            policy_cfg=cfg.planner,
            pretrained_path=cfg.planner.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        if cfg.planner.type in ["pi0_fast", "openvla", "llarva", "qwen"]:
            planner_postprocessor = None

    if cfg.repo_id is not None:
        with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
            info = eval_policy_dataset(
                dataset=dataset,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                planner=planner if cfg.planner is not None else None,
                planner_preprocessor=planner_preprocessor if cfg.planner is not None else None,
                planner_postprocessor=planner_postprocessor if cfg.planner is not None else None,
                batch_size=cfg.eval.batch_size,
                start_seed=cfg.seed,
                split=cfg.split,
                output_dir=cfg.output_dir,
                sub_key=cfg.sub_key,
            )
    else:
        # Create environment-specific preprocessor and postprocessor (e.g., for LIBERO environments)
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

        with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                planner=planner if cfg.planner is not None else None,
                planner_preprocessor=planner_preprocessor if cfg.planner is not None else None,
                planner_postprocessor=planner_postprocessor if cfg.planner is not None else None,
                max_episodes_rendered=10,
                videos_dir=Path(cfg.output_dir) / "videos",
                start_seed=cfg.seed,
                max_parallel_tasks=cfg.env.max_parallel_tasks,
                force_policy_steps=cfg.force_policy_steps,
            )
        # Close all vec envs
        close_envs(envs)

    print("Overall Aggregated Metrics:")
    print(info["overall"])

    for task_group, task_group_info in info.items():
        print(f"\nAggregated Metrics for {task_group}:")
        print(task_group_info)

    # Save info
    with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    # --- ADD THIS BLOCK TO SAVE ANALYSIS TENSORS ---
    if hasattr(policy, "track_analysis") and policy.track_analysis:
        analysis_path = Path(cfg.output_dir) / "pi05_analysis_results.pt"
        try:
            policy.save_analysis(str(analysis_path))
            logging.info(f"Successfully saved analysis offline traces to: {analysis_path}")
        except Exception as e:
            logging.error(f"Failed to save analysis: {e}")

    logging.info("End of eval")


# ---- typed payload returned by one task eval ----
class TaskMetrics(TypedDict):
    sum_rewards: list[float]
    max_rewards: list[float]
    successes: list[bool]
    video_paths: list[str]


ACC_KEYS = ("sum_rewards", "max_rewards", "successes", "video_paths")


def eval_one(
    env: gym.vector.VectorEnv,
    *,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    planner: PreTrainedPolicy | None,
    planner_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None,
    planner_postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    force_policy_steps: int,
) -> TaskMetrics:
    """Evaluates one task_id of one suite using the provided vec env."""

    task_videos_dir = videos_dir

    task_result = eval_policy(
        env=env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        planner=planner,
        planner_preprocessor=planner_preprocessor,
        planner_postprocessor=planner_postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        force_policy_steps=force_policy_steps,
    )

    per_episode = task_result["per_episode"]
    return TaskMetrics(
        sum_rewards=[ep["sum_reward"] for ep in per_episode],
        max_rewards=[ep["max_reward"] for ep in per_episode],
        successes=[ep["success"] for ep in per_episode],
        video_paths=task_result.get("video_paths", []),
    )


def run_one(
    task_group: str,
    task_id: int,
    env,
    *,
    policy,
    env_preprocessor,
    env_postprocessor,
    preprocessor,
    postprocessor,
    planner,
    planner_preprocessor,
    planner_postprocessor,
    n_episodes: int,
    max_episodes_rendered: int,
    videos_dir: Path | None,
    return_episode_data: bool,
    start_seed: int | None,
    force_policy_steps: int,
):
    """
    Run eval_one for a single (task_group, task_id, env).
    Returns (task_group, task_id, task_metrics_dict).
    This function is intentionally module-level to make it easy to test.
    """
    task_videos_dir = None
    if videos_dir is not None:
        task_videos_dir = videos_dir / f"{task_group}_{task_id}"
        task_videos_dir.mkdir(parents=True, exist_ok=True)

    # Call the existing eval_one (assumed to return TaskMetrics-like dict)
    metrics = eval_one(
        env,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        planner=planner,
        planner_preprocessor=planner_preprocessor,
        planner_postprocessor=planner_postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=task_videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        force_policy_steps=force_policy_steps,
    )
    # ensure we always provide video_paths key to simplify accumulation
    if max_episodes_rendered > 0:
        metrics.setdefault("video_paths", [])
    return task_group, task_id, metrics


def eval_policy_all(
    envs: dict[str, dict[int, gym.vector.VectorEnv]],
    policy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    n_episodes: int,
    planner=None,
    planner_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    planner_postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    *,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
    max_parallel_tasks: int = 1,
    force_policy_steps: int = 0,
) -> dict:
    """
    Evaluate a nested `envs` dict: {task_group: {task_id: vec_env}}.
    This implementation flattens tasks, runs them sequentially or via ThreadPoolExecutor,
    accumulates per-group and overall statistics, and returns the same aggregate metrics
    schema as the single-env evaluator (avg_sum_reward / avg_max_reward / pc_success / timings)
    plus per-task infos.
    """
    start_t = time.time()

    # Flatten envs into list of (task_group, task_id, env)
    tasks = [(tg, tid, vec) for tg, group in envs.items() for tid, vec in group.items()]

    # accumulators: track metrics at both per-group level and across all groups
    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in ACC_KEYS})
    overall: dict[str, list] = {k: [] for k in ACC_KEYS}
    per_task_infos: list[dict] = []

    # small inline helper to accumulate one task's metrics into accumulators
    def _accumulate_to(group: str, metrics: dict):
        # metrics expected to contain 'sum_rewards', 'max_rewards', 'successes', optionally 'video_paths'
        # but eval_one may store per-episode lists; we assume metrics uses scalars averaged per task as before.
        # To be robust, accept scalars or lists.
        def _append(key, value):
            if value is None:
                return
            if isinstance(value, list):
                group_acc[group][key].extend(value)
                overall[key].extend(value)
            else:
                group_acc[group][key].append(value)
                overall[key].append(value)

        _append("sum_rewards", metrics.get("sum_rewards"))
        _append("max_rewards", metrics.get("max_rewards"))
        _append("successes", metrics.get("successes"))
        # video_paths is list-like
        paths = metrics.get("video_paths", [])
        if paths:
            group_acc[group]["video_paths"].extend(paths)
            overall["video_paths"].extend(paths)

    # Choose runner (sequential vs threaded)
    task_runner = partial(
        run_one,
        policy=policy,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        planner=planner,
        planner_preprocessor=planner_preprocessor,
        planner_postprocessor=planner_postprocessor,
        n_episodes=n_episodes,
        max_episodes_rendered=max_episodes_rendered,
        videos_dir=videos_dir,
        return_episode_data=return_episode_data,
        start_seed=start_seed,
        force_policy_steps=force_policy_steps,
    )

    if max_parallel_tasks <= 1:
        prefetch_thread: threading.Thread | None = None
        for i, (task_group, task_id, env) in enumerate(tasks):
            if prefetch_thread is not None:
                prefetch_thread.join()
                prefetch_thread = None

            try:
                tg, tid, metrics = task_runner(task_group, task_id, env)
                _accumulate_to(tg, metrics)
                per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
            finally:
                env.close()
                # Prefetch next task's workers *after* closing current env to prevent
                # GPU memory overlap between consecutive tasks.
                if i + 1 < len(tasks):
                    next_env = tasks[i + 1][2]
                    if hasattr(next_env, "_ensure"):
                        prefetch_thread = threading.Thread(target=next_env._ensure, daemon=True)
                        prefetch_thread.start()
    else:
        with cf.ThreadPoolExecutor(max_workers=max_parallel_tasks) as executor:
            fut2meta = {}
            for task_group, task_id, env in tasks:
                fut = executor.submit(task_runner, task_group, task_id, env)
                fut2meta[fut] = (task_group, task_id, env)
            for fut in cf.as_completed(fut2meta):
                tg, tid, env = fut2meta[fut]
                try:
                    tg, tid, metrics = fut.result()
                    _accumulate_to(tg, metrics)
                    per_task_infos.append({"task_group": tg, "task_id": tid, "metrics": metrics})
                finally:
                    env.close()

    # compute aggregated metrics helper (robust to lists/scalars)
    def _agg_from_list(xs):
        if not xs:
            return float("nan")
        arr = np.array(xs, dtype=float)
        return float(np.nanmean(arr))

    # compute per-group aggregates
    groups_aggregated = {}
    for group, acc in group_acc.items():
        groups_aggregated[group] = {
            "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
            "avg_max_reward": _agg_from_list(acc["max_rewards"]),
            "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
            "n_episodes": len(acc["sum_rewards"]),
            "video_paths": list(acc["video_paths"]),
        }

    # overall aggregates
    overall_agg = {
        "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
        "avg_max_reward": _agg_from_list(overall["max_rewards"]),
        "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
        "n_episodes": len(overall["sum_rewards"]),
        "eval_s": time.time() - start_t,
        "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
        "video_paths": list(overall["video_paths"]),
    }

    return {
        "per_task": per_task_infos,
        "per_group": groups_aggregated,
        "overall": overall_agg,
    }

def get_info_success(info):
    if "final_info" in info:
        final_info = info["final_info"]
        if not isinstance(final_info, dict):
            raise RuntimeError(
                "Unsupported `final_info` format: expected dict (Gymnasium >= 1.0). "
                "You're likely using an older version of gymnasium (< 1.0). Please upgrade."
            )
        successes = final_info["is_success"].tolist()
    elif "is_success" in info:
        is_success = info["is_success"]
        successes = (
            is_success.tolist() if hasattr(is_success, "tolist") else [bool(is_success)]
        )
    else:
        successes = [False]

    return successes[0]

def get_dir(current_state, goal_state, sorted=False) -> str:
    delta = goal_state - current_state
    
    components = [
        (abs(delta[0]), "forward" if delta[0] > 0 else "backward"),
        (abs(delta[1]), "right" if delta[1] > 0 else "left"),
        (abs(delta[2]), "up" if delta[2] > 0 else "down"),
    ]
    
    primary_mag, _ = max(components, key=lambda x: x[0])
    
    if primary_mag == 0:
        return ""
        
    if sorted:
        components.sort(key=lambda x: x[0], reverse=True)
        
    directions = []
    
    for mag, dir_name in components:
        ratio = mag / primary_mag
        
        if ratio < 0.25:
            continue
        elif ratio < 0.5:
            directions.append(f"slightly {dir_name}")
        else:
            directions.append(dir_name)

    return ', '.join(directions)

import time
import json
import torch
import numpy as np
from collections import defaultdict, deque
from pathlib import Path

import time
import json
import torch
import numpy as np
from collections import defaultdict, deque
from pathlib import Path

def eval_policy_dataset(
    dataset,  # LeRobotDataset
    policy,   # PreTrainedPolicy
    preprocessor,  # PolicyProcessorPipeline
    postprocessor, # PolicyProcessorPipeline
    planner,  # PreTrainedPolicy | None
    planner_preprocessor,  # PolicyProcessorPipeline | None
    planner_postprocessor, # PolicyProcessorPipeline | None
    batch_size: int,
    start_seed: int | None = None,
    output_dir: Path | None = None,
    split: int | None = None,
    sub_key: str | None = None,
) -> dict:
    start_t = time.time()
    policy.eval()
    policy.reset()

    group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {"sum_rewards": [], "max_rewards": [], "successes": []})
    overall: dict[str, list] = {"sum_rewards": [], "max_rewards": [], "successes": []}
    per_task_infos: list[dict] = []
    
    only_first = False
    if policy.name == "pi05":
        out_dir_path = Path(output_dir)
        eval_info_path = out_dir_path / f"eval_info_{split}.json"
        
        # Kept location for logic dependencies, retaining only dir_lang as a metric
        results_per_task = defaultdict(lambda: {"location": [], "dir_lang": []})
        completed_counts = defaultdict(int)
        
        if eval_info_path.exists():
            with open(eval_info_path, "r") as f:
                saved_data = json.load(f)
                for k, v in saved_data.items():
                    task_idx = int(k) # JSON keys are strings, convert back to int
                    
                    # Load the existing file data into our defaultdict base
                    results_per_task[task_idx]["location"].extend(v.get("location", []))
                    results_per_task[task_idx]["dir_lang"].extend(v.get("dir_lang", []))
                    
                    # Rebuild the completed counts from the loaded data
                    completed_counts[task_idx] = sum(1 for loc in v.get("location", []) if float(loc) == 0.0)

        dataset.reader.completed_episodes_per_task = completed_counts
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.libero import get_libero_dummy_action
        env_config = LiberoEnv(task="libero_90")
        envs = make_env(env_config) 
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_config, policy_cfg=policy.config)
        
        for batch in dataloader:
            if not batch["eval_sub_start"][0].item():
                continue
                
            location = batch["eval_location"][0].item()
            if only_first and location != 0.0:
                continue
            task_id = batch["libero_id"][0].item()
            sub_len = batch["eval_sub_len"][0].item()
            task = batch['task'][0]
            subtask = batch['eval_subtask'][0]
            full_subtask = batch['subtask'][0]
            goal_state = batch["eval_goal_state"][0]
            init_state = batch["eval_init_state"][0]
            prev_actions = batch["eval_prev_actions"]
            gt_dir_lang = batch['eval_move'][0]

            # Action trimming
            gt_actions = batch["action"]
            gt_actions = gt_actions[:, :sub_len, :]

            done = False
            success = False

            policy.reset()
            env = envs["libero_90"][task_id]
            observation, info = env.reset(seed=[start_seed])
            env.envs[0]._env.set_init_state(init_state)
            dummy_action = np.array([get_libero_dummy_action()], dtype=np.float32)
            
            for _ in range(10):
                observation, reward, terminated, truncated, info = env.step(dummy_action)

            prev_action = None # Noop

            # Fast-forward through previous actions
            for action in prev_actions:
                action_transition = {ACTION: action}
                action_transition = env_postprocessor(action_transition)
                action = action_transition[ACTION]

                action_numpy: np.ndarray = action.to("cpu").numpy()
                assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

                observation, reward, terminated, truncated, info = env.step(action_numpy)
                prev_action = action_numpy

                done = terminated | truncated
                success = get_info_success(info)
                if done or success:
                    break
            
            if done or success:
                continue  

            action_queue = deque(maxlen=policy.config.n_action_steps)
            policy_states = []

            observation = preprocess_observation(observation)
            observation = env_preprocessor(observation)
            
            start_state = observation['observation.state'][0]
            dir_langs = []

            # Execute Move
            step = 0
            max_steps = 1.25 * sub_len
            min_steps = 0.75 * sub_len

            while step <= sub_len:
                # Score Dir Lang computation
                if sub_key == "subtask_move":
                    cur_state = observation['observation.state'][0]
                    current_subtask_str = full_subtask
                    if step == sub_len:
                        dir_lang = get_dir(start_state, cur_state, sorted=False)
                        dir_langs.append(dir_lang)
                        
                # Fill action queue via policy
                while len(action_queue) == 0:
                    observation["task"] = [task]
                    observation["subtask"] = [current_subtask_str]
                    observation = preprocessor(observation)

                    actions = []
                    for _ in range(policy.config.n_action_steps):
                        with torch.inference_mode():
                            action = policy.select_action(observation)
                        action = postprocessor(action)

                        action_transition = {ACTION: action}
                        action_transition = env_postprocessor(action_transition)
                        action = action_transition[ACTION]

                        action_numpy: np.ndarray = action.to("cpu").numpy()
                        assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"
                        actions.append(action_numpy)
                    
                    if policy.config.dynamic_action_chunking:
                        for i in range(len(actions) - 1, -1, -1):
                            _prev_action = actions[i - 1] if i > 0 else prev_action
                            if not is_noop(actions[i], _prev_action, threshold=0.01):
                                action_queue.extend(actions[:i+1])
                                break
                    else:
                        action_queue.extend(actions)

                action_numpy = action_queue.popleft()
                prev_action = action_numpy

                observation, reward, terminated, truncated, info = env.step(action_numpy)
                observation = preprocess_observation(observation)
                observation = env_preprocessor(observation)
                
                cur_state = observation['observation.state'][0]
                policy_states.append(cur_state)

                success = get_info_success(info)
                if success:
                    break
                
                step += 1   

            # ----------------- SAVE RESULTS ----------------- #
            results_per_task[task_id]["location"].append(round(location, 2))
            dir_score = max([score_directions(gt_dir_lang, dir_lang) for dir_lang in dir_langs], default=None)
            results_per_task[task_id]["dir_lang"].append(round(dir_score, 4) if dir_score is not None else None)

            out_dir_path.mkdir(parents=True, exist_ok=True)
            with open(eval_info_path, "w") as f:
                json.dump(dict(results_per_task), f, indent=4)

        close_envs(envs)

# def eval_policy_dataset(
#     dataset,  # LeRobotDataset
#     policy,   # PreTrainedPolicy
#     preprocessor,  # PolicyProcessorPipeline
#     postprocessor, # PolicyProcessorPipeline
#     planner,  # PreTrainedPolicy | None
#     planner_preprocessor,  # PolicyProcessorPipeline | None
#     planner_postprocessor, # PolicyProcessorPipeline | None
#     batch_size: int,
#     start_seed: int | None = None,
#     output_dir: Path | None = None,
#     split: int | None = None,
#     sub_key: str = "",
# ) -> dict:
#     start_t = time.time()
#     policy.eval()
#     policy.reset()

#     group_acc: dict[str, dict[str, list]] = defaultdict(lambda: {"sum_rewards": [], "max_rewards": [], "successes": []})
#     overall: dict[str, list] = {"sum_rewards": [], "max_rewards": [], "successes": []}
#     per_task_infos: list[dict] = []
    
#     only_first = False
#     if policy.name == "pi05":
#         out_dir_path = Path(output_dir)
#         eval_info_path = out_dir_path / f"eval_info_{split}.json"
        
#         # Added 'vlm_reward' to tracking dictionaries
#         results_per_task = defaultdict(lambda: {"location": [], "nrmse": [], "dir_full": [], "dir_lang": [], "euc_dist": [], "vlm_reward": []})
#         completed_counts = defaultdict(int)
        
#         if eval_info_path.exists():
#             with open(eval_info_path, "r") as f:
#                 saved_data = json.load(f)
#                 for k, v in saved_data.items():
#                     task_idx = int(k) # JSON keys are strings, convert back to int
                    
#                     # Load the existing file data into our defaultdict base
#                     results_per_task[task_idx]["location"].extend(v.get("location", []))
#                     results_per_task[task_idx]["nrmse"].extend(v.get("nrmse", []))
#                     results_per_task[task_idx]["dir_full"].extend(v.get("dir_full", []))
#                     results_per_task[task_idx]["dir_lang"].extend(v.get("dir_lang", []))
#                     results_per_task[task_idx]["euc_dist"].extend(v.get("euc_dist", []))
#                     results_per_task[task_idx]["vlm_reward"].extend(v.get("vlm_reward", []))
                    
#                     # Rebuild the completed counts from the loaded data
#                     completed_counts[task_idx] = sum(1 for loc in v.get("location", []) if float(loc) == 0.0)

#         dataset.reader.completed_episodes_per_task = completed_counts
#         dataloader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=False,
#         )

#         from lerobot.envs.configs import LiberoEnv
#         from lerobot.envs.libero import get_libero_dummy_action
#         env_config = LiberoEnv(task="libero_90")
#         envs = make_env(env_config) 
#         env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_config, policy_cfg=policy.config)
        
#         for batch in dataloader:
#             if not batch["eval_sub_start"][0].item():
#                 continue
                
#             location = batch["eval_location"][0].item()
#             if only_first and location != 0.0:
#                 continue
#             task_id = batch["libero_id"][0].item()
#             sub_len = batch["eval_sub_len"][0].item()
#             subtask = batch['eval_subtask'][0]
#             gt_move = batch['eval_move'][0] if batch.get('eval_move') else None
#             full_subtask = batch['subtask'][0]
#             goal_state = batch["eval_goal_state"][0]
#             init_state = batch["eval_init_state"][0]
#             prev_actions = batch["eval_prev_actions"]
            
#             # Action trimming
#             gt_actions = batch["action"]
#             gt_actions = gt_actions[:, :sub_len, :]

#             # Retrieve dynamic moves if active
#             if sub_key == "move_dynamic":
#                 num_moves = batch["eval_num_moves"][0].item()
#                 m_goal_states = batch["eval_goal_states"][0][:num_moves].to(goal_state.device)
#                 m_sub_lens = batch["eval_sub_lens"][0][:num_moves]
#                 m_moves_gt = batch["eval_moves_str"][0].split("|||")
#             else:
#                 m_goal_states = [goal_state]
#                 m_sub_lens = [sub_len]
#                 m_moves_gt = [gt_move]
#                 num_moves = 1

#             done = False
#             success = False

#             policy.reset()
#             env = envs["libero_90"][task_id]
#             observation, info = env.reset(seed=[start_seed])
#             env.envs[0]._env.set_init_state(init_state)
#             dummy_action = np.array([get_libero_dummy_action()], dtype=np.float32)
            
#             for _ in range(10):
#                 observation, reward, terminated, truncated, info = env.step(dummy_action)

#             prev_action = None # Noop

#             # Fast-forward through previous actions
#             for action in prev_actions:
#                 action_transition = {ACTION: action}
#                 action_transition = env_postprocessor(action_transition)
#                 action = action_transition[ACTION]

#                 action_numpy: np.ndarray = action.to("cpu").numpy()
#                 assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"

#                 observation, reward, terminated, truncated, info = env.step(action_numpy)
#                 prev_action = action_numpy

#                 done = terminated | truncated
#                 success = get_info_success(info)
#                 if done or success:
#                     break
            
#             if done or success:
#                 continue 

#             action_queue = deque(maxlen=policy.config.n_action_steps)
#             policy_states = []
#             trajectory_frames = []

#             observation = preprocess_observation(observation)
#             observation = env_preprocessor(observation)
            
#             # Capture initial frame
#             if "observation.images.image" in observation:
#                 trajectory_frames.append(observation["observation.images.image"].detach().cpu())
            
#             start_state = observation['observation.state'][0]
#             dir_full = None
#             dir_lang = None

#             # ----------------- OUTER LOOP: SEQUENTIAL MOVES ----------------- #
#             for m_idx in range(num_moves):
#                 cur_goal_state = m_goal_states[m_idx]
#                 cur_sub_len = m_sub_lens[m_idx].item() if isinstance(m_sub_lens[m_idx], torch.Tensor) else m_sub_lens[m_idx]
                
#                 # CRITICAL: Clear stale actions from the previous move's inference
#                 action_queue.clear()
                
#                 # 1) Generate Dynamic move phrase
#                 if sub_key == "move_dynamic":
#                     base_subtask = full_subtask.split(" (")[0]
#                     cur_gt_move = m_moves_gt[m_idx] if m_idx < len(m_moves_gt) else ""
                    
#                     if cur_gt_move and "grasp the object" in cur_gt_move:
#                         current_subtask_str = f"{base_subtask} ({cur_gt_move})"
#                     else:
#                         cur_state = observation['observation.state'][0]
#                         delta = cur_goal_state[:3] - cur_state[:3]
                        
#                         components = [
#                             (abs(delta[0]), "forward" if delta[0] > 0 else "backward"),
#                             (abs(delta[1]), "right" if delta[1] > 0 else "left"),
#                             (abs(delta[2]), "up" if delta[2] > 0 else "down"),
#                         ]
                        
#                         primary_mag = max(comp[0] for comp in components)
#                         if primary_mag == 0:
#                             pred_move = ""
#                         else:
#                             directions = []
#                             for mag, dir_str in components:
#                                 if mag == primary_mag:
#                                     directions.append(dir_str)
#                                 else:
#                                     ratio = mag / primary_mag
#                                     if ratio >= 0.5:
#                                         directions.append(dir_str)
#                                     elif ratio >= 0.25:
#                                         directions.append(f"slightly {dir_str}")
                                        
#                             pred_move = ', '.join(directions)
                        
#                         current_subtask_str = f"{base_subtask} ({pred_move})"
#                 else:
#                     current_subtask_str = full_subtask

#                 # 2) Execute Move Inner Loop
#                 step = 0
#                 max_steps = 1.25 * cur_sub_len
#                 min_steps = 0.75 * cur_sub_len
                
#                 # Patience setup
#                 best_dist = float('inf')
#                 patience_counter = 0
#                 max_patience = 3  # Allowed consecutive steps of divergence

#                 while step < max_steps:
#                     # Score original Dir Lang computation for standard move
#                     if sub_key != "move_dynamic" and step == cur_sub_len:
#                         cur_state = observation['observation.state'][0]
#                         full_pred_delta = cur_state - start_state
#                         full_gt_delta = cur_goal_state - start_state
#                         dir_full = np.dot(full_pred_delta, full_gt_delta) / (np.linalg.norm(full_pred_delta) * np.linalg.norm(full_gt_delta))

#                         dir_lang = None
#                         if m_moves_gt[0] is not None and "grasp the object" not in m_moves_gt[0]:
#                             delta = cur_state[:3] - start_state[:3]
#                             components = [
#                                 (abs(delta[0]), "forward" if delta[0] > 0 else "backward"),
#                                 (abs(delta[1]), "right" if delta[1] > 0 else "left"),
#                                 (abs(delta[2]), "up" if delta[2] > 0 else "down"),
#                             ]

#                             primary_mag = max(comp[0] for comp in components)
#                             if primary_mag > 0:
#                                 directions = []
#                                 for mag, dir_str in components:
#                                     if mag == primary_mag:
#                                         directions.append(dir_str)
#                                     else:
#                                         ratio = mag / primary_mag
#                                         if ratio >= 0.5:
#                                             directions.append(dir_str)
#                                         elif ratio >= 0.25:
#                                             directions.append(f"slightly {dir_str}")

#                                 pred_move_eval = ', '.join(directions)
#                                 dir_lang = score_directions(m_moves_gt[0], pred_move_eval)
                                
#                     # Fill action queue via policy
#                     while len(action_queue) == 0:
#                         observation["subtask"] = [current_subtask_str]
#                         observation = preprocessor(observation)

#                         actions = []
#                         for _ in range(policy.config.n_action_steps):
#                             with torch.inference_mode():
#                                 action = policy.select_action(observation)
#                             action = postprocessor(action)

#                             action_transition = {ACTION: action}
#                             action_transition = env_postprocessor(action_transition)
#                             action = action_transition[ACTION]

#                             action_numpy: np.ndarray = action.to("cpu").numpy()
#                             assert action_numpy.ndim == 2, "Action dimensions should be (batch, action_dim)"
#                             actions.append(action_numpy)
                        
#                         if policy.config.dynamic_action_chunking:
#                             for i in range(len(actions) - 1, -1, -1):
#                                 _prev_action = actions[i - 1] if i > 0 else prev_action
#                                 if not is_noop(actions[i], _prev_action, threshold=0.01):
#                                     action_queue.extend(actions[:i+1])
#                                     break
#                         else:
#                             action_queue.extend(actions)

#                     action_numpy = action_queue.popleft()
#                     prev_action = action_numpy

#                     observation, reward, terminated, truncated, info = env.step(action_numpy)
#                     observation = preprocess_observation(observation)
                    
#                     # Capture subsequent frames for the VLM
#                     if "observation.images.image" in observation:
#                         trajectory_frames.append(observation["observation.images.image"].detach().cpu())

#                     observation = env_preprocessor(observation)
#                     cur_state = observation['observation.state'][0]
#                     policy_states.append(cur_state)

#                     success = get_info_success(info)
#                     if success:
#                         break
                        
#                     # Stop if not converging, utilizing the patience window
#                     if sub_key == "move_dynamic" and step > min_steps:
#                         curr_dist = torch.norm(cur_state - cur_goal_state).item()
                        
#                         # Use a small 1e-4 tolerance to account for physics jitter
#                         if curr_dist > best_dist + 1e-4:
#                             patience_counter += 1
#                             if patience_counter >= max_patience:
#                                 break  # Loop aborts if distance increases 3 steps in a row
#                         else:
#                             patience_counter = 0  # Reset counter if distance stays steady or improves
#                             best_dist = min(best_dist, curr_dist)
                        
#                     step += 1  
                
#                 if success:
#                     break

#             # ----------------- COMPUTE NRMSE & EUC_DIST ----------------- #
#             states_tensor = torch.stack([
#                 s.detach() if isinstance(s, torch.Tensor) else torch.tensor(s) 
#                 for s in policy_states
#             ]).to(goal_state.device)

#             if success:
#                 nrmse = 0.0
#                 euc_dist = 0.0
#             else:
#                 mse = torch.mean((states_tensor - goal_state)**2, dim=1)
#                 rmse = torch.sqrt(mse)
#                 normalization = torch.clamp(goal_state.max() - goal_state.min(), min=1e-8)
                
#                 nrmse_vals = rmse / normalization
#                 nrmse = torch.min(nrmse_vals).item()

#                 euc_dists = torch.norm(states_tensor[:, :3] - goal_state[:3], dim=1)
#                 euc_dist = torch.min(euc_dists).item()

#             # ----------------- COMPUTE VLM REWARD (TOPReward) ----------------- #
#             vlm_reward = None
#             if success:
#                 vlm_reward = 0.0
#             elif planner is not None and len(trajectory_frames) > 0:
#                 # Concatenate frames: (T, 1, C, H, W) -> (T, C, H, W)
#                 frames_tensor = torch.cat(trajectory_frames, dim=0)
                
#                 max_frames = 15
#                 if len(frames_tensor) > max_frames:
#                     indices = torch.linspace(0, len(frames_tensor) - 1, max_frames).long()
#                     frames_tensor = frames_tensor[indices]
                
#                 # Clean the subtask by removing anything from the first opening parenthesis onward
#                 clean_subtask = full_subtask.split(" (")[0].strip()
                
#                 planner_transition = {
#                     "observation.images.image": frames_tensor.to("cuda"),
#                     "task": clean_subtask, # Passed as a string so the preprocessor batches it cleanly
#                     "fps": dataset.meta.fps,
#                 }
                
#                 planner_batch = planner_preprocessor(planner_transition)
#                 with torch.inference_mode():
#                     reward_raw = planner.select_action(planner_batch)
                
#                 planner_output = planner_postprocessor(reward_raw)
#                 print(planner_output)
#                 vlm_reward = planner_output[0]

#             # ----------------- SAVE RESULTS ----------------- #
#             results_per_task[task_id]["location"].append(round(location, 2))
#             results_per_task[task_id]["nrmse"].append(round(nrmse, 4))
#             results_per_task[task_id]["euc_dist"].append(round(euc_dist, 4))
#             results_per_task[task_id]["vlm_reward"].append(round(vlm_reward, 4) if vlm_reward is not None else None)
            
#             if sub_key != "move_dynamic":
#                 results_per_task[task_id]["dir_full"].append(round(float(dir_full), 4) if dir_full is not None else None)
#                 results_per_task[task_id]["dir_lang"].append(round(dir_lang, 4) if dir_lang is not None else None)
#             else:
#                 results_per_task[task_id]["dir_full"].append(None)
#                 results_per_task[task_id]["dir_lang"].append(None)

#             out_dir_path.mkdir(parents=True, exist_ok=True)
#             with open(eval_info_path, "w") as f:
#                 json.dump(dict(results_per_task), f, indent=4)

#         close_envs(envs)

#     elif planner is not None:
#         planner.eval()

#         for _ in range(num_samples):
#             batch = next(dl_iter)
            
#             batch_planner = planner_preprocessor(batch)
#             pred_subtask = planner.select_action(batch_planner)

#             print(f"Task: {batch['task'][0]}\nGT Subtask: {batch['subtask'][0]}\nPred Subtask: {pred_subtask[0]}\n\n")

#             batch["subtask"] = pred_subtask
#             gt_actions = batch["action"]
#             batch = preprocessor(batch)
            
#             print(f"Number action steps: {policy.config.n_action_steps}")
#             pred_actions = []
#             for _ in range(policy.config.n_action_steps):
#                 pred_action = policy.select_action(batch)
#                 pred_action = postprocessor(pred_action)
#                 pred_actions.append(pred_action)

#             print(f"Task: {batch['task'][0]}\nGT Actions: {gt_actions}\nPred Actions: {pred_actions}\n")

#     elif policy.name == "pi0_fast":
#         sample_ix = 0
        
#         while sample_ix < num_samples:
#             batch = next(dl_iter)
            
#             original_tasks = batch["task"].copy()

#             batch = preprocessor(batch)
            
#             gt_subtasks = batch["subtask"] if "subtask" in batch else [""] * len(batch["task"])
#             pred_subtasks = policy.select_action(batch)

#             for b in range(batch_size):
#                 if sample_ix >= num_samples:
#                     break
#                 task_group = original_tasks[b]
#                 gt_subtask = gt_subtasks[b]
#                 pred_subtask = pred_subtasks[b]
                
#                 # Calculate success for this specific sample
#                 is_success = bool(pred_subtask == gt_subtask)
#                 sum_reward = 1.0 if is_success else 0.0
#                 max_reward = 1.0 if is_success else 0.0
                
#                 # Accumulate per group and overall
#                 group_acc[task_group]["successes"].append(is_success)
#                 group_acc[task_group]["sum_rewards"].append(sum_reward)
#                 group_acc[task_group]["max_rewards"].append(max_reward)
                
#                 overall["successes"].append(is_success)
#                 overall["sum_rewards"].append(sum_reward)
#                 overall["max_rewards"].append(max_reward)
                
#                 # Accumulate per episode/sample
#                 per_task_infos.append({
#                     "sample_ix": sample_ix,
#                     "task_group": task_group,
#                     "gt_subtask": gt_subtask,
#                     "pred_subtask": pred_subtask,
#                     "success": is_success,
#                     "sum_reward": sum_reward,
#                     "max_reward": max_reward,
#                 })

#                 logging.info(f"Sample {sample_ix+1}/{num_samples}")
#                 logging.info(f"Original Task: {task_group}")
#                 logging.info(f"GT Subtask:   {gt_subtask}")
#                 logging.info(f"Pred Subtask: {pred_subtask}")
#                 logging.info(f"Match: {is_success}")
#                 logging.info("-" * 40)

#                 sample_ix += 1
            
#         def _agg_from_list(xs):
#             if not xs: return float("nan")
#             return float(np.nanmean(np.array(xs, dtype=float)))

#         # Compute per-group aggregates
#         groups_aggregated = {}
#         for group, acc in group_acc.items():
#             groups_aggregated[group] = {
#                 "avg_sum_reward": _agg_from_list(acc["sum_rewards"]),
#                 "avg_max_reward": _agg_from_list(acc["max_rewards"]),
#                 "pc_success": _agg_from_list(acc["successes"]) * 100 if acc["successes"] else float("nan"),
#                 "n_episodes": len(acc["sum_rewards"]),
#             }

#         # Overall aggregates
#         overall_agg = {
#             "avg_sum_reward": _agg_from_list(overall["sum_rewards"]),
#             "avg_max_reward": _agg_from_list(overall["max_rewards"]),
#             "pc_success": _agg_from_list(overall["successes"]) * 100 if overall["successes"] else float("nan"),
#             "n_episodes": len(overall["sum_rewards"]),
#             "eval_s": time.time() - start_t,
#             "eval_ep_s": (time.time() - start_t) / max(1, len(overall["sum_rewards"])),
#         }
        
#         logging.info(f"Subtask Prediction Accuracy: {overall_agg['pc_success']:.2f}%")
        
#         return {
#             "per_task": per_task_infos,
#             "per_group": groups_aggregated,
#             "overall": overall_agg,
#         }

    # elif policy.name == "pi0_fast":
    #     is_generalize = False
    #     task_to_target_object = {
    #         "pick up the sweet object and put it in the tray": {"OR": ["pudding"]},
    #         "pick up the tallest object and put it in the tray": {"OR": ["ketchup", "bottle"], "NOT": ["alphabet"]},
    #         "pick up the mug next to the book and place it to the right compartment of the caddy": {"AND": ["white", "mug"]}
    #     }
    #     results = {task: [] for task in task_to_target_object.keys()}
    #     for _ in range(num_samples):
    #         batch = next(dl_iter)
            
    #         targets = [task_to_target_object[t] for t in batch["task"]]
    #         original_task = batch["task"].copy()

    #         batch = preprocessor(batch)
            
    #         gt_subtask = batch["subtask"][0]
            
    #         # observation = preprocessor(observation)
    #         pred_subtask = policy.select_action(batch)[0]

    #         score = 0
    #         for target in targets:
    #             if "OR" in target:
    #                 if any(t in pred_subtask for t in target["OR"]):
    #                     score = 1   
    #             if "AND" in target:
    #                 if all(t in pred_subtask for t in target["AND"]):
    #                     score = 1
    #             if "NOT" in target:
    #                 if any(t in pred_subtask for t in target["NOT"]):
    #                     score = 0

    #         results[original_task[0]].append(score)
    #         print(f"Task: {original_task[0]}\nGT Subtask: {gt_subtask}\nPred Subtask: {pred_subtask}\n\n")
            
    #         policy.reset()

    #     for task, scores in results.items():
    #         success_rate = sum(scores) / len(scores) * 100
    #         print(f"Task: {task}\nSuccess Rate: {success_rate:.2f}%\n")
    # elif policy.name == "pi0":
    #     task_to_orig = {
    #         "pick up the sweet object and put it in the tray": "pick up the chocalate pudding and put it in the tray",
    #         "pick up the tallest object and put it in the tray": "pick up the ketchup and put it in the tray",
    #         "pick up the mug next to the book and place it to the right compartment of the caddy": "pick up the white mug and place it to the right compartment of the caddy"
    #     }
    #     for _ in range(num_samples):
    #         batch = next(dl_iter)
    #         gt_actions = batch["action"]
    #         # batch["task"] = [task_to_orig[t] for t in batch["task"]]
    #         batch = preprocessor(batch)
    #         # batch["task"] = batch["subtask"]
            
    #         print(f"Number action steps: {policy.config.n_action_steps}")
    #         pred_actions = []
    #         for _ in range(policy.config.n_action_steps):
    #             pred_action = policy.select_action(batch)
    #             pred_action = postprocessor(pred_action)
    #             pred_actions.append(pred_action)

    #         print(f"Task: {batch['task'][0]}\nGT Actions: {gt_actions}\nPred Actions: {pred_actions}\n")

import math

def parse_direction(d_str):
    """Converts a direction string into a 3D vector [X, Y, Z]."""
    vec = [0.0, 0.0, 0.0] # [Right/Left, Forward/Backward, Up/Down]
    
    for part in d_str.split(','):
        part = part.strip().lower()
        magnitude = 0.5 if 'slightly' in part else 1.0
        
        # X-axis
        if 'right' in part: vec[0] += magnitude
        elif 'left' in part: vec[0] -= magnitude
        # Y-axis
        if 'forward' in part: vec[1] += magnitude
        elif 'backward' in part or 'back' in part: vec[1] -= magnitude
        # Z-axis
        if 'up' in part: vec[2] += magnitude
        elif 'down' in part: vec[2] -= magnitude
            
    return vec

def score_directions(truth_str, policy_str):
    """Scores policy against truth using Cosine Similarity (0.0 to 1.0)."""
    v_truth = parse_direction(truth_str)
    v_policy = parse_direction(policy_str)
    
    dot_product = sum(t * p for t, p in zip(v_truth, v_policy))
    mag_truth = math.sqrt(sum(t * t for t in v_truth))
    mag_policy = math.sqrt(sum(p * p for p in v_policy))
    
    if mag_truth == 0 or mag_policy == 0:
        return 0.0 # Handle edge case of empty/zero vectors
        
    return dot_product / (mag_truth * mag_policy)

def main():
    init_logging()
    register_third_party_plugins()
    eval_main()


if __name__ == "__main__":
    main()
