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
"""Private reader component for LeRobotDataset. Handles random-access reading (HF dataset, delta indices, video decoding)."""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import numpy as np
import torch
import datasets

from .dataset_metadata import LeRobotDatasetMetadata
from .feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from .io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from .video_utils import decode_video_frames


def get_dir(current_state, goal_state, sorted=True) -> str:
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
    

def get_trace(full_trace) -> str:
    """
    Selects exactly 8 points from a full trace. 
    If > 8 points: Uses a greedy RDP to isolate structural keypoints.
    If < 8 points: Resamples the path via arc-length interpolation.
    """
    if isinstance(full_trace, torch.Tensor):
        trace = full_trace.detach().cpu().numpy()
    else:
        trace = np.array(full_trace)
        
    N = len(trace)
    
    if N < 8:
        # Edge cases: 0 or 1 points
        if N == 0:
            selected_points = np.zeros((8, 2))
        elif N == 1:
            selected_points = np.tile(trace[0], (8, 1))
        else:
            # 1. Calculate distances between consecutive points
            diffs = np.diff(trace, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            
            # 2. Cumulative distance along the path
            cum_dists = np.insert(np.cumsum(dists), 0, 0)
            total_dist = cum_dists[-1]
            
            # 3. Handle edge case where all points are exactly overlapping
            if total_dist == 0:
                selected_points = np.tile(trace[0], (8, 1))
            else:
                # 4. Create 8 target distances evenly spaced along the path
                target_dists = np.linspace(0, total_dist, 8)
                
                # 5. Interpolate X and Y independently based on distance traveled
                interp_x = np.interp(target_dists, cum_dists, trace[:, 0])
                interp_y = np.interp(target_dists, cum_dists, trace[:, 1])
                selected_points = np.column_stack((interp_x, interp_y))
                
    elif N == 8:
        # Exact match
        selected_points = trace
        
    else:
        # Greedy RDP downsampling for N > 8
        indices = [0, N - 1]
        
        while len(indices) < 8:
            indices.sort()
            max_dist = -1.0
            best_idx = -1
            
            for i in range(len(indices) - 1):
                start_idx = indices[i]
                end_idx = indices[i + 1]
                
                if end_idx - start_idx <= 1:
                    continue
                    
                p1 = trace[start_idx]
                p2 = trace[end_idx]
                pts = trace[start_idx + 1 : end_idx]
                
                if np.allclose(p1, p2):
                    dists = np.linalg.norm(pts - p1, axis=1)
                else:
                    line_vec = p2 - p1
                    pt_vecs = p1 - pts
                    cross_prods = line_vec[0] * pt_vecs[:, 1] - line_vec[1] * pt_vecs[:, 0]
                    dists = np.abs(cross_prods) / np.linalg.norm(line_vec)
                    
                seg_max_idx = np.argmax(dists)
                seg_max_dist = dists[seg_max_idx]
                
                if seg_max_dist > max_dist:
                    max_dist = seg_max_dist
                    best_idx = start_idx + 1 + seg_max_idx
                    
            if best_idx != -1:
                indices.append(best_idx)
            else:
                for j in range(N):
                    if j not in indices:
                        indices.append(j)
                        break
                        
        indices.sort()
        selected_points = trace[indices]
    
    # Format to final string
    formatted_pts = [f"[{int(round(p[0]))}, {int(round(p[1]))}]" for p in selected_points]
    
    return "[" + ", ".join(formatted_pts) + "]"

def reorder_moves(moves_string):
    moves = [move.strip() for move in moves_string.split(',')]
    
    def get_axis_priority(move):
        move_lower = move.lower()
        if 'forward' in move_lower or 'backward' in move_lower:
            return 0
        elif 'right' in move_lower or 'left' in move_lower:
            return 1
        elif 'up' in move_lower or 'down' in move_lower:
            return 2
        return 3 
        
    sorted_moves = sorted(moves, key=get_axis_priority)
    return ', '.join(sorted_moves)

class DatasetReader:
    """Encapsulates read-side state and methods for LeRobotDataset.

    Owns: hf_dataset, _absolute_to_relative_idx, delta_indices.
    """

    def __init__(
        self,
        meta, # Assuming LeRobotDatasetMetadata
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        dynamic_action_chunking: str = "",
        return_uint8: bool = False,
        chain_close: int = 10,
        chain_dir: bool = False,
        is_planner: bool = False,
        ilfm: bool = False,
        chunk_size = 50,
        train_then: bool = False,
        sorted_dir: bool = False,
        is_absolute: bool = False,
    ):
        """Initialize the reader with metadata, filtering, and transform config."""
        self._meta = meta
        self.root = root
        self.episodes = episodes
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        self._image_transforms = image_transforms
        self.dynamic_action_chunking = dynamic_action_chunking
        self._return_uint8 = return_uint8
        self.chain_close = chain_close
        self.sorted_dir = sorted_dir
        self.chain_dir = chain_dir
        self.is_planner = is_planner
        self.ilfm = ilfm
        self.chunk_size = chunk_size
        self.train_then = train_then
        self.is_absolute = is_absolute

        print(f"Initialized DatasetReader with chain_close={chain_close}, chain_dir={chain_dir}, ILFM={ilfm}, is_absolute={is_absolute} and chunk_size={chunk_size}")
        self.debug = True
        self.hf_dataset = None
        self._absolute_to_relative_idx: dict[int, int] | None = None

        # Setup delta_indices (doesn't depend on hf_dataset)
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s) 
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

    def try_load(self) -> bool:
        """Attempt to load from local cache. Returns True if data is sufficient."""
        try:
            self.hf_dataset = self._load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            self.hf_dataset = None
            return False
        if not self._check_cached_episodes_sufficient():
            self.hf_dataset = None
            return False
        self._build_index_mapping()
        return True

    def load_and_activate(self) -> None:
        """Load HF dataset from disk and build index mapping. Call after data is on disk."""
        self.hf_dataset = self._load_hf_dataset()
        self._build_index_mapping()

    def _build_index_mapping(self) -> None:
        """Build absolute-to-relative index mapping from loaded hf_dataset."""
        self._absolute_to_relative_idx = None
        if self.episodes is not None and self.hf_dataset is not None:
            indices = self.hf_dataset.data.column("index").to_numpy()
            self._absolute_to_relative_idx = dict(zip(indices.tolist(), range(len(indices)), strict=True))

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    def _load_hf_dataset(self):
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self._meta.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        if self.episodes is None:
            requested_episodes = set(range(self._meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        if not requested_episodes.issubset(available_episodes):
            return False

        if len(self._meta.video_keys) > 0:
            for ep_idx in requested_episodes:
                for vid_key in self._meta.video_keys:
                    video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def get_episodes_file_paths(self) -> list[Path]:
        """Return deduplicated file paths (data + video) for selected episodes."""
        episodes = self.episodes if self.episodes is not None else list(range(self._meta.total_episodes))
        fpaths = [str(self._meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self._meta.video_keys) > 0:
            video_files = [
                str(self._meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self._meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        fpaths = list(set(fpaths))
        return fpaths

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor], int]:
        """Compute query indices for delta timestamps."""
        ep = self._meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        ### Dynamic action chunking ###
        rel_idx = self._absolute_to_relative_idx[abs_idx] if self._absolute_to_relative_idx is not None else abs_idx
        rel_ep_end = self._absolute_to_relative_idx[ep_end - 1] + 1 if self._absolute_to_relative_idx is not None else ep_end
        
        sub_len = ep_end - abs_idx
        
        # Short-circuit if we are at or past the episode boundary
        if sub_len <= 0:
            sub_len = 0
            ep_end = abs_idx
        elif self.dynamic_action_chunking:
            if self.dynamic_action_chunking in ["move", "subtask_move"]:
                dac_key = "move_index"
            elif self.dynamic_action_chunking == "trace":
                dac_key = "segments_rdp"
            else:
                dac_key = "subtask_index"
            
            # Zero-copy PyArrow columnar slice
            chunk_len = rel_ep_end - rel_idx
            ep_sub_indices = np.array(self.hf_dataset.data.column(dac_key).slice(rel_idx, chunk_len))

            if len(ep_sub_indices) > 0:
                sub_idx = int(ep_sub_indices[0])
                mismatches = ep_sub_indices != sub_idx
                sub_len = np.argmax(mismatches) if np.any(mismatches) else len(ep_sub_indices)
            else:
                sub_len = 0

            sub_end = abs_idx + sub_len
            ep_end = sub_end

        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding, sub_len

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self._meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Query dataset for indices across keys, skipping video keys."""
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self._meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            
            ds_col = self.hf_dataset.select_columns([key])
            taken_data = [ds_col[i][key] for i in relative_indices]
            
            if isinstance(taken_data[0], torch.Tensor):
                result[key] = torch.stack(taken_data)
            else:
                result[key] = torch.tensor(taken_data)
                
        return result

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        ep = self._meta.episodes[ep_idx]

        def _decode_single(vid_key: str, query_ts: list[float]) -> tuple[str, torch.Tensor]:
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]
            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(
                video_path,
                shifted_query_ts,
                self._tolerance_s,
                self._video_backend,
                return_uint8=self._return_uint8,
            )
            return vid_key, frames.squeeze(0)

        items = list(query_timestamps.items())

        if len(items) <= 1:
            return {vid_key: _decode_single(vid_key, query_ts)[1] for vid_key, query_ts in items}

        with ThreadPoolExecutor(max_workers=len(items)) as pool:
            futures = [pool.submit(_decode_single, k, ts) for k, ts in items]
            return dict(f.result() for f in futures)

    def get_item(self, idx) -> dict:
        """Core __getitem__ logic. Assumes hf_dataset is loaded."""
        total_start = time.perf_counter()
        prof = {}

        t0 = time.perf_counter()
        item = self.hf_dataset[idx]

        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()
        ep_end_idx = self._meta.episodes[ep_idx]["dataset_to_index"]

        prof["1_hf_dataset_initial_fetch"] = time.perf_counter() - t0

        query_indices = None
        sub_end = abs_idx + 1  # Fallback initialization to prevent UnboundLocalError
        
        combined_low_level = False
        first_sub_len = 1
        first_sub_end = sub_end
        
        if self.delta_indices is not None:
            t0 = time.perf_counter()
            query_indices, padding, sub_len = self._get_query_indices(abs_idx, ep_idx)
            if getattr(self, 'debug', False):
                print(f"DEBUG. Subtask length = {sub_len}")
            prof["2_first_get_query_indices"] = time.perf_counter() - t0

            sub_end = abs_idx + sub_len
            
            # Save original boundaries for calculating the first subtask text/direction correctly
            first_sub_len = sub_len
            first_sub_end = sub_end
            ep_true_end = self._meta.episodes[ep_idx]["dataset_to_index"]
            _, _, next_sub_len = self._get_query_indices(first_sub_end, ep_idx)  

            # --- START LOW-LEVEL POLICY COMBINATION LOGIC ---
            t0 = time.perf_counter()
            if self.train_then and not self.is_planner and sub_end < ep_true_end:
                if sub_len <= self.chain_close - 5:
                    combined_low_level = True
                elif sub_len <= self.chain_close + 5: # or first_sub_len + next_sub_len <= self.chunk_size:
                    combined_low_level = torch.rand(1).item() < 0.5
                else:
                    combined_low_level = False
            # --- END LOW-LEVEL POLICY COMBINATION LOGIC ---
            
            if combined_low_level:
                # Find length of the full next subgoal
                _, _, next_sub_len = self._get_query_indices(sub_end, ep_idx)
                sub_len = sub_len + next_sub_len
                sub_end = abs_idx + sub_len
                
                ep_start = self._meta.episodes[ep_idx]["dataset_from_index"]
                
                # Recompute padding and queries for the extended bounds
                query_indices = {
                    key: [max(ep_start, min(sub_end - 1, abs_idx + delta)) for delta in delta_idx]
                    for key, delta_idx in self.delta_indices.items()
                }
                padding = {
                    f"{key}_is_pad": torch.BoolTensor(
                        [(abs_idx + delta < ep_start) | (abs_idx + delta >= sub_end) for delta in delta_idx]
                    )
                    for key, delta_idx in self.delta_indices.items()
                }
            prof["3_low_level_combo_logic"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            if self.ilfm and not self.is_planner:
                if "info" not in item:
                    item["info"] = {}
                item["info"]["action_horizon"] = torch.tensor([sub_len], dtype=torch.float32)
                ep_start = self._meta.episodes[ep_idx]["dataset_from_index"]
                ep_end_idx = self._meta.episodes[ep_idx]["dataset_to_index"]
                
                # Clamp indices to valid episode bounds
                raw_action_indices = [max(ep_start, min(ep_end_idx - 1, abs_idx + i)) for i in range(sub_len)]
                
                if self._absolute_to_relative_idx is not None:
                    rel_indices = [self._absolute_to_relative_idx[i] for i in raw_action_indices]
                else:
                    rel_indices = raw_action_indices
                    
                # Direct PyArrow take for fast action extraction
                pa_actions = self.hf_dataset.data.column("action").take(rel_indices).to_pylist()
                raw_actions = torch.tensor(pa_actions, dtype=torch.float32)
                
                # Shape manipulation for interpolation: [Horizon, Dim] -> [1, Dim, Horizon]
                raw_actions_cf = raw_actions.T.unsqueeze(0) 
                
                interpolated_actions = torch.nn.functional.interpolate(
                    raw_actions_cf, size=self.chunk_size, mode='linear', align_corners=True
                )
                
                # Revert shape: [1, Dim, fixed_horizon] -> [fixed_horizon, Dim]
                item["action"] = interpolated_actions.squeeze(0).T
            prof["4_ilfm_action_extraction"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                if not (self.ilfm and key == "action"):
                    item[key] = val

            if self.dynamic_action_chunking and "action" in item and "action_is_pad" in item:
                if self.ilfm:
                    item["action_is_pad"] = torch.zeros(self.chunk_size, dtype=torch.bool)

                is_pad = item["action_is_pad"]
                if is_pad.any() and not self.is_absolute:
                    # Padding defaults to 0.0 for arm (stay still)
                    noops = torch.zeros_like(item["action"])
                    
                    # Carry over absolute gripper state for padded frames
                    gripper_pad_state = item["action"][..., -1]
                    noops[..., -1] = gripper_pad_state

                    item["action"] = torch.where(is_pad.unsqueeze(-1), noops, item["action"])
            prof["5_query_hf_dataset_and_padding"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if len(self._meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self._image_transforms is not None:
            image_keys = self._meta.camera_keys
            for cam in image_keys:
                item[cam] = self._image_transforms(item[cam])
        prof["6_video_and_image_transforms"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name
        item["task"] = item["task"].split("primitive:")[-1].lstrip()

        # --- START PRIMARY MODE FORMATTING (DECOUPLED FROM SUBTASKS) ---
        rel_idx = self._absolute_to_relative_idx[abs_idx] if self._absolute_to_relative_idx is not None else abs_idx
        rel_first_sub_end = self._absolute_to_relative_idx[first_sub_end] if self._absolute_to_relative_idx is not None else first_sub_end

        rel_first_sub_start = rel_idx
        # --- DYNAMICALLY CALCULATE rel_first_sub_start ---
        ep_start = self._meta.episodes[ep_idx]["dataset_from_index"]
        rel_ep_start = self._absolute_to_relative_idx[ep_start] if self._absolute_to_relative_idx is not None else ep_start
        
        if self.dynamic_action_chunking:
            if self.dynamic_action_chunking in ["move", "subtask_move"]:
                dac_key = "move_index"
            elif self.dynamic_action_chunking == "trace":
                dac_key = "segments_rdp"
            else:
                dac_key = "subtask_index"

            past_len = rel_idx - rel_ep_start
            if past_len > 0:
                past_indices = np.array(self.hf_dataset.data.column(dac_key).slice(rel_ep_start, past_len))
                current_sub_idx = self.hf_dataset.data.column(dac_key)[rel_idx].as_py()
                    
                mismatches_past = past_indices != current_sub_idx
                if np.any(mismatches_past):
                    rel_first_sub_start = rel_ep_start + np.where(mismatches_past)[0][-1] + 1
                else:
                    rel_first_sub_start = rel_ep_start
        else:
            rel_first_sub_start = rel_ep_start

        if self.dynamic_action_chunking == "trace":
            item["subtask"] = get_trace(full_trace=self.hf_dataset[rel_first_sub_start:rel_first_sub_end]['2d_gripper'])
            
        elif self.dynamic_action_chunking == "move":
            has_moves = "move_index" in self._meta.features and self._meta.moves is not None
            dir_str = ""
            if has_moves:
                move_idx = item["move_index"].item()
                item["subtask"] = self._meta.moves.iloc[move_idx].name
            else:
                start_state = torch.tensor(self.hf_dataset.data.column('observation.state')[rel_first_sub_start].as_py(), dtype=item['observation.state'].dtype)
                goal_state = torch.tensor(self.hf_dataset.data.column('observation.state')[rel_first_sub_end-1].as_py(), dtype=item['observation.state'].dtype)
                dir_str = get_dir(start_state, goal_state=goal_state, sorted=self.sorted_dir)
                item["subtask"] = dir_str
                
        elif self.dynamic_action_chunking == "subtask" or self.dynamic_action_chunking == "subtask_move":
            assert "subtask_index" in self._meta.features and self._meta.subtasks is not None, "Mode = 'subtask' or 'subtask_move' but dataset contains no subtasks"
            subtask_idx = item["subtask_index"].item()
            subtask_str = self._meta.subtasks.iloc[subtask_idx].name
            
            if self.dynamic_action_chunking == "subtask":
                item["subtask"] = subtask_str
            else: # "subtask_move"
                has_moves = "move_index" in self._meta.features and self._meta.moves is not None
                dir_str = ""
                if has_moves:
                    move_idx = item["move_index"].item()
                    dir_str = self._meta.moves.iloc[move_idx].name
                else:
                    start_state = torch.tensor(self.hf_dataset.data.column('observation.state')[rel_first_sub_start].as_py(), dtype=item['observation.state'].dtype)
                    goal_state = torch.tensor(self.hf_dataset.data.column('observation.state')[rel_first_sub_end-1].as_py(), dtype=item['observation.state'].dtype)
                    dir_str = get_dir(start_state, goal_state=goal_state, sorted=self.sorted_dir)
                item["subtask"] = f"{subtask_str} ({dir_str})"
        # --- END PRIMARY MODE FORMATTING ---
        prof["7_primary_formatting"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ep_end = self._meta.episodes[ep_idx]["dataset_to_index"]

        # Merge the low-level combined logic with the existing planner chain logic
        if (self.is_planner and first_sub_len < self.chain_close and first_sub_end < ep_end) or combined_low_level:

            rel_next_idx = self._absolute_to_relative_idx[first_sub_end] if self._absolute_to_relative_idx is not None else first_sub_end
            next_item = self.hf_dataset[rel_next_idx]
            
            _, _, next_sub_len = self._get_query_indices(first_sub_end, ep_idx)   
            next_sub_end = first_sub_end + next_sub_len

            rel_next_sub_end = self._absolute_to_relative_idx[next_sub_end] if self._absolute_to_relative_idx is not None else next_sub_end
            
            # --- START PLANNER MODE FORMATTING (DECOUPLED FROM SUBTASKS) ---
            if self.dynamic_action_chunking == "trace":
                item["subtask"] = get_trace(full_trace=self.hf_dataset[rel_first_sub_start:rel_next_sub_end]['2d_gripper'])
            elif self.dynamic_action_chunking == "move":
                has_moves = "move_index" in self._meta.features and self._meta.moves is not None
                next_dir_str = ""
                if has_moves:
                    next_move_idx = next_item["move_index"].item()
                    next_dir_str = self._meta.moves.iloc[next_move_idx].name
                else:
                    goal_state = torch.tensor(self.hf_dataset.data.column('observation.state')[rel_next_sub_end-1].as_py(), dtype=item['observation.state'].dtype)
                    next_dir_str = get_dir(next_item['observation.state'], goal_state=goal_state, sorted=self.sorted_dir)
                item["subtask"] = f"{item['subtask']} THEN {next_dir_str}"
                
            elif self.dynamic_action_chunking in ["subtask", "subtask_move"]:
                assert "subtask_index" in self._meta.features and self._meta.subtasks is not None, "Mode = 'subtask_move' but dataset contains no subtasks"
                next_subtask_idx = next_item["subtask_index"].item()
                next_subtask_base = self._meta.subtasks.iloc[next_subtask_idx].name
                
                if self.dynamic_action_chunking == "subtask":
                    if next_subtask_base != subtask_str:
                        item["subtask"] = f"{item['subtask']} THEN {next_subtask_base}"
                else: # "subtask_move"
                    has_moves = "move_index" in self._meta.features and self._meta.moves is not None
                    next_dir_str = ""
                    if has_moves:
                        next_move_idx = next_item["move_index"].item()
                        next_dir_str = self._meta.moves.iloc[next_move_idx].name
                    else:
                        goal_state = torch.tensor(self.hf_dataset.data.column('observation.state')[rel_next_sub_end-1].as_py(), dtype=item['observation.state'].dtype)
                        next_dir_str = get_dir(next_item['observation.state'], goal_state=goal_state, sorted=self.sorted_dir)
                    
                    if next_subtask_base != subtask_str or self.chain_dir:
                        item["subtask"] = f"{item['subtask']} THEN {next_subtask_base} ({next_dir_str})"
            # --- END PLANNER MODE FORMATTING ---
        prof["8_planner_formatting"] = time.perf_counter() - t0

        prof["9_TOTAL_TIME"] = time.perf_counter() - total_start

        if getattr(self, 'debug', False):
            print("\n" + "="*40)
            print("⏱️  GET_ITEM BOTTLENECK PROFILER")
            print("="*40)
            for k, v in prof.items():
                print(f"{k.ljust(35)}: {v*1000:7.2f} ms")
            print("="*40 + "\n")
            print(f"DEBUG. Subtask = {item['subtask']}" if 'subtask' in item else "DEBUG. No subtask key found.")
            self.debug = False

        return item

from typing import Callable
from pathlib import Path
import torch
import numpy as np
import datasets

class BottomUpDatasetReader(DatasetReader):
    def __init__(
        self,
        meta, # Assuming LeRobotDatasetMetadata
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        dynamic_action_chunking: str = "",
        return_uint8: bool = False,
        chunk_size: int = 10,
        chain_close: int = 10,
        chain_dir: bool = False,
        is_planner: bool = False,
        sorted_dir: bool = False,
    ):
        super().__init__(
            meta, # Assuming LeRobotDatasetMetadata
            root,
            episodes,
            tolerance_s,
            video_backend,
            delta_timestamps,
            image_transforms,
            dynamic_action_chunking="subtask" if dynamic_action_chunking == "subtask_move" else "", # dynamic_action_chunking="subtask",
            return_uint8=return_uint8,
            chain_close=chain_close,
            chain_dir=chain_dir,
            is_planner=is_planner,
            sorted_dir=sorted_dir,
        )

        print(f"Initialized BottomUpDatasetReader with chunk_size={chunk_size}, chain_close={chain_close} and chain_dir={chain_dir}")
        self.chunk_size = chunk_size
        self.chain_close = chain_close
        self.sorted_dir = sorted_dir
        self.mode = dynamic_action_chunking # "move" | "trace"

    def get_item(self, idx) -> dict:
        """Core __getitem__ logic. Assumes hf_dataset is loaded."""
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        sub_end = abs_idx + 1  # Fallback initialization to prevent UnboundLocalError
        
        if self.delta_indices is not None:
            query_indices, padding, sub_len = self._get_query_indices(abs_idx, ep_idx)
            sub_end = abs_idx + min(sub_len, self.chunk_size)
            if self.debug:
                print(f"DEBUG. Subtask length = {sub_len}")
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

            if self.dynamic_action_chunking and "action" in item and "action_is_pad" in item:
                is_pad = item["action_is_pad"]
                if is_pad.any():
                    noops = torch.zeros_like(item["action"])
                    noops[..., -1] = item["action"][..., -1]

                    # Prefetch release (gripper_action = -1) from next chunk
                    # ISSUE: Policy alone decides whether subtask is complete
                    # next_chunk_idx = max(query_indices["action"]) + 1
                    # if next_chunk_idx < self._meta.episodes[ep_idx]["dataset_to_index"]:
                    #     next_action = self.hf_dataset[next_chunk_idx]["action"]
                    #     if next_action[-1].item() == -1:
                    #         gripper_pad_state = -1.0

                    item["action"] = torch.where(is_pad.unsqueeze(-1), noops, item["action"])

        if len(self._meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self._image_transforms is not None:
            image_keys = self._meta.camera_keys
            for cam in image_keys:
                item[cam] = self._image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        if self.mode == "move":
            item["subtask"] = get_dir(current_state=item['observation.state'], goal_state=self.hf_dataset[sub_end-1]['observation.state'], sorted=self.sorted_dir)
        elif self.mode == "trace":
            item["subtask"] = get_trace(full_trace=self.hf_dataset[abs_idx:sub_end]['2d_gripper'])
        elif self.mode == "subtask_move":
            assert "subtask_index" in self._meta.features and self._meta.subtasks is not None, "Mode = 'subtask_move' but dataset contains no subtasks"
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self._meta.subtasks.iloc[subtask_idx].name

            dir_data = ""
            if "move_index" in self._meta.features and self._meta.moves is not None:
                move_idx = item["move_index"].item()
                dir_data = self._meta.moves.iloc[move_idx].name

            if "grasp" in dir_data:
                item["subtask"] += f" ({dir_data})"
            else:
                dir = get_dir(current_state=item['observation.state'], goal_state=self.hf_dataset[sub_end-1]['observation.state'], sorted=self.sorted_dir)
                item["subtask"] += f" ({dir})"

                    
            ep_end = self._meta.episodes[ep_idx]["dataset_to_index"]

            # Currently only planner, need to add actions
            if self.is_planner and sub_len < self.chain_close and sub_end < ep_end:
                # Find relative index for the start of the *next* subtask chunk
                rel_next_idx = self._absolute_to_relative_idx[sub_end] if self._absolute_to_relative_idx is not None else sub_end
                next_item = self.hf_dataset[rel_next_idx]
                
                next_subtask_idx = next_item["subtask_index"].item()
                next_subtask_str = self._meta.subtasks.iloc[next_subtask_idx].name

                if "move_index" in self._meta.features and self._meta.moves is not None:
                    next_move_idx = next_item["move_index"].item()
                    dir_data = self._meta.moves.iloc[next_move_idx].name
                    if "grasp" in dir_data:
                        next_subtask_str += f" ({dir_data})"
                        item["subtask"] = f"{item['subtask']} THEN {next_subtask_str}"

                        return item

                    _, _, next_sub_len = self._get_query_indices(sub_end, ep_idx)
                    next_sub_end = sub_end + min(next_sub_len, self.chunk_size)
                    
                    next_dir = get_dir(next_item['observation.state'], self.hf_dataset[next_sub_end-1]['observation.state'], sorted=self.sorted_dir)
                    next_subtask_str += f" ({next_dir})"

                item["subtask"] = f"{item['subtask']} THEN {next_subtask_str}"

            elif self.is_planner and self.chain_dir and sub_len > self.chunk_size and sub_end < ep_end:
                rel_next_idx = self._absolute_to_relative_idx[sub_end] if self._absolute_to_relative_idx is not None else sub_end
                next_item = self.hf_dataset[rel_next_idx]
                next_subtask_idx = next_item["subtask_index"].item()
                next_subtask_str = self._meta.subtasks.iloc[next_subtask_idx].name

                next_dir = ""
                if "move_index" in self._meta.features and self._meta.moves is not None:
                    next_move_idx = next_item["move_index"].item()
                    next_dir = self._meta.moves.iloc[next_move_idx].name
                    
                if "grasp" not in next_dir:
                    next_sub_len = min(sub_len - self.chunk_size, self.chunk_size)
                    next_sub_end = sub_end + next_sub_len
                    next_dir = get_dir(self.hf_dataset[sub_end]['observation.state'], self.hf_dataset[next_sub_end-1]['observation.state'], sorted=self.sorted_dir)
                
                next_subtask_str += f" ({next_dir})"
                item["subtask"] = f"{item['subtask']} THEN {next_subtask_str}"

        if self.debug:
            print(f"DEBUG. Subtask = {item['subtask']}")
            self.debug = False


        return item


# class EvalDatasetReader(DatasetReader):
#     def __init__(
#         self,
#         meta,
#         root: Path,
#         episodes: list[int] | None,
#         tolerance_s: float,
#         video_backend: str,
#         delta_timestamps: dict[str, list[float]] | None,
#         image_transforms: Callable | None,
#         dynamic_action_chunking: str = "",
#         return_uint8: bool = False,
#         split: int = 0,
#         n_splits: int = 1,
#         completed_episodes_per_task: dict[int, int] | None = None,
#         chain_close: int = 10,
#         sorted_dir: bool = False,
#         chain_dir: bool = False,
#     ):
#         """Initialize the reader with metadata, filtering, and transform config."""
#         self._meta = meta
#         self.root = root
#         self.episodes = episodes if episodes is not None else list(range(self._meta.total_episodes))
#         k, m = divmod(len(self.episodes), n_splits)
#         splits = [
#             self.episodes[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
#             for i in range(n_splits)
#         ]
        
#         self.episodes = splits[split]
#         self._tolerance_s = tolerance_s
#         self._video_backend = video_backend
#         self._image_transforms = image_transforms
#         self.dynamic_action_chunking = dynamic_action_chunking
#         self._return_uint8 = return_uint8
        
#         self.debug = True
#         self.hf_dataset = None
#         self._absolute_to_relative_idx: dict[int, int] | None = None

#         self.delta_indices = None
#         if delta_timestamps is not None:
#             check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
#             self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)
            
#         from collections import defaultdict
#         self.completed_episodes_per_task = completed_episodes_per_task or {}
#         self._task_episode_order = defaultdict(list)

#     def get_item(self, idx) -> dict:
#         """Core __getitem__ logic. Assumes hf_dataset is loaded."""
#         item = self.hf_dataset[idx]
#         ep_idx = item["episode_index"].item()
#         abs_idx = item["index"].item()

#         if self.completed_episodes_per_task:
#             task_id = item.get("libero_id", item.get("task_index")).item()
            
#             if ep_idx not in self._task_episode_order[task_id]:
#                 self._task_episode_order[task_id].append(ep_idx)
            
#             ep_rank = self._task_episode_order[task_id].index(ep_idx)
#             if ep_rank < self.completed_episodes_per_task.get(task_id, 0):
#                 item["eval_sub_start"] = False
#                 return item

#         # Grouping keys appropriately
#         if self.dynamic_action_chunking == "move_dynamic":
#             dac_key = "subtask_index"
#         elif "move" in self.dynamic_action_chunking:
#             dac_key = "move_index"
#         else:
#             dac_key = "subtask_index"
            
#         ep_start = self._meta.episodes[ep_idx]["dataset_from_index"]

#         if abs_idx == ep_start:
#             item["eval_sub_start"] = True
#         else:
#             current_sub = item[dac_key].item()
#             prev_sub = self.hf_dataset[idx - 1][dac_key] if idx > 0 else current_sub
            
#             is_sub_start = (current_sub != prev_sub)
            
#             # only merge
#             if is_sub_start and "subtask" in self.dynamic_action_chunking:
#                 if "move_index" in self._meta.features and self._meta.moves is not None:
#                     prev_move_idx = self.hf_dataset[idx - 1]["move_index"]
#                     if hasattr(prev_move_idx, "item"): 
#                         prev_move_idx = prev_move_idx.item()
                    
#                     prev_move_name = self._meta.moves.iloc[prev_move_idx].name
#                     if "grasp the object" in prev_move_name:
#                         is_sub_start = False
                        
#             item["eval_sub_start"] = is_sub_start

#         if not item["eval_sub_start"]:
#             return item

#         query_indices = None
#         if self.delta_indices is not None:
#             query_indices, padding, sub_end = self._get_query_indices(abs_idx, ep_idx)
#             query_result = self._query_hf_dataset(query_indices)
#             item = {**item, **padding}
#             for key, val in query_result.items():
#                 item[key] = val

#             if self.dynamic_action_chunking and "action" in item and "action_is_pad" in item:
#                 is_pad = item["action_is_pad"]
#                 if is_pad.any():
#                     noops = torch.zeros_like(item["action"])
#                     gripper_pad_state = item["action"][..., -1]

#                     next_chunk_idx = max(query_indices["action"]) + 1
#                     if next_chunk_idx < self._meta.episodes[ep_idx]["dataset_to_index"]:
#                         rel_next_chunk_idx = self._absolute_to_relative_idx[next_chunk_idx] if self._absolute_to_relative_idx is not None else next_chunk_idx
#                         next_action = self.hf_dataset[rel_next_chunk_idx]["action"]
                        
#                         if next_action[-1].item() == -1:
#                             gripper_pad_state = -1.0
                    
#                     noops[..., -1] = gripper_pad_state
#                     item["action"] = torch.where(is_pad.unsqueeze(-1), noops, item["action"])

#         if len(self._meta.video_keys) > 0:
#             current_ts = item["timestamp"].item()
#             query_timestamps = self._get_query_timestamps(current_ts, query_indices)
#             video_frames = self._query_videos(query_timestamps, ep_idx)
#             item = {**video_frames, **item}

#         if self._image_transforms is not None:
#             image_keys = self._meta.camera_keys
#             for cam in image_keys:
#                 item[cam] = self._image_transforms(item[cam])

#         task_idx = item["task_index"].item()
#         item["task"] = self._meta.tasks.iloc[task_idx].name

#         if "subtask_index" in self._meta.features and self._meta.subtasks is not None:
#             subtask_idx = item["subtask_index"].item()
#             item["subtask"] = self._meta.subtasks.iloc[subtask_idx].name
#             item["eval_subtask"] = self._meta.subtasks.iloc[subtask_idx].name

#             if "move_index" in self._meta.features and self._meta.moves is not None:
#                 move_idx = item["move_index"].item()
#                 if "move" in self.dynamic_action_chunking and self.dynamic_action_chunking != "move_dynamic":
#                     item["subtask"] += f" ({self._meta.moves.iloc[move_idx].name})"
#                 item["eval_move"] = self._meta.moves.iloc[move_idx].name
#             else:
#                 item["eval_move"] = None
                
#         ep_end = self._meta.episodes[ep_idx]["dataset_to_index"]

#         # merge grasp with next subtask as subtask models not trained with grasp
#         if "subtask" in self.dynamic_action_chunking and item.get("eval_move") is not None and "grasp the object" in item["eval_move"]:
#             if sub_end < ep_end:
#                 rel_next_sub_start = self._absolute_to_relative_idx[sub_end] if self._absolute_to_relative_idx is not None else sub_end
#                 next_item = self.hf_dataset[rel_next_sub_start]
                
#                 n_subtask_idx = next_item["subtask_index"]
#                 if hasattr(n_subtask_idx, "item"): n_subtask_idx = n_subtask_idx.item()
                
#                 item["subtask"] = self._meta.subtasks.iloc[n_subtask_idx].name
#                 item["eval_subtask"] = item["subtask"]
                
#                 if "move_index" in self._meta.features and self._meta.moves is not None:
#                     n_move_idx = next_item["move_index"]
#                     if hasattr(n_move_idx, "item"): n_move_idx = n_move_idx.item()
                    
#                     if "move" in self.dynamic_action_chunking:
#                         item["subtask"] += f" ({self._meta.moves.iloc[n_move_idx].name})"
#                     item["eval_move"] = self._meta.moves.iloc[n_move_idx].name
                
#                 _, _, next_sub_end = self._get_query_indices(sub_end, ep_idx)
#                 sub_end = next_sub_end

#         # -------------- EXTRACT SEQUENTIAL MOVES FOR DYNAMIC PROCESSING -------------- #
#         if self.dynamic_action_chunking == "move_dynamic":
#             rel_abs_idx = self._absolute_to_relative_idx[abs_idx] if self._absolute_to_relative_idx else abs_idx
#             rel_sub_end = self._absolute_to_relative_idx[sub_end-1] if self._absolute_to_relative_idx else sub_end-1
            
#             # Slice move components
#             move_indices = self.hf_dataset[rel_abs_idx:rel_sub_end + 1]["move_index"]
            
#             # CRITICAL FIX: HF slices return lists. Convert to a 1D numpy array 
#             # to prevent scalar boolean evaluation during the inequality check.
#             if isinstance(move_indices, torch.Tensor):
#                 move_indices = move_indices.numpy()
#             move_indices = np.atleast_1d(np.array(move_indices))
            
#             eval_goal_states = []
#             eval_sub_lens = []
#             eval_moves = []
            
#             boundaries = np.where(move_indices[:-1] != move_indices[1:])[0] + 1
#             boundaries = np.concatenate([boundaries, [len(move_indices)]])
            
#             prev_b = 0
#             for b in boundaries:
#                 move_len = int(b - prev_b)
#                 move_end_abs = abs_idx + b - 1
#                 rel_move_end = self._absolute_to_relative_idx[move_end_abs] if self._absolute_to_relative_idx else move_end_abs
                
#                 eval_goal_states.append(self.hf_dataset[rel_move_end]['observation.state'])
#                 eval_sub_lens.append(move_len)
                
#                 m_idx = move_indices[prev_b].item()
#                 if "move_index" in self._meta.features and self._meta.moves is not None:
#                     eval_moves.append(self._meta.moves.iloc[m_idx].name)
#                 else:
#                     eval_moves.append("")
                    
#                 prev_b = b

#             # Pack to MAX_MOVES preventing heterogeneous size tensor collation failures downstream
#             MAX_MOVES = 20
#             state_dim = eval_goal_states[0].shape[0] if len(eval_goal_states) > 0 else 0
#             padded_goal_states = np.zeros((MAX_MOVES, state_dim), dtype=np.float32)
#             padded_sub_lens = np.zeros((MAX_MOVES,), dtype=np.int64)
#             num_moves = len(eval_sub_lens)
            
#             for i in range(min(num_moves, MAX_MOVES)):
#                 if isinstance(eval_goal_states[i], torch.Tensor):
#                     padded_goal_states[i] = eval_goal_states[i].numpy()
#                 else:
#                     padded_goal_states[i] = eval_goal_states[i]
#                 padded_sub_lens[i] = eval_sub_lens[i]
                
#             item["eval_goal_states"] = padded_goal_states
#             item["eval_sub_lens"] = padded_sub_lens
#             item["eval_num_moves"] = min(num_moves, MAX_MOVES)
#             item["eval_moves_str"] = "|||".join(eval_moves[:MAX_MOVES])

#         # Core assignments
#         rel_sub_end = self._absolute_to_relative_idx[sub_end-1] if self._absolute_to_relative_idx is not None else sub_end-1
#         item["eval_goal_state"] = self.hf_dataset[rel_sub_end]['observation.state']

#         if "init_state_index" in self._meta.features and self._meta.init_states is not None:
#             init_state_idx = item["init_state_index"].item()
#             item["eval_init_state"] = np.array(self._meta.init_states.iloc[init_state_idx].name, dtype=np.float32)

#         if abs_idx == ep_start:
#             item["eval_prev_actions"] = []
#         else:
#             rel_ep_start = self._absolute_to_relative_idx[ep_start] if self._absolute_to_relative_idx is not None else ep_start
#             rel_abs_idx = self._absolute_to_relative_idx[abs_idx] if self._absolute_to_relative_idx is not None else abs_idx
#             item["eval_prev_actions"] = self.hf_dataset[rel_ep_start:rel_abs_idx]["action"]

#         item["eval_location"] = ((abs_idx - ep_start) / (ep_end - ep_start - 1))
#         item["eval_sub_len"] = sub_end - abs_idx

#         return item
    

# from typing import Callable
# from pathlib import Path
# import torch
# import numpy as np

# class EvalBottomUpDatasetReader(EvalDatasetReader):
#     def __init__(
#         self,
#         meta,
#         root: Path,
#         episodes: list[int] | None,
#         tolerance_s: float,
#         video_backend: str,
#         delta_timestamps: dict[str, list[float]] | None,
#         image_transforms: Callable | None,
#         dynamic_action_chunking: str = "move",
#         return_uint8: bool = False,
#         split: int = 0,
#         n_splits: int = 1,
#         completed_episodes_per_task: dict[int, int] | None = None,
#         chunk_size: int = 10,
#         sorted_dir: bool = True,
#     ):
#         super().__init__(
#             meta,
#             root,
#             episodes,
#             tolerance_s,
#             video_backend,
#             delta_timestamps,
#             image_transforms,
#             "subtask",
#             return_uint8,
#             split,
#             n_splits,
#             completed_episodes_per_task,
#         )
#         self.mode = dynamic_action_chunking
#         self.chunk_size = chunk_size
#         self.sorted_dir = sorted_dir
#         print(f"Initialized EvalBottomUpDatasetReader with mode={self.mode}, chunk_size={chunk_size}, sorted_dir={sorted_dir}")

#     def get_item(self, idx) -> dict:
#         """Core __getitem__ logic. Assumes hf_dataset is loaded."""
#         item = self.hf_dataset[idx]
#         ep_idx = item["episode_index"].item()
#         abs_idx = item["index"].item()

#         if self.completed_episodes_per_task:
#             task_id = item.get("libero_id", item.get("task_index")).item()
            
#             if ep_idx not in self._task_episode_order[task_id]:
#                 self._task_episode_order[task_id].append(ep_idx)
            
#             ep_rank = self._task_episode_order[task_id].index(ep_idx)
#             if ep_rank < self.completed_episodes_per_task.get(task_id, 0):
#                 item["eval_sub_start"] = False
#                 return item

#         # Grouping keys appropriately
#         if self.mode == "move_dynamic":
#             dac_key = "subtask_index"
#         elif "move" in self.mode:
#             dac_key = "move_index"
#         else:
#             dac_key = "subtask_index"
            
#         ep_start = self._meta.episodes[ep_idx]["dataset_from_index"]

#         if abs_idx == ep_start:
#             item["eval_sub_start"] = True
#         else:
#             current_sub = item[dac_key].item()
#             prev_sub = self.hf_dataset[idx - 1][dac_key] if idx > 0 else current_sub
            
#             is_sub_start = (current_sub != prev_sub)                
#             item["eval_sub_start"] = is_sub_start

#         if not item["eval_sub_start"]:
#             return item

#         query_indices = None
#         sub_end = abs_idx + 1  # Fallback initialization to prevent UnboundLocalError
        
#         if self.delta_indices is not None:
#             query_indices, padding, sub_len = self._get_query_indices(abs_idx, ep_idx)
#             sub_end = abs_idx + min(sub_len, self.chunk_size)
#             query_result = self._query_hf_dataset(query_indices)
#             item = {**item, **padding}
#             for key, val in query_result.items():
#                 item[key] = val

#             if self.dynamic_action_chunking and "action" in item and "action_is_pad" in item:
#                 is_pad = item["action_is_pad"]
#                 if is_pad.any():
#                     noops = torch.zeros_like(item["action"])
#                     noops[..., -1] = item["action"][..., -1]

#                     item["action"] = torch.where(is_pad.unsqueeze(-1), noops, item["action"])

#         if len(self._meta.video_keys) > 0:
#             current_ts = item["timestamp"].item()
#             query_timestamps = self._get_query_timestamps(current_ts, query_indices)
#             video_frames = self._query_videos(query_timestamps, ep_idx)
#             item = {**video_frames, **item}

#         if self._image_transforms is not None:
#             image_keys = self._meta.camera_keys
#             for cam in image_keys:
#                 item[cam] = self._image_transforms(item[cam])

#         # Add task as a string
#         task_idx = item["task_index"].item()
#         item["task"] = self._meta.tasks.iloc[task_idx].name

#         # add subtask information if available
#         if "subtask_index" in self._meta.features and self._meta.subtasks is not None:
#             subtask_idx = item["subtask_index"].item()
#             item["subtask"] = self._meta.subtasks.iloc[subtask_idx].name
#             item["eval_subtask"] = self._meta.subtasks.iloc[subtask_idx].name

#             dir_data = ""
#             if "move_index" in self._meta.features and self._meta.moves is not None:
#                 move_idx = item["move_index"].item()
#                 dir_data = self._meta.moves.iloc[move_idx].name

#             if "grasp" in dir_data:
#                 item["subtask"] += f" ({dir_data})"
#                 item["eval_move"] = dir_data
#             else:
#                 rel_sub_end_idx = self._absolute_to_relative_idx[sub_end-1] if self._absolute_to_relative_idx is not None else sub_end-1
#                 dir = get_dir(current_state=item['observation.state'], goal_state=self.hf_dataset[rel_sub_end_idx]['observation.state'], sorted=self.sorted_dir)
#                 item["subtask"] += f" ({dir})"
#                 item["eval_move"] = dir
                    
#             ep_end = self._meta.episodes[ep_idx]["dataset_to_index"]

#             if self.debug:
#                 print(f"DEBUG. Does move_index exist? {'move_index' in self._meta.features and self._meta.moves is not None}. Subtask = {item['subtask']}")
#                 self.debug = False

#         # -------------- EXTRACT SEQUENTIAL MOVES FOR DYNAMIC PROCESSING -------------- #
#         if self.mode == "move_dynamic":
#             rel_abs_idx = self._absolute_to_relative_idx[abs_idx] if self._absolute_to_relative_idx else abs_idx
            
#             # FIX 1: Use the TRUE subtask length, not the truncated sub_end
#             full_sub_end_abs = abs_idx + sub_len
#             rel_full_sub_end = self._absolute_to_relative_idx[full_sub_end_abs - 1] if self._absolute_to_relative_idx else full_sub_end_abs - 1
            
#             seq_len = (rel_full_sub_end + 1) - rel_abs_idx
            
#             # FIX 2: Batch slice the dataset states to prevent IO bottlenecking
#             # This pulls all required states for the sequence into memory at once
#             sequence_states = self.hf_dataset[rel_abs_idx : rel_full_sub_end + 1]['observation.state']
#             if isinstance(sequence_states, list): 
#                 sequence_states = torch.stack(sequence_states)
            
#             eval_goal_states = []
#             eval_sub_lens = []
#             eval_moves = []
            
#             boundaries = [min(i + self.chunk_size, seq_len) for i in range(0, seq_len, self.chunk_size)]
            
#             prev_b = 0
#             for b in boundaries:
#                 move_len = int(b - prev_b)
                
#                 # Fetch states from the pre-sliced sequence array, avoiding HF __getitem__ overhead
#                 start_state = sequence_states[prev_b]
#                 goal_state = sequence_states[b - 1]
                
#                 eval_goal_states.append(goal_state)
#                 eval_sub_lens.append(move_len)
                
#                 dir_str = get_dir(
#                     current_state=start_state, 
#                     goal_state=goal_state, 
#                     sorted=self.sorted_dir
#                 )
#                 eval_moves.append(dir_str)
                
#                 prev_b = b

#             MAX_MOVES = 20
#             # FIX 3: Fallback dimension to prevent crash if sequence is empty
#             state_dim = eval_goal_states[0].shape[0] if len(eval_goal_states) > 0 else sequence_states[0].shape[0]
            
#             padded_goal_states = np.zeros((MAX_MOVES, state_dim), dtype=np.float32)
#             padded_sub_lens = np.zeros((MAX_MOVES,), dtype=np.int64)
#             num_moves = len(eval_sub_lens)
            
#             for i in range(min(num_moves, MAX_MOVES)):
#                 if isinstance(eval_goal_states[i], torch.Tensor):
#                     padded_goal_states[i] = eval_goal_states[i].numpy()
#                 else:
#                     padded_goal_states[i] = eval_goal_states[i]
#                 padded_sub_lens[i] = eval_sub_lens[i]
                
#             item["eval_goal_states"] = padded_goal_states
#             item["eval_sub_lens"] = padded_sub_lens
#             item["eval_num_moves"] = min(num_moves, MAX_MOVES)
#             item["eval_moves_str"] = "|||".join(eval_moves[:MAX_MOVES])

#         # Core assignments
#         rel_sub_end = self._absolute_to_relative_idx[sub_end-1] if self._absolute_to_relative_idx is not None else sub_end-1
#         item["eval_goal_state"] = self.hf_dataset[rel_sub_end]['observation.state']

#         if "init_state_index" in self._meta.features and self._meta.init_states is not None:
#             init_state_idx = item["init_state_index"].item()
#             item["eval_init_state"] = np.array(self._meta.init_states.iloc[init_state_idx].name, dtype=np.float32)

#         if abs_idx == ep_start:
#             item["eval_prev_actions"] = []
#         else:
#             rel_ep_start = self._absolute_to_relative_idx[ep_start] if self._absolute_to_relative_idx is not None else ep_start
#             rel_abs_idx = self._absolute_to_relative_idx[abs_idx] if self._absolute_to_relative_idx is not None else abs_idx
#             item["eval_prev_actions"] = self.hf_dataset[rel_ep_start:rel_abs_idx]["action"]

#         item["eval_location"] = ((abs_idx - ep_start) / (ep_end - ep_start - 1))
#         item["eval_sub_len"] = sub_end - abs_idx

#         return item

class EvalDatasetReader(DatasetReader):
    def __init__(
        self,
        meta,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        dynamic_action_chunking: str = "",
        return_uint8: bool = False,
        split: int = 0,
        n_splits: int = 1,
        completed_episodes_per_task: dict[int, int] | None = None,
        chain_close: int = 10,
        sorted_dir: bool = False,
        chain_dir: bool = False,
    ):
        """Initialize the reader with metadata, filtering, and transform config."""
        self._meta = meta
        self.root = root
        self.episodes = episodes if episodes is not None else list(range(self._meta.total_episodes))
        k, m = divmod(len(self.episodes), n_splits)
        splits = [
            self.episodes[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(n_splits)
        ]
        
        self.episodes = splits[split]
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        self._image_transforms = image_transforms
        self.dynamic_action_chunking = dynamic_action_chunking
        self._return_uint8 = return_uint8
        
        self.debug = True
        self.hf_dataset = None
        self._absolute_to_relative_idx: dict[int, int] | None = None

        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)
            
        from collections import defaultdict
        self.completed_episodes_per_task = completed_episodes_per_task or {}
        self._task_episode_order = defaultdict(list)
        self.debug = True

    def get_item(self, idx) -> dict:
        """Core __getitem__ logic. Assumes hf_dataset is loaded."""
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        if self.debug:
            print("Is correct EvalDatasetReader")

        ep_start = self._meta.episodes[ep_idx]["dataset_from_index"]
        item["eval_step"] = item["index"] - ep_start

        if self.completed_episodes_per_task:
            task_id = item.get("libero_id", item.get("task_index")).item()
            
            if ep_idx not in self._task_episode_order[task_id]:
                self._task_episode_order[task_id].append(ep_idx)
            
            ep_rank = self._task_episode_order[task_id].index(ep_idx)
            if ep_rank < self.completed_episodes_per_task.get(task_id, 0):
                item["eval_sub_start"] = False
                return item

        # Grouping keys appropriately
        if "move" in self.dynamic_action_chunking:
            dac_key = "move_index"
        else:
            dac_key = "subtask_index"
            
        ep_start = self._meta.episodes[ep_idx]["dataset_from_index"]

        if abs_idx == ep_start:
            item["eval_sub_start"] = True
        else:
            current_sub = item[dac_key].item()
            prev_sub = self.hf_dataset[idx - 1][dac_key] if idx > 0 else current_sub
            
            current_sub_str = self._meta.moves.iloc[current_sub].name
            is_sub_start = (current_sub != prev_sub) and ("grasp" not in current_sub_str)   
                        
            item["eval_sub_start"] = is_sub_start

        if not item["eval_sub_start"]:
            return item

        query_indices = None
        sub_end = abs_idx + 1  # Fallback initialization

        if self.delta_indices is not None:
            query_indices, padding, sub_len = self._get_query_indices(abs_idx, ep_idx)
            sub_end = abs_idx + sub_len
            
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

            if self.dynamic_action_chunking and "action" in item and "action_is_pad" in item:
                is_pad = item["action_is_pad"]
                if is_pad.any():
                    noops = torch.zeros_like(item["action"])
                    gripper_pad_state = item["action"][..., -1]
                    
                    noops[..., -1] = gripper_pad_state
                    item["action"] = torch.where(is_pad.unsqueeze(-1), noops, item["action"])

        if len(self._meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self._image_transforms is not None:
            image_keys = self._meta.camera_keys
            for cam in image_keys:
                item[cam] = self._image_transforms(item[cam])

        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        if "subtask_index" in self._meta.features and self._meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self._meta.subtasks.iloc[subtask_idx].name
            item["eval_subtask"] = self._meta.subtasks.iloc[subtask_idx].name

            if "move_index" in self._meta.features and self._meta.moves is not None:
                move_idx = item["move_index"].item()
                if "move" in self.dynamic_action_chunking:
                    item["subtask"] += f" ({self._meta.moves.iloc[move_idx].name})"
                item["eval_move"] = self._meta.moves.iloc[move_idx].name
            else:
                item["eval_move"] = None
                
        ep_end = self._meta.episodes[ep_idx]["dataset_to_index"]

        # Core assignments
        rel_sub_end = self._absolute_to_relative_idx[sub_end-1] if self._absolute_to_relative_idx is not None else sub_end-1
        item["eval_goal_state"] = self.hf_dataset[rel_sub_end]['observation.state']

        if "init_state_index" in self._meta.features and self._meta.init_states is not None:
            init_state_idx = item["init_state_index"].item()
            item["eval_init_state"] = np.array(self._meta.init_states.iloc[init_state_idx].name, dtype=np.float32)

        if abs_idx == ep_start:
            item["eval_prev_actions"] = []
        else:
            rel_ep_start = self._absolute_to_relative_idx[ep_start] if self._absolute_to_relative_idx is not None else ep_start
            rel_abs_idx = self._absolute_to_relative_idx[abs_idx] if self._absolute_to_relative_idx is not None else abs_idx
            item["eval_prev_actions"] = self.hf_dataset[rel_ep_start:rel_abs_idx]["action"]

        item["eval_location"] = ((abs_idx - ep_start) / (ep_end - ep_start - 1))
        item["eval_sub_len"] = sub_end - abs_idx

        return item
    

from typing import Callable
from pathlib import Path
import torch
import numpy as np

class EvalBottomUpDatasetReader(EvalDatasetReader):
    def __init__(
        self,
        meta,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        dynamic_action_chunking: str = "move",
        return_uint8: bool = False,
        split: int = 0,
        n_splits: int = 1,
        completed_episodes_per_task: dict[int, int] | None = None,
        chunk_size: int = 10,
        sorted_dir: bool = False,
    ):
        super().__init__(
            meta,
            root,
            episodes,
            tolerance_s,
            video_backend,
            delta_timestamps,
            image_transforms,
            "subtask",
            return_uint8,
            split,
            n_splits,
            completed_episodes_per_task,
        )
        self.mode = dynamic_action_chunking
        self.chunk_size = chunk_size
        self.sorted_dir = sorted_dir
        print(f"Initialized EvalBottomUpDatasetReader with mode={self.mode}, chunk_size={chunk_size}, sorted_dir={sorted_dir}")

    def get_item(self, idx) -> dict:
        """Core __getitem__ logic. Assumes hf_dataset is loaded."""
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        ep_start = self._meta.episodes[ep_idx]["dataset_from_index"]
        item["step"] = item["index"] - ep_start

        if self.completed_episodes_per_task:
            task_id = item.get("libero_id", item.get("task_index")).item()
            
            if ep_idx not in self._task_episode_order[task_id]:
                self._task_episode_order[task_id].append(ep_idx)
            
            ep_rank = self._task_episode_order[task_id].index(ep_idx)
            if ep_rank < self.completed_episodes_per_task.get(task_id, 0):
                item["eval_sub_start"] = False
                return item

        # Grouping keys appropriately
        if "move" in self.mode:
            dac_key = "move_index"
        else:
            dac_key = "subtask_index"
            
        ep_start = self._meta.episodes[ep_idx]["dataset_from_index"]

        if abs_idx == ep_start:
            item["eval_sub_start"] = True
        else:
            current_sub = item[dac_key].item()
            prev_sub = self.hf_dataset[idx - 1][dac_key] if idx > 0 else current_sub
            
            current_sub_str = self._meta.moves.iloc[current_sub].name
            is_sub_start = (current_sub != prev_sub) and ("grasp" not in current_sub_str)             
            item["eval_sub_start"] = is_sub_start

        if not item["eval_sub_start"]:
            return item

        query_indices = None
        sub_end = abs_idx + 1  # Fallback initialization to prevent UnboundLocalError
        
        if self.delta_indices is not None:
            query_indices, padding, sub_len = self._get_query_indices(abs_idx, ep_idx)
            sub_end = abs_idx + min(sub_len, self.chunk_size)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

            if self.dynamic_action_chunking and "action" in item and "action_is_pad" in item:
                is_pad = item["action_is_pad"]
                if is_pad.any():
                    noops = torch.zeros_like(item["action"])
                    noops[..., -1] = item["action"][..., -1]

                    item["action"] = torch.where(is_pad.unsqueeze(-1), noops, item["action"])

        if len(self._meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self._image_transforms is not None:
            image_keys = self._meta.camera_keys
            for cam in image_keys:
                item[cam] = self._image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        # add subtask information if available
        if "subtask_index" in self._meta.features and self._meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self._meta.subtasks.iloc[subtask_idx].name
            item["eval_subtask"] = self._meta.subtasks.iloc[subtask_idx].name

            dir_data = ""
            if "move_index" in self._meta.features and self._meta.moves is not None:
                move_idx = item["move_index"].item()
                dir_data = self._meta.moves.iloc[move_idx].name

            if "grasp" in dir_data:
                item["subtask"] += f" ({dir_data})"
                item["eval_move"] = dir_data
            else:
                rel_sub_end_idx = self._absolute_to_relative_idx[sub_end-1] if self._absolute_to_relative_idx is not None else sub_end-1
                dir = get_dir(current_state=item['observation.state'], goal_state=self.hf_dataset[rel_sub_end_idx]['observation.state'], sorted=self.sorted_dir)
                item["subtask"] += f" ({dir})"
                item["eval_move"] = dir
                    
            ep_end = self._meta.episodes[ep_idx]["dataset_to_index"]

            if self.debug:
                print(f"DEBUG. Does move_index exist? {'move_index' in self._meta.features and self._meta.moves is not None}. Subtask = {item['subtask']}")
                self.debug = False

        # Core assignments
        rel_sub_end = self._absolute_to_relative_idx[sub_end-1] if self._absolute_to_relative_idx is not None else sub_end-1
        item["eval_goal_state"] = self.hf_dataset[rel_sub_end]['observation.state']

        if "init_state_index" in self._meta.features and self._meta.init_states is not None:
            init_state_idx = item["init_state_index"].item()
            item["eval_init_state"] = np.array(self._meta.init_states.iloc[init_state_idx].name, dtype=np.float32)

        if abs_idx == ep_start:
            item["eval_prev_actions"] = []
        else:
            rel_ep_start = self._absolute_to_relative_idx[ep_start] if self._absolute_to_relative_idx is not None else ep_start
            rel_abs_idx = self._absolute_to_relative_idx[abs_idx] if self._absolute_to_relative_idx is not None else abs_idx
            item["eval_prev_actions"] = self.hf_dataset[rel_ep_start:rel_abs_idx]["action"]

        item["eval_location"] = ((abs_idx - ep_start) / (ep_end - ep_start - 1))
        item["eval_sub_len"] = sub_end - abs_idx

        return item