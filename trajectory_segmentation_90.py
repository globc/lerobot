import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.signal import argrelextrema, find_peaks
from scipy.ndimage import gaussian_filter1d
from rdp import rdp

def reduce_trajectory(segments, lengths, states, threshold=25):
    if not segments or not lengths:
        return [], []

    let_go_state = states[0]
    hold_state = -let_go_state

    def filter_state(s_arr, l_arr, st_arr, target_state, max_noise_len):
        res_starts, res_lengths, res_states = [], [], []
        
        for s, l, st in zip(s_arr, l_arr, st_arr):
            if st == target_state and l <= max_noise_len:
                st = -target_state
                
            if res_states and res_states[-1] == st:
                res_lengths[-1] += l
            else:
                res_starts.append(s)
                res_lengths.append(l)
                res_states.append(st)
                
        return res_starts, res_lengths, res_states

    s1, l1, st1 = filter_state(segments, lengths, states, let_go_state, threshold)
    
    s2, l2, _ = filter_state(s1, l1, st1, hold_state, threshold)
    
    return s2, l2

def get_gripper_segments(gripper_actions, noise_threshold=25):
    diffs = gripper_actions[:-1] != gripper_actions[1:]
    segment_indices = np.where(diffs)[0] + 1
    segment_indices = np.insert(segment_indices, 0, 0)
    
    segment_lengths = np.empty(len(segment_indices), dtype=int)
    if len(segment_indices) > 1:
        segment_lengths[:-1] = np.diff(segment_indices)
    segment_lengths[-1] = len(gripper_actions) - segment_indices[-1]

    reduced_indices, reduced_lengths = reduce_trajectory(
        segment_indices.tolist(), 
        segment_lengths.tolist(), 
        gripper_actions[segment_indices].tolist(), 
        threshold=noise_threshold
    )
    
    return reduced_indices, reduced_lengths

def get_state_segments(states):
    deltas = np.diff(states, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    minima_indices = argrelextrema(norms, np.less)[0]
    segment_boundaries = np.insert(minima_indices, 0, 0)

    return segment_boundaries.tolist(), []

def get_kinematic_segments(state, prominence=0.005, min_length=25, sigma=0.0):
    velocities = np.diff(state[:, :6], axis=0)
    velocities = np.insert(velocities, 0, 0, axis=0)
    
    # OPTIMIZATION: Vectorized gaussian filter instead of loop
    if sigma > 0:
        velocities = gaussian_filter1d(velocities, sigma=sigma, axis=0)
            
    speeds = np.linalg.norm(velocities, axis=1)

    minima_indices, _ = find_peaks(-speeds, prominence=prominence)
    
    filtered_boundaries = [0]
    for idx in minima_indices:
        if idx - filtered_boundaries[-1] >= min_length:
            filtered_boundaries.append(idx)
            
    if len(state) - filtered_boundaries[-1] < min_length and len(filtered_boundaries) > 1:
        filtered_boundaries.pop()
    
    boundaries_with_end = np.append(filtered_boundaries, len(state))
    segment_lengths = np.diff(boundaries_with_end).tolist()
    
    return filtered_boundaries, segment_lengths

def get_directory_segments(state, sigma=4.0, merge_threshold=10):
    velocities = np.diff(state[:, :3], axis=0)
    velocities = np.insert(velocities, 0, 0, axis=0)

    # OPTIMIZATION: Vectorized gaussian filter instead of loop
    if sigma > 0:
        velocities = gaussian_filter1d(velocities, sigma=sigma, axis=0)
    
    all_crossings = set()
    
    for j in range(3):
        minima_indices, _ = find_peaks(-np.abs(velocities[:, j]))
        all_crossings.update(minima_indices)

    sorted_crossings = sorted(list(all_crossings))
    merged_crossings = []

    if sorted_crossings:
        current_cluster = [sorted_crossings[0]]
        
        for idx in sorted_crossings[1:]:
            if idx - current_cluster[-1] < merge_threshold:
                current_cluster.append(idx)
            else:
                avg_idx = int(np.round(np.mean(current_cluster)))
                merged_crossings.append(avg_idx)
                
                current_cluster = [idx]
                
        avg_idx = int(np.round(np.mean(current_cluster)))
        merged_crossings.append(avg_idx)
        
    if 0 not in merged_crossings:
        merged_crossings.insert(0, 0)

    return merged_crossings, merged_crossings

def segment_to_directory(segment_state):
    delta = segment_state[-1][:3] - segment_state[0][:3]
    components = [
        (abs(delta[0]), "forward" if delta[0] > 0 else "backward"),
        (abs(delta[1]), "right" if delta[1] > 0 else "left"),
        (abs(delta[2]), "up" if delta[2] > 0 else "down"),
    ]

    primary_mag = max(comp[0] for comp in components)
    if primary_mag == 0:
        return ""
    
    directions = []
    for mag, dir_str in components:
        if mag == primary_mag:
            directions.append(dir_str)
        else:
            ratio = mag / primary_mag
            
            if ratio >= 0.5:
                directions.append(dir_str)
            elif ratio >= 0.25:
                directions.append(f"slightly {dir_str}")

    return ", ".join(directions)

def segment_to_move(segment_state, threshold_scale=1.0):
    if len(segment_state) < 2:
        return None
        
    delta = segment_state[-1][:6] - segment_state[0][:6]
    
    components = [
        (abs(delta[0]), "forward" if delta[0] > 0 else "backward"),
        (abs(delta[1]), "right" if delta[1] > 0 else "left"),
        (abs(delta[2]), "up" if delta[2] > 0 else "down"),
        (abs(delta[3]) / 10, "turn right" if delta[3] > 0 else "turn left"), 
        (abs(delta[4]) / 10, "rotate clockwise" if delta[4] > 0 else "rotate counter-clockwise"),
        (abs(delta[5]) / 10, "tilt up" if delta[5] > 0 else "tilt down")
    ]
    
    components.sort(key=lambda x: x[0], reverse=True)
    
    base_threshold = 0.03/4 * len(segment_state) * threshold_scale
    primary_mag, primary_dir = components[0]
    move = []
    
    if primary_mag < base_threshold * 2 : 
        move.append(f"slightly {primary_dir}")
    elif primary_mag < base_threshold * 3:
        move.append(primary_dir)
    else:
        move.append(f"strongly {primary_dir}")
        
    for other_mag, other_dir in components[1:]:
        if other_mag < base_threshold:
            continue
        elif other_mag < base_threshold * 2:
            move.append(f"slightly {other_dir}")
        elif other_mag < base_threshold * 3:
            move.append(other_dir)
        else:
            move.append(f"strongly {other_dir}")
        
    return ", ".join(move)


def clean_and_merge_segments(boundaries, state, vel_threshold=0.0005):
    if not boundaries:
        return [], []
        
    bnds = list(boundaries)
    if bnds[-1] != len(state):
        bnds.append(len(state))
        
    changed = True
    while changed:
        changed = False
        
        i = 0
        while i < len(bnds) - 1:
            start = bnds[i]
            end = bnds[i+1]
            seg_state = state[start:end]
            
            vels = np.abs(np.diff(seg_state[:, :3], axis=0))
            max_vel = np.max(vels) if len(vels) > 0 else 0
            
            if max_vel < vel_threshold:
                if i > 0:
                    bnds.pop(i) 
                    changed = True
                    break
                elif len(bnds) > 2:
                    bnds.pop(1) 
                    changed = True
                    break
            i += 1
            
        if changed:
            continue 
            
        strings = [segment_to_directory(state[bnds[k]:bnds[k+1]]) for k in range(len(bnds)-1)]
        i = 0
        while i < len(strings) - 1:
            s1 = strings[i]
            s2 = strings[i+1]
            
            s1_base = s1.replace("slightly ", "")
            s2_base = s2.replace("slightly ", "")
            
            if s1 == s2 or s1_base == s2_base:
                bnds.pop(i+1) 
                changed = True
                break
            i += 1
            
    final_strings = [segment_to_directory(state[bnds[k]:bnds[k+1]]) for k in range(len(bnds)-1)]
    return bnds[:-1], final_strings


def get_segment_dicts(observation_batch, action_batch, segment_boundaries):
    subtask_dict = {}
    move_dict = {}
    for i, seg in enumerate(segment_boundaries):
        seg_end = segment_boundaries[i+1] if i+1 < len(segment_boundaries) else len(observation_batch)
        if seg > 0 and action_batch[seg-1, -1] == -1 and action_batch[seg, -1] == 1: 
            arr = action_batch[seg:seg_end, -1]
            changes = np.where(np.diff(arr) != 0)[0] + 1
            boundaries = np.concatenate(([0], changes, [len(arr)]))
            lengths = np.diff(boundaries)
            longest_idx = np.argmax(lengths)

            last_gripper_close = int(boundaries[longest_idx]) 
            
            move_segments_grasp, _ = get_state_segments(observation_batch[seg:seg_end])
            move_segments_grasp = [mov for mov in move_segments_grasp if mov >= last_gripper_close + 5]
            grasp_end = seg + move_segments_grasp[0] if move_segments_grasp else seg

            subtask_dict[seg] = "grasp the object"
            move_dict[seg] = "grasp the object"
            
            seg = grasp_end

        subtask_dict[seg] = ""

        move_segments, _ = get_directory_segments(observation_batch[seg:seg_end], sigma=4.0)
        
        final_boundaries, final_strings = clean_and_merge_segments(
            move_segments, 
            observation_batch[seg:seg_end], 
            vel_threshold=0.0005
        )
        
        for mov, string in zip(final_boundaries, final_strings):
            move_dict[seg + mov] = string

    subtask_dict = {int(k): v for k, v in sorted(subtask_dict.items())}
    move_dict = {int(k): v for k, v in sorted(move_dict.items())}

    if move_dict:
        first_key = next(iter(move_dict))
        if first_key != 0:
            first_val = move_dict.pop(first_key)
            new_move_dict = {0: first_val}
            new_move_dict.update(move_dict)
            move_dict = new_move_dict

    return subtask_dict, move_dict


def main():
    builder = tfds.builder_from_directory("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/data/embodied_features_and_demos_libero/libero_lm_90/1.0.0")
    
    # OPTIMIZATION: Prefetching keeps the CPU fed with data continuously
    dataset = builder.as_dataset(split='train').prefetch(tf.data.AUTOTUNE)

    all_subtask_segments = {}
    all_lengths = []

    ep_idx = 0
    for ep in dataset:
        # OPTIMIZATION: Batch the entire sequence dataset directly into memory 
        # instead of looping frame-by-frame with Python list comprehensions.
        batched_steps = ep["steps"].batch(100000) # Ensure it encompasses max episode length
        step_data = next(iter(tfds.as_numpy(batched_steps)))
        
        action_batch = step_data["action"]
        observation_batch = step_data["observation"]["state"]
        print(observation_batch[0])
        cool = col
        # Pulls from the batched strings array at index 0 directly
        task = step_data["language_instruction"][0].decode('utf-8')
        
        gripper_actions = action_batch[:, -1].astype(int)

        subtask_segments, _ = get_gripper_segments(gripper_actions)

        if len(subtask_segments) < 2:
            subtask_segments, _ = get_kinematic_segments(
                observation_batch, 
                prominence=0.0085,
                min_length=25,
                sigma=4.0
            )
        
        print(f"Subtask segments for episode {ep_idx} ({task})")
        subtask_dict, move_dict = get_segment_dicts(observation_batch, action_batch, subtask_segments)
        subtask_lengths = np.diff(np.append(np.array(list(subtask_dict.keys())), len(action_batch))).tolist()

        all_subtask_segments[ep_idx] = {
            "task": task,
            "num_segments": len(subtask_dict.keys()),
            "subtask_dict": subtask_dict,
            "move_dict": move_dict,
            "subtask_lengths": subtask_lengths
        }
        
        subtask_segments = list(subtask_dict.keys())

        all_lengths.extend(subtask_lengths)
        ep_idx += 1

    # --- Save JSON ---
    json_output_path = "./subtask_segments_dir_merge_new_minima_90.json"
    with open(json_output_path, "w") as f:
        json.dump(all_subtask_segments, f, indent=4)
    print(f"Successfully saved all segments, tasks, counts, and lengths to {json_output_path}")

    # --- Create and Save Plot ---
    plt.figure(figsize=(10, 6))
    plt.hist(all_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Subtask Segment Lengths", fontsize=14)
    plt.xlabel("Segment Length (Frames)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    
    plot_output_path = "./subtask_lengths_dir_merge_new_minima_90.png"
    plt.savefig(plot_output_path, bbox_inches='tight')
    plt.close() 
    print(f"Successfully saved length distribution plot to {plot_output_path}")

if __name__ == "__main__":
    main()