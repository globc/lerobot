from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.signal import argrelextrema, find_peaks
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

def get_grasp_end(states):
    deltas = np.diff(states[:, -2:], axis=0)
    norms = np.linalg.norm(deltas, axis=1)

    matches = np.where(norms < 2.5e-5)[0]
    print(f"Grasp end candidates based on minimal movement: {matches}")
    return int(matches[0]) if len(matches) > 0 else 0

def get_state_segments(states):
    deltas = np.diff(states, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    minima_indices = argrelextrema(norms, np.less)[0]
    segment_boundaries = np.insert(minima_indices, 0, 0)

    return segment_boundaries.tolist(), []

def get_kinematic_segments(state, prominence=0.005):
    positions = state[:, :3]
    velocities = np.diff(positions, axis=0)
    velocities = np.insert(velocities, 0, 0, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    
    inverted_speeds = -speeds
    minima_indices, _ = find_peaks(inverted_speeds, prominence=prominence)
    
    segment_boundaries = np.insert(minima_indices, 0, 0)
    
    boundaries_with_end = np.append(segment_boundaries, len(state))
    segment_lengths = np.diff(boundaries_with_end).tolist()
    
    return segment_boundaries.tolist(), segment_lengths

def get_geometric_segments(state, epsilon=0.015):
    positions = state[:, :3] 
    simplified_points = rdp(positions, epsilon=epsilon)
    
    segment_boundaries = []
    search_start_idx = 0
    
    for point in simplified_points:
        match_idx = np.where(np.all(positions[search_start_idx:] == point, axis=1))[0][0]
        actual_idx = search_start_idx + match_idx
        segment_boundaries.append(actual_idx)
        search_start_idx = actual_idx
        
    segment_boundaries = np.array(segment_boundaries)
    segment_lengths = np.diff(segment_boundaries).tolist()
    
    return segment_boundaries.tolist(), segment_lengths

def segment_to_move(segment_state):
    if len(segment_state) < 2:
        return None
        
    delta = segment_state[-1] - segment_state[0]

    move = []
    base_threshold = 0.05 * len(segment_state)/2 # "half" of segment should be action
    
    components = [
        (abs(delta[0]), "forward" if delta[0] > 0 else "backward"),
        (abs(delta[1]), "right" if delta[1] > 0 else "left"),
        (abs(delta[2]), "up" if delta[2] > 0 else "down"),
        (abs(delta[3])/5, "tilt up" if delta[3] > 0 else "tilt down"),
        (abs(delta[4])/5, "turn right" if delta[4] > 0 else "turn left"),
        (abs(delta[5])/5, "rotate clockwise" if delta[5] > 0 else "rotate counter-clockwise")
    ]
    components.sort(key=lambda x: x[0], reverse=True)
    
    primary_mag, primary_dir = components[0]
    
    if primary_mag < 0.2 * base_threshold:
        move.append(f"slightly {primary_dir}")
    elif primary_mag < 0.3 * base_threshold:
        move.append(primary_dir)
    else:
        move.append(f"strongly {primary_dir}")
        
    for other_mag, other_dir in components[1:]:
        if other_mag < 0.15 * base_threshold:
            continue
        elif other_mag < 0.2 * base_threshold:
            move.append(f"slightly {other_dir}")
        elif other_mag < 0.3 * base_threshold:
            move.append(other_dir)
        else:
            move.append(f"strongly {other_dir}")
        
    return ", ".join(move)

def get_segment_dicts(observation_batch, action_batch, segment_boundaries):
    subtask_dict = {}
    move_dict = {}
    for i, seg in enumerate(segment_boundaries):
        seg_end = segment_boundaries[i+1] if i+1 < len(segment_boundaries) else len(observation_batch)
        if seg > 0 and action_batch[seg-1, -1] == -1 and action_batch[seg, -1] == 1: # open to close
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

        subtask_dict[seg] = None
        move_segments, _ = get_geometric_segments(observation_batch[seg:seg_end])
        move_segments = [mov for mov in move_segments if 5 <= mov <= (seg_end - seg) - 5]
        move_segments.insert(0, 0)
        for j in range(len(move_segments)):
            mov = move_segments[j]
            mov_end = move_segments[j+1] if j+1 < len(move_segments) else (seg_end - seg)
            move_dict[seg + mov] = segment_to_move(observation_batch[seg + mov:seg + mov_end])

    subtask_dict = dict(sorted(subtask_dict.items()))
    move_dict = dict(sorted(move_dict.items()))

    return subtask_dict, move_dict


def main():
    REPO_NAME = "HuggingFaceVLA/libero"
    dataset = LeRobotDataset(REPO_NAME)
    
    fast_action_dataset = dataset.hf_dataset.select_columns(["action", "observation.state"]).with_format("numpy")

    all_subtask_segments = {}
    all_lengths = []

    for ep_idx, ep in enumerate(dataset.meta.episodes):
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        action_batch = fast_action_dataset[ep_start:ep_end]["action"]
        observation_batch = fast_action_dataset[ep_start:ep_end]["observation.state"]
        
        gripper_actions = action_batch[:, -1].astype(int)

        subtask_segments, _ = get_gripper_segments(gripper_actions)

        if len(subtask_segments) < 2:
            subtask_segments, _ = get_kinematic_segments(observation_batch)
        
        print(f"Subtask segments for episode {ep_idx} ({ep['tasks'][0]})")
        subtask_dict, move_dict = get_segment_dicts(observation_batch, action_batch, subtask_segments)
        subtask_lengths = np.diff(np.append(np.array(list(subtask_dict.keys())), len(action_batch))).tolist()

        # Save the task name, count, indices, and lengths to our dictionary
        all_subtask_segments[ep_idx] = {
            "task": ep["tasks"][0],
            "num_segments": len(subtask_dict.keys()),
            "subtask_dict": subtask_dict,
            "move_dict": move_dict,
            "subtask_lengths": subtask_lengths
        }
        
        subtask_segments = list(subtask_dict.keys())

        all_lengths.extend(subtask_lengths)

    # --- Save JSON ---
    json_output_path = "./subtask_segments_new.json"
    with open(json_output_path, "w") as f:
        json.dump(all_subtask_segments, f, indent=4)
    print(f"Successfully saved all segments, tasks, counts, and lengths to {json_output_path}")

    # --- Create and Save Plot ---
    plt.figure(figsize=(10, 6))
    
    # Plot histogram with auto-calculated bins based on the data
    plt.hist(all_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.title("Distribution of Subtask Segment Lengths", fontsize=14)
    plt.xlabel("Segment Length (Frames)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    
    # Save the figure to disk
    plot_output_path = "./subtask_lengths_new.png"
    plt.savefig(plot_output_path, bbox_inches='tight')
    plt.close() # Close to free up memory
    print(f"Successfully saved length distribution plot to {plot_output_path}")

if __name__ == "__main__":
    main()