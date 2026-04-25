from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from sklearn.cluster import HDBSCAN
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import json

import numpy as np

def reduce_trajectory(segments, lengths, states, threshold=25):
    """
    Applies morphological closing using highly optimized parallel lists 
    instead of dictionary copying.
    """
    if not segments or not lengths:
        return [], []

    let_go_state = states[0]
    hold_state = -let_go_state

    def filter_state(s_arr, l_arr, st_arr, target_state, max_noise_len):
        res_starts, res_lengths, res_states = [], [], []
        
        for s, l, st in zip(s_arr, l_arr, st_arr):
            # Flip small segments matching the target state
            if st == target_state and l <= max_noise_len:
                st = -target_state
                
            # Merge with previous if states match
            if res_states and res_states[-1] == st:
                res_lengths[-1] += l
            else:
                res_starts.append(s)
                res_lengths.append(l)
                res_states.append(st)
                
        return res_starts, res_lengths, res_states

    # Pass 1: Fill small "Let Go" gaps
    s1, l1, st1 = filter_state(segments, lengths, states, let_go_state, threshold)
    
    # Pass 2: Remove small "Hold" islands
    s2, l2, _ = filter_state(s1, l1, st1, hold_state, threshold)
    
    return s2, l2


def get_gripper_segments(gripper_actions, noise_threshold=25):
    # FAST EXTRACTION: Calculate state changes without appending arrays
    diffs = gripper_actions[:-1] != gripper_actions[1:]
    segment_indices = np.where(diffs)[0] + 1
    segment_indices = np.insert(segment_indices, 0, 0)
    
    # FAST LENGTHS: Calculate lengths using pre-allocated logic to avoid np.append overhead
    segment_lengths = np.empty(len(segment_indices), dtype=int)
    if len(segment_indices) > 1:
        segment_lengths[:-1] = np.diff(segment_indices)
    segment_lengths[-1] = len(gripper_actions) - segment_indices[-1]
    
    # Extract states directly via numpy indexing
    states = gripper_actions[segment_indices]

    # Convert to standard Python lists for the reduction loop (faster for dynamic appending)
    reduced_indices, reduced_lengths = reduce_trajectory(
        segment_indices.tolist(), 
        segment_lengths.tolist(), 
        states.tolist(), 
        threshold=noise_threshold
    )
    
    return reduced_indices, reduced_lengths


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

def get_hdbscan_segments(poses, max_clusters=2):
    def spatio_temporal_distance(point1, point2):
        return euclidean(point1[:-1], point2[:-1]) + abs(point1[-1] - point2[-1])

    x = [np.append(pose, i / 30.0) for i, pose in enumerate(poses)]

    # Calculate distance matrix using your custom metric
    dist_matrix = pairwise_distances(x, metric=spatio_temporal_distance)
    
    # Agglomerative clustering natively supports n_clusters
    clustering = AgglomerativeClustering(
        n_clusters=max_clusters, 
        metric='precomputed', 
        linkage='average'
    )
    raw_labels = clustering.fit_predict(dist_matrix)

    # Note: Agglomerative doesn't produce "-1" noise, so forward filling 
    # isn't strictly necessary, but you can keep the rest of your logic identical.
    
    segment_indices = np.where(raw_labels[:-1] != raw_labels[1:])[0] + 1
    segment_indices = np.insert(segment_indices, 0, 0).tolist()
    segment_lengths = np.diff(np.append(segment_indices, len(poses))).tolist()

    return segment_indices, segment_lengths


def main():
    REPO_NAME = "meituan/LIBERO-X" # "HuggingFaceVLA/libero"
    dataset = LeRobotDataset(REPO_NAME)
    
    # Isolate the column and format to NumPy directly for speed
    fast_action_dataset = dataset.hf_dataset.select_columns(["action"]).with_format("numpy")

    # Dictionaries and lists to hold our results
    all_segments = {}
    all_lengths = []

    # Add enumerate to get the ep_idx
    for ep_idx, ep in enumerate(dataset.meta.episodes):
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        # Fetch the chunk
        action_batch = fast_action_dataset[ep_start:ep_end]["action"]
        
        # Slice the gripper state
        gripper_actions = action_batch[:, -1].astype(int)
        
        # Run the optimized morphological closing
        segment_indices, segment_lengths = get_gripper_segments(gripper_actions, noise_threshold=0)

        # if len(segment_indices) < 2:
        #     segment_indices, segment_lengths = get_hdbscan_segments(action_batch[:, :-1], max_clusters=2)
            

        # Save the task name, count, indices, and lengths to our dictionary
        all_segments[ep_idx] = {
            "task": ep["tasks"][0],
            "num_segments": len(segment_indices),
            "indices": segment_indices,
            "lengths": segment_lengths
        }
        
        # Add to our lengths distribution for the plot
        all_lengths.extend(segment_lengths)

        # Optional: Print progress
        print(f"Episode {ep_idx}: {ep['tasks'][0]} | Segments: {len(segment_indices)}")

    # --- Save JSON ---
    json_output_path = "./subtask_segments_libero_x_0.json"
    with open(json_output_path, "w") as f:
        json.dump(all_segments, f, indent=4)
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
    plot_output_path = "./subtask_lengths_libero_x_0.png"
    plt.savefig(plot_output_path, bbox_inches='tight')
    plt.close() # Close to free up memory
    print(f"Successfully saved length distribution plot to {plot_output_path}")

if __name__ == "__main__":
    main()