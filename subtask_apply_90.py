from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tensorflow as tf
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import bisect
import h5py

from libero.libero.benchmark.libero_suite_task_map import libero_task_map


def main():
    REPO_NAME = "globcy/libero_dir_90_val"

    output_path = Path("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.cache/huggingface/lerobot") / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Maintain two separate vocabularies
    subtask_vocab = {}
    move_vocab = {}
    init_states = {}
    current_subtask_idx = 0
    current_move_idx = 0
    current_init_state_idx = 0

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "init_state_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            "libero_id": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            "subtask_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
            # Add a dedicated feature for moves
            "move_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    tf.config.set_visible_devices([], device_type="gpu")

    with open("subtask_annotations_dir_90.json", "r") as f:
        annotation_data = json.load(f)

    builder = tfds.builder_from_directory("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/data/embodied_features_and_demos_libero/libero_lm_90/1.0.0")
    raw_dataset = builder.as_dataset(split='train').prefetch(tf.data.AUTOTUNE)

    ep_path_counts = {}
    ep_idx = 0
    for ep in raw_dataset:
        ep_path = ep["episode_metadata"]["file_path"].numpy().decode("utf-8")
        demo_id = str(ep["episode_metadata"]["demo_id"].numpy())
        
        if ep_path_counts.get(ep_path, 0) >= 10:
            continue
        ep_path_counts[ep_path] = ep_path_counts.get(ep_path, 0) + 1

        orig_data_path = Path("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/libero_100/libero_90") / ep_path
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        demo_data = orig_data[f"demo_{demo_id}"]
        init_state = tuple(demo_data["states"][()][0].tolist())
        libero_id = libero_task_map["libero_90"].index(ep_path.removesuffix("_demo.hdf5"))

        if init_state not in init_states:
            init_states[init_state] = current_init_state_idx
            current_init_state_idx += 1

        batched_steps = ep["steps"].batch(100000) # Ensure it encompasses max episode length
        step_data = next(iter(tfds.as_numpy(batched_steps)))
        
        action_batch = step_data["action"]
        observation_batch = step_data["observation"]["state"]
        image_batch = step_data["observation"]["image"]
        wrist_image_batch = step_data["observation"]["wrist_image"]
        task = step_data["language_instruction"][0].decode('utf-8')
        ep_length = len(action_batch)

        ep_segments = annotation_data[str(ep_idx)]
        subtask_dict = ep_segments["subtask_dict"]
        move_dict = ep_segments["move_dict"]
        
        move_segment_indices = sorted([int(k) for k in move_dict.keys()])
        subtask_segment_indices = sorted([int(k) for k in subtask_dict.keys()])
        
        ep_subtask_indices = []
        ep_move_indices = []

        for frame_idx in range(ep_length):
            # Process Subtask
            subtask_idx = bisect.bisect_right(subtask_segment_indices, frame_idx) - 1
            subtask_idx = max(0, subtask_idx)
            subtask = subtask_dict[str(subtask_segment_indices[subtask_idx])]

            if subtask not in subtask_vocab:
                subtask_vocab[subtask] = current_subtask_idx
                current_subtask_idx += 1
            ep_subtask_indices.append(subtask_vocab[subtask])

            # Process Move
            move_idx = bisect.bisect_right(move_segment_indices, frame_idx) - 1
            move_idx = max(0, move_idx)
            move = move_dict[str(move_segment_indices[move_idx])]

            if move not in move_vocab:
                move_vocab[move] = current_move_idx
                current_move_idx += 1
            ep_move_indices.append(move_vocab[move])

        for frame_idx in range(ep_length):
            dataset.add_frame(
                {
                    "observation.images.image": np.flip(image_batch[frame_idx], axis=1),
                    "observation.images.image2": np.flip(wrist_image_batch[frame_idx], axis=1),
                    "observation.state": observation_batch[frame_idx],
                    "action": action_batch[frame_idx],
                    "task": task,
                    "init_state_index": np.array([init_states[init_state]], dtype=np.int64),
                    "libero_id": np.array([libero_id], dtype=np.int64),
                    "subtask_index": np.array([ep_subtask_indices[frame_idx]], dtype=np.int64),
                    "move_index": np.array([ep_move_indices[frame_idx]], dtype=np.int64), # Add to frame
                }
            )
        dataset.save_episode()

        ep_idx += 1

    dataset.finalize()

    # Save both vocabularies to parquet files
    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. Subtasks
    subtasks_df = pd.DataFrame([
        {"subtask": label, "subtask_index": idx}
        for label, idx in subtask_vocab.items()
    ])
    subtasks_df.set_index("subtask", inplace=True)
    subtasks_df.to_parquet(meta_dir / "subtasks.parquet", engine="pyarrow", compression="snappy")

    # 2. Moves
    moves_df = pd.DataFrame([
        {"move": label, "move_index": idx}
        for label, idx in move_vocab.items()
    ])
    moves_df.set_index("move", inplace=True)
    moves_df.to_parquet(meta_dir / "moves.parquet", engine="pyarrow", compression="snappy")

    init_states_df = pd.DataFrame([
        {"init_state": label, "init_state_index": idx}
        for label, idx in init_states.items()
    ])
    init_states_df.set_index("init_state", inplace=True)
    init_states_df.to_parquet(meta_dir / "init_states.parquet", engine="pyarrow", compression="snappy")

    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    dataset.push_to_hub(
        tags=["libero", "panda", "rlds"],
        private=False,
        push_videos=True,
        license="apache-2.0",
    )

if __name__ == "__main__":
    main()