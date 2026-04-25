from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tensorflow as tf
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import bisect

def main():

    REPO_NAME = "globcy/libero_subtask_custom_slow"

    output_path = Path("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.cache/huggingface/lerobot") / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    subtask_vocab = {}
    current_subtask_idx = 0

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": (3, 256, 256),
                "names": ["channel", "height", "width"],
            },
            "observation.images.image2": {
                "dtype": "image",
                "shape": (3, 256, 256),
                "names": ["channel", "height", "width"],
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
            "subtask_index": {
                "dtype": "int64",
                "shape": (1,),
                "names": None,
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    tf.config.set_visible_devices(
        [], device_type="gpu"
    )

    original_dataset = LeRobotDataset("HuggingFaceVLA/libero")

    with open("subtask_segments.json", "r") as f:
        segment_data = json.load(f)
    
    with open("subtask_annotations.json", "r") as f:
        annotation_data = json.load(f)

    for ep_idx, ep in enumerate(original_dataset.meta.episodes):
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        ep_length = ep["length"]

        segment_indices = segment_data[str(ep_idx)]['indices']
        
        ep_subtask_indices = []
        for frame_idx in range(ep_length):
            subtask = annotation_data[str(ep_idx)][str(bisect.bisect_right(segment_indices, frame_idx))]
            if subtask not in subtask_vocab:
                subtask_vocab[subtask] = current_subtask_idx
                current_subtask_idx += 1
            ep_subtask_indices.append(subtask_vocab[subtask])

        episode_data = original_dataset.hf_dataset[ep_start:ep_end]

        for frame_idx in range(ep_length):
            dataset.add_frame(
                {
                    "observation.images.image": episode_data["observation.images.image"][frame_idx],
                    "observation.images.image2": episode_data["observation.images.image2"][frame_idx],
                    "observation.state": episode_data["observation.state"][frame_idx],
                    "action": episode_data["action"][frame_idx],
                    "task": ep["tasks"][0],
                    "subtask_index": np.array([ep_subtask_indices[frame_idx]], dtype=np.int64),
                }
            )
        dataset.save_episode()

    dataset.finalize()

    subtasks_df = pd.DataFrame([
        {"subtask": label, "subtask_index": idx}
        for label, idx in subtask_vocab.items()
    ])
    subtasks_df.set_index("subtask", inplace=True)

    meta_dir = output_path / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    subtasks_df.to_parquet(
        meta_dir / "subtasks.parquet",
        engine="pyarrow",
        compression="snappy"
    )

    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    dataset.push_to_hub(
        tags=["libero", "panda", "rlds"],
        private=False,
        push_videos=True,
        license="apache-2.0",
        # upload_large_folder=True,
    )

if __name__ == "__main__":
    main()
