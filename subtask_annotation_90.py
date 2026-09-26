import json
import torch
import numpy as np
import re
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import ast
import time
import argparse
import math
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
tf.config.set_visible_devices([], 'GPU')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=0, help="Index of this shard (0 to num_shards-1)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of parallel instances")
    return parser.parse_args()

def main():
    args = parse_args()
    
    model_name = "Qwen/Qwen3-VL-32B-Instruct"

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"

    # device_map="auto" will automatically span across the GPUs made visible to this process
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    json_input_path = "./subtask_segments_dir_merge_new_minima_90.json"
    with open(json_input_path, "r") as f:
        segment_data = json.load(f)

    # Save to a shard-specific file to prevent write conflicts
    json_output_path = f"./subtask_annotations_dir_90_{args.shard_id}.json"
    new_segment_data = {}

    BATCH_SIZE = 4
    batch_messages = []
    batch_indices = []

    # --- TFDS DATASET LOADING ---
    builder = tfds.builder_from_directory("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/ecot-lite/data/embodied_features_and_demos_libero/libero_lm_90/1.0.0")
    dataset = builder.as_dataset(split='train').prefetch(tf.data.AUTOTUNE)
    total_episodes = len(dataset)

    # --- SEQUENTIAL SHARDING LOGIC ---
    # Assign a sequential chunk of episodes to each shard
    chunk_size = math.ceil(total_episodes / args.num_shards)
    start_idx = args.shard_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_episodes)
    
    shard_ep_indices_set = set(range(start_idx, end_idx))
    print(f"Shard {args.shard_id}/{args.num_shards} processing {len(shard_ep_indices_set)} episodes (from index {start_idx} to {end_idx-1})")

    processed_in_shard = 0

    # Iterate over TFDS sequentially
    for ep_idx, ep in enumerate(dataset):
        # Skip if this episode doesn't belong to the current shard
        if ep_idx not in shard_ep_indices_set:
            continue
            
        processed_in_shard += 1
        
        # Batch and load this specific episode into memory
        batched_steps = ep["steps"].batch(100000) 
        step_data = next(iter(tfds.as_numpy(batched_steps)))
        
        task = step_data["language_instruction"][0].decode('utf-8')
        
        episode_images = step_data["observation"]["image"]
        
        ep_segments = segment_data[str(ep_idx)]
        original_keys = list(ep_segments["subtask_dict"].keys())
        model_subtask_dict = {str(k+1): value for k, value in enumerate(ep_segments["subtask_dict"].values())}
        segment_lengths = ep_segments["subtask_lengths"]
        num_segments = ep_segments["num_segments"]

        max_trajectory_frames = 25
        frames_per_seg = [max(2, int(round(max_trajectory_frames * (length / sum(segment_lengths))))) for length in segment_lengths]

        message_content = []
        
        for j, (orig_key, seg_length) in enumerate(zip(original_keys, segment_lengths)):
            message_content.append({"type": "text", "text": f"Segment {str(j+1)} ({seg_length / 10:.2f} seconds):"})
        
            num_frames = frames_per_seg[j]
            fracs = [0.0, 1.0] if num_frames == 2 else [f / (num_frames - 1) for f in range(num_frames)]
            
            # Fetch specific frames from the numpy array and convert to PIL directly
            for frac in fracs:
                frame_idx = int(orig_key) + round((seg_length - 1) * frac)
                # Clamp safely to prevent out-of-bounds
                frame_idx = min(frame_idx, len(episode_images) - 1)
                
                img_array = np.flip(episode_images[frame_idx], axis=1)
                
                # Convert numpy array (H, W, C) straight to PIL Image
                pil_img = Image.fromarray(img_array)
                message_content.append({"type": "image", "image": pil_img})

        prompt = (
            f"The above segments show a robot manipulation trajectory that completes the task: '{task}'. "
            f"Decompose the task into {num_segments} subtasks based on the segments provided in sequential order. Pay attention to the robot hand and identify which subtask it is performing in each segment."
            f"You should output strictly in a Python dictionary format: {{segment_number: 'subtask', ...}}, where segment_number is an integer in [1, {num_segments}]. "
            "Note that there are only a few long segments, so do NOT include any micro-subtasks like 'lift'. Instead, subtasks should capture the whole segment and focus on logical steps crucial for completing the task."
        )

        grasp_segments = [int(k) for k, v in model_subtask_dict.items() if "grasp" in v.lower()]
        if grasp_segments:
            prompt += f" Hint: Segments {grasp_segments} involve grasping. Start the corresponding subtasks with 'grasp'. Remember to approach/move towards the object in subtasks {[k-1 for k in grasp_segments]}."
            prompt += "\nExample 1: Task 'put the white mug on the plate and put the chocolate pudding to the right of the plate' with 7 segments, Output: {1: 'approach the white mug', 2: 'grasp the white mug', 3: 'place the white mug on the plate', 4: 'move towards the chocolate pudding', 5: 'grasp the chocolate pudding', 6: 'place the chocolate pudding to the right of the plate', 7: 'push the chocolate pudding to the right of the plate'}\n"
            prompt += "\nExample 2: Task 'open the top drawer and put the bowl inside' with 3 segments, Output: {1: 'move towards the top drawer, open the top drawer and approach the bowl', 2: 'grasp the bowl', 'put the bowl inside the top drawer'}\n"
        else:
            prompt += " Do NOT include any subtasks starting with 'grasp' or 'lift'"
            prompt += "\nExample: Task 'push the plate to the front of the stove' with 2 segments, Output: {1: 'move towards the plate', 2: 'push the plate to the front of the stove'}\n"

        message_content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": message_content}]
        
        batch_messages.append(messages)
        batch_indices.append(ep_idx)

        # Trigger inference if batch is full OR if we've hit the last episode in this shard
        is_last_in_shard = (processed_in_shard == len(shard_ep_indices_set))
        if len(batch_messages) == BATCH_SIZE or is_last_in_shard:
            
            texts = [
                processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for msg in batch_messages
            ]

            image_inputs, video_inputs = process_vision_info(batch_messages)

            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            for attempt in range(3):
                try:
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.95,
                        top_k=20,
                        min_p=0.0,
                        repetition_penalty=1.0
                    )
                        
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    
                    output_texts = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                    
                    for b_idx, out_text in zip(batch_indices, output_texts):
                        match = re.search(r'\{.*\}', out_text, re.DOTALL)
                        if match:
                            dict_str = match.group(0)
                            try:
                                output_dict = ast.literal_eval(dict_str)
                            except (ValueError, SyntaxError):
                                output_dict = {}
                        else:
                            output_dict = {}

                        new_ep_segment = segment_data[str(b_idx)].copy()
                        # Use dict() safely with zip instead of direct .keys() modification
                        keys = list(new_ep_segment["subtask_dict"].keys())
                        values = list(output_dict.values())
                        # Pad values if VLM didn't return enough to prevent zip truncation errors
                        if len(values) < len(keys):
                            values.extend([""] * (len(keys) - len(values)))
                        
                        new_ep_segment["subtask_dict"] = dict(zip(keys, values[:len(keys)]))

                        new_segment_data[str(b_idx)] = new_ep_segment
                    
                    with open(json_output_path, "w") as f:
                        json.dump(new_segment_data, f, indent=4)
                    
                    break # Success, break out of retry loop
                except Exception as e:
                    print(f"Failed on batch ending at ep_idx {ep_idx}, attempt {attempt+1}: {e}")
                    time.sleep(10)

            batch_messages = []
            batch_indices = []

if __name__ == "__main__":
    main()