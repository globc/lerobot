import json
import torch
import numpy as np
import re
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from torchvision.transforms import ToPILImage
import ast
import time
import argparse
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=0, help="Index of this shard (0 to num_shards-1)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of parallel instances")
    return parser.parse_args()

def main():
    args = parse_args()
    
    REPO_NAME = "HuggingFaceVLA/libero"
    dataset = LeRobotDataset(REPO_NAME)

    model_name = "Qwen/Qwen3.5-27B"

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"

    # device_map="auto" will automatically span across the GPUs made visible to this process
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    json_input_path = "./subtask_segments.json"
    with open(json_input_path, "r") as f:
        all_segments = json.load(f)

    # Save to a shard-specific file to prevent write conflicts
    json_output_path = f"./subtask_annotations_{args.shard_id}.json"
    all_subtasks = {}

    BATCH_SIZE = 4
    batch_messages = []
    batch_indices = []
    to_pil = ToPILImage()

    # --- SORTING & DATA SHARDING LOGIC ---
    total_episodes = len(dataset.meta.episodes)
    
    # Sort episode indices based on num_segments in descending order
    sorted_ep_indices = sorted(
        range(total_episodes),
        key=lambda idx: all_segments[str(idx)]["num_segments"],
        reverse=True
    )

    chunk_size = math.ceil(total_episodes / args.num_shards)
    start_idx = args.shard_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_episodes)
    
    # Extract just the episodes this specific shard is responsible for
    shard_ep_indices = sorted_ep_indices[start_idx:end_idx]
    
    print(f"Shard {args.shard_id}/{args.num_shards} processing {len(shard_ep_indices)} episodes (from sorted index {start_idx} to {end_idx-1})")

    # Iterate ONLY over the sorted chunk assigned to this shard
    for i, ep_idx in enumerate(shard_ep_indices):
        ep = dataset.meta.episodes[ep_idx]
        
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]

        ep_segments = all_segments[str(ep_idx)]
        segment_indices = ep_segments["indices"]
        segment_lengths = ep_segments["lengths"]
        num_segments = ep_segments["num_segments"]

        task = ep["tasks"][0]

        prompt = (
            f"The following {num_segments} segments show a robot manipulation trajectory that completes the task: '{task}'. "
            "Based on the segments provided in sequential order, pay attention to the robot hand and identify which subtask it is performing in each segment. "
            f"You should output strictly in a Python dictionary format: {{segment_number: 'subtask', ...}}, where segment_number is an integer in [1, {num_segments}]. "
            "Do NOT include subtasks that are just picking, grasping, lifting, etc. Make sure the robot hand was moved to an object before interacting with it."
        )

        message_content = [{"type": "text", "text": prompt}]
        
        for j, (seg_idx, seg_length) in enumerate(zip(segment_indices, segment_lengths)):
            message_content.append({"type": "text", "text": f"Segment {str(j+1)}:"})
            
            images = [dataset.hf_dataset[ep_start + seg_idx + round(seg_length * frac)]['observation.images.image'] for frac in [0.0, 0.2, 0.4, 0.6, 0.8]]
            images = [img.clamp(0, 1).mul(255.0).round().to(torch.uint8) for img in images]
            
            message_content.extend([{"type": "image", "image": to_pil(img)} for img in images])
        
        messages = [{"role": "user", "content": message_content}]
        
        batch_messages.append(messages)
        batch_indices.append(ep_idx)

        # Trigger inference if batch is full, OR if this is the very last episode in THIS shard
        if len(batch_messages) == BATCH_SIZE or i == len(shard_ep_indices) - 1:
            
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
                                output_dict = {"error": "Failed to parse matched dictionary", "raw_output": out_text}
                        else:
                            output_dict = {"error": "No dictionary found in output", "raw_output": out_text}
                            
                        all_subtasks[str(b_idx)] = output_dict
                    
                    with open(json_output_path, "w") as f:
                        json.dump(all_subtasks, f, indent=4)
                    
                    break # Success, break out of retry loop
                except Exception as e:
                    print(f"Failed on batch ending at ep_idx {ep_idx}, attempt {attempt+1}: {e}")
                    time.sleep(10)

            batch_messages = []
            batch_indices = []

if __name__ == "__main__":
    main()