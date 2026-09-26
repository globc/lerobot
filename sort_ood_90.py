import torch
import numpy as np
import json
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_distances
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ==========================================
# Configuration
# ==========================================
PATH_LIBERO_90 = "globcy/libero_dir_90_val"
PATH_LIBERO_BASE = "globcy/libero_dir"
OUTPUT_JSON_FILE = "libero_ood_90_results.json"

IMAGE_OBS_KEY = "observation.images.image" 

TEXT_WEIGHT = 0.34
VISION_WEIGHT = 0.33
ACTION_WEIGHT = 0.33

# ==========================================
# Helper Functions
# ==========================================
def extract_task_representatives(dataset: LeRobotDataset):
    representatives = {}
    
    num_episodes = getattr(dataset, "num_episodes", None)
    if num_episodes is None:
        num_episodes = dataset.meta.total_episodes

    print(f"Scanning up to {num_episodes} episodes to find unique tasks...")
    
    for ep_idx, ep in enumerate(dataset.meta.episodes):
        start_idx = ep["dataset_from_index"]
        end_idx = ep["dataset_to_index"]

        task_str = ep["tasks"][0]
        task_id = dataset.hf_dataset[start_idx]["libero_id"].item() if "libero_id" in dataset.hf_dataset[start_idx] else task_str
        print(task_id)
        if task_id not in representatives:
            img_tensor = dataset.hf_dataset[start_idx][IMAGE_OBS_KEY]
            if isinstance(img_tensor, torch.Tensor):
                img_np = img_tensor.permute(1, 2, 0).numpy()
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_tensor

            actions = np.array(dataset.hf_dataset[start_idx:end_idx]["action"])

            representatives[task_id] = {
                "text": task_str,
                "image": img_np,
                "actions": actions
            }
            
            if len(representatives) == 90 and "90" in str(dataset.root).lower():
                break

    print(f"-> Found {len(representatives)} unique tasks.\n")
    return representatives

def compute_clip_embeddings(reps_dict, model, processor, device):
    tasks = list(reps_dict.keys())
    texts = [reps_dict[t]["text"] for t in tasks]
    images = [reps_dict[t]["image"] for t in tasks]

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        text_embeds = outputs.text_embeds.cpu().numpy()
        image_embeds = outputs.image_embeds.cpu().numpy()

    return tasks, text_embeds, image_embeds

def compute_action_dtw_matrix(reps_90, reps_base):
    tasks_90 = list(reps_90.keys())
    tasks_base = list(reps_base.keys())
    
    dist_matrix = np.zeros((len(tasks_90), len(tasks_base)))
    
    total_comparisons = len(tasks_90) * len(tasks_base)
    print(f"Computing DTW for {total_comparisons} trajectory pairs...")
    
    for i, t90 in enumerate(tasks_90):
        traj_90 = reps_90[t90]["actions"]
        for j, tbase in enumerate(tasks_base):
            traj_base = reps_base[tbase]["actions"]
            distance, _ = fastdtw(traj_90, traj_base, dist=euclidean)
            dist_matrix[i, j] = distance
            
    return dist_matrix

def min_max_scale(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP model...")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    print("Loading LeRobot datasets...")
    ds_90 = LeRobotDataset(PATH_LIBERO_90)
    ds_base = LeRobotDataset(PATH_LIBERO_BASE)

    print("--- Extracting Libero_90 ---")
    reps_90 = extract_task_representatives(ds_90)
    print("--- Extracting Libero_Base ---")
    reps_base = extract_task_representatives(ds_base)

    print("Generating text and visual embeddings...")
    tasks_90, txt_emb_90, img_emb_90 = compute_clip_embeddings(reps_90, model, processor, device)
    tasks_base, txt_emb_base, img_emb_base = compute_clip_embeddings(reps_base, model, processor, device)

    print("Calculating Semantic and Visual distances...")
    txt_dist_matrix = cosine_distances(txt_emb_90, txt_emb_base)
    img_dist_matrix = cosine_distances(img_emb_90, img_emb_base)
    
    print("Calculating Procedural distances...")
    act_dist_matrix = compute_action_dtw_matrix(reps_90, reps_base)

    min_txt_dists = txt_dist_matrix.min(axis=1)
    min_img_dists = img_dist_matrix.min(axis=1)
    min_act_dists = act_dist_matrix.min(axis=1)

    norm_txt_dists = min_max_scale(min_txt_dists)
    norm_img_dists = min_max_scale(min_img_dists)
    norm_act_dists = min_max_scale(min_act_dists)

    ood_scores = []
    for i, task in enumerate(tasks_90):
        combined_score = (
            (TEXT_WEIGHT * norm_txt_dists[i]) + 
            (VISION_WEIGHT * norm_img_dists[i]) + 
            (ACTION_WEIGHT * norm_act_dists[i])
        )
        
        # VERY IMPORTANT: Cast numpy types to python floats for JSON serialization
        ood_scores.append({
            "task": task,
            "score": float(combined_score),
            "txt_norm": float(norm_txt_dists[i]),
            "img_norm": float(norm_img_dists[i]),
            "act_norm": float(norm_act_dists[i])
        })

    ood_scores.sort(key=lambda x: x["score"], reverse=True)

    # Save to JSON
    with open(OUTPUT_JSON_FILE, "w") as f:
        json.dump(ood_scores, f, indent=4)

    print(f"\nSuccessfully saved results for {len(ood_scores)} tasks to {OUTPUT_JSON_FILE}")