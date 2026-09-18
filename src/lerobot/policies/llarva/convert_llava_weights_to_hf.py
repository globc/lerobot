import os
import torch
from transformers import LlavaConfig, LlavaForConditionalGeneration, AutoConfig, AutoProcessor
from huggingface_hub import snapshot_download

# --- Configuration ---
text_model_id = "lmsys/vicuna-7b-v1.5"
vision_model_id = "openai/clip-vit-large-patch14-336"
repo_id = "globcy/llarva"           # Can be a HF Hub ID or a local directory path
output_dir = "./llarva_hf"    # Where to save the fixed model locally
hub_repo = "globcy/llarva_hf"       # Where you eventually want to push it

# 1. Generate LLaVA config
print("1. Generating LLaVA config...")
text_config = AutoConfig.from_pretrained(text_model_id)
vision_config = AutoConfig.from_pretrained(vision_model_id).vision_config

config = LlavaConfig(
    text_config=text_config,
    vision_config=vision_config,
    ignore_index=-100,
    image_token_index=32000,
    projector_hidden_act="gelu",
    vision_feature_select_strategy="default",
    vision_feature_layer=-2,
)

# 2. Initialize empty Hugging Face LLaVA model
print("2. Initializing empty Hugging Face LLaVA model...")
model = LlavaForConditionalGeneration(config)

# 3. Fetch your old state dict (handles both single files and shards)
print("3. Fetching your old state dict...")
if os.path.isdir(repo_id):
    repo_path = repo_id
else:
    repo_path = snapshot_download(repo_id=repo_id, allow_patterns=["*.bin", "*.safetensors"])

state_dict = {}
for file in os.listdir(repo_path):
    if file.endswith(".bin") or file.endswith(".safetensors"):
        file_path = os.path.join(repo_path, file)
        print(f"   Loading {file}...")
        if file.endswith(".safetensors"):
            from safetensors.torch import load_file
            shard_dict = load_file(file_path)
        else:
            shard_dict = torch.load(file_path, map_location="cpu", weights_only=True)
        state_dict.update(shard_dict)

# 4. Map the keys to correctly match the strict Hugging Face layout
print("4. Mapping keys to match Hugging Face format...")
new_state_dict = {}
for key, value in state_dict.items():
    
    # 1. Fix the duplicate vision_tower prefix
    if key.startswith("model.vision_tower.vision_tower."):
        new_key = key.replace("model.vision_tower.vision_tower.", "model.vision_tower.")
        
    # 2. Map projector layer 0 to linear_1
    elif key.startswith("model.mm_projector.0."):
        new_key = key.replace("model.mm_projector.0.", "model.multi_modal_projector.linear_1.")
        
    # 3. Map projector layer 2 to linear_2
    elif key.startswith("model.mm_projector.2."):
        new_key = key.replace("model.mm_projector.2.", "model.multi_modal_projector.linear_2.")
        
    # 4. Push the base language model components inside 'model.language_model.'
    elif key.startswith("model.embed_tokens.") or key.startswith("model.layers.") or key.startswith("model.norm."):
        new_key = key.replace("model.", "model.language_model.")
        
    # 5. Leave top-level keys (like lm_head.weight) untouched
    else:
        new_key = key
        
    new_state_dict[new_key] = value

# 5. Inject weights into the empty model
print("5. Injecting weights into model...")
model.load_state_dict(new_state_dict, strict=True)

# 6. Save the properly formatted HF model and processor
print(f"6. Saving Hugging Face model locally to {output_dir}...")
model.save_pretrained(output_dir)

# We load a standard LLaVA 1.5 processor so your output directory is fully equipped
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor.save_pretrained(output_dir)

# Uncomment the lines below if you want the script to automatically push to your Hub repo:
# print("7. Pushing directly to the Hugging Face Hub...")
# model.push_to_hub(hub_repo)
# processor.push_to_hub(hub_repo)

print("\nDone! Conversion successful.")