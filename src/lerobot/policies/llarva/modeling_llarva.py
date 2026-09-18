from .configuration_llarva import LlarvaConfig 
from lerobot.configs import PreTrainedConfig 
from ..pretrained import PreTrainedPolicy, T 
import torchvision.transforms as transforms 

from typing import TypedDict, Unpack 
import builtins 
from pathlib import Path 
import torch 
import torch.nn.functional as F 
from torch import Tensor 
import re 
import dataclasses 
from enum import auto, Enum 
from typing import List 
import base64 
from io import BytesIO 
from PIL import Image 
import json 
import torch 
from transformers import AutoProcessor, LlavaForConditionalGeneration 
import numpy as np 

def parse_points(data_string): 
    matches = re.findall(r"\[[^\[\]]+\]", data_string) 
    parsed_points = [] 
    
    for m in matches: 
        # Extract all numbers (handles integers, decimals, and negative signs) 
        nums = re.findall(r"-?\d+(?:\.\d+)?", m) 
        # Only append if we successfully parsed at least an X and Y coordinate 
        if nums and len(nums) >= 2: 
            parsed_points.append([float(nums[0]), float(nums[1])]) 
            
    return parsed_points 

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

class ActionSelectKwargs(TypedDict, total=False): 
    temperature: float | None 
    reduction: str | None 

class LlarvaPolicy(PreTrainedPolicy): 
    name = "llarva" 
    config_class = LlarvaConfig 

    def __init__( 
        self, 
        config: LlarvaConfig, 
        **kwargs, 
    ): 
        super().__init__(config) 
        self.config = config 
        processor_id = "llava-hf/llava-1.5-7b-hf" 

        # --- Load Model and Processor --- 
        print("Loading processor and model...") 
        self.processor = AutoProcessor.from_pretrained(processor_id) 
        
        self.model = LlavaForConditionalGeneration.from_pretrained( 
            self.config.model_name,  
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            attn_implementation="flash_attention_2" 
        ).to(self.config.device) 
        self.model.resize_token_embeddings(len(self.processor.tokenizer)) 

        if self.config.gradient_checkpointing: 
            self.model.gradient_checkpointing_enable() 
            self.model.config.use_cache = False 
            self.model.enable_input_require_grads() 

        self.reset() 

        self.reset() 

    @classmethod 
    def from_pretrained( 
        cls: builtins.type[T], 
        pretrained_name_or_path: str | Path, 
        *, 
        config: PreTrainedConfig | None = None, 
        force_download: bool = False, 
        resume_download: bool | None = None, 
        proxies: dict | None = None, 
        token: str | bool | None = None, 
        cache_dir: str | Path | None = None, 
        local_files_only: bool = False, 
        revision: str | None = None, 
        strict: bool = False, 
        **kwargs, 
    ) -> T: 
        """ 
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are 
        deactivated). To train it, you should first set it back in training mode with `policy.train()`. 
        """ 
        if config is None: 
            config = PreTrainedConfig.from_pretrained( 
                pretrained_name_or_path=pretrained_name_or_path, 
                force_download=force_download, 
                resume_download=resume_download, 
                proxies=proxies, 
                token=token, 
                cache_dir=cache_dir, 
                local_files_only=local_files_only, 
                revision=revision, 
                **kwargs, 
            ) 

        model = cls(config, **kwargs) 

        return model 

    @torch.no_grad() 
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor: 
        self.model.eval() 
        
        prompt = batch["messages"][0] 
        print(prompt) 
        to_pil = transforms.ToPILImage() 

        image = batch["observation.images.image"][0] 
        image = to_pil(image) 
        image = image.convert("RGB") 

        device = next(self.model.parameters()).device 

        inputs = self.processor( 
            text=prompt,  
            images=image,  
            return_tensors="pt" 
        ).to(device) 

        if "pixel_values" in inputs: 
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype) 

        # --- Generate --- 
        print("\nGenerating action...") 
        with torch.inference_mode(): 
            output_ids = self.model.generate( 
                **inputs, 
                max_new_tokens=128 if not self.config.zero_shot else 1024, 
                do_sample=False, 
                use_cache=True 
            ) 

        input_len = inputs["input_ids"].shape[1] 
        output = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip() 
        print(f"LLARVA output: {output}") 
        
        if "The trajectory: " in output: 
            output = output.split("The trajectory: ")[-1] 
            points = parse_points(output)[:self.config.chunk_size] 
            
            if not self.config.zero_shot: 
                points = points # self.unnormalize_trace(points) 
            
            return get_trace(points) 
        
        return output 

    @torch.no_grad() 
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor: 
        return None 

    def get_optim_params(self) -> dict: 
        return self.parameters() 

    def reset(self): 
        pass 

    def unnormalize_trace(self, normalized_trace, target_resolution=(256, 256), source_resolution=(224, 224)): 
        """ 
        Reverses the normalization of a trace to match the original PIL Image's resolution. 
        """ 
        img_width, img_height = target_resolution 
        source_width, source_height = source_resolution 

        unnormalized_trace = [] 
        for point in normalized_trace: 
            if len(point) >= 2: 
                orig_x = round((point[0] / source_width) * img_width) 
                orig_y = round((point[1] / source_height) * img_height) 
                unnormalized_trace.append([orig_x, orig_y]) 

        return unnormalized_trace 

    def normalize_trace(self, trace, source_resolution=(256, 256), target_resolution=(224, 224)): 
        """ 
        Normalizes a trace based on a PIL Image's resolution. 
        """ 
        img_width, img_height = source_resolution 
        target_width, target_height = target_resolution 

        normalized_trace = [] 
        for point in trace: 
            if len(point) >= 2: 
                norm_x = round((point[0] / img_width) * target_width) 
                norm_y = round((point[1] / img_height) * target_height) 
                normalized_trace.append([norm_x, norm_y]) 

        return normalized_trace 


    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]: 
        text_inputs = batch["messages"] 
        to_pil = transforms.ToPILImage() 

        images = batch["observation.images.image"][:, -1, :, :, :] 
        images = [to_pil(img).convert("RGB") for img in images] 
        traces = batch["subtask"] 

        target_texts = [] 
        for trace in traces: 
            normalized_trace = trace # normalized_trace = self.normalize_trace(parse_points(trace)) 
            target_texts.append(f"The next action step: []. The trajectory: {normalized_trace}") 

        full_texts = [prompt + " " + target for prompt, target in zip(text_inputs, target_texts)] 
        
        device = next(self.model.parameters()).device 
        
        # Ensure tokenizer has a pad token (fallback to eos_token if missing) 
        if self.processor.tokenizer.pad_token is None: 
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token 

        # 2. Process the full sequences 
        inputs = self.processor( 
            text=full_texts, 
            images=images, 
            return_tensors="pt", 
            padding=True 
        ).to(device) 

        # Match pixel_values dtype with the loaded model (e.g., float16) 
        if "pixel_values" in inputs: 
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype) 

        # 3. Create labels for causal LM loss computation 
        labels = inputs["input_ids"].clone() 

        # Tokenize just the prompts to find out exactly how many tokens they take up 
        prompt_inputs = self.processor( 
            text=text_inputs, 
            images=images, 
            return_tensors="pt", 
            padding=True 
        ) 

        # 4. Mask the prompt and padding tokens with -100 so they are ignored by the loss function 
        for i in range(labels.shape[0]): 
            # Number of non-padding tokens in the prompt 
            prompt_len = prompt_inputs["attention_mask"][i].sum().item() 
            
            # Ignore the prompt tokens 
            labels[i, :prompt_len] = -100 
            
            # Ignore padding tokens in the full sequence 
            pad_mask = inputs["attention_mask"][i] == 0 
            labels[i, pad_mask] = -100 

        # 5. Forward pass through the model 
        output = self.model( 
            **inputs, 
            labels=labels 
        ) 

        detailed_loss_dict = { 
            "loss": output.loss.item() 
        } 
        
        return output.loss, detailed_loss_dict 
    
    def _get_default_peft_targets(self) -> dict[str, any]: 
        """Return default PEFT target modules for PI0Fast fine-tuning.""" 
        return { 
            "target_modules": "all-linear", 
            "modules_to_save": [], 
        }