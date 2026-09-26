import json
import os
from pathlib import Path
import builtins
from PIL import Image

from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms

from tensorflow_datasets import image
from prismatic import load

import torch
import torch.nn as nn
from typing import Any, Optional
from typing import TYPE_CHECKING

from lerobot.configs import PreTrainedConfig
from prismatic.models.vlms.prismatic import PrismaticVLM
from ..pretrained import PreTrainedPolicy, T
from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_SUBTASK_ATTENTION_MASK, OBS_LANGUAGE_SUBTASK_TOKENS, OBS_LANGUAGE_TOKENS

IGNORE_INDEX = -100
from .configuration_openvla import OpenVLAConfig

class OpenVLAPolicy(PreTrainedPolicy):
    config_class = OpenVLAConfig
    name = "openvla"
    debug = True

    def __init__(self, config: OpenVLAConfig, **kwargs):
        super().__init__(config)
        self.config = config
        config.validate_features()

        self.model = load(
            config.pretrained_checkpoint,
            hf_token=os.environ.get("HF_TOKEN"),
            load_for_training=self.training,
        )

        self.model.vision_backbone.to(dtype=self.model.vision_backbone.half_precision_dtype)
        self.model.llm_backbone.to(dtype=self.model.llm_backbone.half_precision_dtype)
        self.model.to(dtype=self.model.llm_backbone.half_precision_dtype)

        if config.gradient_checkpointing:
            self.model.llm_backbone.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.model.to(config.device)

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

    def reset(self):
        """Reset per-episode state."""
        # OpenVLA is generally stateless between single steps unless acting recurrently
        pass

    def get_optim_params(self) -> dict:
        """Return parameters to pass to the optimizer."""
        return self.parameters()

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return self.select_action(batch, **kwargs)

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Return a single action for the current timestep."""
        self.model.eval()
        
        prompt_text = batch["task"][0]
        prompt_text +=  "RATIONALE:" if self.config.reason else "COMMAND:"
        # prompt_text = f"In: close the top drawer of the cabinet.\nOut: COMMAND:"

        # if not torch.all(input_ids[:, -1] == 29871):
        #     input_ids = torch.cat(
        #         (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        #     )

        to_pil = transforms.ToPILImage()
        image_transform = self.model.vision_backbone.get_image_transform()

        img = batch["observation.images.image"][0]
        img = to_pil(img)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)  # resize to size seen at train time
        img = img.convert("RGB")

        device = next(self.model.parameters()).device
        model_dtype = next(self.model.vision_backbone.parameters()).dtype

        pixel_values = [image_transform(img)]

        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(device=device, dtype=model_dtype) 
            
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(1)]).to(device=device, dtype=model_dtype) 
                for k in pixel_values[0]
            }

        if not hasattr(self.model.__class__, "_is_stateful"):
            self.model.__class__._is_stateful = False

        generated_text = self.model.generate_batch(
            pixel_values=pixel_values,
            texts=[prompt_text],
            max_new_tokens=128,
        )[0]

        generated_text = generated_text.split("COMMAND:")[-1].split("\n")[0].strip(".").lstrip(" ")
        return generated_text
    
    def _preprocess_images(self, batch: dict[str, torch.Tensor], input_ids):
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        """

        device = next(self.parameters()).device
        model_dtype = next(self.model.vision_backbone.parameters()).dtype

        present_img_keys = [key for key in self.config.image_features if key in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        images = batch[present_img_keys[0]]

        # Ensure tensor is on the same device as the model
        if images.device != device:
            images = images.to(device)

        to_pil = transforms.ToPILImage()
        
        image_transform = self.model.vision_backbone.get_image_transform()
        pixel_values = []
        for img in images:
            img = to_pil(img)
            img = img.resize((224, 224), Image.Resampling.LANCZOS)  # resize to size seen at train time
            img = img.convert("RGB")
            pixel_values.append(image_transform(img))

        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(device=device, dtype=model_dtype) 
            
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(input_ids))]).to(device=device, dtype=model_dtype) 
                for k in pixel_values[0]
            }
            
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")
        
        return pixel_values

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """Compute the training loss for OpenVLA (typically Next-Token Prediction Causal LM Loss)."""
        tasks = batch["task"]
        subtasks = batch["subtask"]
            
        eos_token = "</s>"
        tokenizer = self.model.llm_backbone.tokenizer
        input_ids_list = []
        labels_list = []
        for task, subtask in zip(tasks, subtasks):

            num_answer_tokens = len(tokenizer(subtask)["input_ids"])

            input_ids = tokenizer(task + subtask + eos_token, add_special_tokens=True).input_ids

            labels = list(input_ids)

            input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

            num_end_tokens = 1
            labels[: -(num_answer_tokens + num_end_tokens)] = -100

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        input_ids = input_ids[:, : self.config.tokenizer_max_length]
        labels = labels[:, : self.config.tokenizer_max_length]

        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        pixel_values = self._preprocess_images(batch, input_ids)

        # =====================================================================
        # ADD THESE LINES: Move newly created tensors to the model's device
        # =====================================================================
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)
        # =====================================================================

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
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