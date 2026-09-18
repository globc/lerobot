from .configuration_qwen import QwenConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from lerobot.configs import PreTrainedConfig
from ..pretrained import PreTrainedPolicy, T

from typing import TypedDict, Unpack
import builtins
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import Tensor

class ActionSelectKwargs(TypedDict, total=False):
    temperature: float | None
    reduction: str | None

class QwenPolicy(PreTrainedPolicy):
    name = "qwen"
    config_class = QwenConfig

    def __init__(
        self,
        config: QwenConfig,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.model_name, 
            torch_dtype=torch.bfloat16, 
            device_map="cuda",
            attn_implementation="flash_attention_2",
            ignore_mismatched_sizes=True
        )
        if self.config.gradient_checkpointing: 
            self.model.gradient_checkpointing_enable() 
            self.model.config.use_cache = False 
            self.model.enable_input_require_grads() 
            
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        self.model.to("cuda")
        self.debug = True

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
        if self.debug:
            print(f"Input Qwen: {batch['messages']}")
            self.debug = False
        inputs = batch["inputs"]
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Hardcoded for 100% deterministic (greedy) decoding
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=256,
            do_sample=False,  # Enforces greedy decoding; temperature/top_p/top_k are ignored and removed
            repetition_penalty=1.0
        )
            
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_texts

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return None

    def get_optim_params(self) -> list[torch.nn.Parameter]:
        """Return only trainable LoRA parameters for the optimizer."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def reset(self):
        pass

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Forward pass for supervised fine-tuning."""
        self.model.train()
        inputs = batch["inputs"]

        if self.debug:
            print(f"Input Qwen: {batch['messages']}")
            self.debug = False
        
        # Move inputs to device
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        if "labels" not in inputs:
            raise ValueError("Labels are required for training. Ensure QwenProcessorStep generated them.")

        outputs = self.model(**inputs)
        loss = outputs.loss

        # Hugging Face models return mean-reduced scalar loss over non-masked tokens by default
        return loss, {"loss": loss.item()}
    
    def _get_default_peft_targets(self) -> dict[str, any]: 
        """Return default PEFT target modules for PI0Fast fine-tuning."""
        return { 
            "target_modules": 'all-linear', 
            "modules_to_save": [], 
        }