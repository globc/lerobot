from .configuration_qwen import QwenConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from lerobot.configs import PreTrainedConfig
from ..pretrained import PreTrainedPolicy, T

import builtins
from pathlib import Path
import torch
from torch import Tensor


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
            torch_dtype="auto", 
            device_map="auto",
        )
        self.model.to("cuda")


    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        inputs = batch["inputs"]

        generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        max_len = max(len(t) for t in generated_ids_trimmed)
        padded_ids = [
            torch.nn.functional.pad(t, (0, max_len - len(t)), value=0) 
            for t in generated_ids_trimmed
        ]
        
        return torch.stack(padded_ids)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return None

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        pass

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        return None, {}