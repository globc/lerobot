from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig

@PreTrainedConfig.register_subclass("openvla")
@dataclass
class OpenVLAConfig(PreTrainedConfig):
    pretrained_checkpoint: str = "openvla/openvla-7b"
    text_tokenizer_name: str = "meta-llama/Llama-2-7b-hf"
    tokenizer_max_length: int = 2048
    dtype: str = "float32"
    reason: bool = False
    chunk_size: int = 50 # not used
    
    input_features = {
        "observation.images.image": PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(224, 224, 3)
        )
    }

    image_resolution: tuple[int, int] = (
        224,
        224,
    )

    hierarchical: bool = True
    dynamic_action_chunking: str = "subtask"
    device: str | None = None  # Device to use for the model (None = auto-detect)
    gradient_checkpointing: bool = False
    compile_model: bool = False

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    optimizer_lr: float = 2.5e-5
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

    def validate_features(self) -> None:
        return None

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return [0]
    
    @property
    def reward_delta_indices(self) -> None:
        return None