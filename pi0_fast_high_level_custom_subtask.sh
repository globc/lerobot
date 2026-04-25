#!/usr/bin/env bash
. ~/.bashrc

export HOME=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta
export HF_LEROBOT_HOME=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.cache/huggingface/lerobot
export TRITON_CACHE_DIR="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.torchinductor_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda activate lerobot

# policy.chunk_size: Maximum number actions per chunk
# policy.n_action_steps: Number actions of chunk executed

# lerobot-train \
accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  $(which lerobot-train) \
    --dataset.repo_id=globcy/libero_subtask_custom_slow \
    --policy.type=pi0_fast \
    --output_dir=./outputs/pi0_fast_high_level_custom_subtask \
    --job_name=pi0_fast_high_level_custom_subtask \
    --policy.repo_id=globcy/pi0_fast_high_level_custom_subtask \
    --policy.pretrained_path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/p05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.optimizer_lr=2.5e-4 \
    --policy.scheduler_decay_lr=2.5e-5 \
    --policy.empty_cameras=1 \
    --steps=30000 \
    --save_freq=5000 \
    --policy.device=cuda \
    --batch_size=32 \
    --peft.method_type=LORA \
    --peft.r=16 \
    --peft.lora_alpha=32 \
    --peft.lora_dropout=0.1 \

conda deactivate
