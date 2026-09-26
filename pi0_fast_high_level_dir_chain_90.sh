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
  --num_processes=8 \
  $(which lerobot-train) \
    --dataset.repo_id=globcy/libero_dir_90 \
    --bottom_up=false \
    --bottom_up_chunk_size=10 \
    --reader_chain_close=10 \
    --reader_chain_dir=true \
    --is_planner=true \
    --policy.type=pi0_fast \
    --output_dir=./outputs/pi0_fast_high_level_dir_chain_dir_10 \
    --job_name=openvla_high_level_dir_swapped_left_chain_dir_10 \
    --policy.repo_id=globcy/pi0_fast_high_level_dir_chain_dir_10 \
    --policy.hierarchical=true \
    --policy.dynamic_action_chunking=move \
    --policy.pretrained_path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/p05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.optimizer_lr=2.5e-4 \
    --policy.scheduler_decay_lr=2.5e-5 \
    --policy.empty_cameras=1 \
    --steps=30000 \
    --save_freq=15000 \
    --policy.device=cuda \
    --batch_size=16 \
    --peft.method_type=LORA \
    --peft.r=16 \
    --peft.lora_alpha=32 \
    --peft.lora_dropout=0.1 \

conda deactivate
