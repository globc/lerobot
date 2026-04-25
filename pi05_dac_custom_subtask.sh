#!/usr/bin/env bash
. ~/.bashrc

export HOME=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta
export HF_LEROBOT_HOME=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.cache/huggingface/lerobot
export TRITON_CACHE_DIR="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/.triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/.torchinductor_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda activate lerobot

# policy.chunk_size: Maximum number actions per chunk
# policy.n_action_steps: Number actions of chunk executed

# lerobot-train \
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  $(which lerobot-train) \
    --dataset.repo_id=globcy/libero_subtask_custom_slow \
    --policy.type=pi05 \
    --policy.chunk_size=200 \
    --policy.n_action_steps=50 \
    --policy.hierarchical=true \
    --policy.dynamic_action_chunking=true \
    --output_dir=./outputs/pi05_dac_custom_subtask \
    --job_name=pi05_dac_custom_subtask \
    --policy.repo_id=globcy/pi05_dac_custom_subtask \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=false \
    --policy.dtype=bfloat16 \
    --policy.empty_cameras=1 \
    --steps=30000 \
    --save_freq=10000 \
    --policy.device=cuda \
    --batch_size=32

conda deactivate
