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
# policy.dynamic_action_chunking = "move" | "trace" | "subtask_move" | "subtask" | "subtask_move_trace"

# lerobot-train \
accelerate launch \
  --multi_gpu \
  --num_processes=8 \
  $(which lerobot-train) \
    --bottom_up=false \
    --bottom_up_chunk_size=null \
    --dataset.repo_id=globcy/libero_dir \
    --is_planner=false \
    --reader_chain_close=10 \
    --reader_chain_dir=true \
    --sorted_dir=false \
    --policy.type=pi05 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.compile_model=true \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.dynamic_action_chunking="subtask_move" \
    --policy.empty_cameras=1 \
    --policy.gradient_checkpointing=true \
    --policy.hierarchical=true \
    --policy.ilfm=false \
    --policy.ilfm_max_horizon=75 \
    --policy.include_task=false \
    --policy.mask_pad=false \
    --policy.pretrained_path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/pi05_base \
    --policy.push_to_hub=false \
    --policy.train_then=true \
    --wandb.enable=false \
    --steps=30000 \
    --save_freq=10000 \
    --batch_size=32 \
    --output_dir=./outputs_new/pi05_subtask_dir_train_then_fixed \
    --job_name=pi05_subtask_dir_train_then_fixed

conda deactivate
