#!/usr/bin/env bash
. ~/.bashrc

export HOME=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta
export HF_LEROBOT_HOME=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.cache/huggingface/lerobot
export TRITON_CACHE_DIR="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.torchinductor_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MUJOCO_GL=glx

Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

conda activate lerobot

# GPU 0: Combination
CUDA_VISIBLE_DEVICES=0 lerobot-eval \
    --policy.path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/outputs/pi05_no_dac_custom_subtask/checkpoints/030000/pretrained_model \
    --planner.path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/outputs/pi0_fast_high_level_custom_subtask/checkpoints/015000/pretrained_model \
    --policy.hierarchical=true \
    --policy.compile_model=false \
    --policy.chunk_size=50 \
    --eval.batch_size=1 \
    --env.type=libero \
    --env.task=libero_goal \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.n_action_steps=10 \
    --seed=1000 \
    --output_dir=./eval/eval_no_dac/goal_15k_planner/ \
    --env.max_parallel_tasks=1 &
# 23
# GPU 1: Refer
CUDA_VISIBLE_DEVICES=1 lerobot-eval \
    --policy.path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/outputs/pi05_no_dac_custom_subtask/checkpoints/030000/pretrained_model \
    --planner.path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/outputs/pi0_fast_high_level_custom_subtask/checkpoints/015000/pretrained_model \
    --policy.hierarchical=true \
    --policy.compile_model=false \
    --policy.chunk_size=50 \
    --eval.batch_size=1 \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.n_action_steps=10 \
    --seed=1000 \
    --output_dir=./eval/eval_no_dac/spatial_15k_planner/ \
    --env.max_parallel_tasks=1 &

# 23
# GPU 2: Spatial
CUDA_VISIBLE_DEVICES=2 lerobot-eval \
    --policy.path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/outputs/pi05_no_dac_custom_subtask/checkpoints/030000/pretrained_model \
    --planner.path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/outputs/pi0_fast_high_level_custom_subtask/checkpoints/015000/pretrained_model \
    --policy.hierarchical=true \
    --policy.compile_model=false \
    --policy.chunk_size=50 \
    --eval.batch_size=1 \
    --env.type=libero \
    --env.task=libero_object \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.n_action_steps=10 \
    --seed=1000 \
    --output_dir=./eval/eval_no_dac/object_15k_planner/ \
    --env.max_parallel_tasks=1 &

# 22
# GPU 3: Context
CUDA_VISIBLE_DEVICES=3 lerobot-eval \
    --policy.path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/outputs/pi05_no_dac_custom_subtask/checkpoints/030000/pretrained_model \
    --planner.path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/outputs/pi0_fast_high_level_custom_subtask/checkpoints/015000/pretrained_model \
    --policy.hierarchical=true \
    --policy.compile_model=false \
    --policy.chunk_size=50 \
    --eval.batch_size=1 \
    --env.type=libero \
    --env.task=libero_10 \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.n_action_steps=10 \
    --seed=1000 \
    --output_dir=./eval/eval_no_dac/long_15k_planner/ \
    --env.max_parallel_tasks=1 &

# 22
# Wait for all background processes to finish before continuing
wait

conda deactivate