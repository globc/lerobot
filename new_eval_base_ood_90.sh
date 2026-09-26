#!/usr/bin/env bash
. ~/.bashrc

# Check if an argument was provided
if [ -z "$1" ]; then
  echo "Error: No job argument provided. Please pass 0, 1, 2, 3, 4, 5, 6, or 7."
  echo "Usage: sh eval_no_dac.sh <job_id>"
  exit 1
fi

JOB_ID=$1

# Map the job ID to the correct task IDs and level folder
case $JOB_ID in
  0)
    TASK_IDS="[79, 49, 53, 73, 48, 19, 46, 24, 44, 75, 20]"
    LEVEL="level_1a"
    ;;
  1)
    TASK_IDS="[77, 69, 80, 54, 22, 18, 52, 43, 50, 74, 25]"
    LEVEL="level_1b"
    ;;
  2)
    TASK_IDS="[83, 9, 70, 38, 2, 78, 30, 82, 47, 29, 11]"
    LEVEL="level_2a"
    ;;
  3)
    TASK_IDS="[67, 13, 55, 12, 6, 68, 81, 14, 56, 31, 27]"
    LEVEL="level_2b"
    ;;
  4)
    TASK_IDS="[65, 26, 15, 34, 58, 21, 28, 7, 71, 10, 36]"
    LEVEL="level_3a"
    ;;
  5)
    TASK_IDS="[76, 17, 85, 66, 41, 0, 57, 37, 8, 45, 39]"
    LEVEL="level_3b"
    ;;
  6)
    TASK_IDS="[42, 4, 33, 84, 61, 23, 60, 40, 1, 59, 72]"
    LEVEL="level_4a"
    ;;
  7)
    TASK_IDS="[63, 64, 3, 35, 5, 62, 89, 16, 87, 32, 88, 86]"
    LEVEL="level_4b"
    ;;
  *)
    echo "Error: Invalid job ID '$JOB_ID'. Must be 0, 1, 2, 3, 4, 5, 6, or 7."
    exit 1
    ;;
esac

export HOME=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta
export HF_LEROBOT_HOME=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.cache/huggingface/lerobot
export TRITON_CACHE_DIR="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.triton_cache"
export TORCHINDUCTOR_CACHE_DIR="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/.torchinductor_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MUJOCO_GL=glx

# Start virtual display
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99

conda activate lerobot

echo "Starting evaluation for Job $JOB_ID (Tasks: $TASK_IDS, Output: $LEVEL)"

# Since Determined AI allocates 1 GPU per task, it will be visible as GPU 0 inside the container.
# Notice I removed the duplicate `--eval.batch_size=1` from your original script.
CUDA_VISIBLE_DEVICES=0 lerobot-eval \
    --policy.path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/outputs_new/pi05_libero_reproduce/checkpoints/030000/pretrained_model \
    --policy.hierarchical=false \
    --policy.compile_model=false \
    --policy.train_then=false \
    --policy.dynamic_action_chunking="" \
    --policy.compile_model=false \
    --policy.chunk_size=50 \
    --eval.batch_size=1 \
    --env.type=libero \
    --env.task=libero_90 \
    --env.task_ids="$TASK_IDS" \
    --eval.n_episodes=5 \
    --policy.n_action_steps=10 \
    --seed=1000 \
    --output_dir="./eval_new/pi05_repr/$LEVEL/" \
    --env.max_parallel_tasks=1

conda deactivate