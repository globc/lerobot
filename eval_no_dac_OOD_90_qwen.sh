#!/usr/bin/env bash
. ~/.bashrc

# Check if an argument was provided
if [ -z "$1" ]; then
  echo "Error: No job argument provided. Please pass 0, 1, 2, or 3."
  echo "Usage: sh eval_no_dac.sh <job_id>"
  exit 1
fi

JOB_ID=$1

# Map the job ID to the correct task IDs and level folder
case $JOB_ID in
  0)
    TASK_IDS="[0, 20, 30, 48]"
    LEVEL="level_1"
    ;;
  1)
    TASK_IDS="[1, 26, 32, 69]"
    LEVEL="level_2"
    ;;
  2)
    TASK_IDS="[4, 14, 65, 73]"
    LEVEL="level_3"
    ;;
  3)
    TASK_IDS="[16, 18, 39, 55]"
    LEVEL="level_4"
    ;;
  *)
    echo "Error: Invalid job ID '$JOB_ID'. Must be 0, 1, 2, or 3."
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
    --policy.path=/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/cb14syta/lerobot/outputs/pi05_no_dac_custom_subtask/checkpoints/030000/pretrained_model \
    --planner.type=qwen \
    --planner.model_name=Qwen/Qwen3-VL-8B-Instruct \
    --policy.hierarchical=true \
    --policy.compile_model=false \
    --policy.chunk_size=50 \
    --eval.batch_size=1 \
    --env.type=libero \
    --env.task=libero_90 \
    --env.task_ids="$TASK_IDS" \
    --eval.n_episodes=10 \
    --policy.n_action_steps=10 \
    --seed=1000 \
    --output_dir="./eval/eval_no_dac_qwen/OOD_90/$LEVEL/" \
    --env.max_parallel_tasks=1

conda deactivate