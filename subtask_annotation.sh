#!/usr/bin/env bash
. ~/.bashrc

conda activate lerobot

# Worker 0 uses GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 python subtask_annotation_dist.py --shard-id 0 --num-shards 4 &

# Worker 1 uses GPUs 2 and 3
CUDA_VISIBLE_DEVICES=2,3 python subtask_annotation_dist.py --shard-id 1 --num-shards 4 &

# Worker 2 uses GPUs 4 and 5
CUDA_VISIBLE_DEVICES=4,5 python subtask_annotation_dist.py --shard-id 2 --num-shards 4 &

# Worker 3 uses GPUs 6 and 7
CUDA_VISIBLE_DEVICES=6,7 python subtask_annotation_dist.py --shard-id 3 --num-shards 4 &

# Wait for all background processes to finish
wait
echo "All shards completed."