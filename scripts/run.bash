#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --job-name=test
#SBATCH --output=%x_%A.out
#SBATCH --error=%x_%A.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

torchrun --nnodes=1:2 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:6000 torchrun_test.py