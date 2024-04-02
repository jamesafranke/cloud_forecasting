#!/bin/bash
#SBATCH --job-name=cloudViT8
eval "$(/share/data/2pals/jim/code/python/mc3/bin/conda 'shell.bash' 'hook')"

torchrun --nnodes=1 --nproc_per_node=8 torchrun.py \
        --datadir "/share/data/2pals/jim/data/processed/cloudsatsmall" \
        --image_size 128 \
        --patch_size 8 \
        --mlp_dim 1024 \
        --dim 1024 \
        --heads 8 \
        --teacher_temp 0.04 \
        --batch_size 260 \
        --epochs 30


sbatch -p greg-gpu -c8 -C a6000 run.sbatch