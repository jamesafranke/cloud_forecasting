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



#!/bin/bash
#SBATCH --job-name=j0001
eval "$(/share/data/2pals/jim/code/python/mc3/bin/conda 'shell.bash' 'hook')"
unset XDG_RUNTIME_DIR
NODEIP=$(hostname -i)
NODEPORT=$(( $RANDOM + 1024))
echo "ssh command: ssh -N -L 8888:$NODEIP:$NODEPORT `whoami`@beehive.ttic.edu"
jupyter lab --ip=$NODEIP --port=$NODEPORT --no-browser



#!/bin/bash
#SBATCH --job-name=openstl
eval "$(/share/data/2pals/jim/code/python/mc3/bin/conda 'shell.bash' 'hook')"
python stlrun.py


sbatch -p gpu -G1 openstl.sbatch

#!/bin/bash
#SBATCH --job-name=goes
eval "$(/share/data/2pals/jim/code/python/mc3/bin/conda 'shell.bash' 'hook')"
python goes.py


sbatch -p cpu goes.sbatch

sbatch -p cpu beerun.sbatch