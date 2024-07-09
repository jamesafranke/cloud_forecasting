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

sbatch -p gpu -G1 -C48g openstl.sbatch

sbatch -p gpu -G6 -C48g openstl.sbatch


#!/bin/bash
#SBATCH --job-name=goes
eval "$(/share/data/2pals/jim/code/python/mc3/bin/conda 'shell.bash' 'hook')"
cd /share/data/2pals/jim/data/1d-tokenizer
python goes.py


sbatch -p cpu goes.sbatch

sbatch -p cpu beerun.sbatch

sbatch -p gpu -G1 beerun.sbatch


#!/bin/bash
#SBATCH --job-name=remove
cd /share/data/2pals/jim/data/geostat/
rm -r goes16/


#!/bin/bash
#SBATCH --job-name=taming-transformer
eval "$(/share/data/2pals/jim/code/python/mc3/bin/conda 'shell.bash' 'hook')"
conda activate taming
python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1,2,3,4,5,6,7,

#!/bin/bash
#SBATCH --job-name=taming-transformer
eval "$(/share/data/2pals/jim/code/python/mc3/bin/conda 'shell.bash' 'hook')"
python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1,2,3,4,5,6,7,


model:
  base_learning_rate: 4.5e-6
  target: taming.models.vqgan.VQModel
  params:
    embed_dim: 256
    n_embed: 1024
    ddconfig:
      double_z: False
      z_channels: 256
      resolution: 256
      in_channels: 3
      out_ch: 3
      ch: 128
      ch_mult: [ 1,1,2,2,4]  # num_down = len(ch_mult)-1
      num_res_blocks: 2
      attn_resolutions: [16]
      dropout: 0.0

    lossconfig:
      target: taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator
      params:
        disc_conditional: False
        disc_in_channels: 3
        disc_start: 10000
        disc_weight: 0.8
        codebook_weight: 1.0

data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 15
    num_workers: 8
    train:
      target: taming.data.custom.CustomTrain
      params:
        training_images_list_file: /share/data/2pals/jim/data/geostat/train.txt
        size: 256
    validation:
      target: taming.data.custom.CustomTest
      params:
        test_images_list_file: /share/data/2pals/jim/data/geostat/test.txt
        size: 256


find $(pwd)/himawari_256 -name "*.jpg" > train.txt
find $(pwd)/himawari_256_test -name "*.jpg" > test.txt


find $(pwd)/himawari_truecolor -name "*.jpeg" > truecolor.txt


