import os, torch, numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from vit_pytorch import ViT, Dino
from pathlib import Path
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

HOME=str(Path.home())

def parse_args(verbose=False):
    """ arg parser to automate experiment """
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--datadir",       type=str,  default=f'{HOME}/workspace/hack_team_01/data/processed/patch_256', help='training data directory')
    p.add_argument("--batch_size",    type=int,  default=200,  help='minibatch size')
    p.add_argument("--num_workers",   type=int,  default=4,    help='number of workers for pipeline')
    p.add_argument("--epochs",        type=int,  default=30,   help='epochs')
    
    # ViT
    p.add_argument("--image_size",    type=int,  default=256,  help='input image size')
    p.add_argument("--patch_size",    type=int,  default=256,  help='patch size in ViT')
    p.add_argument("--dim",           type=int,  default=1024, help='last dimension of output tensor')
    p.add_argument("--mlp_dim",       type=int,  default=2048, help='channel size in MLP layer')
    p.add_argument("--depth",         type=int,  default=6,    help='number of iteration of attention module')
    p.add_argument("--heads",         type=int,  default=8,    help='number of attention heads')
    p.add_argument("--lr",            type=float,default=3e-4, help='learning rate: default 3e-4')
    
    # DINO
    p.add_argument("--project_hidden_size",      type=int,    default=256,  help='channel dimension of dino')
    p.add_argument("--project_layers",           type=int,    default=4,    help='projection layers')
    # FIXME add following parse args later 
    #p.add_argument("--stemp",                    type=float,  default=0.9,  help='minibatch size')
    
    FLAGS = p.parse_args()
    if verbose:
        for f in FLAGS.__dict__:
            print("\t", f, (25 - len(f)) * " ", FLAGS.__dict__[f])
        print("\n")
    return FLAGS 
    

    
class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = os.listdir(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data[idx])
        img = read_image(img_path)
        return img / 255 

# Parse argument
FLAGS = parse_args(verbose=True)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0") # if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': FLAGS.batch_size,'shuffle': True, 'num_workers': FLAGS.num_workers}

custom_dataset = CustomDataset(data_dir=FLAGS.datadir)
data_loader = DataLoader(custom_dataset, **params)

model = ViT(
    image_size = FLAGS.image_size,
    patch_size = FLAGS.patch_size,
    num_classes = 1000,                # i dont think this matters in self supervised formulation
    dim = FLAGS.dim,
    depth = FLAGS.depth,
    heads = FLAGS.heads,
    mlp_dim = FLAGS.mlp_dim
)

learner = Dino(
    model,
    image_size = FLAGS.image_size,
    hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
    projection_hidden_size = 256,      # projector network hidden dimension
    projection_layers = 4,             # number of layers in projection network
    num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
    student_temp = 0.9,                # student temperature
    teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
    local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
    global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
    moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
    center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
).to(device)

opt = torch.optim.Adam(learner.parameters(), lr = FLAGS.lr)

print(torch.cuda.is_available() , flush=True )
profile_epoch = int(FLAGS.epochs//2)

for epoch in range(FLAGS.epochs):
    
    if epoch == profile_epoch:
      torch.cuda.cudart().cudaProfilerStart() 
    
    for i, batch in enumerate(data_loader):
        batch = batch.to(device)
        # forward
        if epoch == profile_epoch:
          torch.cuda.nvtx.range_push("iteration{}".format(i*(epoch+1)))
        loss = learner(batch)

        # backward 
        if epoch == profile_epoch:  
          torch.cuda.nvtx.range_push("backward")
        opt.zero_grad()
        loss.backward()

        # optimizer
        if epoch == profile_epoch:  
          torch.cuda.nvtx.range_push("opt.step()")
        opt.step()
        learner.update_moving_average() 


        if i % 20 == 0:
            print(f"Step {i} :  {torch.cuda.memory_allocated(device=device)/(1024**3) } [GB]", flush=True)
    torch.cuda.empty_cache()

torch.cuda.cudart().cudaProfilerStop()


os.makedirs(f"{HOME}/ViT_clouds/run/profile",exist_ok=True)
torch.save(model.state_dict(), f'{HOME}/ViT_clouds/run/profile/pretrained-net_modis_{FLAGS.image_size}_{FLAGS.image_size}_patch{FLAGS.patch_size}_mod.pt')