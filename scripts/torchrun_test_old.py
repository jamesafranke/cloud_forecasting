import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from vit_pytorch import ViT, Dino
from pathlib import Path
import logging
import argparse
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

#TODO
torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True # this all bf16 makes slow

HOME=str(Path.home())
logging.basicConfig(level=logging.INFO,format="%(asctime)s;%(levelname)s;%(message)s")
log= logging.getLogger("")

def parse_args(verbose=False):
    """ arg parser to automate experiment """
    p = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description=__doc__)
    p.add_argument("--datadir",       type=str,  default=f'{HOME}/workspace/hack_team_01/data/processed/patch_256', help='training data directory', required=True)
    p.add_argument("--batch_size",    type=int,  default=200,  help='minibatch size')
    p.add_argument("--epochs",        type=int,  default=30,   help='epochs')
    p.add_argument("--save_every_nepoch",        type=int,  default=1,   help='epochs')
        
    # Pipeline
    p.add_argument("--num_workers",       type=int,  default=4,    help='number of workers for pipeline')
    p.add_argument("--prefetch_factor",   type=int,  default=None,    help='number of workers for pipeline')
    
    # ViT
    p.add_argument("--image_size",    type=int,  default=256,  help='input image size')
    p.add_argument("--patch_size",    type=int,  default=256,  help='patch size in ViT')
    p.add_argument("--dim",           type=int,  default=1024, help='last dimension of output tensor')
    p.add_argument("--mlp_dim",       type=int,  default=2048, help='channel size in MLP layer')
    p.add_argument("--depth",         type=int,  default=6,    help='number of iteration of attention module')
    p.add_argument("--heads",         type=int,  default=8,    help='number of attention heads')
    p.add_argument("--lr",            type=float,default=3e-4, help='learning rate: default 3e-4')
    
    # DINO
    # FIXME add following parse args later 
    #p.add_argument("--project_hidden_size",      type=int,    default=256,  help='channel dimension of dino')
    #p.add_argument("--project_layers",           type=int,    default=4,    help='projection layers')
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

# Load parsed config
FLAGS = parse_args(verbose=False)

vit = ViT(image_size = FLAGS.image_size, patch_size = FLAGS.patch_size, num_classes = 1000, dim = FLAGS.dim, 
          depth = FLAGS.depth, heads = FLAGS.heads, mlp_dim = FLAGS.mlp_dim)

learner = Dino(
    vit,
    image_size = FLAGS.image_size,
    hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
    projection_hidden_size = 256,      # projector network hidden dimension
    projection_layers = 4,             # number of layers in projection network
    num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
    student_temp = 0.9,                # student temperature
    teacher_temp = 0.04,               # teacher temp, needs to be annealed from 0.04 to 0.07 over 30 epochs
    local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
    global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
    moving_average_decay = 0.9,        # moving average of encoder -anywhere from 0.9 to 0.999 was ok
    center_moving_average_decay = 0.9, # moving average of teacher centers -anywhere from 0.9 to 0.999 was ok
)

def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    FLAGS = parse_args(rank==0) # Parse argument
    log.info(f"Start running basic DDP example on rank {rank}.")

    # distributed dataloader
    custom_dataset = CustomDataset(data_dir=FLAGS.datadir)
    data_loader = DataLoader(
        dataset=custom_dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        sampler=DistributedSampler(custom_dataset),
        num_workers=FLAGS.num_workers,
        #pin_memory=True,
        prefetch_factor=FLAGS.prefetch_factor,  
    )

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = learner.to(device_id)
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    os.makedirs(f"{HOME}/ViT_clouds/run/profile-v2",exist_ok=True)
    CHECKPOINT_PATH = f'{HOME}/ViT_clouds/run/profile-v2/pretrained-net_modis_{FLAGS.image_size}_{FLAGS.image_size}_patch{FLAGS.patch_size}_mod.pt'


    # configure map_location properly
    if os.path.exists(CHECKPOINT_PATH):
      map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
      ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))
      # Use a barrier() to make sure that process 1 loads the model after process
      # 0 saves it.
      dist.barrier()

    if rank == 0: log.info("Enter Training Loop")
    bepoch = int(FLAGS.epochs//2) # bench mark at mid point in training
    
    opt = torch.optim.Adam(ddp_model.parameters(), lr = 3e-4)
    for epoch in range(1, FLAGS.epochs+1, 1):
        if rank == 0:
            log.info(f"Start Training at Epoch {epoch}")
        
        data_loader.sampler.set_epoch(epoch)

        if epoch == bepoch: torch.cuda.cudart().cudaProfilerStart()

        train_loss = 0
        for i, batch in enumerate(data_loader):
            batch = batch.to(device_id)
            opt.zero_grad()
            
            # Automated Mixed precision
            with torch.autocast(device_type='cuda',dtype=torch.float16):
              if epoch == bepoch: torch.cuda.nvtx.range_push("iteration{}".format(i*(epoch+1)))
              loss = ddp_model(batch)
            
            if epoch == bepoch: torch.cuda.nvtx.range_push("backward")
            loss.backward()
            if epoch == bepoch: torch.cuda.nvtx.range_push("opt.step()")
            opt.step()
            model.update_moving_average() # moving average should be model instead of ddp_model 
            train_loss += loss.detach_().item()
        
        if rank ==0: log.info(f"\n Train loss : {train_loss/len(data_loader)} ")

        # FIXME: the saving criterion should be the best validation loss 
        if epoch % FLAGS.save_every_nepoch == 0:
            if rank == 0:
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient.
                log.info(f'save checkpoint ... at epoch {epoch}')
                torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)
        torch.cuda.empty_cache() # clear cache

        if epoch == bepoch: torch.cuda.cudart().cudaProfilerStop()
    dist.destroy_process_group()

if __name__ == "__main__":
    log.info(f" GPU Device Count =  {torch.cuda.device_count()}" )
    demo_basic()