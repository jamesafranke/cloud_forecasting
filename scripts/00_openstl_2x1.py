import torch, os, numpy as np
from glob import glob
from torch.utils.data import Dataset
from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser

torch.set_float32_matmul_precision('medium')

pre_seq_length = 2
aft_seq_length = 1
batch_size = 10
n_epoch = 30
name = '01_2x1_big_2'

root = '/share/data/2pals/jim/data/openstl/'
os.chdir(root)

class CustomDataset(Dataset):
    def __init__(self, X, Y, data_name='custom'):
        super(CustomDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.mean = None
        self.std = None
        self.data_name = data_name

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels

print('-------- starting dataloading ------------')
flX = glob('/share/data/2pals/jim/data/openstl/2x1stacks/*X.npy')
flX = np.sort(flX)
X_train = np.empty((1,2,3,256,256), dtype=np.float32)
for file in flX[1::2]: X_train = np.concatenate((X_train, np.load(file)), axis=0)

flY = glob('/share/data/2pals/jim/data/openstl/2x1stacks/*Y.npy')
flY = np.sort(flY)
Y_train = np.empty((1,1,3,256,256), dtype=np.float32)
for file in flY[1::2]: Y_train = np.concatenate((Y_train, np.load(file)), axis=0)

dim = 256
X_val = X_train[-50:,:,:,:,:] 
Y_val = Y_train[-50:,:,:,:,:] 

X_train = X_train[1:-50,:,:,:,:]
Y_train = Y_train[1:-50,:,:,:,:]

print('-----------', X_train.shape[0], '--------------training samples')

X_test = np.load(f'{root}testX.npy')[:,-2:,:,:dim,:dim]
Y_test = np.load(f'{root}testY.npy')[:,1,:,:dim,:dim]

train_set = CustomDataset(X=X_train, Y=Y_train)
val_set   = CustomDataset(X=X_val, Y=Y_val)
test_set  = CustomDataset(X=X_test, Y=Y_test)

dataloader_train = torch.utils.data.DataLoader( train_set, batch_size=batch_size, shuffle=True, pin_memory=True )
dataloader_val   = torch.utils.data.DataLoader( val_set, batch_size=batch_size, shuffle=True, pin_memory=True )
dataloader_test  = torch.utils.data.DataLoader( test_set, batch_size=batch_size, shuffle=True, pin_memory=True )

custom_training_config = {
    'pre_seq_length': pre_seq_length,
    'aft_seq_length': aft_seq_length,
    'total_length': pre_seq_length + aft_seq_length,
    'batch_size': batch_size,
    'val_batch_size': batch_size,
    'epoch': n_epoch,
    'lr': 0.001,
    'metrics': ['mse', 'mae'],
    'ex_name': name,
    'dataname': 'custom',
    'in_shape': [2, 3, dim, dim], # frames in sequence, bands, x, y
}

custom_model_config = {
    # For MetaVP models, the most important hyperparameters are: N_S, N_T, hid_S, hid_T, model_type
    'method': 'SimVP',
    'model_type': 'gSTA',
    'N_S': 4,
    'N_T': 8,
    'hid_S': 256,
    'hid_T': 512
}

args = create_parser().parse_args([])
config = args.__dict__

# update default parameters
default_values = default_parser()
for attribute in default_values.keys():
    if config[attribute] is None:
        config[attribute] = default_values[attribute]

config.update(custom_training_config)    # update the training config
config.update(custom_model_config)       # update the model config

exp = BaseExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test), strategy='ddp')

print('>'*35 + ' training ' + '<'*35) 
exp.train()

print('>'*35 + ' testing  ' + '<'*35)
exp.test()