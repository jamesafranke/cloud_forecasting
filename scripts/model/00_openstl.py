import torch, os, pickle, numpy as np
from torch.utils.data import Dataset
from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser

torch.set_float32_matmul_precision('medium')

pre_seq_length = 2
aft_seq_length = 1
batch_size = 4
n_epoch = 250
name = '13_hid256_512_2x1'

root = '/share/data/2pals/jim/data/openstl/'
os.chdir(root)

#with open('dataset.pkl', 'rb') as f: dataset = pickle.load(f)

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

#X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset['X_train'], dataset['X_val'], dataset['X_test'], dataset['Y_train'], dataset['Y_val'], dataset['Y_test']
X_train = np.load(f'goes2019_trainX_1562.npy')
Y_train = np.load(f'goes2019_trainY_1562.npy')

dim = 256
X_val   = X_train[-30:,-2:,:,:dim,:dim]  #np.load(f'{root}goes2019_valX.npy')
Y_val   = Y_train[-30:,1,:,:dim,:dim]  #np.load(f'{root}goes2019_valY.npy')

X_train = X_train[:1480,-2:,:,:dim,:dim]
Y_train = Y_train[:1480,1,:,:dim,:dim]

#X_test  = X_train[-10:,:,:,:dim,:dim]  #np.load(f'{root}goes2019_testX.npy')
#Y_test  = Y_train[-10:,:,:,:dim,:dim]  #np.load(f'{root}goes2019_testY.npy')
X_test = np.load(f'testX.npy')[:,-2:,:,:dim,:dim]
Y_test = np.load(f'testY.npy')[:,1,:,:dim,:dim]


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