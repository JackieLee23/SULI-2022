import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
import os


class InverseNeuralNet(pl.LightningModule):
    def __init__(self, layer_sizes, lr = 0.01, lr_factor = 0.0):
        super(InverseNeuralNet, self).__init__()
        
        modules = []
        for i in range(len(layer_sizes) - 1):
            modules.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            if i != len(layer_sizes) - 2:
                modules.append(nn.ReLU())
        
        self.forward_prop = nn.Sequential(*modules)
        self.learning_rate = lr
        self.lr_factor = lr_factor
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.forward_prop(x)
    
    def training_step(self, batch, batch_idx):
        dos, params = batch
        
        # Forward pass
        predicted = self(dos)
        loss = F.mse_loss(predicted, params)
        
        #log to tensorboard
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        dos, params = batch
        
        # Forward pass
        predicted = self(dos)
        loss = F.mse_loss(predicted, params)
        
        #log to tensorboard
        self.log("val_loss", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.lr_factor == 0.0:
            return optimizer
        
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.lr_factor, min_lr = 1e-7)
            return {
                "optimizer":optimizer,
                "lr_scheduler" : {
                    "scheduler" : sch,
                    "monitor" : "train_loss",

                }
            }
        
        
        
class InverseDataset(Dataset):

    def __init__(self, dos_arr, params_arr):
        
        self.dos_arr = torch.from_numpy(dos_arr).float()
        self.params_arr = torch.from_numpy(params_arr).float()
    
    def __getitem__(self, index):
        return self.dos_arr[index], self.params_arr[index]
    
    def __len__(self):
        return self.params_arr.size(dim=0)
    
    
class InverseDataModule(pl.LightningDataModule):
    def __init__(self, data_loc, batch_size):
        super().__init__()
        self.data = ScaledData(data_loc)
        self.batch_size = batch_size
    
    def setup(self, stage: str = None):
        
        if stage == "fit" or stage is None:
            self.train_dataset = InverseDataset(self.data.train_dos, self.data.train_params)
            self.val_dataset = InverseDataset(self.data.val_dos, self.data.val_params)
            
        if stage == "test" or stage is None:
            self.test_dataset = DosDataset(self.data.test_dos, self.data.test_params)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=2)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=2)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=2)
    

class ScaledData:
    def __init__(self, data_loc):
        self.train_set = np.load(os.path.join(data_loc, 'train-set.npz'))
        self.train_params = self.train_set['params']
        self.train_dos_unscaled = self.train_set['dos']

        self.val_set = np.load(os.path.join(data_loc, 'val-set.npz'))
        self.val_params = self.val_set['params']
        self.val_dos_unscaled = self.val_set['dos']

        self.test_set = np.load(os.path.join(data_loc, 'test-set.npz'))
        self.test_params = self.test_set['params']
        self.test_dos_unscaled = self.test_set['dos']

        #With standard scaling
        scaler = StandardScaler()
        self.train_dos = scaler.fit_transform(self.train_dos_unscaled)
        self.val_dos = scaler.transform(self.val_dos_unscaled)
        self.test_dos = scaler.transform(self.test_dos_unscaled)


def plot_one(ax, mse_params, index, see_baseline):
    ax.set_ylim([-0.5, 1.0])
    tick_pos = np.arange(0, 3)
    ax.plot(tick_pos, mse_params[index][2], label = "Ground Truth")
    ax.plot(tick_pos, mse_params[index][1], label = "ML Predicted")
    
    if see_baseline:
        ax.plot(tick_pos, [0, 0, 0.6], label = "baseline")
    
    ax.set_xticks(tick_pos, ('t1', 't2', 'J'))
    ax.legend()

def see_results(predicted, truth, grid_shape, percentiles, see_baseline = False):
    mse_mat = (predicted - truth) ** 2
    mse_list = np.mean(mse_mat, axis = 1)
    print(f"model mse: {np.mean(mse_list)}")

    mse_params = zip(mse_list, predicted, truth)
    mse_params = sorted(mse_params, key = lambda x: x[0], reverse = True)
    
    dim1, dim2 = grid_shape
    fig, ax = plt.subplots(dim1, dim2, figsize = (15, dim1 * 5))
    
    for i in range(dim1):
        for j in range(dim2):
            if i * dim2 + j < len(percentiles):
                percentile = percentiles[i * dim2 + j]
                index = percentile * (len(mse_params)//100)
                plot_one(ax[i][j], mse_params, index, see_baseline)
                ax[i][j].set_title(f"{percentile} percentile")