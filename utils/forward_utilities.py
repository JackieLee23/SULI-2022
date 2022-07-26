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


class LitNeuralNet(pl.LightningModule):
    def __init__(self, layer_sizes, lr = 0.01, factor = 0.0):
        super(LitNeuralNet, self).__init__()
        
        modules = []
        for i in range(len(layer_sizes) - 1):
            modules.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            
            if i != len(layer_sizes) - 2:
                modules.append(nn.ReLU())
        
        self.forward_prop = nn.Sequential(*modules)
        self.learning_rate = lr
        self.factor = factor
        self.save_hyperparameters()
    
    def training_step(self, batch, batch_idx):
        params, dos = batch
        
        # Forward pass
        predicted = self.forward_prop(params)
        loss = F.mse_loss(predicted, dos)
        
        #log to tensorboard
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        params, dos = batch
        
        # Forward pass
        predicted = self.forward_prop(params)
        loss = F.mse_loss(predicted, dos)
        
        #log to tensorboard
        self.log("val_loss", loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.factor == 0.0:
            return optimizer
        
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.factor, min_lr = 1e-7)
            return {
                "optimizer":optimizer,
                "lr_scheduler" : {
                    "scheduler" : sch,
                    "monitor" : "train_loss",

                }
            }
    
    
class DosDataset(Dataset):

    def __init__(self, data_file):
        dataset = np.load(data_file)
        params_arr = dataset['params']
        dos_arr = dataset['dos']
        
        self.params_arr = torch.from_numpy(params_arr).float()
        self.dos_arr = torch.from_numpy(dos_arr).float()
    
    def __getitem__(self, index):
        return self.params_arr[index], self.dos_arr[index]
    
    def __len__(self):
        return self.params_arr.size(dim=0)
    
    
class DosDataModule(pl.LightningDataModule):
    def __init__(self, data_loc, batch_size):
        super().__init__()
        self.data_loc = data_loc
        self.batch_size = batch_size
    
    def setup(self, stage: str = None):
        
        if stage == "fit" or stage is None:
            self.train_dataset = DosDataset(f"{self.data_loc}/train-set.npz")
            self.val_dataset = DosDataset(f"{self.data_loc}/val-set.npz")
            
            self.mean = self.train_dataset.params_arr.mean(0, keepdim=True)
            self.std = self.train_dataset.params_arr.std(0, unbiased=False, keepdim=True)
            
            self.train_dataset.params_arr -= self.mean
            self.train_dataset.params_arr /= self.std

            self.val_dataset.params_arr -= self.mean
            self.val_dataset.params_arr /= self.std
            
        if stage == "test" or stage is None:
            self.test_dataset = DosDataset(f"{self.data_loc}/test-set.npz")
        
            self.test_dataset.params_arr -= self.mean
            self.test_dataset.params_arr /= self.std
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=2)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=2)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=2)
    
    
class ScaledData:
    def __init__(self, data_loc):
        self.train_set = np.load(os.path.join(data_loc, 'train-set.npz'))
        self.train_params_unscaled = self.train_set['params']
        self.train_dos = self.train_set['dos']

        self.val_set = np.load(os.path.join(data_loc, 'val-set.npz'))
        self.val_params_unscaled = self.val_set['params']
        self.val_dos = self.val_set['dos']

        self.test_set = np.load(os.path.join(data_loc, 'test-set.npz'))
        self.test_params_unscaled = self.test_set['params']
        self.test_dos = self.test_set['dos']

        #With standard scaling
        scaler = StandardScaler()
        self.train_params = scaler.fit_transform(self.train_params_unscaled)
        self.val_params = scaler.transform(self.val_params_unscaled)
        self.test_params = scaler.transform(self.test_params_unscaled)
        
        
def plot_one(ax, mse_params, index, text):
    
    ax.plot(np.linspace(-6, 6, 301), mse_params[index][2], label = "Ground Truth")
    ax.plot(np.linspace(-6, 6, 301), mse_params[index][1], label = "ML Predicted")
    
    if text:
        ax.text(2, 0.2, mse_params[index][0])
    ax.legend()

def see_results(predicted, truth, grid_shape, percentiles, text = False):
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
                plot_one(ax[i][j], mse_params, index, text)
                ax[i][j].set_title(f"{percentile} percentile")
    

