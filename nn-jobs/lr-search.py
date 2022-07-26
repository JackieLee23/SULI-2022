import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import os
import sys
from sklearn.preprocessing import StandardScaler
import pickle
root = "/project/wyin/jlee/ml-project/"
util_loc = os.path.join(root, "utils")
sys.path.append(util_loc)
from utilities import LitNeuralNet, LitDataModule

step_num = int(sys.argv[1])
num_cpus = int(sys.argv[2])
torch.set_num_threads(num_cpus // 2)

#########Code to make hparam iterable#################
lr_arr = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.0005]
######################################################

#######Set changed hyperparemter(s) here###############
layer_sizes = [354, 256, 192, 128, 64, 3]
learning_rate = lr_arr[step_num - 1]
batch_size = 128
schedule_factor = 0.5
max_time = "00:00:15:00"

data_loc = os.path.join(root, "inverse-shifted/data")
save_loc = os.path.join(root, "Tests/utilities-testing")
log_name = "arch-search-15-min"
save_models = True

inputs = "dos"
outputs = "params"

######################################################
log_folder = os.path.join(save_loc, "logs")
val_folder = os.path.join(save_loc, "val-ends")
log_path = os.path.join(log_folder, log_name)
val_path = os.path.join(val_folder, log_name)

if step_num == 1:
    
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
        
    if not os.path.exists(val_folder):
        os.mkdir(val_folder)
       
    if not os.path.exists(log_path):
        os.mkdir(log_path)
        
    if not os.path.exists(val_path):
        os.mkdir(val_path)
    
    print("made paths")


data_module = LitDataModule(data_loc, batch_size, X_name = inputs, y_name = outputs)
logger = TensorBoardLogger(log_path, name = f'{layer_sizes},{learning_rate},{batch_size},{schedule_factor}')
trainer = pl.Trainer(enable_checkpointing=save_models, max_time=max_time, logger = logger, enable_progress_bar = False)
model = LitNeuralNet(layer_sizes, lr = learning_rate, lr_factor = schedule_factor)
trainer.fit(model, datamodule=data_module)


end_res = trainer.validate(model, dataloaders = data_module)
f = open(f"{val_path}/{layer_sizes},{learning_rate},{batch_size},{schedule_factor}","wb")
pickle.dump(end_res,f)
f.close()