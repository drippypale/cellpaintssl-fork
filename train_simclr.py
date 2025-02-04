# %%
import os
import argparse
import socket
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

import source.augment as au
from source import SimCLR, MergedChannelsDataset, get_dfs
# %%

parser = argparse.ArgumentParser()
# Add model and training arguments
parser.add_argument('--arch', default="vit_small_patch16_224", type=str)
parser.add_argument('--gpus', type=int, nargs='+', default=[0,1], help='Number of GPUs to use')
parser.add_argument('--path_df', default="SSL_data/dataload/dataset.csv", type=str)
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--num_workers', type=int, default=6, help='Number of workers for data loading')
parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of training epochs')
parser.add_argument('--ckpt_path', type=str, help='Checkpoint path')
parser.add_argument('--every_n_epochs', type=int, default=10, help='Save checkpoint every n epochs')
# Adam parameters
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--wd', type=float, default=0.1, help='Weight decay')
# Model parameters
parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
parser.add_argument('--temperature', type=float, default=0.2, help='Temperature')
parser.add_argument('--size', type=int, default=224, help='Image size')
parser.add_argument('--scale', type=float, default=None, help='Scale')
parser.add_argument('--seed', default=42, type=int, help='Random seed.')
# Parse the command line arguments
args = parser.parse_args()

# Access the parameter values
path_df = args.path_df
gpus = args.gpus
batch_size = int(args.batch_size /  len(gpus))
num_workers = args.num_workers
max_epochs = args.max_epochs
ckpt_path = args.ckpt_path
every_n_epochs = args.every_n_epochs
lr = args.lr
weight_decay = args.wd
hidden_dim = args.hidden_dim
temperature = args.temperature
size = args.size
scale = args.scale

# Setting the seed
pl.seed_everything(args.seed)

# create checkpoint path
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# %%
train_df, val_df = get_dfs(path=path_df)
# %%
# set augmentations for SimCLR
# channel stats for the final training set
means = [0.13849893, 0.18710597, 0.1586524,  0.15757588, 0.08674719]
stds = [0.13005716, 0.15461144, 0.15929441, 0.16021383, 0.16686504]
# default augmentations for joint-embedding models (DINO, SimCLR)
color_prob = 1
base_transform = au.get_default_augs(color_prob, means, stds)
transform = au.Augment(crop_transform=au.RandomCropWithCells(size=size, scale=scale),
                       base_transforms=base_transform, n_views=2)
# %%
trainset = MergedChannelsDataset(train_df, transform=transform)
valset = MergedChannelsDataset(val_df, transform=transform)

# %%
# set up the DataLoader's
train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
    )

val_loader = DataLoader(
    valset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    num_workers=num_workers,
)

# %%
# keep these training parameters fixed
strategy = 'ddp'
# turn off sanity check (don't run on validation batches before training)
num_sanity_val_steps=0

trainer = pl.Trainer(
    default_root_dir=os.path.join(ckpt_path, "SimCLR"),
    gpus=gpus if str(device) == "cuda:0" else 0,
    sync_batchnorm=True,
    strategy=DDPStrategy(find_unused_parameters=False),
    max_epochs=max_epochs,
    num_sanity_val_steps=num_sanity_val_steps,
    callbacks=[
        ModelCheckpoint(save_weights_only=True, 
                        save_top_k=-1,
                        monitor=None,
                        every_n_epochs=every_n_epochs),
        LearningRateMonitor("epoch"),
    ],
    gradient_clip_val=3.0
)
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

# %%
model = SimCLR(max_epochs=max_epochs, lr=lr, hidden_dim=hidden_dim,
                temperature=temperature, weight_decay=weight_decay,
                vit=args.arch)
trainer.fit(model, train_loader, val_loader)