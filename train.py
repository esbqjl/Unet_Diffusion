from __future__ import print_function
import os, math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from sde import marginal_prob_std
from sde import diffusion_coeff
from sde import loss_fn
from unet import UNet
from dataset import CelebADataset
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader

import tqdm.notebook
# We will resize to 64 x 64 for this assignment
image_size = 64

# Hyperparameters
batch_size = 64
learning_rate = 0.0002
weight_decay = 0.00001 # (L2 penalty)

# Transform used for training
train_transform = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), 
                                                    (0.5, 0.5, 0.5)),
                           ])

# Make the dataset
dataset = CelebADataset(transform=train_transform)

# Run on GPU
device = 'cuda'

#@title Training (double click to expand or collapse)


sigma =  25.0 #@param {'type':'number'}
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
score_model = torch.nn.DataParallel(UNet(marginal_prob_std = marginal_prob_std_fn))
score_model = score_model.to(device)

n_epochs =   50#@param {'type':'integer'}
## size of a mini-batch
batch_size =  32 #@param {'type':'integer'}
## learning rate
lr=1e-4 #@param {'type':'number'}


data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

trainloader = torch.utils.data.DataLoader(dataset, drop_last=True,  batch_size=batch_size, shuffle=True, num_workers=8)

optimizer = Adam(score_model.parameters(), lr=lr)
tqdm_epoch = tqdm.notebook.trange(10)
for epoch in tqdm_epoch:
  avg_loss = 0.
  num_items = 0
  for x, y in trainloader:
    x = x.to(device)    
    loss = loss_fn(score_model, x, marginal_prob_std_fn)
    optimizer.zero_grad()
    loss.backward()    
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  # Print the averaged training loss so far.
  tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
  # Update the checkpoint after each epoch of training.
  torch.save(score_model.state_dict(), 'ckpt.pth')