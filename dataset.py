r"""
    #### Downloading data

In this assignment you will be generating images of faces using the CelebA dataset. The CelebA dataset is in a zip file called "img_align_celeba.zip". The labels are contained in a text file called "list_attr_celeba.txt"

You can find the two files here:

*  https://drive.google.com/file/d/1M6wTJ4UEzY8_tasInp_7BcbGda1PX5Zk/view?usp=sharing
*  https://drive.google.com/file/d/1KM41HV3IrmO9epcEsG0PzJ2aJxTl-GZH/view?usp=sharing

or here on the SCC:

* /projectnb/dl523/img_align_celeba.zip
* /projectnb/dl523/list_attr_celeba.txt
! ls
%cd ./drive/MyDrive
! ls
! cp img_align_celeba.zip /content 
! cp list_attr_celeba.txt /content 
# from path will differ depending on where you saved the zip file in Google Drive
! unzip -DD -q  /content/img_align_celeba.zip -d  /content/
! cp /content/list_attr_celeba.txt /content/img_align_celeba


"""


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
import torchvision
from PIL import Image
from copy import deepcopy
dataroot = "img_align_celeba/" #if on colab
# The CelebA dataset contains 40 binary attribute labels for each image
attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 
 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 
 'Big_Lips', 'Big_Nose', 'Black_Hair', 
 'Blond_Hair', 'Blurry', 'Brown_Hair', 
 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 
 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 
 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 
 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 
 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 
 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 
 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 
 'Young']

def set_random_seed(seed=999):
    # Set random seed for reproducibility
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, transform = None):
        '''Initialize the dataset.'''
        self.transform = transform
        self.root = dataroot
        self.attr_txt = dataroot + 'list_attr_celeba.txt'
        self._parse()
    
    def _parse(self):
        '''
        Parse the celeba text file.
        Pupulate the following private variables:
         - self.ys: A list of 1D tensors with 40 binary attribute labels.
         - self.im_paths: A list of strings (image paths).
        '''
        self.im_paths = [] # list of jpeg filenames 
        self.ys = []       # list of attribute labels
        
        def _to_binary(lst):
            return torch.tensor([0 if lab == '-1' else 1 for lab in lst])
            
        with open(self.attr_txt) as f:
            for line in f:
                assert len(line.strip().split()) == 41
                fl = line.strip().split()
                if fl[0][-4:] == '.jpg': # if not header
                  
                    self.im_paths.append(self.root + fl[0]) # jpeg filename
                    self.ys.append(_to_binary(fl[1:]))      # 1D tensor of 40 binary attributes
        
    def __len__(self):
        '''Return length of the dataset.'''
        return len(self.ys)

    def __getitem__(self, index):
        '''
        Return the (image, attributes) tuple.
        This function gets called when you index the dataset.
        '''
        def img_load(index):
            imraw = Image.open(self.im_paths[index])
            im = self.transform(imraw)
            return im

        target = self.ys[index]
        return img_load(index), target