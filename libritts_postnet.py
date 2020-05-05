#!/usr/bin/env python
# coding: utf-8

import os
import torch
#import pandas as pd
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms, utils
import numpy as np
import librosa
import os
from os import listdir
from os.path import isfile,join
import re
import pickle



#from read_audio import read_pickles, read_mels_libritts, read_embeds_libritts
from read_audio import read_postnet_mels_and_tags

from dataset import PostnetDataset

#from melLM import Encoder

from model import get_unet_generator
from model import get_discriminator
from train import run_trainer

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[ ]:


"""
class Params:
    def __init__(self):
        self.input_size = 80
        self.batch_size = 35
        self.num_epochs = 50
        self.lr = 1e-4
        self.restart_file=''
        self.dump_mels = True
        self.save_epoch = 5
        self.max_tgt_length = 501
        self.is_notebook = False
        self.plots_dir = './plots'
        self.run_name = 'libritts
"""
        

class Params:
    def __init__(self):
        self.batch_size = 64
        self.num_epochs = 1
        self.lr_G= 0.0002
        self.lr_D = 0.0002
        self.nc = 1
        self.nz = 100
        self.ngf = 64
        self.ndf = 64
        #for unet 
        self.nc_out = 1
        self.num_downsample = 4
        self.dataroot = '/home/praveen/projects/Speech/postnet_experiments/Postnet/speech_scripts/single_npy_dumps'
        self.metadata_dir = self.dataroot
        self.plots_dir = './plots'
        self.test_plots_dir = './plots/test'
        self.run_name = 'libritts'
        self.workers = 1
        self.restart_file = '130'
        self.save_epoch=5
        self.cuda = True
        self.lambda_L1 = 100
        self.beta1 = 0.5

# In[ ]:


params = Params()


data_path_train = os.path.join(params.dataroot, 'train')
metadata_file_train = os.path.join(params.metadata_dir,'train.txt')

data_path_test = os.path.join(params.dataroot, 'train')
metadata_file_test = os.path.join(params.metadata_dir, 'test.txt')

print('Creating groupings to class mels and associated assets')        

mel_dataset_train = PostnetDataset(data_path_train, metadata_file_train)
mel_dataset_test = PostnetDataset(data_path_test, metadata_file_test)

#create train loader 
train_loader = DataLoader(mel_dataset_train, batch_size=params.batch_size,shuffle=True,num_workers=1)

test_loader = DataLoader(mel_dataset_test, batch_size = params.batch_size, shuffle=False,num_workers=1)

print('size of train loader', len(train_loader))
print('size of test loader', len(test_loader))
#print('yes yes')

postnet = get_unet_generator(params)
postnet = postnet.cuda()

disc = get_discriminator(params)
disc = disc.cuda()

print(params.batch_size)

postnet_optimizer = optim.Adam(postnet.parameters(), params.lr_G)
disc_optimizer = optim.Adam(disc.parameters(), params.lr_D)

run_trainer(train_loader, test_loader, postnet, disc, params)
