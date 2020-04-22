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

from model import get_postnet
from model import get_discriminator
from train import run_trainer

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[ ]:


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
        self.run_name = 'libritts'
        


# In[ ]:


os.getcwd()
#from main_mel_seq2seq import get_encoder, get_decoder


# In[ ]:


params = Params()


mels_dir = './libritts/mels'

metadata_train = './libritts/train.txt'
metadata_test = './libritts/test.txt'

sv_train, tags_train = read_postnet_mels_and_tags(mels_dir, metadata_train, 'recon')
tv_train, _ = read_postnet_mels_and_tags(mels_dir, metadata_train, 'target')
sv_test, tags_test = read_postnet_mels_and_tags(mels_dir, metadata_test, 'recon')
tv_test, _ = read_postnet_mels_and_tags(mels_dir, metadata_test, 'target')

print('Creating groupings to class mels and associated assets')        


mel_dataset_train = PostnetDataset(sv_train, tv_train)
mel_dataset_test = PostnetDataset(sv_test, tv_test)

#create train loader 
train_loader = DataLoader(mel_dataset_train, batch_size=params.batch_size,shuffle=True,num_workers=1)
test_loader = DataLoader(mel_dataset_test, batch_size=params.batch_size,shuffle=True,num_workers=1)


print('size of train loader', len(train_loader))
print('size of test loader', len(test_loader))

#print('yes yes')

params = Params()

postnet = get_postnet(params)
postnet = postnet.cuda()

disc = get_discriminator(params)
disc = disc.cuda()

print(params.batch_size)

postnet_optimizer = optim.Adam(postnet.parameters(), params.lr)
disc_optimizer = optim.Adam(disc.parameters(), params.lr)

run_trainer(train_loader, test_loader, params, postnet, postnet_optimizer, disc, disc_optimizer)
