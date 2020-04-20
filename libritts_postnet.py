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


from read_audio import read_pickles, read_mels_libritts, read_embeds_libritts

from dataset import Mel
from dataset import MelDataset
from dataset import make_grouping


#from melLM import Encoder
from melLM import EncoderCell
from melLM import BahdanauAttnDecoderRNN
#from melLM import NoAttnDecoder
from melLM import Attn
from melLM import Conv_FB_Highway
#from main_mel_seq2seq import main

from melLM import get_encoder
from melLM import get_decoder
from melLM import get_conv_fb_highway
#from melLM import get_postnet
from train_attn import run_trainer
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[ ]:


class Params:
    def __init__(self):
        self.input_size = 80
        self.num_layers = 2
        self.stack_size = 2
        self.r = 1
        self.hidden_size = 600
        self.batch_size = 35
        self.seq_len_max = 100
        self.num_epochs = 551
        self.lr = 0.3*1e-4
        self.restart_file='25'
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


# In[ ]:


'''
def get_mels(dir, voice):
    
    voice = read_pickles(dir,voice) #dict 
    print('name', voice['name'])
   
    v = voice['mels']    
    return v
'''



mels_dir = './libritts/mels'
embeds_dir = './libritts/embeddings'
metadata_train = './libritts/train.txt'
metadata_test = './libritts/test.txt'

#sv_train = get_mels(mels_root_dir, source_train_voices)
#tv_train = get_mels(mels_root_dir, target_train_voices)
#sv_test = get_mels(mels_root_dir, source_test_voices)
#tv_test = get_mels(mels_root_dir, target_test_voices)

sv_train = read_mels_libritts(mels_dir, metadata_train)
tv_train = read_mels_libritts(mels_dir, metadata_train)
sv_test = read_mels_libritts(mels_dir, metadata_test)
tv_test = read_mels_libritts(mels_dir, metadata_test)

embeds_train = read_embeds_libritts(embeds_dir, metadata_train)
embeds_test = read_embeds_libritts(embeds_dir, metadata_test)

print('Creating groupings to class mels and associated assets')        

mel_dataset_train = MelDataset(sv_train, tv_train, embeds_train, params.stack_size)
mel_dataset_test = MelDataset(sv_test, tv_test, embeds_test, params.stack_size)

maxlen_source = mel_dataset_train.maxlen_source

#create train loader 
train_loader = DataLoader(mel_dataset_train, batch_size=params.batch_size,shuffle=True,num_workers=1)
test_loader = DataLoader(mel_dataset_test, batch_size=params.batch_size,shuffle=True,num_workers=1)


print('size of train loader', len(train_loader))
print('size of test loader', len(test_loader))

#print('yes yes')

params = Params()

encoder = get_encoder(params)
encoder = encoder.cuda()


decoder = get_decoder(params)
decoder = decoder.cuda()

#postnet = get_postnet(params)
#postnet = postnet.cuda()

print('encoder', encoder)


# In[ ]:

# In[ ]:


print(params.batch_size)



for i,[sample,seq_len] in enumerate(train_loader):
    src = sample['source']
    tgt = sample['target']
    mask = sample['mask']
    print('src.shape',src[0].shape)
    #plot_mel(src[0].numpy())
    #plot_mel(tgt[0].numpy())
    break

seq_len = src[0].shape[1]
#print(seq_len)


#seq_len = 208
conv_fb_highway = get_conv_fb_highway(params,seq_len)
conv_fb_highway = conv_fb_highway.cuda()


# In[ ]:


encoder_optimizer = optim.Adam(encoder.parameters(),params.lr)
decoder_optimizer = optim.Adam(decoder.parameters(), params.lr)
conv_fb_highway_optimizer = optim.Adam(conv_fb_highway.parameters(),params.lr)
#postnet_optimizer = optim.Adam(postnet.parameters(), params.lr)


# In[ ]:


run_trainer(train_loader, test_loader,params, encoder,decoder,conv_fb_highway, encoder_optimizer,decoder_optimizer,conv_fb_highway_optimizer)
