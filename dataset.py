from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


import numpy as np
import librosa
import os
from os import listdir
from os.path import isfile,join
import re
import pickle

class Mel:
    def __init__(self,mel):
        self.mel = mel
        self.num_bins = mel.shape[0]
        self.seq_len  = mel.shape[1]


def zero_pad_end(mels):
    num_samples = len(mels)

    maxlen = 0
    for i in range(len(mels)):
        if(maxlen<mels[i].shape[1]):
            maxlen = mels[i].shape[1]

    #pad to maxlen
    for i in range(len(mels)):
        mels[i,:,:maxlen] = 0



class PostnetDataset(Dataset):
    def __init__(self, source_mels, target_mels):
        self.source_mels = source_mels
        self.target_mels = target_mels

    def __len__(self):
        return len(self.source_mels)

    def __getitem__(self, idx):
        sample = {'source':self.source_mels[idx], 'target': self.target_mels[idx]}
        return sample
        
    
