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

def random_crop(source, target, cropsize=64):
    width = source.shape[0]
    height = source.shape[1]
    
    j = np.random.randint(width)
    i = np.random.randint(height)

        
    while (j+cropsize//2>width) or (j-cropsize//2<0):
        #print(j+cropsize,j-cropsize)
        j = np.random.randint(width)
        
    #print('end j',j)
    
    jplus = j+cropsize//2
    jminus = j-cropsize//2
        
    while (i+cropsize//2>height) or (i-cropsize//2<0):
        #print(i+cropsize,i-cropsize)
        i = np.random.randint(height)
        
    #print('end i', i)
    
    iplus = i+cropsize//2
    iminus = i-cropsize//2
        
    #print(jminus,jplus,iminus,iplus)
    
    cropped_source = source[jminus:jplus,iminus:iplus]
    cropped_target = target[jminus:jplus,iminus:iplus]
    
    
    return cropped_source, cropped_target

def pad_mels(source, target, padded_width=512,padded_height=128):
    width = source.shape[0]
    height = source.shape[1]
    padded_source = np.zeros((padded_width,padded_height))
    padded_target = np.zeros((padded_width,padded_height))

    padded_source[:width,:height] = source
    padded_target[:width,:height] = target
    
    return padded_source.astype(float), padded_target.astype(float)
    

def normalize(x):
    min_x = x.min()
    #max_x = x.max()
    max_x = 0.7

    y = 1.0-2.0*(max_x-x)/(max_x-min_x+1)

    return y
    
def get_mels(path,metadata_file):
    import os 
    source = []
    target = []
    
    with open(metadata_file,'r') as metafile:
        entries = [entry.split('\n')[0] for entry in metafile]
    metafile.close()
    
    for file in entries:
        src_file = np.load(os.path.join(path,'recon_'+file+'.npy'))
        tgt_file = np.load(os.path.join(path,'target_'+file+'.npy'))

        cropped_src, cropped_tgt = random_crop(src_file,tgt_file)

        #crop until we get max greater than 0.1
        while(cropped_src.max()<0.1):
            cropped_src, cropped_tgt = random_crop(src_file,tgt_file)


        cropped_src = normalize(cropped_src)
        cropped_tgt = normalize(cropped_tgt)
        
        
        source.append(cropped_src)
        target.append(cropped_tgt)
        
    return source, target


def get_padded_mels(path,metadata_file):
    import os 
    source = []
    target = []
    
    with open(metadata_file,'r') as metafile:
        entries = [entry.split('\n')[0] for entry in metafile]
    metafile.close()
    
    for file in entries:
        src_file = np.load(os.path.join(path,'recon_'+file+'.npy'))
        tgt_file = np.load(os.path.join(path,'target_'+file+'.npy'))

        padded_src, padded_tgt = pad_mels(src_file,tgt_file)

        padded_src = normalize(padded_src)
        padded_tgt = normalize(padded_tgt)
        
        source.append(padded_src)
        target.append(padded_tgt)
        
    return source, target
    


class PostnetDataset(Dataset):
    def __init__(self, data_path, metadata_file):
        self.source_mels, self.target_mels = get_mels(data_path, metadata_file)

    def __len__(self):
        return len(self.source_mels)

    def __getitem__(self, idx):
        return self.source_mels[idx], self.target_mels[idx]
        
    

class PostnetInferenceDataset(Dataset):
    def __init__(self, data_path, metadata_file):
        self.source_mels, self.target_mels = get_padded_mels(data_path, metadata_file)


    def __len__(self):
        return len(self.source_mels)

    def __getitem__(self, idx):
        return self.source_mels[idx], self.target_mels[idx]
    
