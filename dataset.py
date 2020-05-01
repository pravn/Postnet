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
        
        source.append(cropped_src)
        target.append(cropped_tgt)
        
    return source, target
    


class PostnetDataset(Dataset):
    def __init__(self, data_path, metadata_file):
        self.source_mels, self.target_mels = get_mels(data_path, metadata_file)


    def __len__(self):
        return len(self.source_mels)

    def __getitem__(self, idx):
        return self.source_mels[idx], self.target_mels[idx]
        
    
