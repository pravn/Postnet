from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn as nn


import numpy as np
import librosa
import os
from os import listdir
from os.path import isfile,join
import re
import pickle
import random

from train import L2Loss

def test(test_loader, params, postnet, epoch):
    from plotting import plot_mel
    from plotting import plot_loss

    test_loss = 0
    j = 0
    
    for i, sample in enumerate(test_loader):

        j+=1

        if (j==len(test_loader)):
            break

        src = sample['source']
        tgt = sample['target']

        src = Variable(src.float()).cuda()

        src = src.transpose(1,2)
        tgt = tgt.transpose(1,2)

        tgt = Variable(tgt).cuda()

        postnet_output = postnet(src)
        postnet_residual = L2Loss(postnet_output, tgt)

        test_loss += postnet_residual.item()

        #plot things
        if(j==1):
            plot_mel(postnet_output[1].data.cpu().numpy(), 'recon_test_'+str(epoch), params)
            plot_mel(tgt[1].data.cpu().numpy(), 'target_test_'+str(epoch), params)
            plot_mel(src[1].data.cpu().numpy(), 'source_test_'+str(epoch), params)
    

    print('test_loss', test_loss)
    return test_loss
            

def run_tester(test_loader, params, postnet, epoch):
    import os
    from plotting import plot_loss

    loss_array = []

    postnet.eval()

    loss = test(test_loader, params, postnet, epoch)

    return loss

    
        
    
    
    
