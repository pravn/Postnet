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

from train import L1Loss

def test(test_loader, params, postnet, epoch):
    from plotting import plot_mel
    from plotting import write_test_image

    test_loss = 0
    j = 0

    device = torch.device("cuda:0")
    
    for i, (src,tgt) in enumerate(test_loader):

        j+=1

        if (j==len(test_loader)):
            break

        src = Variable(src).cuda()
        src = src.unsqueeze(1)

        tgt = Variable(tgt).cuda()
        tgt = tgt.unsqueeze(1)

        fake = postnet(src)
        postnet_loss = params.lambda_L1*L1Loss(fake, tgt)

        test_loss += postnet_loss.item()

        fake = fake.squeeze(1)
        tgt = tgt.squeeze(1)
        src = src.squeeze(1)

        #plot things
        plots_dir = params.test_plots_dir
        if(j==1):
            write_test_image(fake[1].data.cpu().numpy().T, 'fake_test_'+str(epoch), params)
            write_test_image(tgt[1].data.cpu().numpy().T, 'target_test_'+str(epoch), params)
            write_test_image(src[1].data.cpu().numpy().T, 'source_test_'+str(epoch), params)
    

    print('test_loss', test_loss)
    return test_loss
            

def run_tester(test_loader, params, postnet, epoch):
    import os
    from plotting import plot_loss

    loss_array = []

    postnet.eval()

    loss = test(test_loader, params, postnet, epoch)

    return loss

    
        
    
    
    
