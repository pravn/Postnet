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

def inference(inference_loader, params, postnet):
    from plotting import plot_mel
    from plotting import write_inference_image

    inference_loss = 0
    j = 0

    device = torch.device("cuda:0")
    
    for i, (src,tgt) in enumerate(inference_loader):

        j+=1

        if (j==len(inference_loader)):
            break

        src = Variable(src).cuda()
        src = src.unsqueeze(1)

        tgt = Variable(tgt).cuda()
        tgt = tgt.unsqueeze(1)

        fake = postnet(src)
        postnet_loss = params.lambda_L1*L1Loss(fake, tgt)

        inference_loss += postnet_loss.item()

        fake = fake.squeeze(1)
        tgt = tgt.squeeze(1)
        src = src.squeeze(1)

        #plot things
        plots_dir = params.inference_plots_dir
        if(j==1):
            write_inference_image(fake[1].data.cpu().numpy().T, 'fake_test_', params)
            write_inference_image(tgt[1].data.cpu().numpy().T, 'target_test_', params)
            write_inference_image(src[1].data.cpu().numpy().T, 'source_test_', params)
    

    print('inference_loss', inference_loss)
    return inference_loss
            

def run_inference(inference_loader, params, postnet):
    import os
    from plotting import plot_loss

    loss_array = []

    dump_dir = 'dumps'
    restart_file = str(params.restart_file)
    #restart_file = str(59)
    

    if restart_file!='':
        print('loading postnet, discriminator')
        postnet.load_state_dict(torch.load(dump_dir+'/postnet_epoch_'+restart_file+'.pth'))
        #disc.load_state_dict(torch.load(dump_dir+'/disc_epoch_'+restart_file+'.pth'))

    else:
        raise Exception('Restart file cannot be empty in inference mode')
    

    postnet.eval()

    

    loss = inference(inference_loader, params, postnet)

    return loss

    
        
    
    
    
