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

criterion_L1=nn.L1Loss(reduction='sum')
criterion_L2 = nn.MSELoss(reduction='sum')
criterion_BCE = nn.BCELoss()


def L1Loss(input,target):
    t = target[:input.size(0)][:][:]
    t = t.transpose(0,1)
    inp = input.transpose(0,1)
    L = criterion_L1(inp, t)

    return L

def L2Loss(input,target):
    t = target[:input.size(0)][:][:]
    t = t.transpose(0,1)
    inp = input.transpose(0,1)
    L = criterion_L2(inp, t)

    return L




def train(train_loader,params,postnet,
          postnet_optimizer, disc, disc_optimizer, epoch):

    from plotting import plot_mel
    from plotting import plot_attn
    from plotting import plot_loss

    train_loss = 0
    j = 0
    
    for i, sample in enumerate(train_loader):
        #print('l', len(train_loader))
        j+=1
        if(j==len(train_loader)):
            break


        print('batch', i)

        postnet_optimizer.zero_grad()
        disc_optimizer.zero_grad()
        #disc_optimizer.zero_grad()

        for p in postnet.parameters():
            p.requires_grad = True
        
        for p in disc.parameters():
            p.requires_grad = False

        src = sample['source']
        tgt = sample['target']

        #(B,seq_len, dim)
        src = Variable(src.float()).cuda()

        src = src.transpose(1,2)
        tgt = tgt.transpose(1,2)

        tgt = Variable(tgt).cuda()

        postnet_output = postnet(src)

        print('postnet_output.size()', postnet_output.size())
        d_postnet_output = disc(postnet_output)
        d_tgt = disc(tgt)
        
        postnet_residual = L2Loss(d_postnet_output, d_tgt)
        
        postnet_residual.backward(retain_graph=True)
        postnet_optimizer.step()

        train_loss += postnet_residual.item()

        for p in postnet.parameters():
            p.requires_grad = False

        for p in disc.parameters():
            p.requires_grad = True


        d_postnet_output = disc(postnet_output)
        d_target = disc(tgt)

        fake_label = torch.full((params.batch_size,), 0, device=0)
        target_label = torch.full((params.batch_size,), 1, device=0)

        errD_postnet = criterion_BCE(d_postnet_output, target_label)
        errD_postnet.backward(retain_graph=True)
        D_postnet = d_postnet_output.mean().item()
        
        errD_target = criterion_BCE(d_target, target_label)
        errD_target.backward()
        D_target = d_target.mean().item()

        errD = errD_postnet + errD_target

        disc_optimizer.step()
        


        #plot things
        if(j==1):
            plot_mel(postnet_output[1].data.cpu().numpy(), ' recon_train_'+str(epoch), params)
            plot_mel(tgt[1].data.cpu().numpy(), ' target_train_'+str(epoch), params)
            plot_mel(src[1].data.cpu().numpy(), ' source_train_'+str(epoch), params)
            


    print('train_loss', train_loss)
    
    return train_loss

            

        

def run_trainer(train_loader, test_loader, params,postnet,postnet_optimizer, disc, disc_optimizer):
    import os
    from plotting import plot_loss
    #from test_attn import run_tester
    from test import run_tester

    dump_dir = 'dumps'
    restart_file = str(params.restart_file)
    #restart_file = str(59)
    
    if not os.path.exists('./'+dump_dir):
        os.makedirs(dump_dir)


    if restart_file!='':
        print('loading postnet, discriminator')
        postnet.load_state_dict(torch.load(dump_dir+'/postnet_epoch_'+restart_file+'.pth'))
        disc_optimizer.load_state_dict(torch.load(dump_dir+'/disc_epoch_'+restart_file+'.pth'))
    
    num_epochs = params.num_epochs

    loss_array = []
    test_loss_array = []

    for epoch in range(num_epochs):
        #set these to true since we are training
        #we flip them to false in the tester
        #postnet.training = True
        postnet.train()
        
        print('epoch',epoch)
        loss = train(train_loader,params,postnet,postnet_optimizer, disc, disc_optimizer, epoch)
        test_loss = run_tester(test_loader, params, postnet, epoch)

        loss_array.append(loss)
        plot_loss(loss_array, 'train', params)

        test_loss_array.append(test_loss)
        plot_loss(test_loss_array, 'test', params)

        if epoch%params.save_epoch==0:
            print('saving logs for epoch ', epoch)
            torch.save(postnet.state_dict(), '%s/postnet_epoch_%d.pth' % (dump_dir, epoch))
            torch.save(disc.state_dict(), '%s/disc_epoch_%d.pth' % (dump_dir, epoch))



    
    
