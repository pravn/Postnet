from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


import numpy as np
import librosa
import os
from os import listdir
from os.path import isfile,join
import re
import pickle
import random

criterion_L1=nn.L1Loss(reduction='mean')
criterion_L2 = nn.MSELoss(reduction='mean')
criterion_BCE = nn.BCELoss(reduction='mean')


def L1Loss(input,target):
    L = criterion_L1(input, target)
    return L

def L2Loss(input,target):
    L = criterion_L2(input, target)
    return L

def BCELoss(input,label):
    L = criterion_BCE(input, label)
    return L


def train(train_loader,params,postnet,
          postnet_optimizer, disc, disc_optimizer, epoch):

    from plotting import plot_loss
    from plotting import write_image

    real_label = 1
    fake_label = 0
    
    batch_size = params.batch_size
    train_loss = 0
    j = 0

    device = torch.device("cuda:0")
    
    for i, (src,tgt) in enumerate(train_loader):
        #print('l', len(train_loader))
        j+=1
        if(j==len(train_loader)):
            break

        #print('batch', i)

        #(B,seq_len, dim)
        #src = Variable(src.float()).cuda()
        src = Variable(src).cuda()
        src = src.unsqueeze(1)

        #src = src.transpose(1,2)
        #tgt = tgt.transpose(1,2)

        tgt = Variable(tgt).cuda()
        tgt = tgt.unsqueeze(1)

        

        label = torch.full((batch_size,), real_label, device=device)

        '''
        for p in postnet.parameters():
            p.requires_grad = True
        
        for p in disc.parameters():
            p.requires_grad = False
        '''

        postnet_optimizer.zero_grad()

        fake = postnet(src)
        
        d_fake, feats_fake = disc(fake)
        D_fake_G = d_fake.mean().item()
        

        postnet_loss = params.lambda_L1*L1Loss(fake,tgt)
        label.fill_(real_label)
        gan_postnet_loss = BCELoss(d_fake, label)

        postnet_loss.backward(retain_graph=True)
        #postnet_loss.backward()
        gan_postnet_loss.backward(retain_graph=True)

        postnet_optimizer.step()

        train_loss += postnet_loss.item()

        #=====================================


        '''
        for p in postnet.parameters():
            p.requires_grad = False

        for p in disc.parameters():
            p.requires_grad = True
        '''

        disc_optimizer.zero_grad()
        
            
        d_real, _ = disc(tgt)
        label.fill_(real_label)

        D_real_D = d_real.mean().item()

        #print('d_real.size(),label.size()', d_real.size(),label.size())

        d_loss_real = BCELoss(d_real, label)
        d_loss_real.backward(retain_graph=True)

        fake = postnet(src)
        d_fake, _ = disc(fake.detach())

        D_fake_D = d_fake.mean().item()
        
        label.fill_(fake_label)

        d_loss_fake = BCELoss(d_fake, label)
        d_loss_fake.backward(retain_graph=True)

        disc_optimizer.step() 
        

        fake = fake.squeeze(1)
        tgt = tgt.squeeze(1)
        src = src.squeeze(1)
        
        #plot things
        if(j==1):
            write_image(fake[1].data.cpu().numpy().T, 'fake_train_'+str(epoch), params)
            write_image(tgt[1].data.cpu().numpy().T, 'target_train_'+str(epoch), params)
            write_image(src[1].data.cpu().numpy().T, 'source_train_'+str(epoch), params)

            print('postnet_loss:%.4f\tgan_postnet_loss:%.4f\td_loss_real:%.4f\td_loss_fake:%.4f'
                  %(postnet_loss.item(),gan_postnet_loss.item(),d_loss_real.item(),d_loss_fake.item()))

            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            
            print('D_fake_G:%.4f, D_real_D:%.4f, D_fake_D:%.4f'%(D_fake_G, D_real_D, D_fake_D))


    print('train_loss', train_loss)
    
    return train_loss


def run_trainer(train_loader, test_loader, postnet, disc, params):
    import os
    from plotting import plot_loss
    #from test_attn import run_tester
    from test import run_tester

    postnet_optimizer = optim.Adam(postnet.parameters(), lr=0.0002, betas=(params.beta1,0.999))
    postnet_scheduler = StepLR(postnet_optimizer, step_size=1000, gamma=1.0)
    
    disc_optimizer = optim.Adam(disc.parameters(), lr=0.00005, betas=(params.beta1,0.999))
    disc_scheduler = StepLR(disc_optimizer, step_size=1000, gamma=1.0)

    dump_dir = 'dumps'
    restart_file = str(params.restart_file)
    #restart_file = str(59)
    
    if not os.path.exists('./'+dump_dir):
        os.makedirs(dump_dir)


    if restart_file!='':
        print('loading postnet, discriminator')
        postnet.load_state_dict(torch.load(dump_dir+'/postnet_epoch_'+restart_file+'.pth'))
        disc.load_state_dict(torch.load(dump_dir+'/disc_epoch_'+restart_file+'.pth'))

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



    
    
