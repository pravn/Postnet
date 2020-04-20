from __future__ import print_function, division
import os
import torch
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math

class ConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nonlinear=None):
        super(ConvBNBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        conv1d = nn.Conv1d(in_channels,
                           out_channels,
                           kernel_size,
                           padding=padding)
        norm = nn.BatchNorm1d(out_channels)
        dropout = nn.Dropout(p=0.5)
        if nonlinear == 'relu':
            self.net = nn.Sequential(conv1d, norm, nn.ReLU(), dropout)
        elif nonlinear == 'tanh':
            self.net = nn.Sequential(conv1d, norm, nn.Tanh(), dropout)
        else:
            self.net = nn.Sequential(conv1d, norm, dropout)

    def forward(self, x):
        output = self.net(x)
        return output


class Postnet(nn.Module):
    def __init__(self, mel_dim, num_convs=5):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            ConvBNBlock(mel_dim, 512, kernel_size=5, nonlinear='tanh'))
        for _ in range(1, num_convs - 1):
            self.convolutions.append(
                ConvBNBlock(512, 512, kernel_size=5, nonlinear='tanh'))
        self.convolutions.append(
            ConvBNBlock(512, mel_dim, kernel_size=5, nonlinear=None))

    def forward(self, x):
        for layer in self.convolutions:
            x = layer(x)
        return x

def get_postnet(params):
    p = Postnet(params.input_size)
    return p

