import os
import torch
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
import math


import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
#import torchvision.transforms as transforms
#import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
import functools


''' Adapted from pytorch resnet example
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py'''

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        # print m.__class__.__name__
        m.weight.data.normal_(0.0, 0.02)

class ResnetBlockBasic(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResnetBlockBasic, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.in1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.in2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class convBlock(nn.Module):
    def __init__ (self, inplanes, outplanes, kernel, stride, padding):
        super(convBlock, self).__init__()
        model = []
        model += [nn.Conv2d(inplanes, outplanes, kernel, stride, padding, bias=False)]
        model += [nn.BatchNorm2d(outplanes)]
        model += [nn.LeakyReLU(0.2, inplace=True)]
        #model += [nn.SELU(inplace=True)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class DeconvBlock(nn.Module):
    def __init__ (self, inplanes, outplanes, kernel, stride, padding):
        super(DeconvBlock, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(inplanes, outplanes, kernel, stride, padding, bias=False)]
        model += [nn.BatchNorm2d(outplanes)]
        model += [nn.ReLU(inplace=True)]
        #model += [nn.SELU(inplace=True)]
        #model += [ResnetBlockBasic(outplanes, outplanes)]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer

        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downconv_bogus = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                                   stride=0,padding=1,bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            #upconv = [upconv]+[upnorm]+[uprelu]+[nn.ConvTranspose2d(outer_nc, 3, 
            #                                        kernel_size=4,stride=2,
            #                                        padding=1)]

            # This is the hack to inflate make an assymetrical encoder-decoder
            #upconv_final = nn.ConvTranspose2d(outer_nc, input_nc, kernel_size=4,
            #                                  stride=2, padding=1)
            
            down = [downconv]
            #Default - symmetrical Unet
            up = [uprelu, upconv , nn.Tanh()]
            #For an assymetrical decoder, we add an extra upconv_final layer
            #up = [uprelu, upconv, upnorm, upconv_final, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(Discriminator, self).__init__()

        layers = []

        #(nc, 64, 64)
        layers += [nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                                 nn.LeakyReLU(0.2, True))]
        
        #(ndf, 32, 32)
        layers += [convBlock(ndf, 2 * ndf, 4, 2, 1)]

        #(2 * ndf, 16, 16)
        layers += [convBlock(2 * ndf, 4 * ndf, 4, 2, 1)]

        #(4 * ndf, 8, 8)
        layers += [convBlock(4 * ndf, 8 * ndf, 4, 2, 1)]
 
        layers1 = []
       
        #(8 * ndf, 4, 4)
        layers1 += [nn.Conv2d(8 * ndf, 1, 4, 1, 0, bias=False)]
        layers1 += [nn.Sigmoid()]

        self.layers = nn.Sequential(*layers)
        self.layers1 = nn.Sequential(*layers1)

    def forward(self, x):
        feats = self.layers(x)
        output = self.layers1(feats)
        return output.squeeze(), feats.squeeze()

        
def get_unet_generator(params):
    p = UnetGenerator(params.nc,params.nc,params.num_downsample)
    return p

def get_discriminator(params):
    d = Discriminator(params.nc,params.ndf)
    return d
