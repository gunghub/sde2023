
from __future__ import print_function
import os, math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm_class
import torchvision
from PIL import Image
from copy import deepcopy


def activation(x):
    # swish
    return x*torch.sigmoid(x)
    # Swish is similar to GeLU. People tend to use this more than ReLU nowadays.

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, emb_dim=256):
        '''
        in_channels: Number of image channels in input.
        out_channels: Number of image channels in output.
        emb_dim: Length of conditional embedding vector.
        '''
        super().__init__()
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)

        self.proj = nn.Linear(emb_dim, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t):
        '''
        h and x have dimension B x C x H x W,
        where B is batch size,
              C is channel size,
              H is height,
              W is width.
        t is the conditional embedding.
        t has dimension B x V,
        where V is the embedding dimension.
        '''
        h = x
        h = self.norm1(h)
        h = activation(h)
        h = self.conv1(h)

        h = h + activation(self.proj(t))[:,:,None,None]
        h = activation(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            x = self.shortcut(x)
            
        return x+h


class DownSampling(nn.Module):
    ''' Downsampling block.'''
    def __init__(self, in_channels, out_channels):
        '''
        This block downsamples the feature map size by 2.
        in_channels: Number of image channels in input.
        out_channels: Number of image channels in output.
        '''
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x, t):
        ''' x is the feature maps; t is the conditional embeddings. '''
        x = self.pool(x) # The max pooling decreases feature map size by factor of 2
        x = self.conv(x, t)
        return x

class UpSampling(nn.Module):
    ''' Upsampling block.'''
    def __init__(self, in_channels, out_channels):
        '''
        This block upsamples the feature map size by 2.
        in_channels: Number of image channels in input.
        out_channels: Number of image channels in output.
        '''
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip_x, t):
        ''' 
        x is the feature maps; 
        skip_x is the skipp connection feature maps;
        t is the conditional embeddings.
        '''
        x = self.up(x) # The upsampling increases the feature map size by factor of 2
        x = torch.cat([skip_x, x], dim=1) # concatentate skip connection
        x = self.conv(x, t)
        return x
    
    

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, conditional=True, emb_dim=256,marginal_prob_std=None):
        '''
        '''
        super().__init__()
        self.emb_dim = emb_dim
        self.inc = ResidualBlock(c_in, 64)
        self.down1 = DownSampling(64, 128) 
        self.down2 = DownSampling(128, 256)
        self.down3 = DownSampling(256, 256)

        self.bot1 = ResidualBlock(256, 512)
        self.bot2 = ResidualBlock(512, 512)
        self.bot3 = ResidualBlock(512, 512)
        self.bot4 = ResidualBlock(512, 256)

        self.up1 = UpSampling(512, 128)
        self.up2 = UpSampling(256, 64)
        self.up3 = UpSampling(128, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        self.marginal_prob_std = marginal_prob_std
        # nn.Embedding implements a dictionary of num_classes prototypes
        self.conditional = conditional
        if conditional:
            num_classes = 2

            self.gender = nn.Parameter(torch.randn(num_classes, emb_dim))
            
    def temporal_encoding(self, timestep):
        ''' 
        This implements the sinusoidal temporal encoding for the current timestep. 
        Input timestep is a tensor of length equal to the batch size
        Output emb is a 2D tensor B x V,
            where V is the embedding dimension.
        '''
        
        assert len(timestep.shape) == 1
        half_dim = self.emb_dim // 2
        embedding = math.log(10000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -embedding)
        embedding = embedding.to(device=timestep.device)
        embedding = timestep.float()[:, None] * embedding[None, :]
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
        if self.emb_dim % 2 == 1:  # zero pad
            embedding = torch.nn.functional.pad(embedding, (0,1,0,0))
        return embedding

    def unet_forward(self, x, t):
        # x: B x 3 x 224 x 224
        x1 = self.inc(x, t)    # x1: B x 64 x 64 x 64
        x2 = self.down1(x1, t) # x2: B x 128 x 32 x 32
        x3 = self.down2(x2, t) # x3: B x 256 x 16 x 16
        x4 = self.down3(x3, t) # x3: B x 256 x 8 x 8

        x4 = self.bot1(x4, t) # x4: B x 512 x 8 x 8
        # Removing bot2 and bot3 can save some time at the expense of quality
        x4 = self.bot2(x4, t) # x4: B x 512 x 8 x 8
        x4 = self.bot3(x4, t) # x4: B x 512 x 8 x 8
        x4 = self.bot4(x4, t) # x4: B x 256 x 8 x 8

        x = self.up1(x4, x3, t) # x: B x 128 x 16 x 16
        x = self.up2(x, x2, t)  # x: B x 64 x 32 x 32
        x = self.up3(x, x1, t)  # x: B x 64 x 64 x 64
        output = self.outc(x)   # x: B x 3 x 64 x 64
        
        # print(output.shape)
        # print(self.marginal_prob_std(t).unsqueeze(-1).unsqueeze(-1).shape)

        # output = output / self.marginal_prob_std(t).unsqueeze(-1).unsqueeze(-1)
        return output
    
    def forward(self, x, t, y=None):
        if self.conditional:

            temporal_encoding = self.temporal_encoding(t)

            if y is not None:
                # Calculate the gender vector
                gender_vector = self.gender[y]

                # Add the temporal encoding and gender vector to get the final conditional vector
                c = temporal_encoding + gender_vector
            else:
                c = temporal_encoding
                
        else:
            c = self.temporal_encoding(t)
        return self.unet_forward(x, c)