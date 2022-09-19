import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from coordConv import addCoords, addCoords_1D
import math

cwd = os.getcwd()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Tcoil_S_UpCNN(nn.Module):  
    def __init__(self):
        super(Tcoil_S_UpCNN, self).__init__()

        ##Fully Connected Layers
        self.add_coords = addCoords_1D()
        self.lin1 = nn.Linear(5, 30) 
        self.lin2 = nn.Linear(30, 30)

        ##Transposed Convolution Layers
        self.tconv0 = nn.ConvTranspose1d(30, 30, 32, 1, 0)
        
        self.upsample1 = nn.Upsample(scale_factor=2,mode='nearest')
        self.reflection1 = nn.ReflectionPad1d(1)
        self.conv1 = nn.Conv1d(30, 30, 3, 1, 0)
        self.norm1 = nn.BatchNorm1d(30)
        
        self.upsample2 = nn.Upsample(scale_factor=2,mode='nearest')
        self.reflection2 = nn.ReflectionPad1d(1)
        self.conv2 = nn.Conv1d(30, 30, 3, 1, 0)
        self.norm2 = nn.BatchNorm1d(30)

        self.upsample3 = nn.Upsample(scale_factor=2,mode='nearest')
        self.reflection3 = nn.ReflectionPad1d(1)
        self.conv3 = nn.Conv1d(30, 30, 3, 1, 0)
        self.norm3 = nn.BatchNorm1d(30)

        self.upsample4 = nn.Upsample(scale_factor=2,mode='nearest')
        self.reflection4 = nn.ReflectionPad1d(1)
        self.conv4 = nn.Conv1d(30, 30, 3, 1, 0)
        self.norm4 = nn.BatchNorm1d(30)

        self.upsample5 = nn.Upsample(scale_factor=2,mode='nearest')
        self.reflection5 = nn.ReflectionPad1d(1)
        self.conv5 = nn.Conv1d(31, 12, 3, 1, 0)

    def fully_connected(self, x):
        latent = self.lin1(x)
        latent = F.elu(latent)

        latent = self.lin2(latent)
        latent = F.elu(latent)
        
        z = latent.view(-1, self.lin2.out_features, 1)
        return z

    def transposed_conv(self, z):
        latent = self.tconv0(z)
        latent = F.elu(latent)
        
        latent = self.upsample1(latent)
        latent = self.reflection1(latent)
        latent = self.conv1(latent)
        latent = self.norm1(latent)
        latent = F.elu(latent)
        
        latent = self.upsample2(latent)
        latent = self.reflection2(latent)
        latent = self.conv2(latent)
        latent = self.norm2(latent)
        latent = F.elu(latent)
        
        latent = self.upsample3(latent)
        latent = self.reflection3(latent)
        latent = self.conv3(latent)
        latent = self.norm3(latent)
        latent = F.elu(latent)
        
        latent = self.upsample4(latent)
        latent = self.reflection4(latent)
        latent = self.conv4(latent)
        latent = self.norm4(latent)
        latent = F.elu(latent)

        latent = self.upsample5(latent)
        latent = self.reflection5(latent)
        latent = self.add_coords(latent)
        recons_y = self.conv5(latent)
        return recons_y[..., :1000]

    def forward(self, x):
        z = self.fully_connected(x)
        out = self.transposed_conv(z)
        return out
