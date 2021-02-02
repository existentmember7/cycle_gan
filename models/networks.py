import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, _opt):
        super(Generator, self).__init__()
        self.opt = _opt

        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.n_classes)

        self.init_size = self.opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(self.opt.latent_dim + self.opt.n_classes, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, z, noise,labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, _opt):
        super(Discriminator, self).__init__()
        self.opt = _opt

        self.label_embedding = nn.Embedding(self.opt.n_classes, self.opt.n_classes)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = self.opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2 + self.opt.n_classes, 1), nn.Sigmoid())

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        
        out = self.model(img)
        out = torch.cat((out.view(out.shape[0], -1), self.label_embedding(labels)), -1)
        validity = self.adv_layer(out)

        return validity

