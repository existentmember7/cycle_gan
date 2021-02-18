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

from . import networks
from data import data

from dataset.datasets import *
from utils.utils import *

import itertools

class CycleGANModel():
    def __init__(self, _opt):

        self.opt = _opt

        # Loss fuctions
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        
        # Generator and Discriminator
        # self.generator = networks.Generator(self.opt)
        # self.discriminator = networks.Discriminator(self.opt)

        self.G_AB = networks.GeneratorResNet(self.opt)
        self.G_BA = networks.GeneratorResNet(self.opt)
        self.D_A = networks.Discriminator(self.opt)
        self.D_B = networks.DataLoader(self.opt)

        # Data
        self.dataloader = DataLoader(data.CustomDataset(self.opt), batch_size = self.opt.batch_size, shuffle = True)

        # # Optimizers
        # self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        # self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2)
        )
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

        # Learning rate update schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step
        )
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step
        )
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch, self.opt.decay_epoch).step
        )

        # cuda
        self.cuda = True if torch.cuda.is_available() else False
