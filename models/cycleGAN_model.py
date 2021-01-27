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

class CycleGANModel():
    def __init__(self, _opt):

        self.opt = _opt

        # Loss fuctions
        self.loss_fuction = torch.nn.MSELoss()
        
        # Generator and Discriminator
        self.generator = networks.Generator()
        self.discriminator = networks.Discriminator()

        # Data
        self.dataloader = 0

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr, betas=(self.opt.b1, self.opt.b2))

        # cuda
        self.cuda = True if torch.cuda.is_available() else False
