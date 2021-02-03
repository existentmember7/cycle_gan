from data.data import CustomDataset
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from data import CustomDataset
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from models.cycleGAN_model import CycleGANModel
from options import Option


def sample_image(n_row, batches_done, model):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, model.opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = model.generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

opt = Option()
model = CycleGANModel(opt)
dataset = CustomDataset(opt)
dataloader = DataLoader()

if model.cuda:
    model.generator.cuda()
    model.discriminator.cuda()
    model.loss_fuction.cuda()

FloatTensor = torch.cuda.FloatTensor if model.cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if model.cuda else torch.LongTensor

# ----------
#  Training
# ----------

for epoch in range(model.opt.n_epochs):
    for i, (imgs, labels) in enumerate(model.dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        model.optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, model.opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, model.opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = model.generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = model.discriminator(gen_imgs, gen_labels)
        g_loss = model.loss_fuction(validity, valid)

        g_loss.backward()
        model.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        model.optimizer_D.zero_grad()

        # Loss for real images
        validity_real = model.discriminator(real_imgs, labels)
        d_real_loss = model.loss_fuction(validity_real, valid)

        # Loss for fake images
        validity_fake = model.discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = model.loss_fuction(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        model.optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, model.opt.n_epochs, i, len(model.dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(model.dataloader) + i
        if batches_done % model.opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done, model=model)