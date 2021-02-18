from data import data
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.utils import save_image, make_grid

from dataset.datasets import *
from utils.utils import *

from models.cycleGAN_model import CycleGANModel
from options import options


def sample_images(batches_done, model, val_dataset):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataset))
    model.G_AB.eval()
    model.G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = model.G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = model.G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)

opt = options.Option().opt
model = CycleGANModel(opt)
dataset = data.CustomDataset(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

if model.cuda:
    model.G_AB.cuda()
    model.G_BA.cuda()
    model.D_A.cuda()
    model.D_B.cuda()
    model.criterion_cycle.cuda()
    model.criterion_GAN.cuda()
    model.criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    model.G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, opt.epoch)))
    model.G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    model.D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, opt.epoch)))
    model.D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    model.G_AB.apply(weights_init_normal)
    model.G_BA.apply(weights_init_normal)
    model.D_A.apply(weights_init_normal)
    model.D_B.apply(weights_init_normal)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# ----------
#  Training
# ----------

for epoch in range(model.opt.n_epochs):
    for i, (imgs, labels) in enumerate(model.dataloader):

        # Set model input
        real_A = Variable(imgs["A"].type(Tensor))
        real_B = Variable(imgs["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *model.D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *model.D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        model.G_AB.train()
        model.G_BA.train()

        model.optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = model.criterion_identity(model.G_BA(real_A), real_A)
        loss_id_B = model.criterion_identity(model.G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = model.G_AB(real_A)
        loss_GAN_AB = model.criterion_GAN(model.D_B(fake_B), valid)
        fake_A = model.G_BA(real_B)
        loss_GAN_BA = model.criterion_GAN(model.D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = model.G_BA(fake_B)
        loss_cycle_A = model.criterion_cycle(recov_A, real_A)
        recov_B = model.G_AB(fake_A)
        loss_cycle_B = model.criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

        loss_G.backward()
        model.optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        model.optimizer_D_A.zero_grad()

        # Real loss
        loss_real = model.criterion_GAN(model.D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = model.criterion_GAN(model.D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        model.optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        model.optimizer_D_B.zero_grad()

        # Real loss
        loss_real = model.criterion_GAN(model.D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = model.criterion_GAN(model.D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        model.optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataset) + i
        batches_left = opt.n_epochs * len(dataset) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataset),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, model, dataset)

    # Update learning rates
    model.lr_scheduler_G.step()
    model.lr_scheduler_D_A.step()
    model.lr_scheduler_D_B.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(model.G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
        torch.save(model.G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
        torch.save(model.D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
        torch.save(model.D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))

        # batch_size = imgs["A"].shape[0]

        # # Adversarial ground truths
        # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # # Configure input
        # real_imgs = Variable(imgs["B"].type(FloatTensor))
        # labels = Variable(labels.type(LongTensor))

        # # -----------------
        # #  Train Generator
        # # -----------------

        # model.optimizer_G.zero_grad()

        # # Sample noise and labels as generator input
        # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, model.opt.latent_dim))))
        # gen_labels = Variable(LongTensor(np.random.randint(0, model.opt.n_classes, batch_size)))

        # # Generate a batch of images
        # gen_imgs = model.generator(z, gen_labels)

        # # Loss measures generator's ability to fool the discriminator
        # validity = model.discriminator(gen_imgs, gen_labels)
        # g_loss = model.loss_fuction(validity, valid)

        # g_loss.backward()
        # model.optimizer_G.step()

        # # ---------------------
        # #  Train Discriminator
        # # ---------------------

        # model.optimizer_D.zero_grad()

        # # Loss for real images
        # validity_real = model.discriminator(real_imgs, labels)
        # d_real_loss = model.loss_fuction(validity_real, valid)

        # # Loss for fake images
        # validity_fake = model.discriminator(gen_imgs.detach(), gen_labels)
        # d_fake_loss = model.loss_fuction(validity_fake, fake)

        # # Total discriminator loss
        # d_loss = (d_real_loss + d_fake_loss) / 2

        # d_loss.backward()
        # model.optimizer_D.step()

        # print(
        #     "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
        #     % (epoch, model.opt.n_epochs, i, len(model.dataloader), d_loss.item(), g_loss.item())
        # )

        # batches_done = epoch * len(model.dataloader) + i
        # if batches_done % model.opt.sample_interval == 0:
        #     sample_image(n_row=10, batches_done=batches_done, model=model)