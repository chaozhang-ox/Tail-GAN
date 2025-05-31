"""
WGAN with Gradient Penalty as a benchmark
It will learn the distribution of raw financial time series data, no strategy involved.
"""
import argparse
import os
import pandas as pd
import numpy as np
import random
from os.path import join
import time

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from Dataset import Dataset_IS

# set random seed 
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=3000, help="epochs for training")
parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
parser.add_argument("--lr_D", type=float, default=1e-6, help="learning rate for Discriminator")
parser.add_argument("--lr_G", type=float, default=1e-6, help="learning rate for Generator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=1000, help="dimensionality of the latent space")
parser.add_argument("--len", type=int, default=50000, help="number of examples")
parser.add_argument("--n_rows", type=int, default=5, help="number of rows")
parser.add_argument("--n_cols", type=int, default=100, help="number of columns")
parser.add_argument("--n_critic_G", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--n_critic_D", type=int, default=1, help="number of training steps for generator per iter")
parser.add_argument("--data_name", type=str, default='1_Gauss+1_AR50+1_AR-12+1_GARCH-T5+1_GARCH-T10', help="data name")
parser.add_argument("--tickers", type=list, default=['Gauss', 'AR50', 'AR-12', 'GARCH-T5', 'GARCH-T10'], help="tickers")
parser.add_argument("--noise_name", type=str, default='normal', help="noise name")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--numNN", type=int, default=1, help="number of NNs")
parser.add_argument("--version", type=str, default='Test_WGAN', help="version number")

opt = parser.parse_args()
print(opt)
R_shape = (opt.n_rows, opt.n_cols)

# Specific version
this_version = '_'.join(
    [opt.version,
     'Stk' + str(opt.n_rows),
     opt.data_name,
     opt.noise_name,
     'E' + str(opt.n_epochs),
     'BS' + str(opt.batch_size),
     'Clip' + str(opt.clip_value),
     'LR' + '-'.join([str(opt.lr_D), str(opt.lr_G)]),
     'Esb' + str(opt.numNN)])


# Save Path
your_path = 'your_path'

# Save Path
gen_data_path = join(your_path, "Gens/gen_data_{this_version}")
os.makedirs(gen_data_path, exist_ok=True)

model_path = join(your_path, "Models/model_{this_version}")
os.makedirs(model_path, exist_ok=True)


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(R_shape))),
        )

    def forward(self, z):
        img = self.model(z)
        img = torch.clamp(img, min=-1, max=1)
        img = img.view(img.shape[0], *R_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(R_shape)), 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# ----------
#  Training
# ----------
def Train_Single(opt, dataloader, model_index, seed):
    start_time = time.time()

    torch.manual_seed(seed)

    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))

    loss_d_l = []
    loss_g_l = []

    gen_size = 1000 # opt.len
    for epoch in range(opt.n_epochs):
        epoch_loss_D = []
        epoch_loss_G = []

        for i, R in enumerate(dataloader):

            # Configure input
            real_R = Variable(R.type(Tensor))

            # Sample noise as generator input, (batchsize, latent dim): (1000, 100)
            if 't' in opt.noise_name:
                z = Variable(
                    Tensor(np.random.standard_t(int(opt.noise_name.split('t')[1]), (R.shape[0], opt.latent_dim))))
            else:
                z = Variable(Tensor(np.random.normal(0, 1, (R.shape[0], opt.latent_dim))))

            # Generate a batch of images, (1000, 2, 100)
            gen_R = generator(z)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Train the generator every n_critic iterations
            if i % opt.n_critic_D == 0:

                optimizer_D.zero_grad()

                # Adversarial loss
                loss_D = -torch.mean(discriminator(real_R)) + torch.mean(discriminator(gen_R))

                # Update the Gradient in Discriminator
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

                epoch_loss_D.append(loss_D.item())

                # Clip weights of discriminator
                for p in discriminator.parameters():
                    p.data.clamp_(-opt.clip_value, opt.clip_value)

            # Train the generator every n_critic iterations
            if i % opt.n_critic_G == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Adversarial loss
                gen_R = generator(z)

                loss_G = -torch.mean(discriminator(gen_R))

                # Update the Gradient in Discriminator
                loss_G.backward()
                optimizer_G.step()

                epoch_loss_G.append(loss_G.item())

        D_loss_epoch = np.mean(epoch_loss_D)
        G_loss_epoch = np.mean(epoch_loss_G)

        loss_d_l.append(D_loss_epoch)
        loss_g_l.append(G_loss_epoch)

        if epoch % 100 == 0:
            print("[Epoch %d] [D loss: %.4f] [G loss: %.4f]" % (epoch, D_loss_epoch, G_loss_epoch))
            print("--- %d seconds passed ---" % (time.time() - start_time))

        if 't' in opt.noise_name:
            z = Variable(
                Tensor(np.random.standard_t(int(opt.noise_name.split('t')[1]), (gen_size, opt.latent_dim))))
        else:
            z = Variable(Tensor(np.random.normal(0, 1, (gen_size, opt.latent_dim))))

        gen_R = generator(z)

        np.save(join(gen_data_path, "Fake_id%d_E%d.npy" % (model_index, epoch)), gen_R.cpu().detach().numpy())

        if epoch % 100 == 0:
            # Save the Terminate Model
            discriminator_path = join(model_path, "discriminator_id%d_E%d" % (model_index, epoch))
            generator_path = join(model_path, "generator_id%d_E%d" % (model_index, epoch))
            torch.save(discriminator.state_dict(), discriminator_path)
            torch.save(generator.state_dict(), generator_path)

    # Save Loss Value
    loss_d_l = np.array(loss_d_l)
    loss_g_l = np.array(loss_g_l)
    loss_dge = np.stack([loss_d_l, loss_g_l])
    np.save(join(gen_data_path, 'loss_id%d.npy' % model_index), loss_dge)


def Train(opt):
    """
    Train multiple Tail-GANs
    """
    # Configure data loader
    dataset = Dataset_IS(tickers=opt.tickers, data_path=join(your_path, "gan_data", opt.data_name), length=opt.len)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True)

    for iii in range(opt.numNN):
        seed = np.random.randint(low=1, high=10000)
        print("------ Model %d Starts with Random Seed %d " % (iii, seed))
        Train_Single(opt, dataloader, model_index=iii, seed=seed)


def Screen_Ensemble(thres_perc=50):
    loss_l = []
    for j in range(opt.numNN):
        # load loss, focus on the last generator loss
        loss_np = np.load(join(gen_data_path, 'loss_id%d.npy' % j))
        loss_l.append(loss_np[:, 1].iloc[-1])

    threshold_loss = np.percentile(loss_l, thres_perc)
    select_l = []
    for j in range(opt.numNN):
        if loss_l[j] <= threshold_loss:
            select_l.append(j)
        else:
            pass
    return select_l


if __name__ == "__main__":
    Train(opt)
    # select_l = Screen_Ensemble(thres_perc=50)
    # print("Selected Models: ", select_l)
