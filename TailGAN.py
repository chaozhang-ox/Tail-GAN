"""
Tail-GAN for synthetic data. 
The most important parameters are:
strategies: a list of your trading strategies. In our paper, we use static portfolios, mean-reversion, and trend-following.
n_epochs: number of epochs for training.
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
from Transform import *
from gen_thresholds import gen_thresholds
from util import *

# set random seed 
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=3000, help="epochs for training")
parser.add_argument("--batch_size", type=int, default=1000, help="size of the batches")
parser.add_argument("--lr_D", type=float, default=1e-7, help="learning rate for Discriminator")
parser.add_argument("--lr_G", type=float, default=1e-6, help="learning rate for Generator")
parser.add_argument('--temp', type=float, default=0.01, help='multiplier of temperature')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=1000, help="dimensionality of the latent space")
parser.add_argument("--len", type=int, default=50000, help="number of examples")
parser.add_argument("--n_rows", type=int, default=5, help="number of rows")
parser.add_argument("--n_cols", type=int, default=100, help="number of columns")
parser.add_argument("--n_critic_G", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--n_critic_D", type=int, default=1, help="number of training steps for generator per iter")
parser.add_argument("--static_way", type=str, default='LShort', help="trading way of static portfolios")
parser.add_argument("--strategies", type=list, default=['Port', 'MR', 'TF'], help="a list of strategy names")
parser.add_argument("--n_trans", type=int, default=50, help="number of static portfolios")
parser.add_argument("--Cap", type=int, default=10, help="maximum investment capital")
parser.add_argument("--WH", type=int, default=10, help="window history for strategy")
parser.add_argument("--ratios", type=list, default=[1.0, 1.0], help="ratios for longing or shorting")
parser.add_argument("--thresholds_pct", type=list, default=[[31, 69]], help="thresholds for longing or shorting")
parser.add_argument("--data_name", type=str, default='1_Gauss+1_AR50+1_AR-12+1_GARCH-T5+1_GARCH-T10', help="data name")
parser.add_argument("--tickers", type=list, default=['Gauss', 'AR50', 'AR-12', 'GARCH-T5', 'GARCH-T10'], help="tickers")
parser.add_argument("--noise_name", type=str, default='t5', help="noise name")
parser.add_argument("--alphas", type=list, default=[0.05], help="quantiles")
parser.add_argument("--W", type=float, default=10.0, help="scale parameter for W")
parser.add_argument("--score", type=str, default='quant', help="score function")
parser.add_argument("--numNN", type=int, default=10, help="number of NNs")
parser.add_argument("--project", type=bool, default=True, help="Project into constraint set")
parser.add_argument("--version", type=str, default=f'Test{seed}', help="version number")

opt = parser.parse_args()
print(opt)
R_shape = (opt.n_rows, opt.n_cols)


def Infer_Shape(R_shape):
    """
    Shape of portfolio returns
    :param R_shape: shape of asset returns
    """
    PNL_shape_0 = R_shape[0]
    for strategy in opt.strategies:
        if strategy == 'Port':
            PNL_shape_0 += opt.n_trans
        elif strategy == 'MR':
            PNL_shape_0 += opt.n_rows * len(opt.thresholds_pct)
        elif strategy == 'TF':
            PNL_shape_0 += opt.n_rows * len(opt.thresholds_pct)
        else:
            pass
    PNL_shape = (PNL_shape_0, R_shape[1])
    return PNL_shape

PNL_shape = Infer_Shape(R_shape)

# Specific version
this_version = '_'.join(
    [opt.version,
     'Stk' + str(opt.n_rows),
     opt.data_name,
     opt.noise_name,
     'E' + str(opt.n_epochs),
     'N' + str(opt.len),
     'BS' + str(opt.batch_size),
     opt.static_way,
     '_'.join(opt.strategies),
     'P' + str(opt.n_trans),
     'Cap' + str(opt.Cap),
     'WH' + str(opt.WH),
     'R' + '+'.join([str(a) for a in opt.ratios]),
     'T' + '+'.join(['_'.join(map(str, i)) for i in opt.thresholds_pct]),
     'D' + str(opt.n_critic_D), 'G' + str(opt.n_critic_G),
     'LR' + '-'.join([str(opt.lr_D), str(opt.lr_G)]),
     'Temp' + str(opt.temp),
     'Q' + '+'.join([str(int(100 * a)) for a in opt.alphas]),
     'Esb' + str(opt.numNN)])

# set path
your_path = 'your_path'

# Save Path
gen_data_path = join(your_path, "Gens/gen_data_{this_version}")
os.makedirs(gen_data_path, exist_ok=True)

model_path = join(your_path, "Models/model_{this_version}")
os.makedirs(model_path, exist_ok=True)


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# compute PNL of different strategies from returns
def Compute_PNL(R):
    """
    R is the return matrix
    """
    # convert to Prices
    prices_l = Inc2Price(R)
    port_prices_l = StaticPort(prices_l, opt.n_trans, opt.static_way, insample=True)

    PNL_BH = BuyHold(prices_l, opt.Cap)
    PNL_l = [PNL_BH]

    for strategy in opt.strategies:
        if strategy == 'Port':
            PNL_BHPort = BuyHold(port_prices_l, opt.Cap)
            PNL_l.append(PNL_BHPort)
        elif strategy == 'MR':
            for percentile_l in opt.thresholds_pct:
                thresholds_array = gen_thresholds(opt.data_name, opt.tickers, strategy, percentile_l, 100, opt.WH)
                PNL_MR = MeanRev(prices_l, opt.Cap, opt.WH, LR=opt.ratios[0], SR=opt.ratios[1],
                                 ST=thresholds_array[:, -1], LT=thresholds_array[:, -2])
                PNL_l.append(PNL_MR)
        elif strategy == 'TF':
            for percentile_l in opt.thresholds_pct:
                thresholds_array = gen_thresholds(opt.data_name, opt.tickers, strategy, percentile_l, 100, opt.WH)
                PNL_TF = TrendFollow(prices_l, opt.Cap, opt.WH, LR=opt.ratios[0], SR=opt.ratios[1],
                                     ST=thresholds_array[:, 0], LT=thresholds_array[:, 1])
                PNL_l.append(PNL_TF)
        else:
            pass

    PNL = torch.cat(PNL_l, dim=1)
    return PNL


# Generator is a MLP with 4 layers
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



# Discriminator is a MLP with 3 layers
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.W = opt.W

        self.project = opt.project

        self.alphas = opt.alphas

        self.model = nn.Sequential(
            nn.Linear(opt.batch_size, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 2 * len(opt.alphas)),
        )

    def project_op(self, validity):
        for i, alpha in enumerate(self.alphas):
            v = validity[:, 2*i].clone()
            e = validity[:, 2*i+1].clone()
            indicator = torch.sign(torch.as_tensor(0.5 - alpha))
            validity[:, 2*i] = indicator * ((self.W * v < e).float() * v + (self.W * v >= e).float() * (v + self.W * e) / (1 + self.W ** 2))
            validity[:, 2*i+1] = indicator * ((self.W * v < e).float() * e + (self.W * v >= e).float() * self.W * (v + self.W * e) / (1 + self.W ** 2))
        return validity


    def forward(self, R):
        PNL = Compute_PNL(R)
        PNL_transpose = PNL.T
        PNL_s = PNL_transpose.reshape(*PNL_transpose.shape, 1)
        perm_matrix = deterministic_NeuralSort(PNL_s, opt.temp)
        PNL_sort = torch.bmm(perm_matrix, PNL_s)
        PNL_validity = self.model(PNL_sort.reshape(*PNL_transpose.shape))
        if self.project:
            PNL_validity = self.project_op(PNL_validity)
        else:
            pass

        return PNL, PNL_validity


def G1(v):
    return v


def G2(e, scale=1):
    return scale * torch.exp(e / scale)


def G2in(e, scale=1):
    return scale ** 2 * torch.exp(e / scale)


def G1_quant(v, W=opt.W):
    return - W * v ** 2 / 2


def G2_quant(e, alpha):
    return alpha * e


def G2in_quant(e, alpha):
    return alpha * e ** 2 / 2


# general score function
def S_stats(v, e, X, alpha):
    """
    For a given quantile, here named alpha, calculate the score function value
    """
    if alpha < 0.5:
        rt = ((X<=v).float() - alpha) * (G1(v) - G1(X)) + 1. / alpha * G2(e) * (X<=v).float() * (v - X) + G2(e) * (e - v) - G2in(e)
    else:
        alpha_inverse = 1 - alpha
        rt = ((X>=v).float() - alpha_inverse) * (G1(X) - G1(v)) + 1. / alpha_inverse * G2(-e) * (X>=v).float() * (X - v) + G2(-e) * (v - e) - G2in(-e)
    return torch.mean(rt)


# a specific score function requiring some constraints on VAR and ES, but having better optimization properties
def S_quant(v, e, X, alpha, W=opt.W):
    """
    For a given quantile, here named alpha, calculate the score function value
    """
    if alpha < 0.5:
        rt = ((X<=v).float() - alpha) * (G1_quant(v,W) - G1_quant(X,W)) + 1. / alpha * G2_quant(e,alpha) * (X<=v).float() * (v - X) + G2_quant(e,alpha) * (e - v) - G2in_quant(e,alpha)
    else:
        alpha_inverse = 1 - alpha
        rt = ((X>=v).float() - alpha_inverse) * (G1_quant(v,W) - G1_quant(X,W)) + 1. / alpha_inverse * G2_quant(-e,alpha_inverse) * (X>=v).float() * (X - v) + G2_quant(-e,alpha_inverse) * (v - e) - G2in_quant(-e,alpha_inverse)
    return torch.mean(rt)


class Score(nn.Module):
    def __init__(self):
        super(Score, self).__init__()
        self.alphas = opt.alphas
        self.score_name = opt.score
        if self.score_name == 'quant':
            self.score_alpha = S_quant
        elif self.score_name == 'stats':
            self.score_alpha = S_stats
        else:
            self.score_alpha = None

    def forward(self, PNL_validity, PNL):
        # Score
        loss = 0
        for i, alpha in enumerate(self.alphas):
            PNL_var = PNL_validity[:, [2 * i]]
            PNL_es = PNL_validity[:, [2 * i + 1]]
            loss += self.score_alpha(PNL_var, PNL_es, PNL.T, alpha)

        return loss


# ----------
#  Training
# ----------
def Train_Single(opt, dataloader, model_index, seed):
    start_time = time.time()

    torch.manual_seed(seed)

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()
    criterion = Score()

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
                PNL, PNL_validity = discriminator(real_R)
                gen_PNL, gen_PNL_validity = discriminator(gen_R)
                real_score = criterion(PNL_validity, PNL)
                fake_score = criterion(gen_PNL_validity, PNL)
                loss_D = real_score - fake_score

                # Update the Gradient in Discriminator
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

                epoch_loss_D.append(loss_D.item())

            # Train the generator every n_critic iterations
            if i % opt.n_critic_G == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Adversarial loss
                gen_PNL, gen_PNL_validity = discriminator(gen_R)
                loss_G = criterion(gen_PNL_validity, PNL)

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
        print("------ Model %d Starts with Random Seed %d " % (iii, seed))
        Train_Single(opt, dataloader, model_index=iii, seed=seed)


# Some trained models may not converge well, we only use those models with a good converge
# This selection is based on the training data, so no look-forward bias
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
    select_l = Screen_Ensemble()