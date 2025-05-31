"""
Compute and plot correlation and autocorrelation statistics for the generated data and ground truth.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import numpy as np
import pandas as pd
import torch
import os
from os.path import *
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import random
from scipy import stats
from statsmodels.tsa.stattools import acf

from TailGAN import opt, this_version, gen_data_path, Tensor
from Dataset import Dataset_IS
from Transform import *
from gen_thresholds import gen_thresholds

sample_number = 1000

your_path = 'your_path'  # Replace with your actual path
plot_path = join(your_path, 'Plots')
gen_data_path = join(your_path, 'Gens/')
result_data_path = join(your_path, 'Results/')

fontsize = 40

def Load_Data(opt, sample_number, model_index):
    # Realistic Data
    dataset = Dataset_IS(tickers=opt.tickers, data_path=join(your_path, "gan_data", opt.data_name), length=opt.len)
    real = np.array([d.detach().numpy() for d in dataset.samples])

    # Fake Data
    epoches_l = []
    fake_l = []
    files = os.listdir(gen_data_path)
    epoches = [int(s.split('_E')[1][:-4]) for s in files if s.startswith('Fake')]
    epoches.sort()
    for i in epoches:
        tmp_fake = np.load(join(gen_data_path, 'Fake_id%d_E%d.npy' % (model_index, i)))
        sample_idx = random.sample(range(tmp_fake.shape[0]), sample_number)
        tmp_fake = tmp_fake[sample_idx, :]
        fake_l.append(tmp_fake)
        epoches_l.append(i)

    return real, fake_l, epoches_l


def Mean_Corr(real):
    PNL = np.sum(real, axis=2)
    real_corr = np.corrcoef(PNL.T)
    return real_corr


def Mean_AutoCorr(real, nlags):
    real_autocorr_l = []
    for i in range(real.shape[0]):
        for j in range(real.shape[1]):
            real_autocorr = acf(real[i, j, :], nlags=nlags)
            real_autocorr_l.append(real_autocorr)

    real_autocorr_mean = np.nanmean(np.array(real_autocorr_l).reshape((real.shape[0], real.shape[1], nlags+1)), axis=0)
    return real_autocorr_mean


def Other_Real_Stats(opt, real):
    sub_corr_path = '_'.join([opt.data_name + '_Stats_Corr.csv'])
    sub_auto_path = '_'.join([opt.data_name + '_Stats_Auto.csv'])

    if isfile(join(result_data_path, sub_corr_path)):
        print('GroundTruth Estimates Exist!')
        real_corr_df = pd.read_csv(join(result_data_path, sub_corr_path), index_col=0)
        real_autocorr_df = pd.read_csv(join(result_data_path, sub_auto_path), index_col=0)
    else:
        print('Creating GroundTruth Estimates ......')
        # Correlation
        real_corr_mean = Mean_Corr(real)
        l = opt.tickers
        real_corr_df = pd.DataFrame(np.round(real_corr_mean, 3), columns=l, index=l)
        real_corr_df.to_csv(join(result_data_path, sub_corr_path))

        # AutoCorr
        nlags = 10
        real_autocorr = Mean_AutoCorr(real, nlags)
        l = opt.tickers
        real_autocorr_df = pd.DataFrame(np.round(real_autocorr, 3), index=l, columns=['AC-%d' % i for i in range(nlags+1)])

        real_autocorr_df.to_csv(join(result_data_path, sub_auto_path))

    return real_corr_df, real_autocorr_df


def Other_Fake_Stats(opt, fake_l, epoches_l, inc):
    corr_l = []
    autocorr_l = []
    for i, fake in enumerate(fake_l[::inc]):
        fake_corr = Mean_Corr(fake)
        corr_l.append(fake_corr.reshape(-1))

    index_l = ['E%d' % i for i in epoches_l[::inc]]

    corr_df = pd.DataFrame(np.array(corr_l), index=index_l)
    os.makedirs(join(result_data_path, this_version), exist_ok=True)
    corr_save_path = join(result_data_path, "%s/Stats_Corr.csv" % this_version)
    corr_df.to_csv(corr_save_path)

    # AutoCorr
    for i, fake in enumerate(fake_l[::inc]):
        print('  ' * 6 + 'Epoch ' + str(i) + '  ' * 6)
        fake_autocorr = Mean_AutoCorr(fake, nlags=10)
        autocorr_l.append(fake_autocorr.reshape(-1))

    autocorr_df = pd.DataFrame(np.array(autocorr_l), index=index_l)
    auto_save_path = join(result_data_path, "%s/Stats_Auto.csv" % this_version)
    autocorr_df.to_csv(auto_save_path)
    return corr_l, autocorr_l, index_l


def CorrAuto_Error(opt, real, fake_l, epoches_l):
    inc = 10
    real_corr_df, real_autocorr_df = Other_Real_Stats(opt, real)
    corr_l, autocorr_l, index_l = Other_Fake_Stats(opt, fake_l, epoches_l, inc)

    # # # # # # # # Sample Estimates # # # # # # # # #
    corr_error_l = []
    autocorr_error_l = []
    err_index_l = []

    # # # # # # # # Compute Relative Error # # # # # # # # # #
    # Generated Relative Error
    for i, fake in enumerate(fake_l[::inc]):
        corr_error_l.append(np.sum(np.abs(real_corr_df.values.reshape(-1) - corr_l[i])))
        autocorr_error_l.append(np.sum(np.abs(real_autocorr_df.values.reshape(-1) - autocorr_l[i])))
        err_index_l.append(index_l[i])

    df = pd.DataFrame(np.round(np.array([corr_error_l, autocorr_error_l]).T, 3), index=err_index_l, columns=['Corr', 'AutoCorr'])
    save_path = join(result_data_path, "%s/Stats_CorrAuto_Error.csv" % this_version)
    df.to_csv(save_path)
    print(df)


def Corr_Plot(corr_df, name):
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    sns.heatmap(corr_df, vmin=0, vmax=1, ax=ax,
                cmap=cmap,
                cbar=True,
                # cbar_ax=None if i else cbar_ax,
                square=True, linewidth=.5, xticklabels=False, yticklabels=False)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    ax.set_title('%.2f (%.2f)' % (6.11, 0.7), fontsize=fontsize, color="black")
    f.savefig(join(plot_path, 'Corr_%s.pdf' % name))  # saves the current figure into a pdf page


def AutoCorr_Plot(auto_df, name):
    stock_names =  ['Gaussian', r'AR(1) with $\phi_1=0.5$', r'AR(1) with $\phi_2=-0.12$', r'GARCH(1,1) with $t(5)$', r'GARCH(1,1) with $t(10)$']
    f, ax = plt.subplots(1, 1, figsize=(12, 12))
    # sns.set_theme(style='white')
    auto_df.columns = stock_names
    auto_df.reset_index(drop=True, inplace=True)
    ax.plot(auto_df, '-', linewidth=6)
    ax.set_title('%.2f (%.2f)' % (2.41, 0.39), fontsize=fontsize, color="black")
    ax.set_xlim([0, 10])
    ax.set_ylim([-0.25, 1])
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_xlabel('Lags', fontsize=fontsize)
    # ax.set_ylabel('AutoCorrelation', fontsize=fontsize)

    plt.grid()
    plt.legend(auto_df.columns.values, fontsize=fontsize)
    f.savefig(join(plot_path, 'AutoCorr_%s.pdf' % name))  # saves the current figure into a pdf page


if __name__ == "__main__":
    real, fake_l, epoches_l = Load_Data(opt, sample_number)
    CorrAuto_Error(opt, real, fake_l, epoches_l)

    sub_corr_path = '_'.join([opt.data_name + '_Stats_Corr.csv'])
    sub_auto_path = '_'.join([opt.data_name + '_Stats_Auto.csv'])

    real_corr_df = pd.read_csv(join(result_data_path, sub_corr_path), index_col=0)
    real_autocorr_df = pd.read_csv(join(result_data_path, sub_auto_path), index_col=0)

    corr_save_path = join(result_data_path, "%s/Stats_Corr.csv" % this_version)
    corr_l = pd.read_csv(corr_save_path, index_col=0)
    corr_df = corr_l.iloc[2990:].mean().values.reshape(5, 5)
    
    print(np.sum(np.abs(real_corr_df.values.reshape(-1) - corr_df.reshape(-1))))
    Corr_Plot(corr_df, name='TailGAN')
    Corr_Plot(real_corr_df, name='GT')

    auto_save_path = join(result_data_path, "%s/Stats_Auto.csv" % this_version)
    auto_l = pd.read_csv(auto_save_path, index_col=0)
    auto_df = auto_l.iloc[2990:].mean().values.reshape(5, 11)
    auto_df = auto_df # + np.random.uniform(-0.05, 0.05, auto_df.shape)
    auto_df = pd.DataFrame(auto_df)
    AutoCorr_Plot(auto_df.T, name='TailGAN')
    AutoCorr_Plot(real_autocorr_df.T, name='GT')
