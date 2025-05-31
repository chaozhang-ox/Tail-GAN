"""
Compute Eigen Portfolio and other statistics for the Stk20 dataset.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use("ggplot")
import numpy as np
import pandas as pd
import os
from os.path import *
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from statsmodels.tsa.stattools import acf
from numpy import linalg as LA
import scipy.sparse as sparse

from TailGAN_Eigen import opt, this_version, gen_data_path, Tensor
from Dataset import *
from Transform_Eigen import *


your_path = 'your_path'  # Replace with your actual path
path = join(your_path, 'Stats/Stk20')
os.makedirs(path, exist_ok=True)

trans_parent_data_path = join(your_path, 'Static_Port_Transform')
os.makedirs(trans_parent_data_path, exist_ok=True)

trans_version = 'LShort_Stk20'


def IsStock(vec):
    isstock_l = (np.abs(vec) == 1.0) + (np.abs(vec) == 0.0)
    return np.all(isstock_l)


def Load_Data(opt, sample_number):
    # Realistic Data
    dataset = Dataset_IS(tickers=opt.tickers, data_path=join(your_path, "gan_data", opt.data_name), length=sample_number)
    real = np.array([d.detach().numpy() for d in dataset.samples])
    return real


def Mean_Corr(real):
    real_corr_l = []
    real_std_l = []
    for i in range(real.shape[0]):
        tmp_real_corr = np.corrcoef(real[i, :, :])
        real_corr_l.append(tmp_real_corr)
        tmp_real_std = np.std(real[i, :, :], axis=1)
        real_std_l.append(tmp_real_std)

    real_corr = np.mean(np.array(real_corr_l), axis=0)
    real_std = np.mean(np.array(real_std_l), axis=0)
    return real_corr, real_std


def Other_Real_Stats_P(opt, real):
    save_path = join(path, 'Corr.csv')

    if isfile(save_path):
        print('GroundTruth Estimates Exist!')
        real_corr_df = pd.read_csv(save_path, index_col=0)
        real_std_df = pd.read_csv(join(path, 'Std.csv'), index_col=0)
    else:
        print('Creating GroundTruth Estimates ......')
        real_corr, real_std = Mean_Corr(real)
        l = opt.tickers
        real_corr_df = pd.DataFrame(np.round(real_corr, 3), columns=l, index=l)
        real_corr_df.to_csv(save_path)

        real_std_df = pd.DataFrame(np.round(real_std * 100, 3), index=l, columns=['Std'])
        real_std_df.to_csv(join(path, 'Std.csv'))

    return real_corr_df, real_std_df


def EigenPort(real_corr_df, real_std_df):
    evals, evecs = LA.eigh(real_corr_df)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]

    evec_std = np.dot(np.diagflat(100 / real_std_df.values), evecs)
    trans_pca_port = evec_std / np.abs(evec_std).sum(0)

    # Save Path
    trans_data_path = join(trans_parent_data_path, trans_version)
    os.makedirs(trans_data_path, exist_ok=True)

    store_path = join(trans_data_path, 'CorrStd_EigenPort.npy')
    np.save(store_path, trans_pca_port)

    np.save(join(trans_data_path, 'CorrStd_EigenValues.npy'), evals)
    return evals


def EigPort_Combination(opt, insample):
    trans_data_path = join(trans_parent_data_path, trans_version)
    store_path = join(trans_data_path, 'CorrStd_EigenPort.npy')
    # load the eigen portfolio weights
    trans_pca_port = np.load(store_path)

    n_rows = 20
    # rvs = stats.norm(loc=0, scale=1).rvs
    # unscale_trans2port_mat = sparse.random(n_rows, max(int(n_rows ** 2), 50), density=0.3, data_rvs=rvs).toarray()

    # generate sparse random matrix
    unscale_trans2port_mat = sparse.rand(n_rows, 100, density=0.5).toarray()

    # transform the eigen portfolio to random portfolios by linear combinations
    unscale_pca_trans2port_mat = np.dot(trans_pca_port[:, :n_rows], unscale_trans2port_mat)

    if insample:
        # concat the eigen and random portfolios
        unscale_pca_trans2port_mat = np.column_stack([trans_pca_port, unscale_pca_trans2port_mat])
    else:
        pass

    # Scale
    trans_mat = unscale_pca_trans2port_mat / np.abs(unscale_pca_trans2port_mat).sum(0)

    trans_mat_port = trans_mat[:, ~np.apply_along_axis(IsStock, 0, trans_mat)]
    assert trans_mat_port.shape[1] >= 10

    if insample:
        store_path = join(trans_data_path, 'EigenPortIS.npy')
    else:
        store_path = join(trans_data_path, 'EigenPortOOS.npy')

    np.save(store_path, trans_mat_port)


def Plot_Corr(real_corr_df):
    clms = opt.tickers

    # real_cov_df.index =clms
    # real_cov_df.columns = clms
    # real_corr_df.index = clms
    # real_corr_df.index = clms

    pdf_name = join(path, 'Corr.pdf')

    with PdfPages(pdf_name) as pdf:
        # mask = np.zeros_like(t_stats_df)
        # mask[np.triu_indices_from(mask, k=1)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(15, 15))
            ax = sns.heatmap(real_corr_df, square=True, cmap='coolwarm')
            # ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=16)
            # ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=16)
        pdf.savefig()  # saves the current figure into a pdf page

        plt.close()


def Plot_EigValue(evals):
    # Plot Explained Variance Ratio
    pdf_name = join(path, 'EVR.pdf')

    f, ax = plt.subplots(figsize=(16, 12))
    evr = evals / evals.sum()
    pca_evr_df = pd.DataFrame(evr, columns=['Explained Variance Ratio'])
    pca_evr_df['Eigenvalue Order'] = range(1, 1+len(evr))
    sns.barplot(x='Eigenvalue Order', y='Explained Variance Ratio', data=pca_evr_df, ci=None, ax=ax, color=sns.color_palette("Set2")[2])
    ax.set_ylim([0, 0.35])
    ax.set_xlabel('Eigenvalue Order', fontsize=30)
    ax.set_ylabel('Explained Variance Ratio', fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.grid(axis='y')
    f.savefig(pdf_name)


def OOS_Static():
    trans_data_path = join(trans_parent_data_path, trans_version)
    store_path = join(trans_data_path, 'CorrStd_EigenPort.npy')
    trans_pca_port = np.load(store_path)

    n_rows = 10
    # rvs = stats.norm(loc=0, scale=1).rvs
    # unscale_trans2port_mat = sparse.random(n_rows, 30, density=0.3, data_rvs=rvs).toarray()

    unscale_trans2port_mat = sparse.rand(n_rows, 30, density=0.3).toarray()
    unscale_pca_trans2port_mat = np.dot(trans_pca_port[:, :n_rows], unscale_trans2port_mat)
    unscale_pca_trans2port_mat = np.column_stack([trans_pca_port, unscale_pca_trans2port_mat])
    noise = np.random.normal(loc=0, scale=1, size=unscale_pca_trans2port_mat.shape)

    # noise = np.random.normal(loc=0, scale=1, size=trans_pca_port.shape)

    unscale_trans2port_mat_l = []
    for dis in [0, 1e-3, 1e-2, 1e-1, 1]:
        # tmp_unscale_trans2port_mat = trans_pca_port + noise * dis
        tmp_unscale_trans2port_mat = unscale_pca_trans2port_mat + noise * dis
        unscale_trans2port_mat_l.append(tmp_unscale_trans2port_mat)

    unscale_pca_trans2port_mat = np.column_stack(unscale_trans2port_mat_l)
    # Scale
    trans_mat = unscale_pca_trans2port_mat / np.abs(unscale_pca_trans2port_mat).sum(0)

    trans_mat_port = trans_mat[:, ~np.apply_along_axis(IsStock, 0, trans_mat)]
    assert trans_mat_port.shape[1] >= 10

    store_path = join(trans_data_path, 'OOS_Static.npy')
    np.save(store_path, trans_mat_port)


if __name__ == "__main__":
    sample_number = 1000
    real = Load_Data(opt, sample_number)
    print(real.shape)
    real_corr_df, real_std_df = Other_Real_Stats_P(opt, real)
    print(real_corr_df)
    print(real_std_df)
    Plot_Corr(real_corr_df)
    evals = EigenPort(real_corr_df, real_std_df)
    Plot_EigValue(evals)
    EigPort_Combination(opt, insample=True)
    EigPort_Combination(opt, insample=False)
    # # OOS_Static()