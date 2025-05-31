"""
Compute the absolute error in the out-of-sample estimates of VaR and ES
dynamic + static strategies
OOS data has the same distribution as the IS data, for synthetic data
"""
import numpy as np
import pandas as pd
import torch
import os
from os.path import *
import random

from TailGAN import opt, this_version, gen_data_path, Tensor, Screen_Ensemble
from Dataset import Dataset_IS
from Transform import *
from gen_thresholds import gen_thresholds

sample_number = 1000
your_path = 'your_path'


def Empirical_Stats(data):
    """
    Compute the empirical VaR and ES for either true data or synthetic data
    """
    ep_stats_l = []
    for alpha in opt.alphas:
        var_l = np.percentile(data, alpha * 100, axis=(0))
        var = np.round(var_l, 3)
        var_l = var_l.reshape(1, *var_l.shape)
        if alpha < 0.5:
            tmp_data = data * (data <= var_l)
            tmp_data[tmp_data == 0.0] = np.nan
            es = np.round(np.nanmean(tmp_data, axis=(0)), 3)
        else:
            tmp_data = data * (data >= var_l)
            tmp_data[tmp_data == 0.0] = np.nan
            es = np.round(np.nanmean(tmp_data, axis=(0)), 3)
        ep_stats_l.extend(np.stack([var, es]))

    rt = np.array(ep_stats_l).T
    rt = np.round(rt, 3)
    return rt


def Compute_PNL_NP_IS(R):
    R_tensor = Tensor(R)
    # Price
    prices_l = Inc2Price(R_tensor)
    port_prices_l = StaticPort(prices_l, 50, opt.static_way, insample=True)

    PNL_BH = BuyHold(prices_l, opt.Cap)
    PNL_l = [PNL_BH]
    columns = ['Stk-%d ' % (i + 1) for i in range(PNL_BH.shape[1])]

    thresholds_pct = [[31, 69]]

    strategies = ['Port', 'MR', 'TF']

    for strategy in strategies:
        if strategy == 'Port':
            PNL_BHPort = BuyHold(port_prices_l, opt.Cap)
            PNL_l.append(PNL_BHPort)
            columns.extend(['Trans-%d ' % (i + 1) for i in range(PNL_BHPort.shape[1])])
        elif strategy == 'MR':
            for percentile_l in thresholds_pct:
                thresholds_array = gen_thresholds(opt.data_name, opt.tickers, strategy, percentile_l, 100, opt.WH)
                PNL_MR = MeanRev(prices_l, opt.Cap, opt.WH, LR=opt.ratios[0], SR=opt.ratios[1],
                                 ST=thresholds_array[:, -1], LT=thresholds_array[:, -2])
                PNL_l.append(PNL_MR)
                columns.extend(['MR-%d ' % (i + 1) for i in range(PNL_MR.shape[1])])
        elif strategy == 'TF':
            for percentile_l in thresholds_pct:
                thresholds_array = gen_thresholds(opt.data_name, opt.tickers, strategy, percentile_l, 100, opt.WH)
                PNL_TF = TrendFollow(prices_l, opt.Cap, opt.WH, LR=opt.ratios[0], SR=opt.ratios[1],
                                     ST=thresholds_array[:, 0], LT=thresholds_array[:, 1])
                PNL_l.append(PNL_TF)
                columns.extend(['TF-%d ' % (i + 1) for i in range(PNL_TF.shape[1])])
        else:
            pass

    PNL = torch.cat(PNL_l, dim=1)
    return PNL.cpu().detach().numpy(), columns


def Load_Data(opt):
    # Realistic Data
    dataset = Dataset_IS(tickers=opt.tickers, data_path=join(your_path, "gan_data", opt.data_name), length=opt.len)
    real = np.array([d.detach().numpy() for d in dataset.samples])
    return real


def Load_FakeData(model_index):
    # Fake Data
    epoches_l = []
    fake_l = []
    files = os.listdir(gen_data_path)
    epoches = [int(s.split('_E')[1][:-4]) for s in files if s.startswith('Fake') and s.endswith('.npy') and 'id%d' % model_index in s]
    epoches.sort()
    for i in epoches[::10]:
        tmp_fake_l = []
        idx = epoches.index(i)
        for j in epoches[idx:(idx + 10)]:
            tmp_fake0 = np.load(join(gen_data_path, 'Fake_id%d_E%d.npy' % (model_index, j)))
            tmp_fake_l.append(tmp_fake0)

        tmp_fake = np.concatenate(tmp_fake_l)
        sample_idx = random.sample(range(tmp_fake.shape[0]), sample_number)
        tmp_fake = tmp_fake[sample_idx, :]
        fake_l.append(tmp_fake)
        epoches_l.append(i)

    return fake_l, epoches_l


############ Compute or Load the GroundTruth Estimates from examples ###################
def VaR_ES_Real_Stats(opt):
    index_l = ['Real']
    real_stats_l = []
    OOS_alphas = [0.01, 0.05, 0.10]
    sub_path = '_'.join(['Synthetic',
                         '_'.join(opt.strategies),
                         'P' + str(opt.n_trans),
                         'Cap' + str(opt.Cap),
                         'WH' + str(opt.WH),
                         'Q' + '+'.join([str(a) for a in OOS_alphas]),
                         'R' + '+'.join([str(a) for a in opt.ratios]),
                         'T' + '+'.join(['_'.join(map(str, i)) for i in opt.thresholds_pct]),
                         'estimates.csv'])
    os.makedirs(f'{your_path}/Results_S{sample_number}/GroundTruth/', exist_ok=True)
    real_estimates_path = join(f'{your_path}/Results_S{sample_number}/GroundTruth/', sub_path)

    # # # # # # # # Realistic Estimates # # # # # # # # #
    if isfile(real_estimates_path):
        print('GroundTruth Estimates Exist!')
        real_ve = pd.read_csv(real_estimates_path, index_col=0)
        clms = [i for i in real_ve.columns.to_list() if str(opt.alphas[0]) in i]
        real_ve = real_ve[clms]
        real_stats_l.append(np.concatenate([real_ve.values]))
        is_final_clms = real_ve.columns.to_list()
    else:
        print('Creating GroundTruth Estimates ......')
        real = Load_Data(opt)
        PNL, is_clms_l = Compute_PNL_NP_IS(real)
        is_real_ve = Empirical_Stats(PNL).reshape(-1)
        ep_clms_l = []
        for alpha in opt.alphas:
            ep_clms_l.extend(['VaR_%.2f' % alpha, 'ES_%.2f' % alpha])

        is_final_clms = [i + j for i in is_clms_l for j in ep_clms_l]
        real_stats_l.append(np.concatenate([is_real_ve]))

        # Storage
        df = pd.DataFrame(np.round(np.column_stack(real_stats_l).T, 3), index=index_l, columns=is_final_clms)
        df.to_csv(real_estimates_path)

    return real_stats_l, is_final_clms


# compute the sampling error given the sample number
def VaR_ES_Sampling_Error(opt):
    dataset = Dataset_IS(tickers=opt.tickers, data_path=f"{your_path}/gan_data/{opt.data_name}", length=sample_number)
    real = np.array([d.detach().numpy() for d in dataset.samples])

    real_stats_l, is_final_clms = VaR_ES_Real_Stats(opt)

    # # # # # # # # Sample Estimates # # # # # # # # #
    is_sample_ve_l = []

    for i in range(16):
        sample_idx = random.sample(range(real.shape[0]), sample_number)
        sample_fake = real[sample_idx, :, :]
        fake_PNL, is_clms_l = Compute_PNL_NP_IS(sample_fake)
        fake_ve = Empirical_Stats(fake_PNL).reshape(-1)
        sample_err = np.abs(real_stats_l[0].reshape(-1) - fake_ve) / np.abs(real_stats_l[0].reshape(-1))
        is_sample_ve_l.append(sample_err)

    # Sample Relative Error
    err_l = []
    err_index_l = []

    err_l.append(np.mean(np.stack(is_sample_ve_l, axis=1), axis=1))
    err_l.append(np.std(np.stack(is_sample_ve_l, axis=1), axis=1))

    err_index_l.append('Sample-RE-Mean')
    err_index_l.append('Sample-RE-Std')
    sample_error_df = pd.DataFrame(np.round(np.column_stack(err_l).T, 3), index=err_index_l, columns=is_final_clms)
    print(sample_error_df)
    print(sample_error_df.mean(1))
    sample_error_df.to_csv(f"{your_path}/Results_S{sample_number}/Sample_Error_RE.csv")


################ Compute or Load the Error from Generated Data ##################
def VaR_ES_Fake_Error(opt, fake_l, epoches_l, model_index):
    real_stats_l, is_final_clms = VaR_ES_Real_Stats(opt)

    mean_err_l = []
    std_err_l = []

    # # # # # # # # Generated Estimates # # # # # # # # #
    for i, fake in enumerate(fake_l):
        is_sample_ve_l = []
        print('  ' * 6 + 'Epoch ' + str(i * 10) + '  ' * 6)
        for j in range(16):
            sample_idx = random.sample(range(fake.shape[0]), sample_number)
            sample_fake = fake[sample_idx, :, :]
            fake_PNL, is_clms_l = Compute_PNL_NP_IS(sample_fake)
            fake_ve = Empirical_Stats(fake_PNL).reshape(-1)
            sample_err = np.abs(real_stats_l[0].reshape(-1) - fake_ve) / np.abs(real_stats_l[0].reshape(-1))
            is_sample_ve_l.append(sample_err)

        mean_err_l.append(np.mean(np.stack(is_sample_ve_l, axis=1), axis=1))
        std_err_l.append(np.std(np.stack(is_sample_ve_l, axis=1), axis=1))

    mean_df = pd.DataFrame(np.round(np.column_stack(mean_err_l).T, 3), index=epoches_l, columns=is_final_clms)
    std_df = pd.DataFrame(np.round(np.column_stack(std_err_l).T, 3), index=epoches_l, columns=is_final_clms)
    save_path = f"{your_path}/Results_S{sample_number}/{this_version}_Model_{model_index}/"
    os.makedirs(save_path, exist_ok=True)
    mean_df.to_csv(join(save_path, 'Mean_OOS_RE_Mean.csv'))
    std_df.to_csv(join(save_path, 'Std_OOS_RE.csv'))


# Summarize the results for all selected models
def Eval_Sum(opt):
    path = f'{your_path}/Results_S{sample_number}/{this_version}_Model_{model_index}'
    folders = os.listdir(path)
    folders = [i for i in folders if isdir(join(path, i))]
    folders.sort()

    mean_l = []
    std_l = []
    for k in folders:
        mean_df = pd.read_csv(join(path, k, 'Mean_OOS_RE_Mean.csv'), index_col=0)
        std_df = pd.read_csv(join(path, k, 'Std_OOS_RE.csv'), index_col=0)
        arg_min_ind = mean_df.mean(1).argmin()
        mean_res = mean_df.mean(1).min() * 100
        mean_l.append(mean_res)
        std_res = std_df.mean(1).iloc[arg_min_ind] * 100
        std_l.append(std_res)
    
    mean_l = np.array(mean_l)
    std_l = np.array(std_l)
    print('Overall [Mean: %.2f] [Std: %.2f]' % (np.mean(mean_l), np.mean(std_l)))


if __name__ == "__main__":
    # compute the sampling error
    VaR_ES_Sampling_Error(opt)

    # compute the estimate error from the TailGAN generated data
    select_l = Screen_Ensemble(thres_perc=50)
    for model_index in select_l:
        print('Model Index:', model_index)
        fake_l, epoches_l = Load_FakeData(opt)
        VaR_ES_Fake_Error(opt, fake_l, epoches_l)

    # compute the overall summary of the results
    Eval_Sum(opt)