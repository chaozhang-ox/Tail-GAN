"""
Estimate the thresholds for producing trading signals based on training data
"""

import numpy as np
import pandas as pd

import os
from os.path import *

import torch
from torch import nn

from Transform import Tensor, Inc2Price, movingaverage
parent_data_path = '/data01/Chao_TailGAN/gan_data/'


def gen_thresholds(data_name, tickers, strategy, percentile_l, length, WH):
    thresholds_data_folder = join('/data01/Chao_TailGAN/', 'Thresholds_20221124', data_name)
    os.makedirs(thresholds_data_folder, exist_ok=True)

    if 'MR' in strategy and 'Port' not in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '_MR_%s.npy' % '_'.join(map(str, percentile_l)))
    elif 'MR' in strategy and 'Port' in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '_Port*MR_%s.npy' % '_'.join(map(str, percentile_l)))
    elif 'TF' in strategy and 'Port' not in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '_TF_%s.npy' % '_'.join(map(str, percentile_l)))
    elif 'TF' in strategy and 'Port' in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '__Port*TF_%s.npy' % '_'.join(map(str, percentile_l)))
    else:
        pass

    if isfile(thresholds_path):
        thresholds_array_stocks = np.load(thresholds_path)
    else:
        data_path = join(parent_data_path, data_name)
        data_l = []
        files = os.listdir(data_path)
        files.sort()
        for item in range(length):
            file_path = join(data_path, files[item])
            tmp_data = pd.read_csv(file_path)[tickers].values.T
            data_l.append(tmp_data)

        data = np.stack(data_l)
        data = Tensor(data)

        prices_l = Inc2Price(data)

        prices_l_flat = prices_l.view(prices_l.shape[0] * prices_l.shape[1], -1)

        thresholds_array_list = []
        for stk in range(data.shape[1]):
            if 'MR' in strategy:
                prices_l_ma = torch.mean(prices_l[:, :, :WH + 1], dim=2)
                prices_l_ma_flat = prices_l_ma.view(prices_l_ma.shape[0] * prices_l_ma.shape[1], -1)

                # Compute the z-scores for each day using the historical data up to that day
                zscores_MR = (prices_l_flat - prices_l_ma_flat) / 0.01
                zscores_MR = zscores_MR.cpu().detach().numpy()
                thresholds_array = np.array([np.percentile(zscores_MR, i) for i in percentile_l])
                thresholds_array_list.append(thresholds_array)
            elif 'TF' in strategy:
                prices_l_ma = movingaverage(prices_l, WH)
                prices_l_ma2 = movingaverage(prices_l, WH * 2)
                prices_l_ma_flat = prices_l_ma.reshape(prices_l_ma.shape[0] * prices_l_ma.shape[1], -1)
                prices_l_ma2_flat = prices_l_ma2.reshape(prices_l_ma2.shape[0] * prices_l_ma2.shape[1], -1)
                # Compute the z-scores for each day using the historical data up to that day
                zscores_TF = (prices_l_ma_flat - prices_l_ma2_flat) / 0.01

                zscores_TF = zscores_TF.cpu().detach().numpy()
                thresholds_array = np.array([np.percentile(zscores_TF, i) for i in percentile_l])
                thresholds_array_list.append(thresholds_array)
            else:
                pass

        thresholds_array_stocks = np.stack(thresholds_array_list)
        np.save(thresholds_path, thresholds_array_stocks)
    return thresholds_array_stocks