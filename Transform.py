import numpy as np
import os
import random
import statsmodels.api as sm
from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix
from statsmodels.stats.moment_helpers import cov2corr, corr2cov
from os.path import *

import torch
from torch import nn

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
BoolTensor = torch.cuda.BoolTensor if cuda else torch.BoolTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

trans_parent_data_path = '/data01/Chao_TailGAN/Static_Port_Transform'


# Convert increments to price by setting initial price as 1
def Inc2Price(data):
    price0 = Tensor(data.shape[0], data.shape[1], 1).fill_(1)
    prices_l = torch.cat((price0, data), dim=2)
    prices_l = torch.cumsum(prices_l, dim=2)
    return prices_l


##################################### Static Portfolio #####################################
def StaticPort(prices_l, n_trans, static_way, insample):
    n_rows = prices_l.shape[1]
    if 'Long' in static_way:
        trans_version = '_'.join(
            ['Long',
             'Stk' + str(n_rows)])
    elif 'LShort' in static_way:
        trans_version = '_'.join(
            ['LShort',
             'Stk' + str(n_rows)])
    else:
        trans_version = None

    trans_data_path = join(trans_parent_data_path, trans_version)

    if insample:
        store_path = join(trans_data_path, 'TransMat_IS.npy')
    else:
        store_path = join(trans_data_path, 'TransMat_OOS.npy')

    trans_mat = np.load(store_path)
    trans_mat = Tensor(trans_mat[:, :n_trans])

    swap_prices = prices_l.permute(0, 2, 1)
    broad_trans_mat = trans_mat.reshape(1, *trans_mat.shape)
    broad_trans_mat = broad_trans_mat.repeat(swap_prices.size(0), 1, 1)
    swap_trans_prices = torch.bmm(swap_prices, broad_trans_mat)
    port_prices_l = swap_trans_prices.permute(0, 2, 1)
    return port_prices_l


##################################### Buy and Hold (BH) #####################################
def BuyHold(prices_l, Cap):
    BH_money_l = prices_l * Cap
    sum_PNL_BH = BH_money_l[:, :, -1] - BH_money_l[:, :, 0]
    return sum_PNL_BH


################################## Mean Reversion (MR) based on Moving Average (MA) ##################################
########### Moving Average (MA) ###########
def movingaverage(values, WH):
    mean_conv = nn.Conv1d(1, 1, WH)
    kernel_weights = np.ones((1, 1, WH)) / WH
    mean_conv.weight.data = Tensor(kernel_weights)
    mean_conv.weight.requires_grad = False
    mean_conv.bias.data = Tensor(np.zeros(1))
    mean_conv.bias.requires_grad = False
    output_l = [mean_conv(values[:, [ch], :].float()) for ch in range(values.shape[1])]
    all_output = values.clone()
    all_output[:, :, WH-1:] = torch.cat(output_l, dim=1)
    return all_output


def Position_MR(zscores, Cap, LR, SR, ST_tensor, LT_tensor):
    # # # # # # # # SHORT SCENARIO  # # # # # # # #
    # Changing Points Vector
    zero_cross = ((zscores[:, :-1] < 0) & (zscores[:, 1:] >= 0)) | ((zscores[:, :-1] > 0) & (zscores[:, 1:] <= 0))
    zero_cross = torch.cat((BoolTensor(zero_cross.shape[0] * [True]).reshape(-1, 1), zero_cross), dim=1)

    sigma_plus = (zscores[:, :-1] < ST_tensor) & (zscores[:, 1:] >= ST_tensor)
    sigma_plus = torch.cat((BoolTensor(sigma_plus.shape[0] * [False]).reshape(-1, 1), sigma_plus), dim=1)

    # Short TimeStamp
    short_l = -1 * zero_cross + 1 * sigma_plus + 1 * (zero_cross & sigma_plus)
    short_flat = short_l.flatten()

    index_l = LongTensor(np.arange(0, len(short_flat), 1))

    short_nonzero = short_flat[short_flat != 0]
    index_nonzero = index_l[short_flat != 0]

    short_time = torch.cat((BoolTensor([False]), (short_nonzero[:-1] < 0) & (short_nonzero[1:] > 0)))
    short_ts = index_nonzero[short_time]

    # Clear TimeStamp
    clear_time = torch.cat((BoolTensor([False]), (short_nonzero[:-1] > 0) & (short_nonzero[1:] < 0)))
    clear_ts = index_nonzero[clear_time]

    # Short Position
    pos_short = torch.zeros(len(short_flat))
    pos_short[short_ts[short_ts % zscores.shape[-1] != 0]] = -1

    pos_clear = torch.zeros(len(short_flat))
    pos_clear[clear_ts[clear_ts % zscores.shape[-1] != 0]] = 1

    short_pos = torch.cumsum(pos_short.reshape(zscores.shape), dim=1) + torch.cumsum(pos_clear.reshape(zscores.shape), dim=1)
    short_pos = short_pos.type(Tensor)

    # # # # # # # # LONG SCENARIO  # # # # # # # #
    sigma_minus = (zscores[:, :-1] > LT_tensor) & (zscores[:, 1:] <= LT_tensor)
    sigma_minus = torch.cat((BoolTensor(sigma_minus.shape[0] * [False]).reshape(-1, 1), sigma_minus), dim=1)

    # Long TimeStamp
    long_l = -1 * zero_cross + 1 * sigma_minus + 1 * (zero_cross & sigma_minus)
    long_flat = long_l.flatten()

    long_nonzero = long_flat[long_flat != 0]
    index_nonzero = index_l[long_flat != 0]

    long_time = torch.cat((BoolTensor([False]), (long_nonzero[:-1] < 0) & (long_nonzero[1:] > 0)))
    long_ts = index_nonzero[long_time]

    # Clear TimeStamp
    clear_time = torch.cat((BoolTensor([False]), (long_nonzero[:-1] > 0) & (long_nonzero[1:] < 0)))
    clear_ts = index_nonzero[clear_time]

    # Long Position
    pos_long = torch.zeros(len(long_flat))
    pos_long[long_ts[long_ts % zscores.shape[-1] != 0]] = 1

    pos_clear = torch.zeros(len(long_flat))
    pos_clear[clear_ts[clear_ts % zscores.shape[-1] != 0]] = -1

    long_pos = torch.cumsum(pos_long.reshape(zscores.shape), dim=1) + torch.cumsum(pos_clear.reshape(zscores.shape), dim=1)
    long_pos = long_pos.type(Tensor)

    position = Cap * SR * short_pos + Cap * LR * long_pos
    position[:, -1] = 0
    return position


def MeanRev(prices_l, Cap, WH, LR, SR, ST, LT):
    prices_l_flat = prices_l.view(prices_l.shape[0] * prices_l.shape[1], -1)

    ST_tensor = torch.cat(prices_l.shape[0] * [Tensor(ST)]).reshape(-1, 1)
    LT_tensor = torch.cat(prices_l.shape[0] * [Tensor(LT)]).reshape(-1, 1)

    # moving average
    # prices_l_ma_raw = movingaverage(prices_l, WH)
    prices_l_ma = torch.mean(prices_l[:, :, :WH+1], dim=2)
    prices_l_ma_flat = prices_l_ma.view(prices_l_ma.shape[0] * prices_l_ma.shape[1], -1)

    # Compute the z-scores for each day using the historical data up to that day
    zscores = (prices_l_flat - prices_l_ma_flat) / 0.01

    position = Position_MR(zscores, Cap, LR, SR, ST_tensor, LT_tensor)

    PNL_MR_l = position[:, :-1] * (prices_l_flat[:, 1:] - prices_l_flat[:, :-1])
    PNL_MR = PNL_MR_l.reshape(prices_l.shape[0], prices_l.shape[1], -1)
    sum_PNL_MR = torch.sum(PNL_MR, dim=2)
    return sum_PNL_MR


#################################### Trend Following (TF) with price difference ####################################
def Position_TF(zscores, Cap, LR, SR, ST_tensor, LT_tensor):
    # # # # # # # # SHORT SCENARIO  # # # # # # # #
    # Changing Points Vector
    trend_cross = ((zscores[:, :-1] < 0) & (zscores[:, 1:] >= 0)) | ((zscores[:, :-1] > 0) & (zscores[:, 1:] <= 0))
    trend_cross = torch.cat((BoolTensor(trend_cross.shape[0] * [True]).reshape(-1, 1), trend_cross), dim=1)

    sigma_minus = (zscores[:, :-1] > ST_tensor) & (zscores[:, 1:] <= ST_tensor)
    sigma_minus = torch.cat((BoolTensor(sigma_minus.shape[0] * [False]).reshape(-1, 1), sigma_minus), dim=1)

    short_l = -1 * trend_cross + 1 * sigma_minus + 1 * (trend_cross & sigma_minus)
    short_flat = short_l.flatten()

    index_l = LongTensor(np.arange(0, len(short_flat), 1))

    short_nonzero = short_flat[short_flat != 0]
    index_nonzero = index_l[short_flat != 0]

    # Short TimeStamp
    short_time = torch.cat((BoolTensor([False]), (short_nonzero[:-1] < 0) & (short_nonzero[1:] > 0)))
    short_ts = index_nonzero[short_time]

    # Clear TimeStamp
    clear_time = torch.cat((BoolTensor([False]), (short_nonzero[:-1] > 0) & (short_nonzero[1:] < 0)))
    clear_ts = index_nonzero[clear_time]

    # Short Position
    pos_short = torch.zeros(len(short_flat))
    pos_short[short_ts[short_ts % zscores.shape[-1] != 0]] = -1

    pos_clear = torch.zeros(len(short_flat))
    pos_clear[clear_ts[clear_ts % zscores.shape[-1] != 0]] = 1

    short_pos = torch.cumsum(pos_short.reshape(zscores.shape), dim=1) + torch.cumsum(pos_clear.reshape(zscores.shape), dim=1)
    short_pos = short_pos.type(Tensor)

    # # # # # # # # LONG SCENARIO  # # # # # # # #
    sigma_plus = (zscores[:, :-1] < LT_tensor) & (zscores[:, 1:] >= LT_tensor)
    sigma_plus = torch.cat((BoolTensor(sigma_plus.shape[0] * [False]).reshape(-1, 1), sigma_plus), dim=1)

    long_l = -1 * trend_cross + 1 * sigma_plus + 1 * (trend_cross & sigma_plus)
    long_flat = long_l.flatten()

    long_nonzero = long_flat[long_flat != 0]
    index_nonzero = index_l[long_flat != 0]

    # Long TimeStamp
    long_time = torch.cat((BoolTensor([False]), (long_nonzero[:-1] < 0) & (long_nonzero[1:] > 0)))
    long_ts = index_nonzero[long_time]

    # Clear TimeStamp
    clear_time = torch.cat((BoolTensor([False]), (long_nonzero[:-1] > 0) & (long_nonzero[1:] < 0)))
    clear_ts = index_nonzero[clear_time]

    # Long Position
    pos_long = torch.zeros(len(long_flat))
    pos_long[long_ts[long_ts % zscores.shape[-1] != 0]] = 1

    pos_clear = torch.zeros(len(long_flat))
    pos_clear[clear_ts[clear_ts % zscores.shape[-1] != 0]] = -1

    long_pos = torch.cumsum(pos_long.reshape(zscores.shape), dim=1) + torch.cumsum(pos_clear.reshape(zscores.shape), dim=1)
    long_pos = long_pos.type(Tensor)

    position = Cap * SR * short_pos + Cap * LR * long_pos
    position[:, -1] = 0
    return position


def TrendFollow(prices_l, Cap, WH, LR, SR, ST, LT):
    prices_l_flat = prices_l.reshape(prices_l.shape[0] * prices_l.shape[1], -1)

    ST_tensor = torch.cat(prices_l.shape[0] * [Tensor(ST)]).reshape(-1, 1)
    LT_tensor = torch.cat(prices_l.shape[0] * [Tensor(LT)]).reshape(-1, 1)

    prices_l_ma = movingaverage(prices_l, WH)
    prices_l_ma2 = movingaverage(prices_l, WH * 2)
    prices_l_ma_flat = prices_l_ma.reshape(prices_l_ma.shape[0] * prices_l_ma.shape[1], -1)
    prices_l_ma2_flat = prices_l_ma2.reshape(prices_l_ma2.shape[0] * prices_l_ma2.shape[1], -1)

    # Compute the z-scores for each day using the historical data up to that day
    zscores = (prices_l_ma_flat - prices_l_ma2_flat) / 0.01

    position = Position_TF(zscores, Cap, LR, SR, ST_tensor, LT_tensor)

    PNL_TF_l = position[:, :-1] * (prices_l_flat[:, 1:] - prices_l_flat[:, :-1])
    PNL_TF = PNL_TF_l.reshape(prices_l.shape[0], prices_l.shape[1], -1)
    sum_PNL_TF = torch.sum(PNL_TF, dim=2)
    return sum_PNL_TF