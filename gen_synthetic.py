"""
Generate synthetic data
"""

import pandas as pd
import numpy as np
import os
from os.path import join, isfile
import random
import statsmodels.api as sm
from sklearn.datasets import make_sparse_spd_matrix, make_spd_matrix
from statsmodels.stats.moment_helpers import cov2corr, corr2cov

your_path = 'your_path'
parent_data_path = join(your_path, 'gan_data')


def transform_1d(row_type, r, n_cols, alpha, beta, w):
    """

    :param row_type: model type
    :param r: a vector of gaussian noise
    :param n_cols: number of timestamps
    :param alpha: alpha in GARCH model
    :param beta: beta in GARCH model
    :param w: w in GARCH model
    :return:
    """
    if row_type == 'Gauss':
        tmp_data = r[1:]
    elif 'AR' in row_type and 'GARCH' not in row_type:
        tmp_data = 0 * r

        coef = float(row_type.split('AR')[1])

        for k in range(1, r.shape[0]):
            tmp_data[k] = coef / 100 * tmp_data[k - 1] + r[k - 1]

        tmp_data = np.delete(tmp_data, 0)

    elif 'GARCH' in row_type:
        # GARCH with Gaussian noise
        if row_type  == 'GARCH':
            eps = r
        # GARCH with Student noise
        elif '-T' in row_type:
            deg_free = int(row_type.split('-T')[1])
            eps = r / np.sqrt(np.random.chisquare(df=deg_free, size=r.shape[0]) / deg_free)
        else:
            eps = 0 * r

        delt_squ = 0 * eps
        tmp_data = 0 * eps

        for k in range(1, r.shape[0]):
            delt_squ[k] = w + alpha * tmp_data[k - 1] ** 2 + beta * delt_squ[k - 1]
            tmp_data[k] = np.sqrt(delt_squ[k]) * eps[k]

        tmp_data = np.delete(tmp_data, 0)

    else:
        tmp_data = 0 * r

    return tmp_data[n_cols:]


def extract_data_info(data_name):
    raw_row_type_list = data_name.split('+')
    row_type_list = []
    for row_type_str in raw_row_type_list:
        row_type = row_type_str.split('_')[1]
        row_type_num = int(row_type_str.split('_')[0])
        row_type_list.extend([row_type] * row_type_num)
    return row_type_list


def gen_data(data_name, length, n_rows, n_cols):
    print('+ ' * 5 + data_name + ' +' * 5)
    data_path = join(parent_data_path, data_name)
    os.makedirs(data_path, exist_ok=True)

    row_type_list = extract_data_info(data_name)
    assert len(row_type_list) == n_rows

    basic_info_path = join(data_path, 'basic_info.npz')

    if isfile(basic_info_path):
        basic_info = np.load(basic_info_path)
        mean = basic_info['mean']
        cov = basic_info['cov']
        std = np.sqrt(np.diag(cov))
        autocoef_l = basic_info['autocoef']
        alpha_l = basic_info['alpha']
        beta_l = basic_info['beta']
        w_l = basic_info['w']
    else:
        mean = np.zeros(n_rows)
        corr = np.abs(make_sparse_spd_matrix(n_rows, alpha=0.0, norm_diag=True))
        std =  np.random.uniform(0.3, 0.5, n_rows) / np.sqrt(250 * n_cols)
        cov = corr2cov(corr, std)
        autocoef_l = np.random.uniform(-0.15, 0.15, n_rows)
        alpha_l = np.random.uniform(0.08, 0.12, n_rows)
        beta_l = np.random.uniform(0.825, 0.875, n_rows)
        w_l = np.random.uniform(0.03, 0.07, n_rows)
        np.savez(basic_info_path, mean=mean, cov=cov, autocoef=autocoef_l, alpha=alpha_l, beta=beta_l, w=w_l)

    for item in range(length):
        if item % int(length / 10) == 0:
            print(item)
        tmp_data_l = []
        data_r = np.random.multivariate_normal(mean, cov, size=(2 * n_cols + 1)).T

        for i, row_type in enumerate(row_type_list):
            tmp_data = transform_1d(row_type, data_r[i, :], n_cols, alpha_l[i], beta_l[i], w_l[i])
            tmp_data_l.append(tmp_data)

        data_raw = np.stack(tmp_data_l)
        data = data_raw / np.tile(np.std(data_raw, 1), (n_cols, 1)).T * np.tile(std, (n_cols, 1)).T

        df = pd.DataFrame(data).T
        df.columns = row_type_list

        df.to_csv(join(data_path, "%d.csv" % (item + 1)), index=False)


if __name__ == "__main__":
    # asset types: Gauss, AR(+0.5), AR(-0.12), GARCH, GARCH-T
    data_name = '1_Gauss+1_AR50+1_AR-12+1_GARCH-T5+1_GARCH-T10' 
    length = 500000     # number of samples
    n_rows = 5          # number of assets
    n_cols = 100        # number of timestamps
    gen_data(data_name, length, n_rows, n_cols)