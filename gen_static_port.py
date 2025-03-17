"""
Create transformation matrix used to generate static portfolios
"""

import os
import pandas as pd
import numpy as np
import random
from os.path import join, isfile
import scipy
import scipy.sparse as sparse
import scipy.stats as stats

# Save Path
your_path = 'your_path'
trans_parent_data_path = join(your_path, 'Static_Port_Transform')
os.makedirs(trans_parent_data_path, exist_ok=True)


def IsStock(vec):
    isstock_l = (np.abs(vec) == 1.0) + (np.abs(vec) == 0.0)
    return np.all(isstock_l)


def Gen_StaticPort(static_way, n_stocks, n_ports):
    if 'Long' in static_way:
        trans_version = '_'.join(
            ['Long',
             'Stk' + str(n_stocks)])
    elif 'LShort' in static_way:
        trans_version = '_'.join(
            ['LShort',
             'Stk' + str(n_stocks)])
    else:
        trans_version = None

    trans_data_path = join(trans_parent_data_path, trans_version)
    os.makedirs(trans_data_path, exist_ok=True)

    for inout in range(2):
        if inout == 0:
            store_path = join(trans_data_path, 'TransMat_IS.npy')
        else:
            store_path = join(trans_data_path, 'TransMat_OOS.npy')

        price_start = np.ones(n_stocks)

        if 'Long' in static_way:
            # Initialization
            unscale_trans2port_mat = sparse.rand(n_stocks, max(int(n_stocks ** 2), n_ports), density=1.0).toarray()
            assert np.all(unscale_trans2port_mat.sum(0) > 0) and np.all(unscale_trans2port_mat.sum(1) > 0)

        elif 'LShort' in static_way:
            # Initialization
            rvs = stats.norm(loc=0, scale=1).rvs
            unscale_trans2port_mat = sparse.random(n_stocks, max(int(n_stocks ** 2), n_ports), density=0.9, data_rvs=rvs).toarray()
        else:
            pass

        # Scale
        scale_trans_mat = unscale_trans2port_mat / np.abs(unscale_trans2port_mat).sum(0)
        # Position
        position = np.diag(1 / price_start.dot(np.abs(scale_trans_mat)))

        trans_mat = np.dot(scale_trans_mat, position)
        trans_mat_port = trans_mat[:, ~np.apply_along_axis(IsStock, 0, trans_mat)]
        assert trans_mat_port.shape[1] >= 10
        # trans_mat_port = trans_mat_port[:, :10]

        print(pd.DataFrame(trans_mat_port))

        # Store
        np.save(store_path, trans_mat_port)


if __name__ == '__main__':
    Gen_StaticPort('LShort', n_stocks=5, n_ports=50)