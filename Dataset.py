"""
Characterizes a dataset for PyTorch
"""

import torch
from torch.utils import data
import os
import pandas as pd
import numpy as np


class Dataset_IS(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, tickers, data_path, length):
        """Initialization"""
        self.tickers = tickers
        self.data_path = data_path
        self.length = length
        self.samples = []
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith('csv')]
        files.sort()
        for f in files[:self.length]:
            f_path = os.path.join(data_path, f)
            f_array = pd.read_csv(f_path)[tickers].values.T
            f_tensor = torch.from_numpy(f_array)
            self.samples.append(f_tensor)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.samples[index]


class Dataset_OOS(data.Dataset):
    def __init__(self, tickers, data_path, length):
        """Initialization"""
        self.tickers = tickers
        self.data_path = data_path
        self.length = length
        self.samples = []
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith('csv')]
        files.sort()
        for f in files[self.length:]:
            f_path = os.path.join(data_path, f)
            f_array = pd.read_csv(f_path)[tickers].values.T
            f_tensor = torch.from_numpy(f_array)
            self.samples.append(f_tensor)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.samples[index]
