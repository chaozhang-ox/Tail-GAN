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


class Dataset_Month(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, tickers, data_path, month):
        """Initialization"""
        self.tickers = tickers
        self.data_path = data_path
        self.month = month
        self.samples = []
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith('csv') and self.month in f]
        files.sort()
        self.length = len(files)
        for f in files:
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


class Dataset_Period(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, tickers, data_path, start_date, end_date):
        """Initialization"""
        self.tickers = tickers
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.samples = []
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith('csv') and self.start_date <= f.split('_')[0] <= self.end_date]
        files.sort()
        for f in files:
            f_path = os.path.join(data_path, f)
            f_array = pd.read_csv(f_path)[tickers].values.T
            f_tensor = torch.from_numpy(f_array)
            self.samples.append(f_tensor)

        self.length = len(self.samples)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.samples[index]


class Dataset_MonthWeek(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, tickers, data_path, month):
        """Initialization"""
        self.tickers = tickers
        self.data_path = data_path
        self.month = month
        self.samples = []
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith('csv') and self.month < f and self.month not in f]
        files.sort()
        files = files[:1500]
        print(files[0])
        print(files[-1])
        self.length = len(files)
        for f in files:
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


class Dataset_3Month1Week(data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, tickers, data_path, month):
        """Initialization"""
        self.tickers = tickers
        self.data_path = data_path
        self.month = month
        self.samples = []
        files = os.listdir(self.data_path)
        files = [f for f in files if f.endswith('csv') and (self.month > f or self.month in f)]
        files.sort()
        files = files[-1500:]
        print(files[0])
        print(files[-1])
        self.length = len(files)
        for f in files:
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