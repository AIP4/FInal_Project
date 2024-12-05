import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from interpolate import simple_interpolate
from utils import prepare_data


class MLPDataset(Dataset):
    def __init__(self, data):
        self.data = data.drop(columns=["diff"])
        self.feature_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.target_column = "PM10"
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data.iloc[idx][self.feature_columns].values.astype(float)
        y = self.data.iloc[idx][self.target_column]
        return torch.FloatTensor(x), torch.FloatTensor([y])