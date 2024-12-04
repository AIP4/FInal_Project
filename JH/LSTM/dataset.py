import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from interpolate import simple_interpolate
from utils import prepare_data


class FinedustDataset(Dataset):
    """
    3시간 단위로 연속하지 않는 행들을 제거하는 데이터셋
    Default: 이전 일주일을 입력으로 다음 3일 예측 (timesteps = (24 / 3) X 7 = 56)
    """
    def __init__(self, data, window_size=56, prediction_length=24, time_window=3):
        self.data = data
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.sequences = []
        self.time_column = self.data['diff'].values

        self.data = self.data.drop(columns=["diff"])
        
        self.dust_column = "PM10"
        self.feature_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        # Y 값과 TM 제외
        self.feature_columns.remove(self.dust_column)

        for i in range(len(self.data) - self.window_size - self.prediction_length + 1):
            time_window = self.time_column[i:i + self.window_size + self.prediction_length]
            is_continuous = all(time_window == time_window)
            if is_continuous:
                x = self.data[self.feature_columns].iloc[i:i+self.window_size].values
                y = self.data[self.dust_column].iloc[
                    i+self.window_size:i+self.window_size+self.prediction_length].values
                self.sequences.append((x, y))
            else:
                continue
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)
