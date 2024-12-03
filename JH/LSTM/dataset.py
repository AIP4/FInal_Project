import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class FinedustDataset(Dataset):
    """
    3시간 단위로 연속하지 않는 행들을 제거하는 데이터셋
    Default: 이전 일주일을 입력으로 다음 일주일 예측 (timesteps = (24 / 3) X 7 = 56)
    """
    def __init__(self, data, window_size=56, prediction_length=56):
        self.data = data
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.sequences = []
        self.time_column = self.data['diff'].values
        
        self.dust_column = "PM10"
        self.feature_columns = self.data.columns.drop([self.dust_column, 'TM'])

        for i in range(len(self.data) - self.window_size - self.prediction_length + 1):
            time_window = self.time_column[i:i + self.window_size + self.prediction_length]
            is_continuous = all(time_window == 3)
            if is_continuous:
                x = self.data[self.feature_columns].iloc[i:i+self.window_size].values
                y = self.data[self.dust_column].iloc[
                    i+self.window_size:i+self.window_size+self.prediction_length].values
                self.sequences.append((x, y))
            else:
                continue
        
    def __len__(self):
        return len(self.data) - self.window_size - self.prediction_length + 1
    
    def __getitem__(self, idx):
        # window size: window_size
        x, y = self.sequences[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# if __name__ == "__main__":
#     data_path = "../../collect_data/filtered/kma/merged/kma_andong_meta.csv"
#     dataset = FinedustDataset(data_path)
    
#     print(dataset[0])