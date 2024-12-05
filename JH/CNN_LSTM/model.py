import torch
import torch.nn as nn
import einops


### Conv1d Model
class PM10CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, K=1):
        super(PM10CNN, self).__init__()
        self.act = nn.GELU()

        self.conv_list = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels if i == 0 else out_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size, 
                      padding=1) for i in range(K)
            ])
    
    def forward(self, x):
        outputs = []
        x = einops.rearrange(x, 'b l c -> b c l')  # (batch_size, seq_length, in_channels)
        
        for conv in self.conv_list:
            x = self.act(conv(x))  # Conv1d 적용 후 GELU 활성화
            outputs.append(x)  # 출력 저장
        
        stacked_outputs = torch.stack(outputs, dim=1)  # (batch_size, K, out_channels, seq_length)
        ### Concatenate K outputs
        stacked_outputs = einops.rearrange(stacked_outputs, 'b k c l -> b l (k c)')  # (batch_size, seq_length, K*out_channels)
        return stacked_outputs


### LSTM Model
class FinedustLSTM(nn.Module):
    def __init__(self, input_size, num_layers, hidden_size, output_size, dropout_prob=0.3):
        super(FinedustLSTM, self).__init__()
        self.lstm = LSTMEmbedding(input_size, hidden_size, num_layers, dropout_prob=dropout_prob)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.lstm(x)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        return x

class LSTMEmbedding(nn.Module):
    '''LSTM Embedding 생성'''
    def __init__(self, input_size, hidden_size, num_layers, act_fct='gelu', dropout_prob=0.3):
        super(LSTMEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM Layer
        self.lstm = (nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
                     if num_layers > 1 else nn.LSTM(input_size, hidden_size, 1, batch_first=True))
        
        # Additional fully connected layers
        # 시계열 데이터는 일반적으로 Trend, Seasonality, Noise로 구분되므로 3개로 Mapping해보자
        self.fc1 = nn.Linear(hidden_size, hidden_size * 3)  # 첫 번째 선형 계층
        self.fc2 = nn.Linear(hidden_size * 3, hidden_size)  # 두 번째 선형 계층
        
        # Activation and Dropout
        self.act = nn.GELU()         # 활성화 함수로 GELU 사용
        self.dropout = nn.Dropout(p=dropout_prob)  # 드롭아웃 추가

    def forward(self, x):
        # Initializing hidden state for first input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))  # LSTM output
        out = out[:, -1, :]  # 마지막 시퀀스의 출력만 사용
        
        # Fully connected layers with GELU and Dropout
        out = self.dropout(self.act(self.fc1(out)))  # 첫 번째 선형 계층
        out = self.dropout(self.act(self.fc2(out)))  # 두 번째 선형 계층
        
        return out