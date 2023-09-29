# Define RNN Class

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
    
        # 1. RNN층의 정의
        '''
        input_size : 개장가, 최고가, 최저가 (3개의 숫자를 이용한 3차원 벡터)
        hidden_size : 입력 텐서에 가중치를 적용해 특징 추출(가중치를 이용해 hidden_dim 차원으로 확장)
        num_layers : 쌓을 RNN 층 개수(too many->기울기 소실, 무한대 발산) 일반적으로 5 or 3사용
        batch_first : batch 차원이 가장 앞에 오게 함 (32, 30, 3)
        '''
        self.rnn = nn.RNN(input_size=3, hidden_size=8, num_layers=5, batch_first=True)

        # 2. 주가를 예측하는 MLP층 정의
        self.fc1 = nn.Linear(in_features=240, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

        self.relu = nn.ReLU() # 활성화 함수 정의

    def forward(self, x, h0):
        x, hn = self.rnn(x, h0) # 1. RNN층의 출력

        # 2. MLP 층의 입력으로 사용되게 모양 변경
        x = torch.reshape(x, (x.shape[0], -1))

        # MLP층을 이용해 종가 예측
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # 예측한 종가를 1차원 벡터로 표현
        x = torch.flatten(x)

        return x
