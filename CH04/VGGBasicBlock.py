# VGG 기본 블록 정의
import torch
import torch.nn as nn

class BasicBlock(nn.Module): # 1. 기본 블록 정의
    # 기본 블록을 구성하는 층 정의
    def __init__(self, in_channels, out_channels, hidden_dim):
        # 2. nn.Module 클래스의 요소 상속
        super(BasicBlock, self).__init__()

        # 3. 힙성곱층 정의
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels,
                               kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # stride는 커널 이동 거리
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x): # 4. 기본 블록의 순전파 정의
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        return x