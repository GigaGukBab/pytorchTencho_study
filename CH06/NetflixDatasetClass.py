from data import *

# 넷플릭스 데이터셋 정의

import numpy as np
from torch.utils.data.dataset import Dataset

class Netflix(Dataset): # 1. Define class
    def __init__(self):
        # 2. Read data
        self.csv = pd.read_csv("/Users/Jinwoo/JinwooWorkspace/pytorchTencho_study/CH06/CH06.csv")

        # Normalize input data (Min-Max Scaling)
        self.data = self.csv.iloc[:, 1:4].values  # 3. 종가를 제외한 데이터
        self.data = self.data / np.max(self.data) # 4. 0 ~ 1 사이로 정규화

        # 5. 종가 데이터 정규화
        self.label = data["Close"].values
        self.label = self.label / np.max(self.label)

    # 사용 가능한 batch 개수를 반환하는 __len__() 함수
    def __len__(self):
        return len(self.data) - 30 # 1. 사용 가능한 batch 개수
    
    def __getitem__(self, i):
        data = self.data[i:i+30] # 1. 입력 데이터 30일치 읽기
        label = self.label[i+30] # 2. 종가 데이터 30일치 읽기

        return data, label