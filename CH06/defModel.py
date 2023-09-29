from data import *
from RNNClass import *
from NetflixDatasetClass import *

import tqdm

from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU 사용 가능")
elif device =="cpu":
    print("GPU 사용 불가")

model = RNN().to(device) # 모델 정의
dataset = Netflix()      # 데이터셋 정의

# 데이터로더 정의
loader = DataLoader(dataset, batch_size=32)  # 배치 크기를 32로 설정

# 최적화 정의
optim = Adam(params=model.parameters(), lr=0.0001) # 사용할 최적화 설정