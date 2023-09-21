# %%
# # 보스턴 집값 예측하기 : 회귀 분석
# %%
# 데이터 살펴보기
import pandas as pd

# 데이터셋의 URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"

# 데이터셋의 컬럼명
column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

# pandas를 사용하여 데이터 로드
data = pd.read_csv(url, delim_whitespace=True, header=None, names=column_names)

# 처음 5행을 출력하여 데이터 확인
print(data.head())
# %%
# 학습 코드 구현

import torch
import torch.nn as nn
from torch.optim.adam import Adam

# 1. 모델 정의
model = nn.Sequential(
    nn.Linear(13, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

X = data.iloc[:, :13].values # 2. 정답을 제외한 특징을 X에 입력
Y = data['MEDV'].values # 데이터프레임의 target값을 추출

batch_size = 100
learning_rate = 0.001

# 3. 가중치를 수정하는 최적화 함수 정의
optim = Adam(model.parameters(), lr=learning_rate)

# 에포크 반복
for epoch in range(200):

    # 배치 반복
    for i in range(len(X)//batch_size):
        start = i * batch_size # 4. 배치 크기에 맞게 인덱스 지정
        end = start + batch_size

        # 파이토치 실수형 텐서로 변환
        x = torch.FloatTensor(X[start:end])
        y = torch.FloatTensor(Y[start:end])

        optim.zero_grad() # 5. 가중치의 기울기를 0으로 초기화
        preds = model(x) # 6. 모델의 예측값 계산
        loss = nn.MSELoss()(preds, y) # 7. MSE 손실, 계산
        loss.backward() # 8. 오차 역전파
        optim.step() # 9. 최적화 진행

    if epoch % 20 == 0:
        print(f"epoch{epoch} loss:{loss}")
# %%
# 모델 성능 평가하기

prediction = model(torch.FloatTensor(X[0, :13]))
real = Y[0]
print(f"prediction:{prediction.item()} real:{real}")