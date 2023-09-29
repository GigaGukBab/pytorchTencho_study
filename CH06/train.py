from data import *
from RNNClass import *
from NetflixDatasetClass import *
from defModel import *

for epoch in range(200):
    iterator = tqdm.tqdm(loader)
    for data, label in iterator:
        optim.zero_grad()

        # 1. 초기 은닉 상태
        h0 = torch.zeros(5, data.shape[0], 8).to(device)

        # 2. 모델의 예측값
        pred = model(data.type(torch.FloatTensor).to(device), h0)

        # 3. 손실값 계산
        loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))
        loss.backward() # 오차 역전파
        optim.step()    # 최적화 진행

        iterator.set_description(f"eopoch{epoch} loss:{loss}")

torch.save(model.state_dict(), "./rnn.pt") # 모델 저장