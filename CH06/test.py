from data import *
from RNNClass import *
from NetflixDatasetClass import *
from defModel import *

loader = DataLoader(dataset, batch_size=1) # 예측값을 위한 데이터로더

preds = [] # 예측값들을 저장하는 리스트
total_loss = 0

with torch.no_grad():
    # 모델의 가중치 불러오기
    model.load_state_dict(torch.load("rnn.pt", map_location=device))

    for data, label in loader:
        h0 = torch.zeros(5, data.shape[0], 8).to(device)  # 1. 초기 은닉 상태 정의

        # 모델의 예측값 출력
        pred = model(data.type(torch.FloatTensor).to(device), h0)
        preds.append(pred.item()) # 2. 예측값을 리스트에 추가

        # 손실 계산
        loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))

        # 3. 손실의 평균치 계산
        total_loss += loss/len(loader)

print(total_loss.item())

# plot test result

plt.plot(preds, label="prediction")
plt.plot(dataset.label[30:], label="actual")
plt.legend()
# 그림을 저장하기 위한 코드
plt.savefig("./test_result_plot.png")  # 현재 디렉터리에 그림 파일을 저장