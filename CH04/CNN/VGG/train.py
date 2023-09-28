# 4.3.3 모델 학습하기
from VGGBasicBlock import *
from defCNN import *

# 데이터 증강 정의
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import *
from torchvision.datasets.cifar import CIFAR10
from torch.optim.adam import Adam
import tqdm

transforms = Compose([
    RandomCrop((32, 32), padding=4), # 1. 랜덤 크롭핑
    RandomHorizontalFlip(p=0.5), # 2. y축으로 좌우대칭
    ToTensor(), # 3. 텐서로 변환
    # 4. 이미지 정규화
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
])

# 데이터 로드 및 모델 정의
# 1. 학습용 데이터와 평가용 데이터 불러오기
training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

# 2. 데이터로더 정의
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 3. 학습을 진행할 프로세서 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU 사용 가능")
elif device =="cpu":
    print("GPU 사용 불가")

# 4. CNN 모델 정의
model = CNN(num_classes=10)

# 5. 모델을 device로 보냄
model.to(device)

# 모델 학습하기
if __name__ == "__main__":
    # 1. 학습률 정의
    lr = 1e-3

    # 2. 최적화 기법 정의
    optim = Adam(model.parameters(), lr=lr)

    # 학습 루프 정의
    for epoch in range(100):
        for data, label in train_loader: # 3. 데이터 호출
            optim.zero_grad() # 4. 기울기 초기화

            preds = model(data.to(device)) # 5. 모델의 예측

            # 오차역전파의 최적화
            loss = nn.CrossEntropyLoss()(preds, label.to(device))
            loss.backward()
            optim.step()

        if epoch == 0 or epoch % 10 == 9: # 10번마다 손실 출력
            print(f"epoch{epoch+1} loss:{loss}")

    # 모델 저장
    torch.save(model.state_dict(), "CIFAR10-myCNN-VGG.pt")