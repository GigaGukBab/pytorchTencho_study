# 4.4 전이 학습 모델 VGG로 분류하기
from ResNet50 import *

## 4.4.2 모델 학습하기

### 데이터 전처리와 증강
import tqdm

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
import numpy as np

from torch.optim.adam import Adam

transforms = Compose([
    Resize(224),
    RandomCrop((224, 224), padding=4),
    RandomHorizontalFlip(p=0.5),
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
])

### 데이터로더 정의
training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

### 원본 데이터셋에서 10%만 사용하기 위한 인덱스 계산
train_indices = np.random.choice(len(training_data), size=int(0.1 * len(training_data)), replace=False)
test_indices = np.random.choice(len(test_data), size=int(0.1 * len(test_data)), replace=False)

### Subset 객체 생성
training_data_subset = Subset(training_data, train_indices)
test_data_subset = Subset(test_data, test_indices)

### DataLoader에 Subset 객체 사용
train_loader = DataLoader(training_data_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data_subset, batch_size=32, shuffle=False)

print(f"Training data size: {len(training_data_subset)}")
print(f"Test data size: {len(test_data_subset)}")

### 학습 루프 정의
lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(30):
    iterator = tqdm.tqdm(train_loader) # 1. 학습 로그 출력
    for data, label in iterator:
        optim.zero_grad()

        preds = model(data.to(device)) # 모델의 예측값 출력

        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        # 2. tqdm이 출력할 문자열
        iterator.set_description(f"epoch:{epoch+1} loss:{loss}")

torch.save(model.state_dict(), "CIFAR_pretrained_ResNet50.pt") # 모델 저장