# 4.4 전이 학습 모델 VGG로 분류하기
from VGG16 import *

## 4.4.2 모델 학습하기

### 데이터 전처리와 증강
import tqdm

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data import DataLoader, random_split

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

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


print(f"Training data size: {len(training_data)}")
print(f"Test data size: {len(test_data)}")


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

torch.save(model.state_dict(), "CIFAR_pretrained_VGG16.pt") # 모델 저장