# 4.2.2 이미지 정규화

## 데이터 전처리에 정규화 추가
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize

### 데이터 전처리 정의
transforms = Compose([
    T.ToPILImage(),
    RandomCrop((32, 32), padding=4),
    RandomHorizontalFlip(p=0.5),
    T.ToTensor(),

    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)), # 1. 데이터 정규화
    T.ToPILImage()
])

### 학습용 데이터 정의
training_data = CIFAR10(
    root="./",
    train=True,
    download=True,
    transform=transforms
)

### 평가용 데이터 정의
test_data = CIFAR10(
    root="./",
    train=False,
    download=True,
    transform=transforms
)

### 이미지 표시
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(transforms(training_data.data[i]))
plt.show()