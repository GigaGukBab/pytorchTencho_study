# 4.2.1 데이터 증강

## 데이터 전처리에 크롭핑과 좌우대칭 추가
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop

transforms = Compose([ # 1. 데이터 전처리 함수
    T.ToPILImage(),
    RandomCrop((32, 32), padding=4), # 2. 랜덤으로 이미지 일부 제거 및 패딩
    RandomHorizontalFlip(p=0.5),     # 3. y축으로 기준으로 대칭
])

training_data = CIFAR10(
    root="./",
    train=True,
    download=True,
    transform=transforms # transform에는 데이터를 변환하는 함수가 들어감
)

test_data = CIFAR10(
    root="./",
    train=False,
    download=True,
    transform=transforms
)

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(transforms(training_data.data[i]))
plt.show()