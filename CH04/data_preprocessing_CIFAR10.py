# %%
# 4.2 데이터 전처리하기

# 데이터 살펴보기
import matplotlib.pyplot as plt

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor

# CIFAR-10 데이터셋 불러오기
training_data = CIFAR10(
    root="./",             # 1
    train=True,            # 2
    download=True,         # 3
    transform=ToTensor()   # 4
)

test_data = CIFAR10(
    root="./",
    train=False,
    download=True,
    transform=ToTensor()
)

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(training_data.data[i])
plt.show()
# %%
# 4.2.1 데이터 증강
# 데이터 전처리에 크롭핑과 좌우대칭 추가
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
# %%
# 4.2.2 이미지 정규화
# 데이터 전처리에 정규화 추가
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, Normalize

# 데이터 전처리 정의
transforms = Compose([
    T.ToPILImage(),
    RandomCrop((32, 32), padding=4),
    RandomHorizontalFlip(p=0.5),
    T.ToTensor(),

    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)), # 1. 데이터 정규화
    T.ToPILImage()
])

# 학습용 데이터 정의
training_data = CIFAR10(
    root="./",
    train=True,
    download=True,
    transform=transforms
)

# 평가용 데이터 정의
test_data = CIFAR10(
    root="./",
    train=False,
    download=True,
    transform=transforms
)

# 이미지 표시
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(transforms(training_data.data[i]))
plt.show()
# %%
# 데이터셋의 평균과 표준편차

import torch

training_data = CIFAR10(
    root="./",
    train=True,
    download=True,
    transform=ToTensor()
)

# item[0]은 이미지, item[1]은 정답 레이블
imgs = [item[0] for item in training_data]

# 1. imgs를 하나로 합침
imgs = torch.stack(imgs, dim=0).numpy()

# rgb 각 평균
mean_r = imgs[:,0,:,:].mean()
mean_g = imgs[:,1,:,:].mean()
mean_b = imgs[:,2,:,:].mean()
print(mean_r, mean_g, mean_b)

# rgb 각 표준편차
std_r = imgs[:,0,:,:].std()
std_g = imgs[:,1,:,:].std()
std_b = imgs[:,2,:,:].std()
print(std_r, std_g, std_b)
# %%
