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