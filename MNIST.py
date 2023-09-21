# %%
import matplotlib.pyplot as plt

from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

# 학습용 데이터와 평가용 데이터 분리
training_data = MNIST(root="./", train=True, download=True, transform=ToTensor())
test_data = MNIST(root="./", train=False, download=True, transform=ToTensor())
# %%
print(len(training_data)) # 학습에 사용할 데이터 개수
print(len(test_data))     # 평가에 사용할 데이터 개수
# %%
for i in range(9):  # 샘플 이미지 9개 출력
    plt.subplot(3, 3, i+1)
    plt.imshow(training_data.data[i])
plt.show()
# %%
