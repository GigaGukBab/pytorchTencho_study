# 데이터셋의 평균과 표준편차
from imageNorm import *
import torch

training_data = CIFAR10(
    root="./",
    train=True,
    download=True,
    transform=T.ToTensor()
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