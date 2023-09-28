# 전이 학습 모델 ResNet50으로 분류하기

## 사전 학습된 모델 불러오기

### 사전 학습된 모델 준비
import torch
import torch.nn as nn

from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU 사용 가능")
elif device =="cpu":
    print("GPU 사용 불가")

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
fc = nn.Sequential(
        nn.Linear(2048, 512),  # ResNet50의 마지막 층의 출력은 2048
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, 10),
    )

model.fc = fc
model.to(device)