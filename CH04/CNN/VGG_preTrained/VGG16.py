# 4.4 전이 학습 모델 VGG로 분류하기

## 4.4.1 사전 학습된 모델 불러오기

### 사전 학습된 모델 준비
import torch
import torch.nn as nn

from torchvision.models.vgg import vgg16

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU 사용 가능")
elif device =="cpu":
    print("GPU 사용 불가")

model = vgg16(pretrained = True) # 1. vgg16 모델 객체 생성
fc = nn.Sequential( # 2. 분류층 정의
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(),
        nn.Dropout(), # 3. 드롭아웃층 정의
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, 10),
    )

model.classifier = fc # 4. VGG의 classifier를 덮어씀
model.to(device)