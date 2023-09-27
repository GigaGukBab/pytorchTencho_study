# %%
# 4.4 전이 학습 모델 VGG로 분류하기

# %%
# 4.4.1 사전 학습된 모델 불러오기

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
# %%
# 4.4.2 모델 학습하기
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

# 데이터로더 정의
training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


print(f"Training data size: {len(training_data)}")
print(f"Test data size: {len(test_data)}")


# 학습 루프 정의
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

torch.save(model.state_dict(), "CIFAR_pretrained.pt") # 모델 저장
# %%
# 4.4.3 모델 성능 평가하기

model.load_state_dict(torch.load("CIFAR_pretrained.pt", map_location=device))

num_corr = 0

with torch.no_grad():
    for data, label in test_loader:

        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum()
        num_corr += corr

    print(f"Accuracy:{num_corr/len(test_data)}")
# %%
