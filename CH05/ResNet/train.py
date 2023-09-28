from pytorchTencho_study.CH05.ResNet.ResNetBasicBlock import *
from pytorchTencho_study.CH05.ResNet.ResNet import *
import tqdm

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torchvision.transforms import RandomHorizontalFlip, RandomCrop
from torchvision.transforms import Normalize
from torch.utils.data.dataloader import DataLoader

from torch.optim.adam import Adam

transforms = Compose([
    RandomCrop((32, 32), padding=4), # 1. Random Cropping
    RandomHorizontalFlip(p=0.5),     # 2. Random y축 대칭
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
])

# 데이터로더 정의
training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("GPU 사용 가능")
elif device =="cpu":
    print("GPU 사용 불가")

model = ResNet(num_classes=10)
model.to(device)

# Define Training Loop
lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(30):
    iterator = tqdm.tqdm(train_loader)
    for data, label in iterator:
        # 1. 최적화를 위해 기울기 초기화
        optim.zero_grad()

        # 2. 모델의 예측값
        preds = model(data.to(device))

        # 3. 손실 계산 및 역전파
        loss = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

        iterator.set_description(f"epoch:{epoch+1} loss:{loss}")

torch.save(model.state_dict(), "ResNet.pt") # 모델 저장