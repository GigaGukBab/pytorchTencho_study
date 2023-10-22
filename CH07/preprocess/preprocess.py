import tqdm
import torch

from torchvision.transforms import Compose, ToTensor, Resize
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from datasetInit import Pets
from plot import path_to_image, path_to_annotation

device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 전처리 정의
transform = Compose([Resize((128, 128)), ToTensor()])

# 학습용 데이터
train_set = Pets(path_to_img=path_to_image,
                 path_to_anno=path_to_annotation,
                 transforms=transform)

# 평가용 데이터
test_set = Pets(path_to_img=path_to_image,
                path_to_anno=path_to_annotation,
                transforms=transform,
                train=False)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set)