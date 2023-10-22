import glob # 이미지를 불러올 때 사용하는 라이브러리
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image

__all__ = ["Pets"]

class Pets(Dataset):
    def __init__(self, path_to_img,
                       path_to_anno,
                       train=True,
                       transforms=None,
                       input_size=(128, 128)):
    
        # 1. 정답과 입력 이미지를 이름순으로 정렬
        self.images = sorted(glob.glob(path_to_img + "/*.jpg"))
        self.annotations = sorted(glob.glob(path_to_anno + "/*.png"))

        # 2. dataset을 train과 test로 나누기
        self.X_train = self.images[:int(0.8 * len(self.images))]
        self.X_test = self.images[int(0.8 * len(self.images)):]
        self.Y_train = self.annotations[:int(0.8 * len(self.annotations))]
        self.Y_test = self.annotations[int(0.8 * len(self.annotations)):]

        self.train = train # 학습용 데이터 평가용 데이터 결정 여부
        self.transforms = transforms # 사용할 데이터 증강
        self.input_size = input_size # 입력 이미지 크기

    def __len__(self): # 데이터 개수를 나타냄
        if self.train:
            return len(self.X_train) # 학습용 데이터셋 길이
        else:
            return len(self.X_test)  # 평가용 데이터셋 길이
        
    def preprocess_mask(self, mask): # 정답을 변환하는 함수
        mask = mask.resize(self.input_size)
        mask = np.array(mask).astype(np.float32)
        mask[mask != 2.0] = 1.0
        mask[mask == 2.0] = 0.0
        mask = torch.tensor(mask)
        return mask
    
    def __getitem__(self, i): # i번째 데이터와 정답을 반환
        if self.train:
            X_train = Image.open(self.X_train[i])
            X_train = self.transforms(X_train)
            Y_train = Image.open(self.Y_train[i])
            Y_train = self.transforms(Y_train)

            return X_train, Y_train
        else: # 평가용 데이터
            X_test = Image.open(self.X_test[i])
            X_test = self.transforms(X_test)
            Y_test = Image.open(self.Y_test[i])
            Y_test = self.preprocess_mask(Y_test)

            return X_test, Y_test
