# 4.3.4 모델 성능 평가하기
from defCNN import *
from VGGBasicBlock import *
from train import *

model.load_state_dict(torch.load("CIFAR10-myCNN-VGG.pt", map_location=device))

num_corr = 0

with torch.no_grad():

    iterator = tqdm.tqdm(test_loader, desc='Testing')
    for data, label in iterator:
        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum().item()
        num_corr += corr

    print(f"Accuracy:{num_corr/len(test_data)}")