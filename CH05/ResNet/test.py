from pytorchTencho_study.CH05.ResNet.ResNetBasicBlock import *
from pytorchTencho_study.CH05.ResNet.ResNet import *
from pytorchTencho_study.CH05.ResNet.train import *

model.load_state_dict(torch.load("ResNet.pt", map_location=device))

num_corr = 0

with torch.no_grad():
    for data, label in test_loader:

        output = model(data.to(device))
        preds = output.data.max(1)[1]
        corr = preds.eq(label.to(device).data).sum()
        num_corr += corr

    print(f"Accuracy:{num_corr/len(test_data)}")