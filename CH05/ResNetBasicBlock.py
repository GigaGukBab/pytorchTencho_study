import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BasicBlock, self).__init__()

        # 1. Define Convolution Layer
        self.c1 = nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size, padding=1)
        self.c2 = nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size, padding=1)
        
        self.downsample = nn.Conv2d(in_channels, out_channels,
                                    kernel_size=1)
        
        # 2. Define Batch Normalization Layer
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        # 3. Saving the initial input for skip connection
        x_ = x

        # The "F(x)" part in the ResNet BasicBlock
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)

        # 4. Matching the number of channels between the output of a convolution and the input
        x_ = self.downsample(x_)

        # 5. Adding the result of the convolutional layer to the saved input (skip connection)
        x += x_
        x = self.relu(x)

        return x