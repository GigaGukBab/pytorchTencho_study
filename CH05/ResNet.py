from ResNetBasicBlock import *

# Define ResNet Model
class ResNet(nn.Module):
    def __init__(self, num_classes = 10):
        super(ResNet, self).__init__()

        # 1. Basic Block
        self.b1 = BasicBlock(in_channels=3, out_channels=64)
        self.b2 = BasicBlock(in_channels=64, out_channels=128)
        self.b3 = BasicBlock(in_channels=128, out_channels=256)

        # 2. Replace max pooling with average pooling
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 3. Classifier
        self.fc1 = nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)

        self.relu = nn.ReLU()

    # Define forward pass of ResNet
    def forward(self, x):
        # 1. Pass through the basic block and pooling layer
        x = self.b1
        x = self.pool(x)
        x = self.b2
        x = self.pool(x)
        x = self.b3
        x = self.pool(x)

        # 2. Flatten for use as input to the classifier
        x = torch.flatten(x, start_dim=1)

        # 3. Print predictions using the classifier.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x