import torch.nn as n
import torch.nn.functional as f


class VGG(n.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = n.Conv2d(3, 64, 3,padding=1, bias=False)
        self.conv2 = n.Conv2d(64, 64, 3,stride=2, padding=1, bias=False)
        self.bn2 = n.BatchNorm2d(64)
        self.conv3 = n.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = n.BatchNorm2d(128)
        self.conv4 = n.Conv2d(128, 128, 3, stride=2, padding=1, bias=False)
        self.bn4 = n.BatchNorm2d(128)
        self.conv5 = n.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn5 = n.BatchNorm2d(256)
        self.conv6 = n.Conv2d(256, 256, 3, stride=2, padding=1, bias=False)
        self.bn6 = n.BatchNorm2d(256)
        self.conv7 = n.Conv2d(256, 512, 3, padding=1, bias=False)
        self.bn7 = n.BatchNorm2d(512)
        self.conv8 = n.Conv2d(512, 512, 3, stride=2, padding=1, bias=False)
        self.bn8 = n.BatchNorm2d(512)
        self.fc1 = n.Linear(512*16*16, 1024)
        self.fc2 = n.Linear(1024, 2)
        self.drop = n.Dropout2d(0.3)
        
    def forward(self, x):
        block1 = f.leaky_relu(self.conv1(x))
        block2 = f.leaky_relu(self.bn2(self.conv2(block1)))
        block3 = f.leaky_relu(self.bn3(self.conv3(block2)))
        block4 = f.leaky_relu(self.bn4(self.conv4(block3)))
        block5 = f.leaky_relu(self.bn5(self.conv5(block4)))
        block6 = f.leaky_relu(self.bn6(self.conv6(block5)))
        block7 = f.leaky_relu(self.bn7(self.conv7(block6)))
        block8 = f.leaky_relu(self.bn8(self.conv8(block7)))
        block8 = block8.reshape(-1, block8.size(1) * block8.size(2) * block8.size(3))
        block9 = self.fc2(f.leaky_relu(self.fc1(block8)))
        return block9