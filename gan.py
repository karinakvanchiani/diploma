import torch
import torch.nn as nn, n
import torch.nn.functional as f


class ResidualDenseBlock(n.Module):
    def __init__(self,in_channel=64, inc_channel=32, beta=0.2):
        super().__init__()
        self.conv1 = n.Conv2d(in_channel, inc_channel, 3, 1, 1)
        self.conv2 = n.Conv2d(in_channel + inc_channel, inc_channel, 3, 1, 1)
        self.conv3 = n.Conv2d(in_channel + 2 * inc_channel, inc_channel, 3, 1, 1)
        self.conv4 = n.Conv2d(in_channel + 3 * inc_channel, inc_channel, 3, 1, 1)
        self.conv5 = n.Conv2d(in_channel + 4 * inc_channel,  in_channel, 3, 1, 1)
        self.lrelu = n.LeakyReLU()
        self.b = beta
        
    def forward(self, x):
        block1 = self.lrelu(self.conv1(x))
        block2 = self.lrelu(self.conv2(torch.cat((block1, x), dim = 1)))
        block3 = self.lrelu(self.conv3(torch.cat((block2, block1, x), dim = 1)))
        block4 = self.lrelu(self.conv4(torch.cat((block3, block2, block1, x), dim = 1)))
        out = self.conv5(torch.cat((block4, block3, block2, block1, x), dim = 1))
        
        return x + self.b * out


class ResidualInResidualDenseBlock(n.Module):
    def __init__(self, in_channel=64, out_channel=32, beta=0.2):
        super().__init__()
        self.RDB = ResidualDenseBlock(in_channel, out_channel)
        self.b = beta
    
    def forward(self, x):
        out = self.RDB(x)
        out = self.RDB(out)
        out = self.RDB(out)
        
        return x + self.b * out


class Generator(nn.Module):
    def __init__(self,in_channel=3, out_channel=3, noRRDBBlock=23):
        super().__init__()   
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)

        RRDB = ResidualInResidualDenseBlock()
        RRDB_layer = []
        for i in range(noRRDBBlock):
            RRDB_layer.append(RRDB)
        self.RRDB_block =  nn.Sequential(*RRDB_layer)

        self.RRDB_conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.upconv = nn.Conv2d(64, 64, 3, 1, 1)

        self.out_conv = nn.Conv2d(64, 3, 3, 1, 1)
    
    def forward(self, x):
        first_conv = self.conv1(x)
        RRDB_full_block = torch.add(self.RRDB_conv2(self.RRDB_block(first_conv)),first_conv)
        upconv_block1 = self.upconv(f.interpolate(RRDB_full_block, scale_factor = 2))
        upconv_block2 = self.upconv(f.interpolate(upconv_block1, scale_factor = 2))
        out = self.out_conv(upconv_block2)
        
        return out


class Discriminator(n.Module):
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
        self.fc2 = n.Linear(1024, 1)
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
        block9 = f.leaky_relu(self.fc1(block8))
        return block9