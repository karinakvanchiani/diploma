import torch
import torch.nn as nn
import torch.nn.functional as f


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, features=32):
        super(UNet, self).__init__()

        self.encoder1 = UNet.one_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet.one_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet.one_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UNet.one_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = UNet.one_block(features * 8, features * 16)
        

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet.one_block((features * 8) * 2, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet.one_block((features * 4) * 2, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet.one_block((features * 2) * 2, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet.one_block(features * 2, features)

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e1_pool = self.pool1(e1)
        e2 = self.encoder2(e1_pool)
        e2_pool = self.pool2(e2)
        e3 = self.encoder3(e2_pool)
        e3_pool = self.pool3(e3)
        e4 = self.encoder4(e3_pool)
        e4_pool = self.pool4(e4)

        bottleneck = self.bottleneck(e4_pool)

        d4 = self.decoder4(torch.cat((self.upconv4(bottleneck), e4), dim=1))
        d3 = self.decoder3(torch.cat((self.upconv3(d4), e3), dim=1))
        d2 = self.decoder2(torch.cat((self.upconv2(d3), e2), dim=1))
        d1 = self.decoder1(torch.cat((self.upconv1(d2), e1), dim=1))
        return torch.sigmoid(self.conv(d1))


    @staticmethod
    def one_block(in_features, out_features):

        block = nn.Sequential(nn.Conv2d(in_channels=in_features, 
                                        out_channels=out_features, 
                                        kernel_size=3, 
                                        padding=1, 
                                        bias=False),
                              nn.BatchNorm2d(num_features=out_features),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(in_channels=out_features, 
                                        out_channels=out_features, 
                                        kernel_size=3, 
                                        padding=1, 
                                        bias=False),
                              nn.BatchNorm2d(num_features=out_features),
                              nn.ReLU(inplace=True))

        return block