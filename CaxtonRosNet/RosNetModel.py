from torch import nn
import torch
from RosNetModelCompenents import DoubleConv, PoolingBlock, DownSample, AttentionModule_stage1, UpSample
from RosNetModelCompenents import AttentionModule_stage2, AttentionModule_stage3

class RosNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RosNet, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pooling = PoolingBlock()

        #First block
        self.down1 = DownSample(in_channels = 64, out_channels=128)
        self.bottleneck1 = DoubleConv(128, 256)
        self.attention1 = AttentionModule_stage1(256, 256, size1=(56, 56), size2=(28, 28), size3=(14, 14))
        self.up1 = UpSample(256, 128)

        # Second block
        self.down2 = DownSample(128, 256)
        self.bottleneck2 = DoubleConv(256, 512)
        self.attention2 = AttentionModule_stage2(512, 512)
        self.up2 = UpSample(512, 256)

        # Third block
        self.down3 = DownSample(256, 512)
        self.bottleneck3 = DoubleConv(512, 1024)
        self.attention3 = AttentionModule_stage3(1024, 1024)
        self.up3 = UpSample(1024, 512)

        # Final convolution
        self.final_conv = nn.Conv2d(512, 512, 1)
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        # four linear heads
        self.linear1 = nn.Linear(512, 3)
        self.linear2 = nn.Linear(512, 3)
        self.linear3 = nn.Linear(512, 3)
        self.linear4 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.double_conv(x)
        print(x.shape)
        x_down1, x_pooled1 = self.down1(x)
        print(x_down1.shape, x_pooled1.shape)
        b1 = self.bottleneck1(x_pooled1)
        print(b1.shape)
        att1 = self.attention1(b1)
        x_up1 = self.up1(att1, x_down1)

        x_down2, x_pooled2 = self.down2(x_up1)
        b2 = self.bottleneck2(x_pooled2)
        att2 = self.attention2(b2)
        x_up2 = self.up2(att2, x_down2)

        x_down3, x_pooled3 = self.down3(x_up2)
        b3 = self.bottleneck3(x_pooled3)
        att3 = self.attention3(b3)
        x_up3 = self.up3(att3, x_down3)

        x = self.final_conv(x_up3)
        x = self.final_pool(x)
        x = torch.flatten(x, 1)


        out1 = self.linear1(x)
        out2 = self.linear2(x)
        out3 = self.linear3(x)
        out4 = self.linear4(x)

        return out1, out2, out3, out4


