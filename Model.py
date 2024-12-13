import torch
import torch.nn as nn
import torch.nn.functional as F
from Models_components import InitialConvolutionBlock, PoolingBlock, ResidualBlock, AttentionModule_stage1
from Models_components import AttentionModule_stage2, AttentionModule_stage3

class ResidualAttentionNetwork(nn.Module):
    def __init__(self):
        super(ResidualAttentionNetwork, self).__init__()

        self.initial_conv = InitialConvolutionBlock()
        self.pool = PoolingBlock() #paramters were not specified in the article, they are our choice
        self.residual0 = ResidualBlock(input_channels=64, output_channels=256)
        self.attention1 = AttentionModule_stage1(in_channels=256, out_channels=256, size1=(56, 56), size2=(28, 28),
                                                 size3=(14, 14))
        self.residual1 = ResidualBlock(input_channels=256, output_channels=256)
        self.attention2 = AttentionModule_stage2(in_channels=256, out_channels=256, size1=(28, 28), size2=(14, 14))
        self.residual2 = ResidualBlock(input_channels=256, output_channels=256)
        self.attention3 = AttentionModule_stage3(in_channels=256, out_channels=256, size1=(14, 14))
        self.residual3 = ResidualBlock(input_channels=256, output_channels=256)
        self.residual4 = ResidualBlock(input_channels=256, output_channels=256)
        self.residual5 = ResidualBlock(input_channels=256, output_channels=256)
        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 3)
        self.fc2 = nn.Linear(256, 3)
        self.fc3 = nn.Linear(256, 3)
        self.fc4 = nn.Linear(256, 3)

    def forward(self,x):
        x = self.initial_conv(x)
        x = self.pool(x)
        x = self.residual0(x)
        x = self.attention1(x)
        x = self.residual1(x)
        x = self.attention2(x)
        x = self.residual2(x)
        x = self.attention3(x)
        x = self.residual3(x)
        x = self.residual4(x)
        x = self.residual5(x)
        x = self.final_pool(x)
        x = torch.flatten(x, 1)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)
        out4 = self.fc4(x)
        return out1, out2, out3, out4

