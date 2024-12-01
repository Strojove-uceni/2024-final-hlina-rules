import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import UpsamplingBilinear2d

class InitialConvolutionBlock(nn.Module):
    def __init__(self):
        super(InitialConvolutionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return x


class PoolingBlock(nn.Module):
    def __init__(self):
        super(PoolingBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        mid_channels = int(self.output_channels/4)
        self.stride = stride

        # First layer
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=1, stride=1, bias=False)

        # Second layer
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1,
                               bias=False)

        # Third layer
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(output_channels // 4, output_channels, kernel_size=1, stride=1, bias=False)

        # Shortcut connection
        self.conv4 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=stride,
                               bias=False) if input_channels != output_channels or stride != 1 else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.conv4:
            residual = self.conv4(x)

        out += residual
        return out

class AttentionModule_stage1(nn.Module):
    #three times Maxpooling
    def __init__(self, in_channels, out_channels, size1, size2=None, size3=None, retrieve_mask=False):
        super(AttentionModule_stage1, self).__init__()

        self.initital_residual = ResidualBlock(in_channels, out_channels)
        self.trunk_branch = nn.Sequential(ResidualBlock(in_channels, out_channels),
                                          ResidualBlock(in_channels, out_channels),)

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_down1 = ResidualBlock(in_channels, out_channels)
        self.skip1 = ResidualBlock(in_channels, out_channels)

        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_down2 = ResidualBlock(in_channels, out_channels)
        self.skip2 = ResidualBlock(in_channels, out_channels)

        if size3:
            self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.residual_down3 = ResidualBlock(in_channels, out_channels)
            self.upsample3 = UpsamplingBilinear2d(size=size3)
            self.post_residual3 = ResidualBlock(in_channels, out_channels)

        self.upsample2 = UpsamplingBilinear2d(size=size2)
        self.post_residual2 = ResidualBlock(in_channels, out_channels)

        self.upsample1 = UpsamplingBilinear2d(size=size1)
        self.final_residual = nn.Sequential(
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                            nn.BatchNorm2d(out_channels),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                            nn.Sigmoid()
                                            )

        self.last_residual = ResidualBlock(in_channels, out_channels)
        self.retrieve_mask = retrieve_mask

    def forward(self, x):
        x = self.initital_residual(x)

        #trunk branch:
        trunk_output = self.trunk_branch(x)

        #mask branch:
        mask = self.pool1(x)
        mask = self.residual_down1(mask)
        skip1 = self.skip1(mask)

        mask = self.pool2(mask)
        mask = self.residual_down2(mask)
        skip2 = self.skip2(mask)

        if hasattr(self, 'pool3'):
            mask = self.pool3(mask)
            mask = self.residual_down3(mask)
            mask = self.upsample3(mask)
            mask = mask + skip2
            mask = self.post_residual3(mask)

        mask = self.upsample2(mask)
        mask = mask + skip1
        mask = self.post_residual2(mask)

        mask = self.upsample1(mask)
        mask = self.final_residual(mask)

        output = (1+mask)*trunk_output
        output = self.last_residual(output)

        if self.retrieve_mask:
            return output, mask
        return output


class AttentionModule_stage2(nn.Module):
    # two times Maxpooling
    def __init__(self, in_channels, out_channels, size1=(28, 28), size2=(14, 14), retrieve_mask=False):
        super(AttentionModule_stage2, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        # Trunk branch
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        # Mask branch
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual1_blocks = ResidualBlock(in_channels, out_channels)
        self.skip1_connection_residual_block = ResidualBlock(in_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual2_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.interpolation2 = nn.UpsamplingBilinear2d(size=(28, 28))
        self.residual3_blocks = ResidualBlock(in_channels, out_channels)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=(56, 56))

        self.residual4_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

        self.retrieve_mask = retrieve_mask

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_residual1 = self.residual1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_residual1)

        out_mpool2 = self.mpool2(out_residual1)
        out_residual2 = self.residual2_blocks(out_mpool2)

        out_interp2 = self.interpolation2(out_residual2) + out_residual1
        out = out_interp2 + out_skip1_connection
        out_residual3 = self.residual3_blocks(out)
        out_interp1 = self.interpolation1(out_residual3) + out_trunk
        out_residual4 = self.residual4_blocks(out_interp1)
        out = (1 + out_residual4) * out_trunk
        out_last = self.last_blocks(out)

        if self.retrieve_mask:
            return out_last, out_residual4
        return out_last


class AttentionModule_stage3(nn.Module):
    # only one Maxpooling
    def __init__(self, in_channels, out_channels, size1=(14, 14), retrieve_mask=False):
        super(AttentionModule_stage3, self).__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        # Trunk branch
        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        # Mask branch
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=(56, 56))
        self.residual2_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

        self.retrieve_mask = retrieve_mask

    def forward(self, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_residual1 = self.residual1_blocks(out_mpool1)

        out_interp1 = self.interpolation1(out_residual1) + out_trunk
        out_residual2 = self.residual2_blocks(out_interp1)
        out = (1 + out_residual2) * out_trunk
        out_last = self.last_blocks(out)

        if self.retrieve_mask:
            return out_last, out_residual2
        return out_last
