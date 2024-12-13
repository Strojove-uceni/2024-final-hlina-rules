from torch import nn
import torch
from torch.nn import UpsamplingBilinear2d

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class PoolingBlock(nn.Module):
    def __init__(self):
        super(PoolingBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.double_conv = DoubleConv(self.in_channels, self.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.double_conv(x)
        pooled = self.pool(x)

        return x, pooled

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.output_channels = out_channels
        mid_channels = int(self.output_channels/4)
        self.stride = stride

        #First layer:
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias = False)

        #Second layer
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias = False)

        #Third layer
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, stride=1,padding=0, bias=False)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                               bias=False) if in_channels != out_channels or stride != 1 else None
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
            residual = self.conv4(out)

        out += residual
        return out

class AttentionModule_stage1(nn.Module):
    #three times Maxpooling
    def __init__(self, in_channels, out_channels, size1, size2, size3, retrieve_mask=False):
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

        # Trunk branch:
        trunk_output = self.trunk_branch(x)

        # Mask branch:
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
            # Align mask with skip2 dimensions
            if mask.size() != skip2.size():
                mask = torch.nn.functional.interpolate(mask, size=skip2.shape[2:], mode='bilinear', align_corners=False)
            mask = mask + skip2
            mask = self.post_residual3(mask)

        mask = self.upsample2(mask)
        # Align mask with skip1 dimensions
        if mask.size() != skip1.size():
            mask = torch.nn.functional.interpolate(mask, size=skip1.shape[2:], mode='bilinear', align_corners=False)
        mask = mask + skip1
        mask = self.post_residual2(mask)

        mask = self.upsample1(mask)
        # Align mask with trunk_output dimensions
        if mask.size() != trunk_output.size():
            mask = torch.nn.functional.interpolate(mask, size=trunk_output.shape[2:], mode='bilinear',
                                                   align_corners=False)
        mask = self.final_residual(mask)

        output = (1 + mask) * trunk_output
        output = self.last_residual(output)

        if self.retrieve_mask:
            return output, mask
        return output


class AttentionModule_stage2(nn.Module):
    # two times Maxpooling
    def __init__(self, in_channels, out_channels, retrieve_mask=False):
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

        # Downsample and process through residual blocks
        out_mpool1 = self.mpool1(x)
        out_residual1 = self.residual1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_residual1)

        out_mpool2 = self.mpool2(out_residual1)
        out_residual2 = self.residual2_blocks(out_mpool2)

        # Upsample and add
        out_interp2 = self.interpolation2(out_residual2)

        # Align dimensions of out_interp2 and out_residual1
        if out_interp2.size() != out_residual1.size():
            out_interp2 = torch.nn.functional.interpolate(out_interp2, size=out_residual1.shape[2:], mode='bilinear',
                                                          align_corners=False)

        out = out_interp2 + out_residual1

        # Add skip connection
        if out.size() != out_skip1_connection.size():
            out = torch.nn.functional.interpolate(out, size=out_skip1_connection.shape[2:], mode='bilinear',
                                                  align_corners=False)
        out += out_skip1_connection

        out_residual3 = self.residual3_blocks(out)
        out_interp1 = self.interpolation1(out_residual3)

        # Align dimensions of out_interp1 and out_trunk
        if out_interp1.size() != out_trunk.size():
            out_interp1 = torch.nn.functional.interpolate(out_interp1, size=out_trunk.shape[2:], mode='bilinear',
                                                          align_corners=False)

        out_residual4 = self.residual4_blocks(out_interp1)
        out = (1 + out_residual4) * out_trunk
        out_last = self.last_blocks(out)

        if self.retrieve_mask:
            return out_last, out_residual4
        return out_last


class AttentionModule_stage3(nn.Module):
    # only one Maxpooling
    def __init__(self, in_channels, out_channels, retrieve_mask=False):
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

        # Adjust dimensions of out_residual1 to match out_trunk
        out_interp1 = self.interpolation1(out_residual1)
        if out_interp1.size() != out_trunk.size():
            out_interp1 = torch.nn.functional.interpolate(out_interp1, size=out_trunk.shape[2:], mode='bilinear',
                                                          align_corners=False)

        out_interp1 = out_interp1 + out_trunk
        out_residual2 = self.residual2_blocks(out_interp1)
        out = (1 + out_residual2) * out_trunk
        out_last = self.last_blocks(out)

        if self.retrieve_mask:
            return out_last, out_residual2
        return out_last







