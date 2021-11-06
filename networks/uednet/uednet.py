""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.common_func.get_backbone import get_model


class EDBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EDBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.down2 = nn.MaxPool2d((2, 2))
        # self.transC2 = nn.ConvTranspose2d(out_c, out_c, 2, 2, 2)
        self.transC2 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     nn.Conv2d(out_c, out_c, 3, 1, 1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU())
        self.down4 = nn.MaxPool2d((4, 4))
        # self.transC4 = nn.ConvTranspose2d(out_c, out_c, 2, 4, 3)
        self.transC4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     nn.Conv2d(out_c, out_c, 3, 1, 1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU(),
                                     nn.Upsample(scale_factor=2),
                                     nn.Conv2d(out_c, out_c, 3, 1, 1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU()
                                     )
        self.down8 = nn.MaxPool2d((8, 8))
        # self.transC8 = nn.ConvTranspose2d(out_c, out_c, 2, 8, 5)
        self.transC8 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     nn.Conv2d(out_c, out_c, 3, 1, 1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU(),
                                     nn.Upsample(scale_factor=2),
                                     nn.Conv2d(out_c, out_c, 3, 1, 1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU(),
                                     nn.Upsample(scale_factor=2),
                                     nn.Conv2d(out_c, out_c, 3, 1, 1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU()
                                     )
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_c, out_c, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x2 = self.down2(x)
        x2 = self.relu(self.bn(self.conv(x2)))
        x2 = self.transC2(x2)

        x4 = self.down4(x)
        x4 = self.relu(self.bn(self.conv(x4)))
        x4 = self.transC4(x4)

        x8 = self.down8(x)
        x8 = self.relu(self.bn(self.conv(x8)))
        x8 = self.transC8(x8)

        x_conv = self.relu(self.bn(self.conv(x)))
        x_out = x2+x4+x8+x_conv

        return x_out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_cannels, num_classannels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_cannels, num_classannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classannels, num_classannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_cannels, num_classannels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_cannels, num_classannels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_cannels, num_classannels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_cannels // 2, in_cannels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_cannels, num_classannels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_cannels, num_classannels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_cannels, num_classannels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UED50(nn.Module):
    def __init__(self, in_c, num_class, bilinear=True):
        super(UED50, self).__init__()
        self.in_c = 3
        self.num_class = num_class
        self.bilinear = bilinear
        self.backbone = get_model('resnet50')

        self.inc = DoubleConv(in_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(2048+1024, 256, bilinear)
        self.up2 = Up(512+256, 128, bilinear)
        self.up3 = Up(256+128, 64, bilinear)
        self.up4 = Up(64+64, 64, bilinear)

        self.ed1 = EDBlock(64, 64)
        self.ed2 = EDBlock(256, 256)
        self.ed3 = EDBlock(512, 512)
        self.ed4 = EDBlock(1024, 1024)
        self.ed5 = EDBlock(2048, 2048)

        self.outc = OutConv(64, num_class)

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]   #64->2048
        x1 = self.ed1(x1)
        x2 = self.ed2(x2)
        x3 = self.ed3(x3)
        x4 = self.ed4(x4)
        x5 = self.ed5(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = F.interpolate(x, size=(size[2], size[3]), mode='bilinear')
        out = self.outc(out)
        return out

class Res_UNet_34(nn.Module):
    def __init__(self, in_c, num_class, bilinear=True):
        super(Res_UNet_34, self).__init__()
        self.in_c = 3
        self.num_class = num_class
        self.bilinear = bilinear
        self.backbone = get_model('resnet34')

        self.inc = DoubleConv(in_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512+256, 256, bilinear)
        self.up2 = Up(256+128, 128, bilinear)
        self.up3 = Up(128+64, 64, bilinear)
        self.up4 = Up(64+64, 64, bilinear)

        self.outc = OutConv(64, num_class)

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = F.interpolate(x, size=(size[2], size[3]), mode='bilinear')
        out = self.outc(out)
        return out


class Res_UNet_101(nn.Module):
    def __init__(self, in_c, num_class, bilinear=True):
        super(Res_UNet_101, self).__init__()
        self.in_c = 3
        self.num_class = num_class
        self.bilinear = bilinear
        self.backbone = get_model('resnet101')

        self.inc = DoubleConv(in_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(2048+1024, 256, bilinear)
        self.up2 = Up(512+256, 128, bilinear)
        self.up3 = Up(256+128, 64, bilinear)
        self.up4 = Up(64+64, 64, bilinear)

        self.outc = OutConv(64, num_class)

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = F.interpolate(x, size=(size[2], size[3]), mode='bilinear')
        out = self.outc(out)
        return out


class Res_UNet_152(nn.Module):
    def __init__(self, in_c, num_class, bilinear=True):
        super(Res_UNet_152, self).__init__()
        self.in_c = 3
        self.num_class = num_class
        self.bilinear = bilinear
        self.backbone = get_model('resnet152')

        self.inc = DoubleConv(in_c, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(2048+1024, 256, bilinear)
        self.up2 = Up(512+256, 128, bilinear)
        self.up3 = Up(256+128, 64, bilinear)
        self.up4 = Up(64+64, 64, bilinear)

        self.outc = OutConv(64, num_class)

    def forward(self, x):
        size = x.size()
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = F.interpolate(x, size=(size[2], size[3]), mode='bilinear')
        out = self.outc(out)
        return out