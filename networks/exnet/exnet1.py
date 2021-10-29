import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common_func.get_backbone import get_model
from ..common_func.base_func import _ConvBNReLU


class _FCNHead(nn.Module):
    def __init__(self, in_cannels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_cannels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_cannels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


class _ASPPConv(nn.Module):
    def __init__(self, in_cannels, num_classannels, atrous_rate, norm_layer):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_cannels, num_classannels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(num_classannels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_cannels, num_classannels, norm_layer):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_cannels, num_classannels, 1, bias=False),
            norm_layer(num_classannels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_cannels, atrous_rates, norm_layer=nn.BatchNorm2d):
        super(_ASPP, self).__init__()
        in_channels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_cannels, in_channels, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_cannels, in_channels, rate1, nn.BatchNorm2d)
        self.b2 = _ASPPConv(in_cannels, in_channels, rate2, nn.BatchNorm2d)
        self.b3 = _ASPPConv(in_cannels, in_channels, rate3, nn.BatchNorm2d)
        self.b4 = _AsppPooling(in_cannels, in_channels, nn.BatchNorm2d)

        self.project = nn.Sequential(
            nn.Conv2d(5 * in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


class EDBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EDBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.down2 = nn.MaxPool2d((2, 2))
        # self.transC2 = nn.ConvTranspose2d(out_c, out_c, 3, 2, 2)
        self.transC2 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(out_c, out_c, 3, 1, 1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU())
        self.down4 = nn.MaxPool2d((4, 4))
        # self.transC4 = nn.ConvTranspose2d(out_c, out_c, 3, 4, 4)
        self.transC4 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=4),
                                     nn.Conv2d(out_c, out_c, 3, 1, 1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU())
        self.down8 = nn.MaxPool2d((8, 8))
        # self.transC8 = nn.ConvTranspose2d(out_c, out_c, 3, 8, 8)
        self.transC8 = nn.Sequential(nn.Upsample(scale_factor=8),
                                     nn.Conv2d(out_c, out_c, 3, 1, 1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU())
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
        x_out = x2 + x4 + x8 + x_conv

        return x_out


class EXNet(nn.Module):
    r"""EXNet
    Parameters
    ----------
    num_class : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'xception').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """

    def __init__(self, in_c, num_class, **kwargs):
        super(EXNet, self).__init__()
        aux = True
        self.aux = aux
        self.num_class = num_class
        self.in_c = in_c

        self.pretrained = get_model('efficientnet_b2', self.in_c)

        # deeplabv3 plus
        self.head = _DeepLabHead(num_class, c1_channels=24, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(728, num_class)

        self.edb1 = EDBlock(24, 48)
        self.down1 = nn.Sequential(nn.Conv2d(48, 48, 3, 2, 1),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU())
        self.edb2 = EDBlock(48, 120)
        self.down2 = nn.Sequential(nn.Conv2d(120, 120, 3, 2, 1),
                                   nn.BatchNorm2d(120),
                                   nn.ReLU())
        self.edb3 = EDBlock(120, 352)
        self.down3 = nn.Sequential(nn.Conv2d(352, 352, 3, 2, 1),
                                   nn.BatchNorm2d(352),
                                   nn.ReLU())

        self.aspp = _ASPP(352, [5, 7, 11])
        self.last = nn.Sequential(nn.Conv2d(256 + 24, num_class, 3, 1, 1), nn.Softmax())

    def base_forward(self, x):
        # Entry flow
        features = self.pretrained(x)
        return features

    def forward(self, x):
        size = x.size()[2:]
        c0, c1, c2, c3, c4 = self.base_forward(x)
        m1 = self.edb1(c1)
        m1 = self.down1(m1)
        c2 = m1 + c2
        m2 = self.edb2(c2)
        m2 = self.down2(m2)
        c3 = m2 + c3
        m3 = self.edb3(c3)
        m3 = self.down3(m3)
        c4 = m3 + c4

        top_f = self.aspp(c4)
        top_f = F.interpolate(top_f, scale_factor=8, mode='bilinear', align_corners=True)

        concat_f = torch.cat([c1, top_f], dim=1)

        out_f = self.last(concat_f)

        out_f = F.interpolate(out_f, size, mode='bilinear', align_corners=True)
        return out_f


class _DeepLabHead(nn.Module):
    def __init__(self, num_class, c1_channels=128, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(352, [12, 24, 36], nn.BatchNorm2d)
        self.c1_block = _ConvBNReLU(c1_channels, 48, 3, padding=1, norm_layer=nn.BatchNorm2d)
        self.block = nn.Sequential(
            _ConvBNReLU(304, 256, 3, padding=1, norm_layer=nn.BatchNorm2d),
            nn.Dropout(0.5),
            _ConvBNReLU(256, 256, 3, padding=1, norm_layer=nn.BatchNorm2d),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_class, 1))

    def forward(self, x, c1):
        size = c1.size()[2:]
        c1 = self.c1_block(c1)
        x = self.aspp(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return self.block(torch.cat([x, c1], dim=1))


if __name__ == '__main__':
    net = EXNet(in_c=3, num_class=6)
    # nrte = meca(64, 3)

    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print(y.shape)