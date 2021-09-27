import torch.nn as nn
import torch

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_cannels, num_classannels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_cannels, num_classannels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(num_classannels, num_classannels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_cannels, num_classannels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_cannels, num_classannels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(num_classannels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_cannels, num_classannels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_cannels, num_classannels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_cannels, num_classannels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_cannels//2,in_cannels//2,(2,2),2)
        self.nConvs = _make_nConv(in_cannels, num_classannels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, num_class=9):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.num_class = num_class
        # Question here
        in_cannels = 64
        self.inc = ConvBatchNorm(n_channels, in_cannels)
        self.down1 = DownBlock(in_cannels, in_cannels*2, nb_Conv=2)
        self.down2 = DownBlock(in_cannels*2, in_cannels*4, nb_Conv=2)
        self.down3 = DownBlock(in_cannels*4, in_cannels*8, nb_Conv=2)
        self.down4 = DownBlock(in_cannels*8, in_cannels*8, nb_Conv=2)
        self.up4 = UpBlock(in_cannels*16, in_cannels*4, nb_Conv=2)
        self.up3 = UpBlock(in_cannels*8, in_cannels*2, nb_Conv=2)
        self.up2 = UpBlock(in_cannels*4, in_cannels, nb_Conv=2)
        self.up1 = UpBlock(in_cannels*2, in_cannels, nb_Conv=2)
        self.outc = nn.Conv2d(in_cannels, num_class, kernel_size=(1,1))
        if num_class == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
            # print("111")
        else:
            logits = self.outc(x)
            # print("222")
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        return logits


