import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange

class GNLB(nn.Module):
    def __init__(self, mid_c=256):
        super(GNLB, self).__init__()
        self.softmax = nn.Softmax2d()
        self.mid_c = mid_c
        self.conv0 = nn.Conv2d(1, self.mid_c, 3, 1, 1)
        self.conv1 = nn.Conv2d(3, self.mid_c, 3, 1, 1)

        self.conv_mid = nn.Conv2d(self.mid_c, self.mid_c, 1, 1, 0)
        self.conv_out = nn.Conv2d(self.mid_c, 3, 1, 1, 0)
        self.conv_last = nn.Conv2d(3, 3, 1, 1, 0)

    def same_block(self, x):
        x_up = rearrange(x, 'b c h w -> b c (h w)')
        x_down = rearrange(x, 'b c h w -> b (h w) c')
        x_out = torch.matmul(x_down, x_up)
        x_out = self.softmax(x_out.unsqueeze(1))
        return x_out.squeeze(1)


    def forward(self, x0, x1_src):
        x0 = self.conv0(x0)
        x1 = self.conv1(x1_src)
        x0_ = self.same_block(x0)
        x1_ = self.same_block(x1)
        x01 = torch.matmul(x0_, x1_)
        x01 = self.softmax(x01.unsqueeze(1)).squeeze(1)
        x1_down = rearrange(x1, 'b c h w -> b (h w) c')
        x01_cat = torch.matmul(x01, x1_down)
        x01_cat = self.softmax(x01_cat.unsqueeze(1)).squeeze(1)
        img_size = int(x01_cat.size(1)**0.5)
        x01_cat = rearrange(x01_cat, 'b (s1 s2) c -> b c s1 s2', s1=img_size, s2=img_size)
        x01_cat = self.conv_mid(x01_cat)
        x01_cat = self.conv_out(x01_cat)
        x_out = x01_cat+x1_src
        x_out = self.conv_last(x_out)
        return x_out


if __name__ == '__main__':
    x0 = torch.rand(2, 1, 64, 64)
    x1 = torch.rand(2, 3, 64, 64)
    model = GNLB()
    y = model(x0, x1)

    print('ok')