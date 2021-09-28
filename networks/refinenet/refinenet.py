import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


def un_pool(input, scale):
    return F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=True)


class MyLoss(nn.Module):
    def __init__(self, w1=0.5, w2=0.5):
        super().__init__()

        self.w1 = w1
        self.w2 = w2
        self.BCELoss = nn.BCELoss(reduce=True, reduction='sum')
        self.MSELoss = nn.MSELoss()

    def forward(self, seg_out, depth_out, seg_target, depth_target):
        loss = self.w1 * self.BCELoss(seg_out, seg_target) + self.w2 * self.MSELoss(depth_out, depth_target)
        if loss > 10000:
            loss1 = self.BCELoss(seg_out, seg_target)
            loss2 = self.MSELoss(depth_out, depth_target)
        return loss


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()

        _, max_h, max_w = max(shapes, key=lambda x: x[1])

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, h, w = shape
            if max_h % h != 0:
                raise ValueError("max_size not divisble by shape {}".format(i))

            self.scale_factors.append(max_h // h)
            self.add_module(
                "resolve{}".format(i),
                nn.Conv2d(
                    feat,
                    out_feats,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False))

    def forward(self, *xs):

        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = un_pool(output, self.scale_factors[0])

        for i, x in enumerate(xs[1:], 1):
            tmp_out = self.__getattr__("resolve{}".format(i))(x)
            if self.scale_factors[i] != 1:
                tmp_out = un_pool(tmp_out, self.scale_factors[i])
            output = output + tmp_out

        return output


class ChainedResidualPool(nn.Module):
    def __init__(self, feats, block_count=4):
        super().__init__()

        self.block_count = block_count
        self.relu = nn.ReLU(inplace=False)
        for i in range(0, block_count):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(0, self.block_count):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class ChainedResidualPoolImproved(nn.Module):
    def __init__(self, feats, block_count=4):
        super().__init__()

        self.block_count = block_count
        self.relu = nn.ReLU(inplace=False)
        for i in range(0, block_count):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False),
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(0, self.block_count):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion,
                 chained_residual_pool, *shapes):
        super().__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module(
                "rcu{}".format(i),
                nn.Sequential(
                    residual_conv_unit(feats), residual_conv_unit(feats)))

        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None

        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPool, *shapes)


class RefineNetBlockImprovedPooling(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPoolImproved, *shapes)

class BaseRefineNet4Cascade(nn.Module):
    """Multi-path 4-Cascaded RefineNet for image segmentation
    Args:
        input_shape ((int, int, int)): (channel, h, w) assumes input has
            equal height and width
        refinenet_block (block): RefineNet Block
        num_classes (int, optional): number of classes
        features (int, optional): number of features in net
        resnet_factory (func, optional): A Resnet model from torchvision.
            Default: models.resnet101
        pretrained (bool, optional): Use pretrained version of resnet
            Default: True
        freeze_resnet (bool, optional): Freeze resnet model
            Default: True
    Raises:
        ValueError: size of input_shape not divisible by 32
    """
    def __init__(self,
                 input_shape,
                 refinenet_block=RefineNetBlock,
                 num_classes=2,
                 features=256,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):

        super().__init__()

        input_channel, input_h, input_w = input_shape

        if input_h % 32 != 0:
            raise ValueError("{} not divisble by 32".format(input_shape))

        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refinenet4 = refinenet_block(2 * features,
                                         (2 * features, input_h // 32, input_w // 32))
        self.refinenet3 = refinenet_block(features,
                                         (2 * features, input_h // 32, input_w // 32),
                                         (features, input_h // 16, input_w // 16))
        self.refinenet2 = refinenet_block(features,
                                         (features, input_h // 16, input_w // 16),
                                         (features, input_h // 8, input_w // 8))
        self.refinenet1 = refinenet_block(features,
                                         (features, input_h // 8, input_w // 8),
                                         (features, input_h // 4, input_w // 4))

        self.segBranch = nn.Sequential(
            ResidualConvUnit(features),
            ResidualConvUnit(features),
            nn.Conv2d(features, num_classes, kernel_size=1, stride=1,
                      padding=0, bias=True),
            nn.Sigmoid())

        self.depthBranch = nn.Sequential(
            ResidualConvUnit(features),
            ResidualConvUnit(features),
            nn.Conv2d(features, 1, kernel_size=1, stride=1,
                      padding=0, bias=True))

        self.initialize_weights()

        resnet = resnet_factory(pretrained=pretrained)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool, resnet.layer1)

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if freeze_resnet:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # self.output_conv = nn.Sequential(
        #     ResidualConvUnit(features), ResidualConvUnit(features),
        #     nn.Conv2d(
        #         features,
        #         num_classes,
        #         kernel_size=1,
        #         stride=1,
        #         padding=0,
        #         bias=True))

    def forward(self, x):

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)

        path_1 = un_pool(path_1, 4)

        seg = self.segBranch(path_1)
        depth = self.depthBranch(path_1)

        # out = self.output_conv(path_1)
        return seg, depth

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def named_parameters(self):
    #     """Returns parameters that requires a gradident to update."""
    #     return (p for p in super().named_parameters() if p[1].requires_grad)


class RefineNet4Cascade(BaseRefineNet4Cascade):
    """Multi-path 4-Cascaded RefineNet for image segmentation
    Args:
        input_shape ((int, int, int)): (channel, h, w) assumes input has
            equal height and width
        refinenet_block (block): RefineNet Block
        num_classes (int, optional): number of classes
        features (int, optional): number of features in net
        resnet_factory (func, optional): A Resnet model from torchvision.
            Default: models.resnet101
        pretrained (bool, optional): Use pretrained version of resnet
            Default: True
        freeze_resnet (bool, optional): Freeze resnet model
            Default: True
    Raises:
        ValueError: size of input_shape not divisible by 32
    """
    def __init__(self,
                 input_shape,
                 num_classes=10,
                 features=256,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):

        super().__init__(
            input_shape,
            refinenet_block=RefineNetBlock,
            num_classes=num_classes,
            features=features,
            resnet_factory=resnet_factory,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet)


class RefineNet4CascadePoolingImproved(BaseRefineNet4Cascade):
    """Multi-path 4-Cascaded RefineNet for image segmentation with improved pooling
    Args:
        input_shape ((int, int, int)): (channel, h, w) assumes input has
            equal height and width
        refinenet_block (block): RefineNet Block
        num_classes (int, optional): number of classes
        features (int, optional): number of features in net
        resnet_factory (func, optional): A Resnet model from torchvision.
            Default: models.resnet101
        pretrained (bool, optional): Use pretrained version of resnet
            Default: True
        freeze_resnet (bool, optional): Freeze resnet model
            Default: True
    Raises:
        ValueError: size of input_shape not divisible by 32
    """
    def __init__(self,
                 input_shape,
                 num_classes=10,
                 features=256,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):

        super().__init__(
            input_shape,
            refinenet_block=RefineNetBlockImprovedPooling,
            num_classes=num_classes,
            features=features,
            resnet_factory=resnet_factory,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet)


if __name__ == '__main__':
    import torch
    import time
    x = torch.rand((1, 3, 256, 256))
    model = BaseRefineNet4Cascade((3, 256, 256))
    t1 = time.time()
    y = model(x)
    print(time.time()-t1)
    x = x.cuda()
    model.cuda()
    t2 = time.time()
    z = model(x)
    print(time.time()-t2)
