"""
This file contains the code for encoder and up-sample blocks of the saliency models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from imagenet_models.imgnet_utils import load_state_dict_from_url
from imagenet_models.gate_function import virtual_gate
import math

# ################################ ResNet with Gate architectures for Encoder ################################
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cfg=None, num_gate=0):
        super(BasicBlock, self).__init__()
        if cfg is None:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            if num_gate == 1:
                self.gate = virtual_gate(planes)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride
            self.num_gate = num_gate
        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv3x3(inplanes, cfg[0], stride)
            self.bn1 = norm_layer(cfg[0])
            self.relu = nn.ReLU(inplace=True)
            if num_gate == 1:
                self.gate = virtual_gate(cfg[0])
            self.conv2 = conv3x3(cfg[0], planes)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride
            self.num_gate = num_gate

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.num_gate == 1:
            out = self.gate(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cfg=None, num_gate=2):
        super(Bottleneck, self).__init__()
        if cfg is None:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            if num_gate > 1:
                self.gate1 = virtual_gate(width)

            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)

            if num_gate >= 1:
                self.gate2 = virtual_gate(width)

            # self.gate2 = self.gate1

            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride
        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            # width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv1x1(inplanes, cfg[0])
            self.bn1 = norm_layer(cfg[0])
            # self.gate1 = soft_gate(cfg[0])
            if num_gate > 1:
                self.gate1 = virtual_gate(cfg[0])
            else:
                self.gate1 = None
            self.conv2 = conv3x3(cfg[0], cfg[1], stride, groups, dilation)
            self.bn2 = norm_layer(cfg[1])
            if num_gate >= 1:
                self.gate2 = virtual_gate(cfg[1])

            # self.gate2 = self.gate1

            self.conv3 = conv1x1(cfg[1], planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)

            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.gate1(out)
        if self.gate1 is not None:
            out = self.gate1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.gate2 is not None:
            out = self.gate2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class EncoderResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cfg=None, num_gate=2):
        super(EncoderResNet, self).__init__()

        if block is Bottleneck:
            # print('Bottleneck')
            self.factor = 2
        elif block is BasicBlock:
            # print('BasicBlock')
            self.factor = 1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_gate = num_gate

        # print(self.factor)
        if cfg == None:

            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
        else:

            start = 0
            end = int(self.factor * layers[0])
            self.layer1 = self._make_layer(block, 64, layers[0], cfg=cfg[start:end])
            start = end
            end = end + int(self.factor * layers[1])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0], cfg=cfg[start:end])
            start = end
            end = end + int(self.factor * layers[2])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1], cfg=cfg[start:end])
            start = end
            end = end + int(self.factor * layers[3])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2], cfg=cfg[start:end])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cfg=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        if cfg is None:
            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer, num_gate=self.num_gate))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, num_gate=self.num_gate))
        else:

            index = 0
            layers = []

            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer,
                                cfg=cfg[int(self.factor * index):int(self.factor * index + self.factor)],
                                num_gate=self.num_gate))
            index += 1
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer,
                                    cfg=cfg[int(self.factor * index):int(self.factor * index + self.factor)],
                                    num_gate=self.num_gate))
                index += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        scale1 = self.layer1(out)
        scale2 = self.layer2(scale1)
        scale3 = self.layer3(scale2)
        scale4 = self.layer4(scale3)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return scale1, scale2, scale3, scale4

    def count_structure(self):
        structure = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                structure.append(m.width)
        self.structure = structure
        print(structure)
        return sum(structure), structure

    def set_vritual_gate(self, arch_vector):
        i = 0
        start = 0
        for m in self.modules():
            if isinstance(m, virtual_gate):
                end = start + self.structure[i]
                m.set_structure_value(arch_vector.squeeze()[start:end])
                start = end

                i += 1

    def get_gate_grads(self):
        all_grad = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                # print(m.weights.grad.data)
                all_grad.append(m.get_grads().clone())
        # print(all_grad[0])
        return all_grad


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = EncoderResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def my_resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def my_resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def my_resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


# ############ Added by Alireza
encoder_resnet_dict = {'resnet-34': my_resnet34, 'resnet-50': my_resnet50, 'resnet-101': my_resnet101}

# ################################ Up-sample Blocks Architectures ################################
class PixelShuffleBlock(nn.Module):
    def forward(self, x):
        return F.pixel_shuffle(x, 2)


def CNNBlock(in_channels, out_channels,
             kernel_size=3, layers=1, stride=1,
             follow_with_bn=True, activation_fn=lambda: nn.ReLU(True), affine=True):
    assert layers > 0 and kernel_size % 2 and stride > 0
    current_channels = in_channels
    _modules = []
    for layer in range(layers):
        _modules.append(nn.Conv2d(current_channels, out_channels, kernel_size, stride=stride if layer == 0 else 1,
                                  padding=int(kernel_size / 2), bias=not follow_with_bn))
        current_channels = out_channels
        if follow_with_bn:
            _modules.append(nn.BatchNorm2d(current_channels, affine=affine))
        if activation_fn is not None:
            _modules.append(activation_fn())
    return nn.Sequential(*_modules)


def SubpixelUpsampler(in_channels, out_channels, kernel_size=3, activation_fn=lambda: torch.nn.ReLU(inplace=False),
                      follow_with_bn=True):
    _modules = [
        CNNBlock(in_channels, out_channels * 4, kernel_size=kernel_size, follow_with_bn=follow_with_bn),
        PixelShuffleBlock(),
        activation_fn(),
    ]
    return nn.Sequential(*_modules)


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UpSampleBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, passthrough_channels, stride=1):
        super(UpSampleBlock, self).__init__()
        self.upsampler = SubpixelUpsampler(in_channels=in_channels, out_channels=out_channels)
        self.follow_up = Block(out_channels + passthrough_channels, out_channels)

    def forward(self, x, passthrough):
        out = self.upsampler(x)
        out = torch.cat((out, passthrough), 1)
        return self.follow_up(out)


# ################################ MobileNet with Gate architectures for Encoder ################################
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, cfg=None, gate_flag=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        # hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if cfg is None:
            hidden_dim = int(round(inp * expand_ratio))
        else:
            hidden_dim = cfg

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if gate_flag:
                # self.gate = virtual_gate(hidden_dim)
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw
                    virtual_gate(hidden_dim),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:

                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # dw

                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
        self.gate_flag = gate_flag

    def forward(self, x):
        if not self.gate_flag:
            if self.identity:
                return x + self.conv(x)
            else:
                return self.conv(x)
        else:
            if self.identity:
                out = self.conv.__getitem__(0)(x)
                out = self.conv.__getitem__(1)(out)
                out = self.conv.__getitem__(2)(out)

                out = self.conv.__getitem__(3)(out)

                out = self.conv.__getitem__(4)(out)
                out = self.conv.__getitem__(5)(out)
                out = self.conv.__getitem__(6)(out)

                out = self.conv.__getitem__(3)(out)

                out = self.conv.__getitem__(7)(out)
                out = self.conv.__getitem__(8)(out)
                return out + x
            else:
                out = self.conv.__getitem__(0)(x)
                out = self.conv.__getitem__(1)(out)
                out = self.conv.__getitem__(2)(out)

                out = self.conv.__getitem__(3)(out)

                out = self.conv.__getitem__(4)(out)
                out = self.conv.__getitem__(5)(out)
                out = self.conv.__getitem__(6)(out)

                out = self.conv.__getitem__(3)(out)

                out = self.conv.__getitem__(7)(out)
                out = self.conv.__getitem__(8)(out)
                return out


class EncoderMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, custom_cfg=False, gate_flag=False, cfgs=None):
        super(EncoderMobileNetV2, self).__init__()
        # setting of inverted residual blocks

        if custom_cfg:

            self.cfgs = cfgs
        else:
            self.cfgs = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # building first layer
        # input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)

        if custom_cfg:
            input_channel = cfgs[0][-1][0]

        else:
            input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)

        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual

        if custom_cfg is False:
            for t, c, n, s in self.cfgs:
                output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
                for i in range(n):
                    if t == 1:
                        layers.append(
                            block(input_channel, output_channel, s if i == 0 else 1, expand_ratio=t, gate_flag=False))
                    else:
                        layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, gate_flag=gate_flag))
                    input_channel = output_channel
        else:
            for t, c, n, s, p_list in self.cfgs:
                # strides = [stride] + [1]*(num_blocks-1)
                output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
                for i in range(n):
                    stride = s if i == 0 else 1
                    # features.append(block(input_channel, output_channel, stride, expand_ratio=t, cfg = p_list[i], gate_flag=False))
                    # input_channel = output_channel
                    if t == 1:
                        layers.append(
                            block(input_channel, output_channel, stride, expand_ratio=t, gate_flag=False))
                    else:
                        layers.append(block(input_channel, output_channel, stride, expand_ratio=t, cfg=p_list[i],
                                            gate_flag=gate_flag))
                    input_channel = output_channel

        self.features = nn.ModuleList(layers)

        self.scale_transition_layers = [3, 6, 13, 18]

        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)

        self._initialize_weights()

    def forward(self, x):
        scales = []
        for i in range(len(self.features)):
            x = self.features[i](x)
            # logging.warning(f'{i}: \t ' + str(x.shape))
            if i in self.scale_transition_layers:
                scales.append(x)
                # logging.warning('{}: '.format(i) + str(x.shape) + ' appended!')
                # continue
            # logging.warning('{}: '.format(i) + str(x.shape))

        x = self.conv(x)
        # logging.warning(f'{i + 1}: \t ' + str(x.shape))
        scales.append(x)
        # logging.warning('{}: '.format(i + 1) + str(x.shape) + ' appended!')
        return scales[0], scales[1], scales[2], scales[3]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def count_structure(self):
        structure = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                structure.append(m.width)
        self.structure = structure
        return sum(structure), structure

    def set_vritual_gate(self, arch_vector):
        i = 0
        start = 0
        for m in self.modules():
            if isinstance(m, virtual_gate):
                end = start + self.structure[i]
                m.set_structure_value(arch_vector.squeeze()[start:end])
                start = end

                i += 1

    def get_gate_grads(self):
        all_grad = []
        for m in self.modules():
            if isinstance(m, virtual_gate):
                # print(m.weights.grad.data)
                all_grad.append(m.get_grads().clone())
        # print(all_grad[0])
        return all_grad


encoder_mobilenet_dict = {'MobileNetV2': EncoderMobileNetV2}
