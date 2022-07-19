"""
This script contains the implementation of U-Net architecture for our selector model.
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet():
    return ResNet(BasicBlock, [2, 2, 2, 2])


class EncoderResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(EncoderResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        scale1 = self.layer1(out)
        scale2 = self.layer2(scale1)
        scale3 = self.layer3(scale2)
        scale4 = self.layer4(scale3)

        return scale1, scale2, scale3, scale4

    def load_checkpoint(self, checkpoint_dir):
        logging.warning('*' * 20)
        logging.warning('Encoder ResNet:')
        logging.warning('loading pretrained checkpoint from: {}'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')
        if 'module' in list(checkpoint['model'].keys())[0]:
            new_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                if 'linear' in k:
                    continue
                name = k.replace('module.', '')  # remove `module.`
                if 'resnet' in name:
                    name = name.replace('resnet.', '')
                new_dict[name] = v
            self.load_state_dict(new_dict)
        else:
            for k in list(checkpoint['model'].keys()):
                if 'linear' in k:
                    del checkpoint['model'][k]
            if 'resnet' in list(checkpoint['model'].keys())[0]:
                new_dict = OrderedDict()
                for k, v in checkpoint['model'].items():
                    name = k.replace('resnet.', '')
                    new_dict[name] = v
            self.load_state_dict(checkpoint['model'])


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


def encoder_resnet():
    return EncoderResNet(BasicBlock, [2, 2, 2, 2])
