import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.autograd import Function
from collections import OrderedDict
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

zsize = 48
batch_size = 11
iterations = 500
learningRate = 0.0001

import torchvision.models as models


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
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
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
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, return_hidden=False,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.return_hidden = return_hidden

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        self.replace_stride_with_dilation = replace_stride_with_dilation
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
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
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

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def modify(self, block, n_classes):
        #self.layer5 = self._make_layer(block, 512, layers[0], stride=1, dilate=self.replace_stride_with_dilation[4])
        #self.layer6 = self._make_layer(block, 512, layers[1], stride=1, dilate=self.replace_stride_with_dilation[5])
        #self.fc = nn.Linear(6*6*2048, 1000)
        self.fc = nn.Linear(512 * block.expansion, 512 * block.expansion)
        self.classifier = nn.Linear(512 * block.expansion, n_classes)
        self.conv0 = nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn0 = self._norm_layer(3)

    def add_classifier(self, block, old_class_num, new_class_num):
        if new_class_num > old_class_num:
            weight = torch.zeros((new_class_num, 512 * block.expansion), requires_grad=True) + 0.01
            print(weight.size())
            weight[:old_class_num, :] = self.classifier.weight
            bias = torch.zeros((new_class_num), requires_grad=True) + 0.1
            bias[:old_class_num] = self.classifier.bias

            self.classifier = nn.Linear(512 * block.expansion, new_class_num)
            self.classifier.weight = torch.nn.Parameter(weight)
            self.classifier.bias = torch.nn.Parameter(bias)
        else:
            self.classifier.weight.requires_grad = True
            self.classifier.bias.requires_grad = True

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        #x = self.maxpool(x)
        x = self.layer2(x)
        #x = self.maxpool(x)
        x = self.layer3(x)
        #x = self.maxpool(x)
        x = self.layer4(x)
        #x = self.maxpool(x)
        #x = self.layer5(x)
        #x = self.maxpool(x)



        x = self.avgpool(x)
        #print(x.size())
        x = torch.flatten(x, 1)
        #print(x.size())
        #x_flat = x.view(-1, 6 * 6 * 2048)
        feature = self.relu(self.fc(x))

        if self.return_hidden:
            return feature, self.classifier(feature)
        else:
            return self.classifier(feature)

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model



"""

class Encoder(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # , return_indices = True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.avgpool = nn.AvgPool2d(6, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1000)
        # self.fc = nn.Linear(num_classes,16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def modify(self, block, layers, n_classes):
        self.layer5 = self._make_layer(block, 512, layers[0], stride=1)
        self.layer6 = self._make_layer(block, 512, layers[1], stride=1)
        self.fc = nn.Linear(4*4*2048, 1024)
        self.classifier = nn.Linear(1024, n_classes)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                  bias=False)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        #x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = self.layer5(x)
        x = self.maxpool(x)
        x = self.layer6(x)
        x = self.maxpool(x)

        #x = self.avgpool(x)

        #print(x.size())

        x_flat = x.view(-1, 4 * 4 * 2048)

        #x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x_flat))
        #print("feature dimension:", x.size())
        return self.classifier(x)
"""

##########################################
class Classification(nn.Module):
    def __init__(self, num_class):
        super(Classification, self).__init__()
        encoder = Encoder(Bottleneck, [3, 4, 6, 3])

        encoder.load_state_dict(torch.load(
            './pretrained/resnet50-19c8e357.pth'))  # ,map_location=lambda storage, loc: storage.cuda(1)),strict=False)

        encoder.modify(Bottleneck, [3, 3], num_class)
        self.encoder = encoder.cuda()

    def forward(self, x):
        return self.encoder(x)


class Resnet18(nn.Module):
    def __init__(self, num_class):
        super(Resnet18, self).__init__()
        #cnn = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], True, True)
        cnn = ResNet(BasicBlock, [2, 2, 2, 2])

        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet18'], progress=True)
        cnn.load_state_dict(state_dict)

        #cnn.load_state_dict(torch.load( './pretrained/resnet18-5c106cde.pth'))  ## https://download.pytorch.org/models/resnet18-5c106cde.pth

        cnn.modify(BasicBlock,  num_class)
        self.cnn = cnn.cuda()

    def add_classes(self, old_class_num, new_class_num):
        self.cnn.add_classifier(Bottleneck, old_class_num, new_class_num)

    def forward(self, x):
        return self.cnn(x)

class Resnet50(nn.Module):
    def __init__(self, num_class, return_hidden=False):
        super(Resnet50, self).__init__()
        #cnn = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], True, True)
        cnn = ResNet(Bottleneck, [3, 4, 6, 3], return_hidden=return_hidden)

        state_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50'], progress=True)
        cnn.load_state_dict(state_dict)

        #cnn.load_state_dict(torch.load( './pretrained/resnet18-5c106cde.pth'))  ## https://download.pytorch.org/models/resnet18-5c106cde.pth

        cnn.modify(Bottleneck, num_class)
        self.cnn = cnn.cuda()

    def add_classes(self, old_class_num, new_class_num):
        self.cnn.add_classifier(Bottleneck, old_class_num, new_class_num)

    def forward(self, x):
        return self.cnn(x)

######## self defined CNN ###########

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class CNNclassifier(nn.Module):
    def initial_conv(self, in_channels, out_channels):
        return nn.Sequential(nn.Conv2d(in_channels, int(out_channels / 2), 3, padding=1),
                             nn.BatchNorm2d(int(out_channels / 2)),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(int(out_channels / 2), out_channels, 3, padding=1),
                             nn.BatchNorm2d(out_channels),
                             nn.ReLU(inplace=True))


    def __init__(self, num_channels=8, num_latents=4096, num_class=10):
        super(CNNclassifier, self).__init__()

        print('num_channel', num_channels)
        print('num_latent', num_latents)
        self.num_channels = num_channels
        self.conv_initial = self.initial_conv(1, num_channels)
        self.conv_final = nn.Conv2d(num_channels, 1, 3, padding=1)

        self.conv_rest_x_384 = self._make_layer(ResidualBlock, num_channels, num_channels * 2)
        self.conv_rest_x_192 = self._make_layer(ResidualBlock, num_channels * 2, num_channels * 4)
        self.conv_rest_x_96 = self._make_layer(ResidualBlock, num_channels * 4, num_channels * 8)
        self.conv_rest_x_48 = self._make_layer(ResidualBlock, num_channels * 8, num_channels * 16)
        self.conv_rest_x_24 = self._make_layer(ResidualBlock, num_channels * 16, num_channels * 32)
        self.conv_rest_x_12 = self._make_layer(ResidualBlock, num_channels * 32, num_channels * 64)
        self.conv_rest_x_6 = self._make_layer(ResidualBlock, num_channels * 64, num_channels * 128)
        self.conv_rest_x_3 = self._make_layer(ResidualBlock, num_channels * 128, num_channels * 256)

        self.contract = nn.MaxPool2d(2, stride=2)
        self.linear_enc = nn.Linear(3 * 3 * num_channels * 256, num_latents)
        self.classifier = nn.Linear(num_latents, num_class)

        self.num_latents = num_latents
        self.num_classes = num_class

    def _make_layer(self, block, inplanes, planes, blocks=1, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_768 = self.conv_initial(x)  # conv_initial 1->64->128
        x_384 = self.contract(x_768)
        #print(x_384.size())
        x_384 = self.conv_rest_x_384(x_384)  # rest 128->128->256
        x_192 = self.contract(x_384)
        #print(x_192.size())
        x_192 = self.conv_rest_x_192(x_192)  # rest 256->256->512
        x_96 = self.contract(x_192)
        #print(x_96.size())
        x_96 = self.conv_rest_x_96(x_96)  # rest 512->512->256
        x_48 = self.contract(x_96)
        #print(x_48.size())
        x_48 = self.conv_rest_x_48(x_48)
        x_24 = self.contract(x_48)
        #print(x_24.size())
        x_24 = self.conv_rest_x_24(x_24)
        x_12 = self.contract(x_24)
        #print(x_12.size())
        x_12 = self.conv_rest_x_12(x_12)
        x_6 = self.contract(x_12)
        x_6 = self.conv_rest_x_6(x_6)
        x_3 = self.contract(x_6)
        x_3 = self.conv_rest_x_3(x_3)
        x_flat = x_3.view(-1,
                           3 * 3 * self.num_channels * 256)  # dimesion becomes 1x... View is used to optimize, since the tensor is not copied but just seen differently
        pre = self.linear_enc(x_flat)
        #std = 1.e-6 + nn.functional.softplus(self.linear_enc(x_flat))
        return pre #, std







