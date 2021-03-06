"""
This is originally from https://github.com/megvii-research/DPGN/blob/master/backbone.py
"""

import torch.nn as nn


class ResNet12Block(nn.Module):
    """
    ResNet block
    """

    def __init__(self, inplanes, planes):
        super(ResNet12Block, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = x
        residual = self.conv(residual)
        residual = self.bn(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        return out


class ResNet18Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, downsample=False):

        super(ResNet18Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.conv(residual)
            residual = self.bn(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu3(out)
        if self.downsample:
            out = self.maxpool(out)

        return out


class WRNBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(WRNBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet12(nn.Module):
    """
    ResNet12 backbone
    """

    def __init__(self, emb_size, block=ResNet12Block, cifar_flag=False):
        super(ResNet12, self).__init__()
        cfg = [64, 128, 256, 512]
        # layers = [1, 1, 1, 1]
        iChannels = int(cfg[0])
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.LeakyReLU()
        self.emb_size = emb_size
        self.layer1 = self._make_layer(block, cfg[0], cfg[0])
        self.layer2 = self._make_layer(block, cfg[0], cfg[1])
        self.layer3 = self._make_layer(block, cfg[1], cfg[2])
        self.layer4 = self._make_layer(block, cfg[2], cfg[3])
        self.avgpool = nn.AvgPool2d(7)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        layer_second_in_feat = cfg[2] * 5 * 5 if not cifar_flag else cfg[2] * 2 * 2
        self.layer_second = nn.Sequential(nn.Linear(in_features=layer_second_in_feat,
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))

        self.layer_last = nn.Sequential(nn.Linear(in_features=cfg[3],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes):
        layers = []
        layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 3 -> 64
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 64 -> 64
        x = self.layer1(x)
        # 64 -> 128
        x = self.layer2(x)
        # 128 -> 256
        inter = self.layer3(x)
        # 256 -> 512
        x = self.layer4(inter)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # 512 -> 128
        x = self.layer_last(x)
        inter = self.maxpool(inter)
        # 256 * 5 * 5
        inter = inter.view(inter.size(0), -1)
        # 256 * 5 * 5 -> 128
        inter = self.layer_second(inter)
        out = []
        out.append(x)
        out.append(inter)
        # no FC here
        return out


class ResNet18(nn.Module):
    def __init__(self, emb_size, block=ResNet18Bottleneck):
        super(ResNet18, self).__init__()
        cfg = [64, 128, 256, 512]
        layers = [2, 2, 2, 2]
        self.emb_size = emb_size
        self.inplanes = iChannels = int(cfg[0])
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.layer1 = self._make_layer(block, cfg[0], cfg[0], layers[0])
        self.layer2 = self._make_layer(block, cfg[0], cfg[1], layers[1])
        self.layer3 = self._make_layer(block, cfg[1], cfg[2], layers[2])
        self.layer4 = self._make_layer(block, cfg[2], cfg[3], layers[3])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2)
        self.layer_second = nn.Sequential(nn.Linear(in_features=cfg[2] * 5 * 5,
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))
        self.layer_last = nn.Sequential(nn.Linear(in_features=cfg[3],
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, block_num):
        layers = []
        layers.append(block(inplanes, planes, True))
        for i in range(1, block_num):
            layers.append(block(planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        inter = self.layer3(x)
        x = self.layer4(inter)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layer_last(x)
        out = []
        out.append(x)
        inter = self.maxpool(inter)
        inter = inter.view(inter.size(0), -1)
        inter = self.layer_second(inter)
        out.append(inter)
        # no FC here
        return out


class WRN(nn.Module):

    def __init__(self, emb_size, block=WRNBottleneck, layers=[4, 4, 4]):
        super(WRN, self).__init__()
        cfg = [120, 240, 480]
        self.emb_size = emb_size
        self.inplanes = iChannels = int(cfg[0] / 2)
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, cfg[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, cfg[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2], layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(3)
        self.layer_second = nn.Sequential(nn.Linear(in_features=cfg[1] * block.expansion * 9,
                                                    out_features=self.emb_size,
                                                    bias=True),
                                          nn.BatchNorm1d(self.emb_size))

        self.layer_last = nn.Sequential(nn.Linear(in_features=cfg[2] * block.expansion,
                                                  out_features=self.emb_size,
                                                  bias=True),
                                        nn.BatchNorm1d(self.emb_size))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        inter = self.layer2(x)
        x = self.layer3(inter)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.layer_last(x)
        inter = self.maxpool(inter)
        inter = inter.view(inter.size(0), -1)
        inter = self.layer_second(inter)
        out = []
        out.append(x)
        out.append(inter)
        # no FC here
        return out


class ConvNet(nn.Module):
    """
    Conv4 backbone
    """

    def __init__(self, emb_size, cifar_flag=False):
        super(ConvNet, self).__init__()
        # set size
        self.hidden = 128
        # self.last_hidden = self.hidden * 25 if not cifar_flag else self.hidden
        self.last_hidden = self.hidden * 25
        self.emb_size = emb_size

        # set layers
        # 82*82
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=self.hidden,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # 41*41
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels=self.hidden,
                                              out_channels=int(self.hidden * 1.5),
                                              kernel_size=3,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=int(self.hidden * 1.5)),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
        # 20*20
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels=int(self.hidden * 1.5),
                                              out_channels=self.hidden * 2,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.4))
        # 10*10

        # for local feature
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels=self.hidden * 2,
                                              out_channels=self.hidden * 4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        # 5 * 5
        self.layer_last = nn.Sequential(nn.Linear(in_features=self.last_hidden * 4,
                                                  out_features=self.emb_size, bias=True),
                                        nn.BatchNorm1d(self.emb_size))

        # for global feature
        # 10*10
        self.conv_5 = nn.Sequential(nn.Conv2d(in_channels=self.hidden * 2,
                                              out_channels=self.hidden * 4,
                                              kernel_size=3,
                                              padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden * 4),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Dropout2d(0.5))
        # 5*5
        self.conv_6 = nn.Sequential(nn.Conv2d(in_channels=self.hidden * 4,
                                              out_channels=self.hidden,
                                              kernel_size=1,
                                              bias=False),
                                    nn.BatchNorm2d(num_features=self.hidden),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        self.layer_second_fc = nn.Sequential(nn.Linear(in_features=self.last_hidden,
                                                      out_features=self.emb_size, bias=True),
                                            nn.BatchNorm1d(self.emb_size))

    def forward(self, input_data):
        out = []
        out_1 = self.conv_1(input_data)
        out_2 = self.conv_2(out_1)
        out_3 = self.conv_3(out_2)

        # for local feature
        out_4 = self.conv_4(out_3)
        out.append(self.layer_last(out_4.view(out_4.size(0), -1)))

        # for global feature
        output_data0 = self.conv_5(out_3)
        output_data0 = self.conv_6(output_data0)
        out.append(self.layer_second_fc(output_data0.view(output_data0.size(0), -1)))

        return out
