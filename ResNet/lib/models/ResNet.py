# Document function description
#   ResNet and ResNeXt network
import torch
import torch.nn as nn

BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    """ResNet BasicBlock"""
    expansion = 1  # input_channels == output_channels

    def __init__(self, in_channel, out_channel, stride=(1, 1), padding=(1, 1), downsample=None, **kwargs) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3),
                                stride=stride, padding=(1, 1), bias=False)  # bias is useless because of BN layer
        self.bn1 = nn.BatchNorm2d(out_channel, momentum=BN_MOMENTUM)
        
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=(3, 3),
                                stride=(1, 1), padding=(1, 1), bias=False)  # stride is fixed (1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel, momentum=BN_MOMENTUM)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # different block connected by dotted line

    def forward(self, x):
        identity = x
    
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
    
        if self.downsample:
            identity = self.downsample(x)
            
        output = self.relu(output + identity)
        
        return output
    
    
class Bottleneck(nn.Module):
    """Bottlenck in ResNet and ResNeXt"""
    expansion = 4  # channels of the third layer is fourfold the channels of others in block
    def __init__(self, in_channel, out_channel, stride=(1, 1), downsample=None,
                    groups=1, width_per_group=64) -> None:
        super(Bottleneck, self).__init__()
        width = int(out_channel * (width_per_group / 64.)) * groups  # ResNeXt Group Conv, Double channels
        
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=(1, 1), 
                                stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(width, momentum=BN_MOMENTUM)
        
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                                kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(width, momentum=BN_MOMENTUM)
        
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion, kernel_size=(1, 1),
                                stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion, momentum=BN_MOMENTUM)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.conv3(output)
        output = self.bn3(output)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        output = self.relu(output + identity)
        
        return output


class ResNet(nn.Module):
    """ResNet or ResNeXt"""
    def __init__(self, block, block_num, num_classes=1000, include_top=True, groups=1, width_per_group=64) -> None:
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        
        self.groups = groups
        self.width_per_group = width_per_group 
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=(7, 7),
                                stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        
        self.layer1 = self._make_layer(block, 64, block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride=2)
        
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self._weights_init()
    
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel * block.expansion, kernel_size=(1, 1),
                            stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride, 
                            groups=self.groups, width_per_group=self.width_per_group))
        
        self.in_channel = channel * block.expansion
        
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))

        return nn.Sequential(*layers)            
        
    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.maxpool(output)

        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        if self.include_top:
            output = self.avgpool(output)
            output = torch.flatten(output, 1)
            output = self.fc(output)

        return output
    

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                    num_classes=num_classes,
                    include_top=include_top,
                    groups=groups,
                    width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                    num_classes=num_classes,
                    include_top=include_top,
                    groups=groups,
                    width_per_group=width_per_group)
    
    
if __name__ == '__main__':
    # module = resnet34()
    # module = resnet50()
    module = resnext50_32x4d()
    print(module)