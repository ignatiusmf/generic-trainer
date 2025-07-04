import torchvision.transforms as transforms
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_simple(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10,model_type="unset"):
        super(ResNet_simple, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # 1/2 width & height
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(64*block.expansion, num_classes)

        self.model_type = model_type

    def __repr__(self):
        return self.model_type

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

        # 0 - layer1 ... 3 - output
    def forward(self, x):
        attention_map=[]
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        attention_map.append(out1)
        out2 = self.layer2(out1)
        attention_map.append(out2)
        out3 = self.layer3(out2)
        attention_map.append(out3)

        out = self.gap(out3)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        attention_map.append(out)
        return attention_map

def ResNet112(num_classes=10,model_type='ResNet112'):
    model = ResNet_simple(BasicBlock, [18,18,18],num_classes=num_classes,model_type=model_type)
    # print(f"Total parameters for {model_type}: {sum(p.numel() for p in model.parameters()):,}")
    return model

def ResNet56(num_classes=10,model_type='ResNet56'):
    model = ResNet_simple(BasicBlock, [9,9,9],num_classes=num_classes,model_type=model_type)
    # print(f"Total parameters for {model_type}: {sum(p.numel() for p in model.parameters()):,}")
    return model

def ResNet20(num_classes=10,model_type='ResNet20'):
    model = ResNet_simple(BasicBlock, [3,3,3],num_classes=num_classes,model_type=model_type)
    # print(f"Total parameters for {model_type}: {sum(p.numel() for p in model.parameters()):,}")
    return model

def ResNetBaby(num_classes=10,model_type='ResNetBaby'):
    model = ResNet_simple(BasicBlock, [1,1,1],num_classes=num_classes,model_type=model_type)
    # print(f"Total parameters for {model_type}: {sum(p.numel() for p in model.parameters()):,}")
    return model