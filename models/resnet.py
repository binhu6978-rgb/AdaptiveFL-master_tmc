import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, track_running_stats=True):
        super(ResidualBlock, self).__init__()
        self.left1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True)
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel, track_running_stats=track_running_stats)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left1(x)
        out = self.left2(out)
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_channels=3, num_classes=10,
                 track_running_stats=True, slim_idx=0, scale=1.0, dataset='cifar'):
        super(ResNet, self).__init__()

        self.dataset = dataset

        if self.dataset == 'widar':
            self.reshape = nn.Sequential(
                nn.ConvTranspose2d(22, num_channels, 7, stride=1),
                nn.ReLU(),
                nn.ConvTranspose2d(num_channels, num_channels, kernel_size=7, stride=1),
                nn.ReLU()
            )

        # 把第二份接口的 slim_idx + scale 转成第一份实现里用的 rate
        # 共 5 段：stem + 4 个 residual stages
        rate = []
        for idx in range(5):
            tmp_scale = 1.0 if idx < slim_idx else scale
            rate.append(tmp_scale)

        self.inchannel = int(64 * rate[0])

        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(num_channels, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.inchannel, track_running_stats=track_running_stats),
                nn.ReLU()
            ),

            self._make_layer(ResidualBlock, int(96 * rate[1]), 2, stride=1,
                             track_running_stats=track_running_stats),

            self._make_layer(ResidualBlock, int(128 * rate[2]), 2, stride=2,
                             track_running_stats=track_running_stats),

            self._make_layer(ResidualBlock, int(256 * rate[3]), 2, stride=2,
                             track_running_stats=track_running_stats),

            self._make_layer(ResidualBlock, int(512 * rate[4]), 2, stride=2,
                             track_running_stats=track_running_stats)
        )

        self.classifier = nn.Linear(int(512 * rate[4]), num_classes)

    def _make_layer(self, block, channels, num_blocks, stride, track_running_stats):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride, track_running_stats))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.dataset == 'widar':
            out = self.features(self.reshape(x))
        else:
            out = self.features(x)

        result = {'representation': out}
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        result['output'] = out
        return result


def ResNet18_cifar(num_channels=3, num_classes=10, track_running_stats=True, slim_idx=0, scale=1.0):
    return ResNet(ResidualBlock, num_channels, num_classes, track_running_stats, slim_idx, scale, 'cifar')


def ResNet18_widar(num_channels=3, num_classes=22, track_running_stats=True, slim_idx=0, scale=1.0):
    return ResNet(ResidualBlock, num_channels, num_classes, track_running_stats, slim_idx, scale, 'widar')


if __name__ == '__main__':
    net_1 = ResNet18_cifar(num_classes=10, track_running_stats=True, slim_idx=2, scale=0.45)
    print(net_1)