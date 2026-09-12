import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torchinfo import summary

def make_divisible(v, divisor=8, min_value=None):
    # MobileNet 系常用的通道对齐，避免太奇怪的通道数
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, groups=1, track_running_stats=True):
        padding = (kernel - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio, track_running_stats=True):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(in_ch, hidden_dim, kernel=1, stride=1,
                                     track_running_stats=track_running_stats))
        # dw
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, kernel=3, stride=stride,
                                 groups=hidden_dim, track_running_stats=track_running_stats))
        # pw-linear
        layers.append(nn.Conv2d(hidden_dim, out_ch, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_ch, track_running_stats=track_running_stats))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class MobileNetV2(nn.Module):
    """
    CIFAR-friendly MobileNetV2:
    - first conv stride=1 (CIFAR 32x32 不用一上来就 downsample)
    - width scaling via `scale`
    - stage-wise "freeze full width before slim_idx"
    """
    def __init__(self, channels=3, num_classes=10,
                 trs=True, slim_idx=0, scale=1.0, dataset='cifar',
                 round_nearest=8):
        super().__init__()
        self.dataset = dataset

        # MobileNetV2 cfg: t(expand), c(out), n(repeat), s(stride)
        # 这是经典配置，针对 CIFAR 我们通常保留结构，只把首层 stride=1
        inverted_residual_setting = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 1],  # CIFAR 通常把这里 stride 从 2 改 1，减少过早下采样
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # 索引：我们把每个 stage（含首层 conv 和最后 1x1）算作一个“缩放单元”
        # idx < slim_idx 的部分用 full width (scale=1.0)
        idx = 0

        # first conv
        first_out = 32
        out_ch = make_divisible(first_out * (1.0 if idx < slim_idx else scale), round_nearest)
        self.conv1 = ConvBNReLU(channels, out_ch, kernel=3, stride=1,
                                track_running_stats=trs)
        in_ch = out_ch
        idx += 1

        # stages
        features = []
        for (t, c, n, s) in inverted_residual_setting:
            tmp_scale = 1.0 if idx < slim_idx else scale
            out_ch = make_divisible(c * tmp_scale, round_nearest)

            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(
                    in_ch, out_ch, stride=stride, expand_ratio=t,
                    track_running_stats=trs
                ))
                in_ch = out_ch
            idx += 1

        self.features = nn.Sequential(*features)

        # last 1x1 conv
        tmp_scale = 1.0 if idx < slim_idx else scale
        last_ch = make_divisible(1280 * tmp_scale, round_nearest) if tmp_scale != 1.0 else 1280
        self.conv_last = ConvBNReLU(in_ch, last_ch, kernel=1, stride=1,
                                    track_running_stats=trs)
        idx += 1

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_ch, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = self.conv_last(out)

        result = {"representation": out}

        out = self.pool(out)
        out = torch.flatten(out, 1)
        result["features"] = out

        logits = self.classifier(out)
        result["output"] = logits
        return result

def MobileNetV2_cifar(num_channels=3, num_classes=10, track_running_stats=True, slim_idx=0, scale=1.0):
    return MobileNetV2(num_channels=num_channels, num_classes=num_classes,
                       track_running_stats=track_running_stats,
                       slim_idx=slim_idx, scale=scale, dataset='cifar')



if __name__ == "__main__":
    import torch
    import numpy as np
    
    def count_params(model: torch.nn.Module):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def count_buffers(model: torch.nn.Module):
        # BN running_mean/var 之类也算“存储开销”，但不算参数
        total = sum(b.numel() for b in model.buffers())
        return total

    def pretty(n):
        if n >= 1e6:
            return f"{n/1e6:.3f}M"
        if n >= 1e3:
            return f"{n/1e3:.3f}K"
        return str(n)

    widths = [0.5, 0.75, 1.0]  # 你也可以改成 [0.5, 0.71, 1.0]
    slim_idx = 0
    num_classes = 10

    print(f"Testing widths={widths}, slim_idx={slim_idx}, num_classes={num_classes}\n")

    for w in widths:
        net = MobileNetV2(3, num_classes=num_classes, scale=w)
        total, trainable = count_params(net)
        buffers = count_buffers(net)

        # 估一个“模型存储大小”（参数 + buffer），float32 4 bytes
        storage_bytes = (total + buffers) * 4
        storage_mb = storage_bytes / (1024**2)

        print(
            f"scale={w:>4} | params={pretty(total)} (trainable={pretty(trainable)}) "
            f"| buffers={pretty(buffers)} | approx_storage={storage_mb:.2f} MB"
        )

