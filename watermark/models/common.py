# watermark/models/common.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activ='relu', norm=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.activ = {
            'relu': nn.ReLU(inplace=True),
            'silu': nn.SiLU(inplace=True),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            None: None,
            'None': None,
        }[activ]

        self.norm = nn.BatchNorm2d(out_channels) if norm == 'bn' else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.activ: x = self.activ(x)
        return x


class DecBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, activ='relu', norm=None):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2))
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv1 = Conv2d(in_channels, out_channels, 2, 1, 0, activ=activ, norm=norm)
        self.conv2 = Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1, activ=activ, norm=norm)

    def forward(self, x, skip):
        x = self.conv1(self.pad(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)
        return x


# -------------------- 추가된 부분 --------------------

class ImageViewLayer(nn.Module):
    def __init__(self, hidden_dim=16, channel=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.channel = channel

    def forward(self, x):
        return x.view(-1, self.channel, self.hidden_dim, self.hidden_dim)


class ImageRepeatLayer(nn.Module):
    def __init__(self, num_repeats):
        super().__init__()
        self.num_repeats = num_repeats

    def forward(self, x):
        return x.repeat(1, 1, self.num_repeats, self.num_repeats)


class Watermark2Image(nn.Module):
    def __init__(self, watermark_len, resolution=256, hidden_dim=16, num_repeats=2, channel=3):
        super().__init__()
        assert resolution % hidden_dim == 0, "Resolution should be divisible by hidden_dim"
        pad_length = resolution // 4  # 유지

        self.transform = nn.Sequential(
            nn.Linear(watermark_len, hidden_dim * hidden_dim * channel),  # [768, 100]
            ImageViewLayer(hidden_dim, channel),
            nn.Upsample(scale_factor=(resolution // hidden_dim // num_repeats // 2, resolution // hidden_dim // num_repeats // 2)),
            ImageRepeatLayer(num_repeats),
            transforms.Pad(pad_length),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.transform(x)

# ---------------------------------------------------