# watermark/models/common.py
"""
InvisMark 소스를 로컬에서 테스트 하기 위해 사용한 파일,
테스트 목적의 파일이라 현재 사용되지 않음.

포함:
- Conv2d: Conv + (선택)BN + (선택)활성화
- DecBlock: 업샘플 + 스킵연결(decoder용 블록)
- Watermark2Image: 워터마크 벡터 → 간단한 이미지 feature map 투사(프로토타입)

주의:
- 이미지 I/O, 텐서 변환, 문자열/비트 변환 유틸은 openmark_app/io_utils.py 및
  watermark/embedder.py로 이전되어 여기서는 제공하지 않습니다(중복 제거).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# ---- basic conv block ----
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s=1, p=0, bias=True, activ='relu', norm=None):
        super().__init__()
        self.conv  = nn.Conv2d(in_channels, out_channels, k, s, p, bias=bias)
        self.norm  = nn.BatchNorm2d(out_channels) if norm == 'bn' else None
        self.activ = {
            'relu': nn.ReLU(inplace=True),
            'silu': nn.SiLU(inplace=True),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            None: None, 'None': None,
        }[activ]

    def forward(self, x):
        x = self.conv(x)
        if self.norm:  x = self.norm(x)
        if self.activ: x = self.activ(x)
        return x

# ---- simple decoder block (upsample + skip) ----
class DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, activ='relu', norm=None):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode="nearest")
        self.pad   = nn.ZeroPad2d((0,1,0,1))
        self.conv1 = Conv2d(in_ch, out_ch, 2, 1, 0, activ=activ, norm=norm)
        self.conv2 = Conv2d(out_ch + skip_ch, out_ch, 3, 1, 1, activ=activ, norm=norm)

    def forward(self, x, skip):
        x = self.conv1(self.pad(self.up(x)))
        x = torch.cat([x, skip], dim=1)
        return self.conv2(x)

# ---- experimental: watermark vector -> image-like feature ----
class _ImageView(nn.Module):
    def __init__(self, hidden_dim=16, ch=3):
        super().__init__()
        self.h, self.ch = hidden_dim, ch
    def forward(self, x):
        return x.view(-1, self.ch, self.h, self.h)

class _Repeat(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = r
    def forward(self, x):
        return x.repeat(1, 1, self.r, self.r)

class Watermark2Image(nn.Module):
    """
    워터마크 벡터(예: 길이 L) -> 간단한 이미지 feature map 프로젝션.
    순수 실험/프로토타입 용도.
    """
    def __init__(self, watermark_len: int, resolution=256, hidden_dim=16, num_repeats=2, ch=3):
        super().__init__()
        assert resolution % hidden_dim == 0, "resolution must be divisible by hidden_dim"
        pad = resolution // 4
        self.proj = nn.Sequential(
            nn.Linear(watermark_len, hidden_dim * hidden_dim * ch),
            _ImageView(hidden_dim, ch),
            nn.Upsample(scale_factor=(resolution // hidden_dim // num_repeats // 2),
                        mode="nearest"),
            _Repeat(num_repeats),
            transforms.Pad(pad),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)