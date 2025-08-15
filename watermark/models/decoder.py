# OpenMark / watermark / models / decoder.py
from __future__ import annotations
import torch
import torch.nn as nn
from ..configs import LOCAL_INVIS_NUM_BITS, LOCAL_INVIS_IMAGE_SHAPE

class Decoder(nn.Module):
    """
    코드북 기반 로컬 디코더(무학습):
    - 입력: x (B,3,H,W)
    - 출력: probs (B,Nbits) in (0,1)
      (상관값을 시그모이드로 누르고 0.5 임계로 이진화)
    """
    def __init__(self, config=None, seed: int = 1234, temperature: float = 35.0):
        super().__init__()
        self.nbits = int(LOCAL_INVIS_NUM_BITS)
        H, W = LOCAL_INVIS_IMAGE_SHAPE
        C = 3
        self.temperature = float(temperature)

        # 인코더와 동일한 코드북 재생성(같은 seed)
        g = torch.Generator(device="cpu").manual_seed(seed)
        codebook = torch.randn(self.nbits, C, H, W, generator=g)
        codebook = codebook - codebook.mean(dim=(1,2,3), keepdim=True)
        codebook = codebook / (codebook.flatten(1).norm(dim=1, keepdim=True).view(-1,1,1,1) + 1e-8)

        self.register_buffer("codebook", codebook, persistent=False)

        # 간단한 고주파 강조 필터(선택) — 상관 검출력 약간 향상
        kernel = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32) / 8.0
        self.register_buffer("hpf", kernel.view(1,1,3,3).repeat(3,1,1,1), persistent=False)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # [0,1] → 중심화; 고주파 필터로 잔차 강조
        x0 = x - 0.5
        # depthwise conv: 각 채널에 동일 커널
        return torch.nn.functional.conv2d(x0, self.hpf, padding=1, groups=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        반환: (B,Nbits) in (0,1)
        """
        B, _, H, W = x.shape
        xh = self._preprocess(x)                         # (B,3,H,W)

        # 상관값: <xh, P_i> / HW
        # (B,3,H,W) vs (N,3,H,W) → (B,N)
        corr = torch.einsum("bchw,nchw->bn", xh, self.codebook) / (H*W)

        # 확률로 매핑 (스케일 온도)
        probs = torch.sigmoid(self.temperature * corr)
        return probs