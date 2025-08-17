# watermark/models/decoder.py
from __future__ import annotations
import torch
import torch.nn as nn
from ..configs import LOCAL_INVIS_NUM_BITS, LOCAL_INVIS_IMAGE_SHAPE

def _make_codebook(nbits: int, H: int, W: int, C: int = 3, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    cb = torch.randn(nbits, C, H, W, generator=g)
    cb = cb - cb.mean(dim=(1, 2, 3), keepdim=True)
    cb = cb / (cb.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8)
    return cb

class Decoder(nn.Module):
    """
    코드북 상관 기반 무학습 디코더:
      probs = σ(τ * <HPF(x-0.5), P_i> / (H*W)) ∈ (0,1), 임계 0.5로 이진화
    """
    def __init__(self, config=None, seed: int = 1234, temperature: float = 35.0):
        super().__init__()
        self.nbits = int(LOCAL_INVIS_NUM_BITS)
        H, W = LOCAL_INVIS_IMAGE_SHAPE
        self.temperature = float(temperature)
        self.register_buffer("codebook", _make_codebook(self.nbits, H, W), persistent=False)
        k = torch.tensor([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=torch.float32) / 8.0
        self.register_buffer("hpf", k.view(1,1,3,3).repeat(3,1,1,1), persistent=False)  # depthwise용

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # [0,1] -> 중심화 후 고주파 강조
        x0 = x - 0.5
        return torch.nn.functional.conv2d(x0, self.hpf, padding=1, groups=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) in [0,1]  ->  probs: (B,Nbits) in (0,1)
        B, _, H, W = x.shape
        xh = self._preprocess(x)
        corr = torch.einsum("bchw,nchw->bn", xh, self.codebook) / (H * W)  # 평균 상관
        probs = torch.sigmoid(self.temperature * corr)
        return probs