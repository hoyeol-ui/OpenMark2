# watermark/models/encoder.py
from __future__ import annotations
import torch
import torch.nn as nn
from ..configs import LOCAL_INVIS_NUM_BITS, LOCAL_INVIS_IMAGE_SHAPE

class Encoder(nn.Module):
    def __init__(self, config=None, seed: int = 1234, epsilon: float = 0.01):
        super().__init__()
        self.nbits = int(LOCAL_INVIS_NUM_BITS)
        H, W = LOCAL_INVIS_IMAGE_SHAPE
        C = 3
        self.epsilon = float(epsilon)

        # 코드북 생성
        g = torch.Generator(device="cpu").manual_seed(seed)
        codebook = torch.randn(self.nbits, C, H, W, generator=g)
        codebook = codebook - codebook.mean(dim=(1, 2, 3), keepdim=True)
        codebook = codebook / (codebook.flatten(1).norm(dim=1, keepdim=True).view(-1, 1, 1, 1) + 1e-8)
        self.register_buffer("codebook", codebook, persistent=False)

    def forward(self, x: torch.Tensor, payload_bits: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W), payload_bits: (B,N) in {0,1}
        s = (payload_bits * 2.0 - 1.0).to(x.dtype)                  # {-1, +1}
        residual = torch.einsum("bn,nchw->bchw", s, self.codebook)  # (B,3,H,W)
        residual = residual / (self.nbits ** 0.5)                   # ★ 에너지 정규화
        y = torch.clamp(x + self.epsilon * residual, 0.0, 1.0)
        print("[DBG][enc] eps=", float(self.epsilon), "res_norm=",
               float(residual.abs().mean().item()))
        return y