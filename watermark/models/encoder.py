# watermark/models/encoder.py
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

class Encoder(nn.Module):
    """
    입력 x(0~1)에 비트열 payload_bits({0,1})를 코드북 합성으로 삽입:
      y = clip(x + eps * sum_i s_i * P_i / sqrt(N), 0,1), s_i∈{-1,+1}
    """
    def __init__(self, config=None, seed: int = 1234, epsilon: float = 0.01):
        super().__init__()
        self.nbits = int(LOCAL_INVIS_NUM_BITS)
        H, W = LOCAL_INVIS_IMAGE_SHAPE
        self.epsilon = float(epsilon)
        self.register_buffer("codebook", _make_codebook(self.nbits, H, W), persistent=False)

    def forward(self, x: torch.Tensor, payload_bits: torch.Tensor, debug: bool = False) -> torch.Tensor:
        # x: (B,3,H,W), payload_bits: (B,N) in {0,1}
        s = payload_bits.to(x.dtype).mul(2.0).sub(1.0)                   # {0,1} -> {-1,+1}
        residual = torch.einsum("bn,nchw->bchw", s, self.codebook)       # (B,3,H,W)
        residual = residual / (self.nbits ** 0.5)                        # 에너지 정규화
        y = (x + self.epsilon * residual).clamp_(0.0, 1.0)
        if debug:
            with torch.no_grad():
                print(f"[enc] eps={self.epsilon:.4f} | res|mean={residual.abs().mean().item():.6f}")
        return y