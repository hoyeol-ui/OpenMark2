# watermark/embedder.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import os
import cv2
import numpy as np
import torch

from .configs import (
    METHOD, DEVICE,
    LOCAL_INVIS_IMAGE_SHAPE, LOCAL_INVIS_NUM_BITS,
)
from watermark.models.encoder import Encoder
from watermark.models.decoder import Decoder

# ---------------------- small utils ----------------------
def _to01(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.from_numpy(bgr).to(device=device, dtype=torch.float32) / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]

def _from01(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

def _hex_to_bytes(uuid_hex: str, out_len: int) -> bytes:
    try:
        raw = bytes.fromhex(uuid_hex)
    except Exception:
        raw = uuid_hex.encode("utf-8")
    return (raw + b"\x00" * max(0, out_len - len(raw)))[:out_len]

def _bytes_to_bits(b: bytes, n_bits: int) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))  # big-endian와 동일하게 사용
    if len(bits) >= n_bits: return bits[:n_bits]
    return np.concatenate([bits, np.zeros(n_bits - len(bits), dtype=np.uint8)], 0)

def _bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.astype(np.uint8).flatten()
    n_out = (len(bits) + 7) // 8
    # packbits는 big-endian 비트 순서를 쓰므로 MSB-first와 정합됨
    return np.packbits(np.pad(bits, (0, n_out * 8 - len(bits)))).tobytes()

def _delta_bridge(orig_bgr: np.ndarray, wm_small: np.ndarray, base_small: np.ndarray, eps01: float = 2/255) -> np.ndarray:
    H, W = orig_bgr.shape[:2]
    delta = (wm_small.astype(np.int16) - base_small.astype(np.int16))
    delta = cv2.resize(delta, (W, H), interpolation=cv2.INTER_LINEAR)
    delta = cv2.GaussianBlur(delta, (0, 0), 0.5, 0.5, borderType=cv2.BORDER_REFLECT)
    amp = eps01 * 255.0
    out = np.clip(orig_bgr.astype(np.int16) + np.clip(delta, -amp, amp), 0, 255).astype(np.uint8)
    return out

# ---------------------- singleton (encoder/decoder) ----------------------
_LOCAL = {"enc": None, "dec": None, "shape": None, "nbits": None}

def _init_local():
    if _LOCAL["enc"] is not None: return
    try:
        enc = Encoder(epsilon=0.01).to(DEVICE).eval()
    except TypeError:
        enc = Encoder().to(DEVICE).eval()
    dec = Decoder().to(DEVICE).eval()
    _LOCAL.update(enc=enc, dec=dec,
                  shape=tuple(LOCAL_INVIS_IMAGE_SHAPE),
                  nbits=int(LOCAL_INVIS_NUM_BITS))

def _decode_from_bgr(bgr: np.ndarray) -> str:
    """배열에서 즉시 디코딩 → hex 문자열(실제 길이만)."""
    _init_local()
    h, w = _LOCAL["shape"]
    work = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        probs = _LOCAL["dec"](_to01(work, DEVICE))
        if probs.min() < 0 or probs.max() > 1: probs = torch.sigmoid(probs)
        bits = (probs[0].detach().cpu().numpy() >= 0.5).astype(np.uint8)
    n_bits = _LOCAL["nbits"]
    return _bits_to_bytes(bits[:n_bits])[: (n_bits + 7)//8].hex()

# ---------------------- public APIs ----------------------
def embed_watermark(src_path: str, dst_path: str, payload_uuid_hex: str) -> None:
    """
    원본 해상도 유지: 내부 (H,W)=(LOCAL_INVIS_IMAGE_SHAPE)에서 삽입 후 Δ만 원본에 투영.
    """
    if METHOD != "local_invis":
        # 다른 방식(imw 등) 분기 추가 가능
        pass

    _init_local()
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    orig = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if orig is None: raise FileNotFoundError(src_path)

    hs, ws = _LOCAL["shape"]
    base_small = cv2.resize(orig, (ws, hs), interpolation=cv2.INTER_AREA)

    # payload -> bits tensor
    n_bits = _LOCAL["nbits"]
    byte_len = (n_bits + 7) // 8
    bits_np = _bytes_to_bits(_hex_to_bytes(payload_uuid_hex, byte_len), n_bits)
    bits_t  = torch.from_numpy(bits_np).to(device=DEVICE, dtype=torch.float32).unsqueeze(0)  # [1,N]

    # encode
    with torch.no_grad():
        x = _to01(base_small, DEVICE)
        try:
            y = _LOCAL["enc"](x, bits_t)
        except TypeError:
            y = _LOCAL["enc"](x, bits=bits_t)
        except Exception:
            y = _LOCAL["enc"](x)
    wm_small = _from01(y)

    # delta bridge
    out = _delta_bridge(orig, wm_small, base_small, eps01=2/255)

    # verification + gain ladder
    hex_len = byte_len * 2
    target  = _hex_to_bytes(payload_uuid_hex, byte_len).hex()
    if _decode_from_bgr(out)[:hex_len] != target[:hex_len]:
        base16 = orig.astype(np.int16); diff = (out.astype(np.int16) - base16)
        for g in (1.15, 1.3, 1.5, 1.7):
            cand = np.clip(base16 + diff * g, 0, 255).astype(np.uint8)
            if _decode_from_bgr(cand)[:hex_len] == target[:hex_len]:
                out = cand; break

    cv2.imwrite(dst_path, out)

def decode_watermark(src_path: str) -> str:
    """원본 → 내부 해상도로 다운스케일 → 디코더 → hex 문자열(실제 길이만)."""
    if METHOD != "local_invis":
        pass
    _init_local()

    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(src_path)

    hs, ws = _LOCAL["shape"]
    work = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        probs = _LOCAL["dec"](_to01(work, DEVICE))
        if probs.min() < 0 or probs.max() > 1: probs = torch.sigmoid(probs)
        bits = (probs[0].detach().cpu().numpy() >= 0.5).astype(np.uint8)

    n_bits = _LOCAL["nbits"]
    return _bits_to_bytes(bits[:n_bits])[: (n_bits + 7)//8].hex()