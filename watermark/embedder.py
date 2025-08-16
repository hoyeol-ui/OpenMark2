# watermark/embedder.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from .configs import (
    METHOD,
    LOCAL_INVIS_IMAGE_SHAPE,   # (H, W) e.g., (256, 256)
    LOCAL_INVIS_NUM_BITS,      # e.g., 64/128/256
)

from watermark.models.encoder import Encoder
from watermark.models.decoder import Decoder


# ----------------- utils -----------------
def _device_auto() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _bgr_to_tensor01(bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(bgr).to(device=device, dtype=torch.float32) / 255.0  # [H,W,3]
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x

def _tensor01_to_bgr(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

def _uuid_hex_to_bytes(uuid_hex: str, out_len: int) -> bytes:
    try:
        raw = bytes.fromhex(uuid_hex)  # 16 bytes for UUID
    except Exception:
        raw = uuid_hex.encode("utf-8")
    if len(raw) == out_len:
        return raw
    if len(raw) > out_len:
        return raw[:out_len]
    return raw + b"\x00" * (out_len - len(raw))

def _bytes_to_bits(b: bytes, n_bits: int) -> np.ndarray:
    bit_list = []
    for by in b:
        for i in range(8):
            bit_list.append((by >> (7 - i)) & 1)
    arr = np.array(bit_list, dtype=np.uint8)
    if len(arr) >= n_bits:
        return arr[:n_bits]
    pad = np.zeros(n_bits - len(arr), dtype=np.uint8)
    return np.concatenate([arr, pad], axis=0)

def _bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = bits.astype(np.uint8).flatten()
    n_bits = len(bits)
    n_bytes = (n_bits + 7) // 8
    out = bytearray(n_bytes)
    for i in range(n_bits):
        if bits[i]:
            out[i // 8] |= (1 << (7 - (i % 8)))
    return bytes(out)

def _apply_delta_bridge(
    orig_bgr: np.ndarray,
    wm_bgr_small: np.ndarray,
    base_bgr_small: np.ndarray,
    max_eps_01: float = 2 / 255,
) -> np.ndarray:
    H, W = orig_bgr.shape[:2]
    delta_small = (wm_bgr_small.astype(np.int16) - base_bgr_small.astype(np.int16)).astype(np.int16)
    delta_full = cv2.resize(delta_small, (W, H), interpolation=cv2.INTER_LINEAR)
    delta_full = cv2.GaussianBlur(delta_full, (0, 0), 0.5, 0.5, borderType=cv2.BORDER_REFLECT)
    max_amp = max_eps_01 * 255.0
    delta_full = np.clip(delta_full, -max_amp, max_amp)
    out = np.clip(orig_bgr.astype(np.int16) + delta_full, 0, 255).astype(np.uint8)
    return out


# ------------- local_invis singleton -------------
_LOCAL = {"enc": None, "dec": None, "device": None, "shape": None, "nbits": None}

def _init_local_invis():
    if _LOCAL["enc"] is not None:
        return
    device = _device_auto()
    try:
        enc = Encoder(epsilon=0.01).to(device).eval()
    except TypeError:
        enc = Encoder().to(device).eval()
    dec = Decoder().to(device).eval()

    _LOCAL.update(
        enc=enc,
        dec=dec,
        device=device,
        shape=tuple(LOCAL_INVIS_IMAGE_SHAPE),
        nbits=int(LOCAL_INVIS_NUM_BITS),
    )

def _decode_from_bgr_arr(bgr: np.ndarray) -> str:
    """파일 저장 없이 배열에서 즉시 디코딩 → hex 문자열 반환(실제 길이만)."""
    _init_local_invis()
    h_small, w_small = _LOCAL["shape"]
    work = cv2.resize(bgr, (w_small, h_small), interpolation=cv2.INTER_AREA)
    x = _bgr_to_tensor01(work, _LOCAL["device"])
    with torch.no_grad():
        probs = _LOCAL["dec"](x)
        if probs.min() < 0 or probs.max() > 1:
            probs = torch.sigmoid(probs)
        bits = (probs[0].detach().cpu().numpy() >= 0.5).astype(np.uint8)
    n_bits = _LOCAL["nbits"]
    bits = bits[:n_bits]
    out_bytes = _bits_to_bytes(bits)[: (n_bits + 7)//8]
    return out_bytes.hex()


# ----------------- public API -----------------
def embed_watermark(src_path: str, dst_path: str, payload_uuid_hex: str) -> None:
    """
    원본 해상도 유지 + (내부) 256×256에서 InvisMark 삽입 후 '델타'만 원본에 적용.
    저장은 PNG 권장.
    """
    if METHOD != "local_invis":
        # 필요시 다른 METHOD 분기 추가 가능
        pass

    _init_local_invis()
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    orig_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if orig_bgr is None:
        raise FileNotFoundError(src_path)

    h_small, w_small = _LOCAL["shape"]
    base_bgr_small = cv2.resize(orig_bgr, (w_small, h_small), interpolation=cv2.INTER_AREA)

    # payload → bits
    n_bits = _LOCAL["nbits"]
    byte_len = (n_bits + 7) // 8
    payload_bytes = _uuid_hex_to_bytes(payload_uuid_hex, out_len=byte_len)
    bits_np = _bytes_to_bits(payload_bytes, n_bits=n_bits)
    bits_t = torch.from_numpy(bits_np).to(device=_LOCAL["device"], dtype=torch.float32).unsqueeze(0)  # [1,N]

    # encode
    x = _bgr_to_tensor01(base_bgr_small, _LOCAL["device"])
    with torch.no_grad():
        try:
            y = _LOCAL["enc"](x, bits_t)
        except TypeError:
            y = _LOCAL["enc"](x, bits=bits_t)
        except Exception:
            y = _LOCAL["enc"](x)
    wm_bgr_small = _tensor01_to_bgr(y)

    # delta bridge → 원본에 적용
    out_bgr = _apply_delta_bridge(orig_bgr, wm_bgr_small, base_bgr_small, max_eps_01=2/255)

    # --- 삽입 직후 복원 검증 & 자동 보정 ---
    hex_len = ((n_bits + 7)//8) * 2
    target_hex = payload_bytes.hex()
    dec_try = _decode_from_bgr_arr(out_bgr)
    if dec_try[:hex_len] != target_hex[:hex_len]:
        base = orig_bgr.astype(np.int16)
        diff = (out_bgr.astype(np.int16) - base)
        for gain in (1.15, 1.3, 1.5, 1.7):
            cand = np.clip(base + (diff * gain), 0, 255).astype(np.uint8)
            if _decode_from_bgr_arr(cand)[:hex_len] == target_hex[:hex_len]:
                out_bgr = cand
                break
    # --- 끝 ---

    cv2.imwrite(dst_path, out_bgr)

def decode_watermark(src_path: str) -> str:
    """원본 → 256×256 다운스케일 → 디코더 → hex 문자열(실제 길이만)."""
    if METHOD != "local_invis":
        pass
    _init_local_invis()

    img_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(src_path)

    h_small, w_small = _LOCAL["shape"]
    work = cv2.resize(img_bgr, (w_small, h_small), interpolation=cv2.INTER_AREA)
    x = _bgr_to_tensor01(work, _LOCAL["device"])

    with torch.no_grad():
        probs = _LOCAL["dec"](x)
        if probs.min() < 0 or probs.max() > 1:
            probs = torch.sigmoid(probs)
        bits = (probs[0].detach().cpu().numpy() >= 0.5).astype(np.uint8)

    n_bits = _LOCAL["nbits"]
    bits = bits[:n_bits]
    out_bytes = _bits_to_bytes(bits)[: (n_bits + 7)//8]
    return out_bytes.hex()