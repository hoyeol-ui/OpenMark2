# watermark/embedder.py
from __future__ import annotations
import os
import cv2
import numpy as np

from . import configs as _cfg

METHOD = getattr(_cfg, "METHOD", "imw")  # "imw" | "local_invis"

# === 공통 ===
from .models.common import (
    cv2_read_bgr, bgr_to_tensor, tensor_to_bgr,
    str_to_bits, bits_to_str
)

# === imwatermark ===
IMW_METHOD = getattr(_cfg, "IMW_METHOD", "dwtDct")
IMW_PAYLOAD_BYTES = int(getattr(_cfg, "IMW_PAYLOAD_BYTES", 64))

# === local invis ===
LOCAL_BITS = int(getattr(_cfg, "LOCAL_INVIS_NUM_BITS", 256))
INPUT_RANGE = getattr(_cfg, "LOCAL_INVIS_INPUT_RANGE", "0_1")
DEVICE = getattr(_cfg, "DEVICE", "cpu")
IMG_SIZE = getattr(_cfg, "IMG_SIZE", None)

# (선택) 가중치 경로: 하나(paper.ckpt) 또는 enc/dec 개별 경로를 지원
WEIGHTS_PATH = getattr(_cfg, "WEIGHTS_PATH", None)
ENCODER_CKPT = getattr(_cfg, "INVIS_ENCODER_CKPT", None)
DECODER_CKPT = getattr(_cfg, "INVIS_DECODER_CKPT", None)

# -----------------------------
# imwatermark 엔진
# -----------------------------
def _imw_embed(input_path: str, output_path: str, payload_text: str) -> None:
    from imwatermark import WatermarkEncoder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = cv2_read_bgr(input_path)
    payload_bytes = payload_text.encode("utf-8")

    enc = WatermarkEncoder()
    enc.set_watermark("bytes", payload_bytes)
    out = enc.encode(img, IMW_METHOD)

    if not cv2.imwrite(output_path, out):
        raise RuntimeError(f"[imw_embed] Failed to write: {output_path}")

def _imw_decode(image_path: str) -> str:
    from imwatermark import WatermarkDecoder
    img = cv2_read_bgr(image_path)
    dec = WatermarkDecoder("bytes", IMW_PAYLOAD_BYTES)
    data = dec.decode(img, IMW_METHOD)
    try:
        return data.decode("utf-8", errors="strict")
    except Exception:
        return data.decode("utf-8", errors="ignore")

# -----------------------------
# local_invis 엔진 (학습모델 로드 + 자동 길이 맞춤)
# -----------------------------
try:
    import torch
    from .models.encoder import Encoder
    from .models.decoder import Decoder
    import watermark.configs as cfg_for_ckpt
    if hasattr(cfg_for_ckpt, "ModelConfig"):
        torch.serialization.add_safe_globals([cfg_for_ckpt.ModelConfig])
except Exception:
    torch = None
    Encoder = Decoder = None

_ENCODER = None
_DECODER = None
_WEIGHTS_LOADED = False

def _init_local_invis():
    """
    Encoder/Decoder 초기화 + (가능하면) 학습 가중치 로드
    """
    global _ENCODER, _DECODER, _WEIGHTS_LOADED
    if torch is None:
        raise ImportError("PyTorch가 필요합니다. `pip install torch` 후 재시도하세요.")
    if _ENCODER is None:
        _ENCODER = Encoder().to(DEVICE).eval()
    if _DECODER is None:
        _DECODER = Decoder().to(DEVICE).eval()

    if not _WEIGHTS_LOADED:
        loaded_any = False
        # 1) 단일 ckpt
        if WEIGHTS_PATH and os.path.exists(str(WEIGHTS_PATH)):
            try:
                ckpt = torch.load(str(WEIGHTS_PATH), map_location=DEVICE, weights_only=False)
                if "encoder_state_dict" in ckpt:
                    _ENCODER.load_state_dict(ckpt["encoder_state_dict"], strict=False)
                    loaded_any = True
                if "decoder_state_dict" in ckpt:
                    sd = ckpt["decoder_state_dict"]
                    sd = {k: v for k, v in sd.items() if not k.startswith("extractor.classifier")}
                    _DECODER.load_state_dict(sd, strict=False)
                    loaded_any = True
            except Exception:
                pass
        # 2) enc/dec 개별 ckpt
        if ENCODER_CKPT and os.path.exists(str(ENCODER_CKPT)):
            try:
                state = torch.load(str(ENCODER_CKPT), map_location=DEVICE, weights_only=False)
                _ENCODER.load_state_dict(state.get("model", state), strict=False)
                loaded_any = True
            except Exception:
                pass
        if DECODER_CKPT and os.path.exists(str(DECODER_CKPT)):
            try:
                state = torch.load(str(DECODER_CKPT), map_location=DEVICE, weights_only=False)
                _DECODER.load_state_dict(state.get("model", state), strict=False)
                loaded_any = True
            except Exception:
                pass
        _WEIGHTS_LOADED = loaded_any  # 실패해도 False → 무학습 코드북으로 동작

def _maybe_resize(img_bgr: np.ndarray) -> np.ndarray:
    if IMG_SIZE is None:
        return img_bgr
    h, w = IMG_SIZE[0], IMG_SIZE[1]  # (H,W)
    return cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

# 옵션 A: 256 기준으로만 임베드/검증
TARGET_PSNR = 36.0  # 256 기준 목표 PSNR(dB)

def _measure_psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))

def _fit_bits(payload_text: str, max_bits: int) -> np.ndarray:
    return str_to_bits(payload_text, n_bits=max_bits)

def _local_embed(input_path: str, output_path: str, payload_text: str) -> None:
    """
    옵션 A: 256(IMG_SIZE)에서 임베드/튜닝/저장 모두 수행 (업스케일 금지)
    """
    _init_local_invis()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1) 입력(원본) → 256으로 리사이즈
    src = cv2_read_bgr(input_path)
    img_small = _maybe_resize(src)                     # (H,W) = IMG_SIZE
    x = bgr_to_tensor(img_small, device=DEVICE, input_range=INPUT_RANGE)  # (1,3,H,W)

    # 2) 페이로드 비트 (LOCAL_BITS에 맞춤)
    bits = _fit_bits(payload_text, max_bits=LOCAL_BITS)
    bits_t = torch.from_numpy(bits).to(DEVICE).unsqueeze(0)

    # 3) 최소 픽셀 변화(≥1/255)를 보장하기 위해 res_norm 추정
    with torch.inference_mode():
        _ENCODER.epsilon = 1.0
        y1 = _ENCODER(x, bits_t)                       # ε=1 기준
    # 평균 잔차 (float [0,1] 스케일)
    res_norm = float((y1 - x).abs().mean().item())
    if res_norm < 1e-12:
        res_norm = 1e-12  # 안전 가드

    PIXEL_DELTA_MIN = 1.0 / 255.0                      # 최소 1 단계 변화
    min_eps = PIXEL_DELTA_MIN / res_norm               # ε 하한 추정

    # 4) ε 이분 탐색 (256 기준 PSNR >= TARGET_PSNR)
    lo = max(0.002, min_eps)                           # 너무 작으면 변화 소실
    hi = min(max(lo * 16.0, 0.05), 2.0)               # 합리적 상한
    best_img = None

    for _ in range(12):
        mid = (lo + hi) / 2.0
        _ENCODER.epsilon = mid

        with torch.inference_mode():
            y = _ENCODER(x, bits_t)                    # (1,3,H,W) in [0,1]
        wm_small = tensor_to_bgr(y, input_range=INPUT_RANGE)  # uint8 (H,W,3)

        # 256 기준 PSNR
        p = _measure_psnr_uint8(img_small, wm_small)

        # 변화가 전혀 없으면(양자화로 0) → ε 키우기
        if p == float("inf"):
            lo = mid
            continue

        if p >= TARGET_PSNR:
            best_img = wm_small.copy()
            hi = mid            # 목표 달성 → 더 약하게
        else:
            lo = mid            # 목표 미달 → 더 강하게

    final = best_img if best_img is not None else wm_small
    if not cv2.imwrite(output_path, final):
        raise RuntimeError(f"[local_embed] Failed to write: {output_path}")

def _local_decode(image_path: str) -> str:
    _init_local_invis()
    img = cv2_read_bgr(image_path)         # 저장본은 256 크기
    img = _maybe_resize(img)               # 혹시 아닐 때를 대비
    x = bgr_to_tensor(img, device=DEVICE, input_range=INPUT_RANGE)

    with torch.inference_mode():
        probs = _DECODER(x).squeeze(0).detach().cpu().numpy()  # (Nbits,)
    bits = (probs > 0.5).astype(np.uint8)
    return bits_to_str(bits)

# -----------------------------
# 공개 API
# -----------------------------
def embed_watermark(input_path: str, output_path: str, payload_text: str) -> None:
    if METHOD == "imw":
        return _imw_embed(input_path, output_path, payload_text)
    elif METHOD == "local_invis":
        return _local_embed(input_path, output_path, payload_text)
    else:
        raise ValueError(f"Unknown METHOD={METHOD}")

def decode_watermark(image_path: str) -> str:
    if METHOD == "imw":
        return _imw_decode(image_path)
    elif METHOD == "local_invis":
        return _local_decode(image_path)
    else:
        raise ValueError(f"Unknown METHOD={METHOD}")