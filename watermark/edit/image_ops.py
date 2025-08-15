# watermark/edit/image_ops.py
from __future__ import annotations
from typing import Optional, Tuple
import cv2
import numpy as np

def _sharpen(img: np.ndarray, amount: float = 0.0) -> np.ndarray:
    if amount <= 0:
        return img
    # Unsharp masking 스타일: 강도(amount) 0~2 권장
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharp

def resize_keep_max_side(img: np.ndarray, max_w: int, max_h: int,
                         interp=cv2.INTER_LANCZOS4) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w and h <= max_h:
        return img
    ratio = min(max_w / w, max_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

def fit_aspect(img: np.ndarray, target_wh: Tuple[int, int],
               strategy: str = "crop", bg_color=(0, 0, 0),
               interp=cv2.INTER_LANCZOS4) -> np.ndarray:
    """주어진 (W,H)로 비율 맞추기. strategy='crop'|'pad'."""
    tgt_w, tgt_h = target_wh
    h, w = img.shape[:2]
    src_aspect = w / h
    tgt_aspect = tgt_w / tgt_h

    if abs(src_aspect - tgt_aspect) < 1e-3:
        return cv2.resize(img, (tgt_w, tgt_h), interpolation=interp)

    if strategy == "crop":
        # 중앙 크롭 후 리사이즈
        if src_aspect > tgt_aspect:
            # 가로가 더 넓음 → 좌우 크롭
            new_w = int(h * tgt_aspect)
            x0 = (w - new_w) // 2
            crop = img[:, x0:x0+new_w]
        else:
            # 세로가 더 길음 → 상하 크롭
            new_h = int(w / tgt_aspect)
            y0 = (h - new_h) // 2
            crop = img[y0:y0+new_h, :]
        return cv2.resize(crop, (tgt_w, tgt_h), interpolation=interp)

    # padding
    scale = min(tgt_w / w, tgt_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.full((tgt_h, tgt_w, 3), bg_color, dtype=resized.dtype)
    x = (tgt_w - new_w) // 2
    y = (tgt_h - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas

def edit_pipeline(
    img: np.ndarray,
    preset_size: Optional[Tuple[int, int]] = None,
    custom_size: Optional[Tuple[int, int]] = None,
    fit_strategy: str = "crop",
    max_side: Optional[Tuple[int, int]] = None,
    sharpen_amount: float = 0.0,
) -> np.ndarray:
    """
    편집 파이프라인:
    1) (선택) 최대 가로/세로 제한
    2) (선택) 프리셋 또는 사용자 지정 해상도로 비율 맞춤 (크롭/패딩)
    3) (선택) 샤프닝
    """
    out = img.copy()

    if max_side:
        out = resize_keep_max_side(out, max_w=max_side[0], max_h=max_side[1])

    target = custom_size or preset_size
    if target:
        out = fit_aspect(out, target_wh=target, strategy=fit_strategy)

    out = _sharpen(out, amount=sharpen_amount)
    return out