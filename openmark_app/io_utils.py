from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path

def imread_rgb(path: str | Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def imwrite_rgb(path: str | Path, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

def resize_like(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if a.shape[:2] == ref.shape[:2]:
        return a
    return cv2.resize(a, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a-b)**2)
    return float("inf") if mse < 1e-12 else 20.0*np.log10(255.0/np.sqrt(mse))