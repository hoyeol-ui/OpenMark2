from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path

def imread_rgb(path: str | Path) -> np.ndarray:
    """
    RGB 이미지 읽기 (OpenCV 기본 BGR → RGB 변환).
    - 입력: 파일 경로
    - 반환: RGB uint8 배열 (H, W, 3)
    - 예외: 파일을 못 읽으면 FileNotFoundError
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def imwrite_rgb(path: str | Path, rgb: np.ndarray) -> None:
    """
    RGB 이미지를 디스크에 저장 (내부에서 RGB → BGR로 변환).
    - 입력: 저장 경로, RGB uint8 배열 (H, W, 3)
    - 반환: 없음 (저장 실패 시 OpenCV가 False 반환하지만 여기서는 예외 처리 안 함)
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

def resize_like(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    참조(ref)와 동일한 (H, W)로 리사이즈.
    - 입력: a(변환 대상), ref(크기 기준)
    - 반환: a를 ref 크기로 보간한 이미지
    - 참고: 크기가 같으면 원본 a 그대로 반환. Lanczos4로 품질 우선 보간.
    """
    if a.shape[:2] == ref.shape[:2]:
        return a
    return cv2.resize(a, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """
    PSNR(신호대잡음비) 계산: 두 이미지 유사도(화질)를 dB로 표현.
    - 입력: a, b (RGB uint8 또는 실수형 배열)
    - 반환: PSNR (float, dB). 완전히 동일하면 inf.
    - 비고: MSE가 매우 작을 때(1e-12 미만)는 inf로 처리.
    """
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse < 1e-12 else 20.0 * np.log10(255.0 / np.sqrt(mse))