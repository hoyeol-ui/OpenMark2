# openmark_app/vision.py
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .io_utils import resize_like, psnr

# ---- Optional CLIP (lazy import & cache) -------------------------------------
_CLIP_OK = True
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    import open_clip
except Exception:
    _CLIP_OK = False


def clip_available() -> bool:
    return _CLIP_OK


# ==== Visual diagnostics =======================================================
OVERLAY_SCALE = 24.0  # 오버레이 대비(시각 강조 배율)

# ==== HVS-aware masking (사람 눈 가시성 최소화) ==============================
HVS_TILE = 10          # 블록 평균 제거 크기(밝기/틴트 억제)
HVS_Y_WEIGHT = 0.22    # 밝기(Y) 채널 가중(낮출수록 덜 보임)
HVS_CHROMA_GAIN = 1.25 # 색상(Cr,Cb) 채널 보정
HVS_EDGE_BASE = 0.45   # 평탄부 최소 마스크
HVS_EDGE_GAIN = 0.70   # 에지 가중(EDGE_BASE+GAIN≈1.0 권장)
HVS_BN_BLEND = 0.25    # 블루노이즈 마스크 가중(저주파 얼룩 억제)
HVS_BLUR_SIGMA = 0.8   # 최종 미세 블러 σ
HVS_BN_SEED = 1234     # 재현성


# ==== Small helpers ============================================================
def _pil_to_tensor_01(img: Image.Image, size: int = 224):
    """PIL → [0,1] torch tensor (3xHxW), CLIP 표준 전처리(Resize+CenterCrop)."""
    return T.Compose(
        [
            T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
        ]
    )(img)


def _clip_norm(x: "torch.Tensor") -> "torch.Tensor":
    """CLIP 입력 정규화(mean/std)."""
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device)[
        :, None, None
    ]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device)[
        :, None, None
    ]
    return (x - mean) / std


# 하나만 로드해서 재사용
_CLIP_STATE: dict = {"model": None, "tok": None, "device": None}


def _ensure_clip() -> None:
    """OpenCLIP 모델/토크나이저를 1회 로드해 캐시에 보관."""
    if _CLIP_STATE["model"] is not None:
        return
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    tok = open_clip.get_tokenizer("ViT-B-32")
    _CLIP_STATE.update(model=model, tok=tok, device=device)


def _apply_hvs_mask(rgb_in: np.ndarray, delta_full: np.ndarray, max_amp: float) -> np.ndarray:
    """
    델타를 사람 눈에 덜 보이게 후처리(HVS 마스킹 + 색도 보정).
    입력/출력: RGB uint8/np.int16 혼용 허용, 반환은 int16 범위 내.
    """
    rgb_base = rgb_in.astype(np.float32)
    delta = delta_full.astype(np.float32)

    # (A) 블록 DC 제거: 대면적 밝기/색 틴트 억제
    mean_delta = cv2.blur(delta, (HVS_TILE, HVS_TILE))
    delta = (delta - mean_delta).astype(np.float32)

    # (B) YCrCb에서 밝기(Y) 억제, 크로마(Cr/Cb) 소폭 강화
    rgb_pert = np.clip(rgb_base + delta, 0, 255).astype(np.uint8)
    ycc_base = cv2.cvtColor(rgb_in, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    ycc_pert = cv2.cvtColor(rgb_pert, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    dycc = ycc_pert - ycc_base
    dycc[..., 0] *= HVS_Y_WEIGHT     # 밝기 영향 축소
    dycc[..., 1:] *= HVS_CHROMA_GAIN # 색상 성분 약간 강화
    tmp_pert = np.clip(ycc_base + dycc, 0, 255).astype(np.uint8)
    tmp_rgb = cv2.cvtColor(tmp_pert, cv2.COLOR_YCrCb2RGB).astype(np.float32)
    delta = (tmp_rgb - rgb_base)

    # (C) 텍스처 마스킹: 평탄부↓, 에지/디테일↑
    gray = cv2.cvtColor(rgb_in, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge /= (edge.max() + 1e-6)
    mask_edge = HVS_EDGE_BASE + HVS_EDGE_GAIN * edge  # 0.45~1.15
    delta *= mask_edge[..., None]

    # (D) 블루노이즈 마스크: 저주파 얼룩 억제
    h, w = gray.shape
    rng = np.random.default_rng(HVS_BN_SEED)
    wn = rng.standard_normal((h, w)).astype(np.float32)
    bn = cv2.Laplacian(wn, cv2.CV_32F, ksize=3)
    bn = np.abs(bn)
    bn /= (bn.max() + 1e-6)
    delta *= ((1.0 - HVS_BN_BLEND) + HVS_BN_BLEND * bn)[..., None]

    # (E) 미세 블러 정리 + 진폭 제한
    delta = cv2.GaussianBlur(
        delta, (0, 0), sigmaX=HVS_BLUR_SIGMA, sigmaY=HVS_BLUR_SIGMA, borderType=cv2.BORDER_REFLECT
    )
    delta = np.clip(delta, -max_amp, max_amp)
    return delta.astype(np.int16)


# ==== main: disruption =========================================================
def disrupt_once(
    rgb_in: np.ndarray,
    text_prompt: str = "a photo of an object",
    # 권장 기본값(강도↑ / 시각 품질 가드 포함 — UI와 동일 계열)
    steps: int = 2,
    eps: float = 4 / 255,
    alpha: float = 1.3 / 255,
    target_psnr: float = 41.5,
) -> np.ndarray:
    """
    경량 EOT-PGD로 CLIP 정렬을 교란.
    - 해상도 유지, 시각 품질은 HVS 마스킹 + PSNR 가드로 보호
    - 입력/출력: RGB uint8 [H,W,3]
    """
    if not _CLIP_OK:
        return rgb_in.copy()

    # 1) CLIP 준비 (1회 로드 캐시)
    _ensure_clip()
    model, tok, device = _CLIP_STATE["model"], _CLIP_STATE["tok"], _CLIP_STATE["device"]

    # 2) 텍스트 임베딩
    with torch.inference_mode():
        tfeat = model.encode_text(tok([text_prompt]).to(device))
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    tfeat = tfeat.detach().clone()

    # 3) 224 기준 텐서
    x0 = _pil_to_tensor_01(Image.fromarray(rgb_in), size=224).to(device)  # [3,224,224] in [0,1]
    x = x0.clone().detach()
    delta = torch.zeros_like(x, requires_grad=True)

    # 4) PGD (light EOT: scale jitter)
    for _ in range(max(1, steps)):
        s = float(np.random.uniform(0.92, 1.00))
        sz = int(224 * s)
        x_aug = F.interpolate((x + delta).unsqueeze(0), size=(sz, sz), mode="bilinear", align_corners=False)
        x_aug = F.interpolate(x_aug, size=(224, 224), mode="bilinear", align_corners=False).squeeze(0).clamp(0, 1)

        xn = _clip_norm(x_aug)
        with torch.enable_grad():
            img_feat = model.encode_image(xn.unsqueeze(0))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        if tfeat.dtype != img_feat.dtype:
            tfeat = tfeat.to(img_feat.dtype)
        if tfeat.device != img_feat.device:
            tfeat = tfeat.to(img_feat.device)

        loss = (img_feat @ tfeat.T).squeeze(0)[0]  # 유사도 최소화
        loss.backward()

        with torch.no_grad():
            delta.add_(alpha * delta.grad.sign())
            delta.clamp_(-eps, eps)
            (x + delta).clamp_(0, 1)
        delta.grad.zero_()

    # 5) Δ 브리징(224→원본) + HVS 마스킹
    base_224 = (x0.clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    out_224 = ((x + delta).clamp(0, 1).detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    delta_224 = (out_224.astype(np.int16) - base_224.astype(np.int16)).astype(np.int16)

    h, w = rgb_in.shape[:2]
    delta_full = cv2.resize(delta_224, (w, h), interpolation=cv2.INTER_LINEAR)
    delta_full = _apply_hvs_mask(rgb_in, delta_full, max_amp=eps * 255.0)

    # 6) 합성 + 품질 가드
    out_full = np.clip(rgb_in.astype(np.int16) + delta_full, 0, 255).astype(np.uint8)
    cur = psnr(rgb_in, out_full)
    if cur < target_psnr:
        for s in (0.75, 0.5, 0.35):
            tmp = np.clip(rgb_in.astype(np.int16) + (delta_full * s), 0, 255).astype(np.uint8)
            if psnr(rgb_in, tmp) >= target_psnr:
                out_full = tmp
                break
        else:
            out_full = np.clip(rgb_in.astype(np.int16) + (delta_full * 0.25), 0, 255).astype(np.uint8)

    return out_full


# ==== diagnostics ==============================================================
def make_diagnostics(orig_rgb: np.ndarray, pd_rgb: np.ndarray):
    """Residual heatmap / FFT / Overlay 와 PSNR 노트 생성."""
    pd_rgb = resize_like(pd_rgb, orig_rgb)
    diff = (pd_rgb.astype(np.float32) - orig_rgb.astype(np.float32))

    # heatmap (잔차 크기)
    mag = np.abs(diff).mean(axis=2)
    mag = mag / (mag.max() + 1e-8)
    heat = cv2.applyColorMap((mag * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    # FFT(결과 스펙트럼)
    f = np.fft.fftshift(np.fft.fft2(cv2.cvtColor(pd_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)))
    amp = np.log1p(np.abs(f))
    amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
    fft_pd = cv2.applyColorMap((amp * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    fft_pd = cv2.cvtColor(fft_pd, cv2.COLOR_BGR2RGB)

    # overlay(시각 강조)
    overlay = np.clip(orig_rgb.astype(np.float32) + OVERLAY_SCALE * diff, 0, 255).astype(np.uint8)
    note = f"PSNR (orig vs PD): {psnr(orig_rgb, pd_rgb):.2f} dB"
    return heat, fft_pd, overlay, note