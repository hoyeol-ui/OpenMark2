from __future__ import annotations
import numpy as np, cv2
from PIL import Image
from .io_utils import resize_like, psnr

# ---- Optional CLIP ----
_CLIP_OK = True
try:
    import torch, torch.nn.functional as F
    import torchvision.transforms as T
    import open_clip
except Exception:
    _CLIP_OK = False

def clip_available() -> bool:
    return _CLIP_OK

def _pil_to_tensor_01(img: Image.Image, size=224):
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
    ])(img)

def _clip_norm(x):
    import torch
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device)[:,None,None]
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device)[:,None,None]
    return (x - mean) / std

def disrupt_once(
    rgb_in: np.ndarray,
    text_prompt: str = "a photo of an object",
    # CHANGED: 기본값을 보수적으로 낮춤
    steps: int = 1,                 # was: 3
    eps: float = 2/255,             # was: 8/255
    alpha: float = 1/255,           # was: 3/255
    # NEW: 육안 품질 가드레일(목표 PSNR)
    target_psnr: float = 42.0,
) -> np.ndarray:
    """
    경량 EOT-PGD로 CLIP 정렬을 교란하되, 육안 품질을 보수적으로 보호.
    입력/출력: RGB uint8 [H,W,3] (해상도 유지)
    """

    if not _CLIP_OK:
        # CLIP이 없으면 그대로 반환 (Protect-only 데모)
        return rgb_in.copy()

    import torch
    import torch.nn.functional as F
    import open_clip
    import cv2
    import numpy as np
    from .io_utils import psnr  # NEW: PSNR 가드레일에 사용

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- CLIP 준비 (동일) ---
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    with torch.inference_mode():
        tok = tokenizer([text_prompt]).to(device)
        tfeat = model.encode_text(tok)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    # 이미지 텐서 준비 (동일)
    pil = Image.fromarray(rgb_in)
    x0 = _pil_to_tensor_01(pil, size=224).to(device)  # [0,1], 3x224x224
    x = x0.clone().detach()
    delta = torch.zeros_like(x, requires_grad=True)

    # *** dtype/디바이스 매칭 ***
    tfeat = tfeat.detach().clone()

    # --- PGD 루프 (거의 동일, 기본 파라미터만 완화) ---
    for _ in range(max(1, steps)):
        # (EOT 변형 동일)
        s = float(np.random.uniform(0.92, 1.00))
        size = int(224 * s)
        x_aug = F.interpolate((x + delta).unsqueeze(0), size=(size, size),
                              mode="bilinear", align_corners=False)
        x_aug = F.interpolate(x_aug, size=(224, 224), mode="bilinear",
                              align_corners=False).squeeze(0).clamp(0, 1)

        xn = _clip_norm(x_aug)
        with torch.enable_grad():
            img_feat = model.encode_image(xn.unsqueeze(0))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        if tfeat.dtype != img_feat.dtype:
            tfeat = tfeat.to(img_feat.dtype)
        if tfeat.device != img_feat.device:
            tfeat = tfeat.to(img_feat.device)

        # 유사도 최소화
        sim = (img_feat @ tfeat.T).squeeze(0)[0]
        loss = sim
        loss.backward()

        with torch.no_grad():
            delta += alpha * delta.grad.sign()
            delta.clamp_(-eps, eps)
            (x + delta).clamp_(0, 1)
        delta.grad.zero_()

    # === 여기서부터가 핵심 변경(육안 품질 가드레일) =========================
    # base_224 / out_224 계산
    base_224 = (x0.clamp(0,1).detach().cpu().permute(1,2,0).numpy() * 255.0).astype(np.uint8)  # NEW
    out_224  = ((x + delta).clamp(0,1).detach().cpu().permute(1,2,0).numpy() * 255.0).astype(np.uint8)

    # (1) 델타만 추출
    delta_224 = (out_224.astype(np.int16) - base_224.astype(np.int16)).astype(np.int16)  # NEW

    # (2) 원본 해상도로 델타만 보간 (부드럽게)
    h, w = rgb_in.shape[:2]
    delta_full = cv2.resize(delta_224, (w, h), interpolation=cv2.INTER_LINEAR)  # NEW

    # (3) 저주파화: 작은 sigma의 가우시안 블러로 고주파 노이즈 억제
    delta_full = cv2.GaussianBlur(delta_full, (0, 0), sigmaX=0.5, sigmaY=0.5,
                                  borderType=cv2.BORDER_REFLECT)  # NEW

    # (4) 진폭 제한: eps는 [0,1] 스케일 → [0,255]로 환산하여 클리핑
    max_amp = eps * 255.0  # NEW
    delta_full = np.clip(delta_full, -max_amp, max_amp)  # NEW

    # (5) 적용 & PSNR 목표치 가드레일
    out_full = np.clip(rgb_in.astype(np.int16) + delta_full, 0, 255).astype(np.uint8)  # NEW

    cur_psnr = psnr(rgb_in, out_full)  # NEW
    if cur_psnr < target_psnr:  # NEW
        # 단계적으로 스케일 다운 (최대 3회)
        for scale in (0.75, 0.5, 0.35):  # NEW
            tmp = np.clip(rgb_in.astype(np.int16) + (delta_full * scale), 0, 255).astype(np.uint8)
            if psnr(rgb_in, tmp) >= target_psnr:
                out_full = tmp
                break
        # 그래도 미달이면 최저 강도로 안전하게
        if psnr(rgb_in, out_full) < target_psnr:  # NEW
            out_full = np.clip(rgb_in.astype(np.int16) + (delta_full * 0.25), 0, 255).astype(np.uint8)

    return out_full

def make_diagnostics(orig_rgb: np.ndarray, pd_rgb: np.ndarray):
    pd_rgb = resize_like(pd_rgb, orig_rgb)
    diff = (pd_rgb.astype(np.float32) - orig_rgb.astype(np.float32))

    mag = np.abs(diff).mean(axis=2)
    mag = mag / (mag.max() + 1e-8)
    heat = (mag*255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(pd_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    amp = np.log1p(np.abs(f))
    amp = (amp - amp.min())/(amp.max()-amp.min()+1e-8)
    fft_pd = (amp*255).astype(np.uint8)
    fft_pd = cv2.applyColorMap(fft_pd, cv2.COLORMAP_INFERNO)
    fft_pd = cv2.cvtColor(fft_pd, cv2.COLOR_BGR2RGB)

    overlay = np.clip(orig_rgb.astype(np.float32) + 12.0*diff, 0, 255).astype(np.uint8)
    note = f"PSNR (orig vs PD): {psnr(orig_rgb, pd_rgb):.2f} dB"
    return heat, fft_pd, overlay, note