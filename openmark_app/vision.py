from __future__ import annotations
import numpy as np, cv2
from PIL import Image
from .io_utils import resize_like, psnr

# ---- Optional CLIP (lazy init & cache) ----
_CLIP_OK = True
try:
    import torch, torch.nn.functional as F
    import torchvision.transforms as T
    import open_clip
except Exception:
    _CLIP_OK = False

def clip_available() -> bool:
    return _CLIP_OK

# ---- diagnostics tuning ----
OVERLAY_SCALE = 12.0  # 오버레이 대비(시각 강조)

# ---- small helpers ----
def _pil_to_tensor_01(img: Image.Image, size: int = 224):
    return T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),
    ])(img)

def _clip_norm(x):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device)[:, None, None]
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device)[:, None, None]
    return (x - mean) / std

# 하나만 로드해서 재사용
_CLIP_STATE = {"model": None, "tok": None, "device": None}
def _ensure_clip():
    if _CLIP_STATE["model"] is not None:  # already loaded
        return
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32",
                        pretrained="laion2b_s34b_b79k", device=device)
    tok = open_clip.get_tokenizer("ViT-B-32")
    _CLIP_STATE.update(model=model, tok=tok, device=device)

# ---- main: disruption ----
def disrupt_once(
    rgb_in: np.ndarray,
    text_prompt: str = "a photo of an object",
    steps: int = 1,
    eps: float = 2/255,
    alpha: float = 1/255,
    target_psnr: float = 42.0,
) -> np.ndarray:
    """
    경량 EOT-PGD로 CLIP 정렬을 교란하되, 해상도/육안 품질을 보수적으로 보호.
    입력/출력: RGB uint8 [H,W,3]
    """
    if not _CLIP_OK:
        return rgb_in.copy()

    _ensure_clip()
    model, tok, device = _CLIP_STATE["model"], _CLIP_STATE["tok"], _CLIP_STATE["device"]

    # 텍스트 임베딩
    with torch.inference_mode():
        tfeat = model.encode_text(tok([text_prompt]).to(device))
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    tfeat = tfeat.detach().clone()

    # 224 base
    x0 = _pil_to_tensor_01(Image.fromarray(rgb_in), size=224).to(device)   # [3,224,224] in [0,1]
    x = x0.clone().detach()
    delta = torch.zeros_like(x, requires_grad=True)

    # PGD with light EOT (scale jitter)
    for _ in range(max(1, steps)):
        s = float(np.random.uniform(0.92, 1.00))
        sz = int(224 * s)
        x_aug = F.interpolate((x + delta).unsqueeze(0), size=(sz, sz), mode="bilinear", align_corners=False)
        x_aug = F.interpolate(x_aug, size=(224, 224), mode="bilinear", align_corners=False).squeeze(0).clamp(0, 1)

        xn = _clip_norm(x_aug)
        with torch.enable_grad():
            img_feat = model.encode_image(xn.unsqueeze(0))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        if tfeat.dtype != img_feat.dtype:   tfeat = tfeat.to(img_feat.dtype)
        if tfeat.device != img_feat.device: tfeat = tfeat.to(img_feat.device)

        # 유사도 최소화
        loss = (img_feat @ tfeat.T).squeeze(0)[0]
        loss.backward()

        with torch.no_grad():
            delta.add_(alpha * delta.grad.sign())
            delta.clamp_(-eps, eps)
            (x + delta).clamp_(0, 1)
        delta.grad.zero_()

    # Δ를 원본 해상도로 브리징(+저주파화+진폭 제한)
    base_224 = (x0.clamp(0,1).detach().cpu().permute(1,2,0).numpy() * 255.0).astype(np.uint8)
    out_224  = ((x + delta).clamp(0,1).detach().cpu().permute(1,2,0).numpy() * 255.0).astype(np.uint8)
    delta_224 = (out_224.astype(np.int16) - base_224.astype(np.int16)).astype(np.int16)

    h, w = rgb_in.shape[:2]
    delta_full = cv2.resize(delta_224, (w, h), interpolation=cv2.INTER_LINEAR)
    delta_full = cv2.GaussianBlur(delta_full, (0,0), 0.5, 0.5, borderType=cv2.BORDER_REFLECT)
    max_amp = eps * 255.0
    delta_full = np.clip(delta_full, -max_amp, max_amp)

    out_full = np.clip(rgb_in.astype(np.int16) + delta_full, 0, 255).astype(np.uint8)

    # 품질 가드레일(PSNR 목표치 미달 시 자동 축소)
    cur = psnr(rgb_in, out_full)
    if cur < target_psnr:
        for s in (0.75, 0.5, 0.35):
            tmp = np.clip(rgb_in.astype(np.int16) + (delta_full * s), 0, 255).astype(np.uint8)
            if psnr(rgb_in, tmp) >= target_psnr:
                out_full = tmp
                break
        else:  # 모두 실패
            out_full = np.clip(rgb_in.astype(np.int16) + (delta_full * 0.25), 0, 255).astype(np.uint8)

    return out_full

# ---- diagnostics ----
def make_diagnostics(orig_rgb: np.ndarray, pd_rgb: np.ndarray):
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