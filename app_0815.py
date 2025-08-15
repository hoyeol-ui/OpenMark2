from __future__ import annotations
import uuid, io
from pathlib import Path
from typing import Tuple, Optional, List
from gradio.themes import Soft

import cv2
import numpy as np
from PIL import Image

import gradio as gr

# 우리 프로젝트 모듈
from watermark.configs import UPLOAD_DIR, OUTPUT_DIR
from watermark.embedder import embed_watermark, decode_watermark

# ---- Optional: CLIP (없으면 Protect-only로 폴백) ----
_CLIP_OK = True
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    import open_clip
except Exception:
    _CLIP_OK = False

ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}

# ---------------------------
# Util
# ---------------------------
def _ensure_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _imwrite_rgb(path: str, rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)

def _resize_like(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if a.shape[:2] == ref.shape[:2]:
        return a
    return cv2.resize(a, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)

def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a-b)**2)
    return float("inf") if mse < 1e-12 else 20.0*np.log10(255.0/np.sqrt(mse))

# ---------------------------
# Fast Disrupt (single image)
# ---------------------------
def _pil_to_tensor_01(img: Image.Image, size=224):
    tfm = T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                     T.CenterCrop(size),
                     T.ToTensor()])
    return tfm(img)  # [0,1], 3x224x224

def _clip_norm(x: torch.Tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device)[:,None,None]
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device)[:,None,None]
    return (x - mean) / std

def disrupt_once(rgb_in: np.ndarray,
                 text_prompt: str = "a photo of an object",
                 steps: int = 3, eps: float = 8/255, alpha: float = 3/255
                 ) -> np.ndarray:
    """
    경량 EOT-PGD로 시각 품질 유지하며 CLIP 정렬을 교란.
    - 입력/출력: RGB uint8 [H,W,3] (결과 해상도는 입력과 동일)
    """
    if not _CLIP_OK:
        # CLIP이 없으면 그대로 반환 (Protect-only 데모)
        return rgb_in.copy()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32",
                        pretrained="laion2b_s34b_b79k", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # 텍스트 특징 (단일 프롬프트)
    with torch.inference_mode():
        tok = tokenizer([text_prompt]).to(device)
        tfeat = model.encode_text(tok)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

    # *** 여기가 핵심: 추론모드 텐서를 그래프 밖 "일반 텐서"로 변환 ***
    tfeat = tfeat.detach().clone()  # inference 텐서 → 일반 텐서
    # 이미지 특징 dtype과 맞추기(half/float 혼용 방지)
    # 아래 dtype 매칭은 img_feat 산출 이후에도 한 번 더 안전하게 적용합니다.

    # CLIP 입력 준비
    pil = Image.fromarray(rgb_in)
    x0 = _pil_to_tensor_01(pil, size=224).to(device)  # [0,1]
    x = x0.clone().detach()
    delta = torch.zeros_like(x, requires_grad=True)

    for _ in range(max(1, steps)):
        # (EOT 변형은 기존과 동일)
        s = np.random.uniform(0.92, 1.00)
        size = int(224 * s)
        x_aug = F.interpolate((x + delta).unsqueeze(0), size=(size, size),
                              mode="bilinear", align_corners=False)
        x_aug = F.interpolate(x_aug, size=(224, 224), mode="bilinear", align_corners=False).squeeze(0).clamp(0, 1)

        # --- CLIP forward (gradient 필요) ---
        xn = _clip_norm(x_aug)
        # 혹시 모를 전역 추론 모드에 대비
        with torch.enable_grad():
            img_feat = model.encode_image(xn.unsqueeze(0))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # *** dtype/디바이스 매칭 + inference 텐서 금지 ***
        if tfeat.dtype != img_feat.dtype:
            tfeat = tfeat.to(img_feat.dtype)
        if tfeat.device != img_feat.device:
            tfeat = tfeat.to(img_feat.device)

        # 유사도(단일 프롬프트): 올바른 정렬 최소화
        sim = (img_feat @ tfeat.T).squeeze(0)[0]
        loss = sim
        loss.backward()

        # PGD step
        with torch.no_grad():
            delta += alpha * delta.grad.sign()
            delta.clamp_(-eps, eps)
            (x + delta).clamp_(0, 1)
        delta.grad.zero_()

    # 224→원본 해상도 복원(기존과 동일)
    x_out = (x + delta).clamp(0, 1).detach().cpu()
    out_224 = (x_out.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    out_full = cv2.resize(out_224, (rgb_in.shape[1], rgb_in.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return out_full

# ---------------------------
# Diagnostics (heatmap/FFT/overlay)
# ---------------------------
def make_diagnostics(orig_rgb: np.ndarray, pd_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    pd_rgb = _resize_like(pd_rgb, orig_rgb)
    diff = (pd_rgb.astype(np.float32) - orig_rgb.astype(np.float32))

    # Heatmap (잔차)
    mag = np.abs(diff).mean(axis=2)
    mag = mag / (mag.max() + 1e-8)
    heat = (mag*255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    # FFT(원본/결과의 스펙트럼 비교) – 결과만 보여줘도 충분
    def fft_image(xrgb: np.ndarray):
        gray = cv2.cvtColor(xrgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f = np.fft.fftshift(np.fft.fft2(gray))
        amp = np.log1p(np.abs(f))
        amp = (amp - amp.min())/(amp.max()-amp.min()+1e-8)
        return (amp*255).astype(np.uint8)

    fft_pd = cv2.applyColorMap(fft_image(pd_rgb), cv2.COLORMAP_INFERNO)
    fft_pd = cv2.cvtColor(fft_pd, cv2.COLOR_BGR2RGB)

    # Overlay (원본 위에 잔차 강조)
    scale = 6.0  # 보기용 가중
    overlay = np.clip(orig_rgb.astype(np.float32) + scale*diff, 0, 255).astype(np.uint8)

    # 품질 수치
    psnr_val = _psnr(orig_rgb, pd_rgb)
    note = f"PSNR (orig vs PD): {psnr_val:.2f} dB"

    return heat, fft_pd, overlay, note

# ---------------------------
# One-shot pipeline
# ---------------------------
def run_one_shot(file, steps=3, eps=8/255, alpha=3/255):
    """
    업로드 → Protect(워터마크) → Disrupt → 결과/UUID/시각화 반환
    """
    if file is None:
        return None, None, [], "Please upload an image first.", ""

    _ensure_dirs()

    in_path = Path(file.name if hasattr(file, "name") else str(file))
    ext = in_path.suffix.lower()
    if ext not in ALLOWED_EXTS:
        return None, None, [], "Only PNG/JPG are supported.", ""

    # 1) 업로드 보관
    up_path = UPLOAD_DIR / in_path.name
    with open(in_path, "rb") as fsrc, open(up_path, "wb") as fdst:
        fdst.write(fsrc.read())

    # 2) Protect: 워터마크(UUID)
    payload = uuid.uuid4().hex
    wm_only_path = OUTPUT_DIR / f"{in_path.stem}_WM.png"
    embed_watermark(str(up_path), str(wm_only_path), payload)

    # 3) Disrupt: 빠른 CLIP-PGD (폴백 가능)
    wm_rgb = _imread_rgb(str(wm_only_path))
    pd_rgb = disrupt_once(wm_rgb, steps=int(steps), eps=float(eps), alpha=float(alpha))
    pd_path = OUTPUT_DIR / f"{in_path.stem}_PD.png"
    _imwrite_rgb(str(pd_path), pd_rgb)

    # 4) UUID 텍스트 저장 (증빙)
    uuid_path = OUTPUT_DIR / f"{in_path.stem}_uuid.txt"
    uuid_path.write_text(payload, encoding="utf-8")

    # 5) 시각화
    orig_rgb = _imread_rgb(str(up_path))
    heat, fft_pd, overlay, note = make_diagnostics(orig_rgb, pd_rgb)

    return (
        str(pd_path),               # 결과 이미지 (Protect+Disrupt)
        str(uuid_path),             # UUID 파일
        [heat, fft_pd, overlay],    # 갤러리: Heatmap, FFT, Overlay
        ("(CLIP not found → Protect only) " if not _CLIP_OK else "") + note,
        payload                     # 화면에도 UUID 표기
    )

# ---------------------------
# Decode helper (증빙 확인용)
# ---------------------------
def do_decode(file) -> str:
    if not file:
        return "Choose an image."
    try:
        msg = decode_watermark(file.name if hasattr(file, "name") else str(file))
        return msg if msg else "(Empty result)"
    except Exception as e:
        return f"Decode failed: {e}"

# ---------------------------
# Gradio UI  (교체 영역 시작)
# ---------------------------

KOREAN_FONT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap');
* { font-family: 'Noto Sans KR', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Apple SD Gothic Neo', 'Noto Sans KR', '맑은 고딕', 'Malgun Gothic', sans-serif; }
:root { --brand-blue: #2563eb; } /* Tailwind blue-600 비슷한 톤 */
.gradio-container { color: #111827; } /* 설명(본문) 검은색 계열 */
.footer { opacity: .9; }
"""

theme = Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks(
    title="OpenMark — 보이지 않는 보호 + 방해 (원샷)",
    theme=theme,
    css=KOREAN_FONT_CSS,
) as demo:
    gr.Markdown(
        "## OpenMark — **Protect + Disrupt (원샷)**\n"
        "이미지를 업로드하면 **보이지 않는 워터마크(UUID)** 와 **학습 방해(Disrupt)** 가  적용된 결과를 받을 수 있어요.  \n"
        "아래 ‘결과보기’에서는 컴퓨터가 학습을 위해 인식하는 모습(잔차 히트맵, **FFT** 스펙트럼, 오버레이)을 볼 수 있어요."
    )

    with gr.Row():
        # 좌측: 입력 + 옵션
        with gr.Column(scale=2):
            in_file = gr.File(
                label="이미지 업로드 (PNG/JPG)",
                file_count="single",
                file_types=["image"],
            )

            steps = gr.Slider(
                minimum=1, maximum=8, value=3, step=1,
                label="방해 강도 (단계)",
                info="빠르게는 1, 강하게는 4~6 권장(시간 증가)"
            )

            with gr.Accordion("고급 설정 (선택)", open=False):
                eps = gr.Slider(
                    1/255, 12/255, value=8/255, step=1/255,
                    label="최대 변화량 ε (L∞)",
                    info="픽셀 당 허용 최대 변화. 너무 크면 품질 저하"
                )
                alpha = gr.Slider(
                    1/255, 6/255, value=3/255, step=1/255,
                    label="스텝 크기 α",
                    info="PGD 업데이트 크기 (보통 ε의 1/3~1/4)"
                )

            btn_run = gr.Button("원샷 적용 (Protect + Disrupt)", variant="primary")

            gr.Markdown(
                """
                💡 **Tip**: 초고속 체험은 단계(**steps**)를 **1**로 설정해 보세요.  
                단계가 높을수록 방해효과는 강해지지만 시간이 오래 걸립니다.

                ### 📘 도움말 (간단 설명)

                <span style="display:inline-block; background-color:#3b82f6; color:white; 
                             padding:4px 8px; border-radius:6px; font-weight:bold;">
                    UUID(고유번호)
                </span><br>
                 사진마다 붙는 주민등록번호 같은 고유 번호  
                 나중에 '이 사진은 내 거다!' 하고 증명할 수 있어요.

                <span style="display:inline-block; background-color:#3b82f6; color:white; 
                             padding:4px 8px; border-radius:6px; font-weight:bold;">
                    Disrupt(방해)
                </span><br>
                 컴퓨터가 사진을 배우지 못하게 헷갈리게 만드는 기술  
                 사람 눈에는 거의 안 보이지만, LLM은 제대로 학습 못 하게 돼요.

                <span style="display:inline-block; background-color:#3b82f6; color:white; 
                             padding:4px 8px; border-radius:6px; font-weight:bold;">
                    Residual Heatmap(잔차 히트맵)
                </span><br>
                  밝을수록 **더 많이 바뀐 픽셀**<br>
                  눈으론 잘 안 보이는 미세 변화를 색으로 보여줘

                <span style="display:inline-block; background-color:#3b82f6; color:white; 
                             padding:4px 8px; border-radius:6px; font-weight:bold;">
                    FFT(주파수 스펙트럼)
                </span><br>
                  이미지의 **숨은 패턴 X-ray**
                  가운데가 밝으면 자연스러운 편<br> 링/줄무늬가 보이면 **교란 흔적**이 남은 거예요

                <span style="display:inline-block; background-color:#3b82f6; color:white; 
                             padding:4px 8px; border-radius:6px; font-weight:bold;">
                    Overlay
                </span><br>
                  최종적으로 인공지능이 인식하는 이미지  
                """,
                elem_id="help-box"
            )
        # 우측: 결과/UUID/진단/UUID 텍스트
        with gr.Column(scale=3):
            out_img = gr.Image(
                label="결과 이미지 (워터마크 + 방해)",
                interactive=False
            )

            with gr.Row():
                out_uuid = gr.File(label="UUID 다운로드 (.txt)")
                uuid_text = gr.Textbox(
                    label="UUID (기록용)",
                    interactive=False,
                    placeholder="생성된 UUID가 여기에 표시됩니다."
                )

            gr.Markdown("#### 진단 보기 — 모델이 ‘보는’ 흔적 (옵션)")
            diag = gr.Gallery(
                label="Residual Heatmap / FFT / Overlay",
                columns=3,
                height=320,
                show_label=True
            )
            info = gr.Markdown("")  # PSNR 등 수치

    # 실행 연결
    btn_run.click(
        fn=run_one_shot,
        inputs=[in_file, steps, eps, alpha],
        outputs=[out_img, out_uuid, diag, info, uuid_text],
    )

    gr.Markdown("---")
    gr.Markdown("### 디코드 확인 (선택) — 결과 이미지에서 UUID 복원해보기")

    with gr.Row():
        dec_in = gr.File(
            label="OpenMark 보호가 적용된 이미지 선택",
            file_count="single",
            file_types=["image"],
        )
        dec_btn = gr.Button("디코드 실행", variant="secondary")
        dec_out = gr.Textbox(label="복원된 UUID", interactive=False)

    dec_btn.click(fn=do_decode, inputs=[dec_in], outputs=[dec_out])



if __name__ == "__main__":
    demo.launch()