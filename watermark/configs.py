# OpenMark / watermark / configs.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import torch

# ---- 경로
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR   = PROJECT_ROOT / "uploads"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"

# ---- 파일/방식
ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}  # 필요 시 확장 가능
METHOD = "local_invis"                    # "imw" | "local_invis"

# ---- (옵션) 고정 해상도: None이면 원본 유지
IMG_SIZE: Tuple[int, int] | None = None   # 예: (256, 256)

# ---- imwatermark
IMW_METHOD        = "dwtDct"
IMW_PAYLOAD_BYTES = 64                    # UUID+여유

# ---- Invis(로컬 스텁/모델 인터페이스)
LOCAL_INVIS_NUM_BITS   = 64               # 삽입/복원 비트 수
LOCAL_INVIS_IMAGE_SHAPE = (256, 256)      # 내부 작업 해상도(H, W)
LOCAL_INVIS_INPUT_RANGE = "0_1"           # 또는 "-1_1"
PAYLOAD_BYTES = LOCAL_INVIS_NUM_BITS // 8 # 실제 바이트 수

# ---- 가중치/디바이스
WEIGHTS_PATH = PROJECT_ROOT / "watermark" / "models" / "weights" / "paper.ckpt"
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# ---- 업로드 이미지 크기 정책
RECOMMENDED_SIDE_RANGE: Tuple[int, int] = (800, 3000)  # 권장: 800~3000 px
MAX_SIDE_LIMIT: int = 6000                              # 최대 한 변
AUTO_DOWNSCALE_IF_OVER_MAX: bool = True                 # 초과 시 자동 축소

def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def human_size_policy() -> str:
    lo, hi = RECOMMENDED_SIDE_RANGE
    return f"권장 이미지 크기: 한 변 {lo}~{hi}px · 최대 한 변 {MAX_SIDE_LIMIT}px"

# ---- (참고) 학습용 Config 클래스: torch.load 호환 위해 유지
class ModelConfig:
    log_dir: str = './logs/'
    ckpt_path: str = './ckpts/'
    saved_ckpt_path: str = ''
    world_size: int = 1
    lr: float = 2e-4
    num_epochs: int = 50
    log_interval: int = 400
    num_encoded_bits: int = 100
    image_shape: Tuple[int, int] = (256, 256)
    num_down_levels: int = 4
    num_initial_channels: int = 32
    batch_size: int = 32
    beta_min: float = 1e-4
    beta_max: float = 10.0
    beta_start_epoch: float = 1
    beta_epochs: int = 15
    warmup_epochs: int = 1
    discriminator_feature_dim: int = 16
    num_discriminator_layers: int = 4
    watermark_hidden_dim: int = 16
    psnr_threshold: float = 55.0
    enc_mode: str = "ecc"
    ecc_t: int = 16
    ecc_m: int = 8
    num_classes: int = 2
    beta_transform: float = 0.5
    num_noises: int = 2
    noise_start_epoch: int = 20