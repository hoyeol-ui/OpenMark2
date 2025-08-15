# OpenMark / watermark / configs.py
from pathlib import Path
from typing import Tuple
import torch

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 입출력
UPLOAD_DIR = PROJECT_ROOT / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# === 실행 엔진 토글 ===
# - "imw"          : imwatermark(DWT+DCT) 방식
# - "local_invis"  : 로컬 PyTorch 스텁(아키텍처 교체하기 쉬운 뼈대)
METHOD = "local_invis"   # 우선 imw로 동작 확인 후 "local_invis"로 테스트

# === imwatermark ===
IMW_METHOD = "dwtDct"
IMW_PAYLOAD_BYTES = 64  # UUID+여유

# === (옵션) InvisMark 가중치 경로 자리 (나중에 실제 모델로 바꿀 때 사용)
WEIGHTS_PATH = PROJECT_ROOT / "watermark" / "models" / "weights" / "paper.ckpt"

# 디바이스
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# (옵션) 이미지 고정 크기 필요 시
IMG_SIZE = (256, 256)  # 예: (256, 256)

# === 로컬 Invis(스텁) 설정 ===
# 문자열을 비트로 변환해 임베드/디코드할 때 사용할 총 비트 수(=모델 출력 차원)
# 논문은 최대 256bit. 여기선 기본 256bit(=32바이트)로 맞춤.
LOCAL_INVIS_NUM_BITS = 64
LOCAL_INVIS_IMAGE_SHAPE = (256, 256)  # (H, W) 테스트용 기본 크기
LOCAL_INVIS_INPUT_RANGE = "0_1"       # 또는 "-1_1"


# === (참고) 네가 갖고 있던 학습용 Config 클래스 - torch.load 안전 로딩을 위해 유지 ===
class ModelConfig:
    log_dir: str = './logs/'
    ckpt_path: str = './ckpts/'
    saved_ckpt_path: str = ''
    world_size: int = 1
    lr: float = 0.0002
    num_epochs: int = 50
    log_interval: int = 400
    num_encoded_bits: int = 100
    image_shape: Tuple[int, int] = (256, 256)
    num_down_levels: int = 4
    num_initial_channels: int = 32
    batch_size: int = 32
    beta_min: float = 0.0001
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