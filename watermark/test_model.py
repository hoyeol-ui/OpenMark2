import torch
import torch.serialization
import os
import sys

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 정확한 모듈 경로로 import해야 함
import watermark.configs as configs
from watermark.models.encoder import Encoder
from watermark.models.decoder import Decoder

# ✅ torch에게 configs.ModelConfig를 안전하게 허용
torch.serialization.add_safe_globals([configs.ModelConfig])

# config 인스턴스 생성
config = configs.ModelConfig()

# 모델 정의
encoder = Encoder(config)
decoder = Decoder(config)

# 체크포인트 로드
# ckpt_path = "weights/paper.ckpt"
# ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # 🔥 여기를 명시적으로 설정해야 함!

ckpt = torch.load("weights/paper.ckpt", map_location=torch.device("cpu"), weights_only=False)
print(ckpt["decoder_state_dict"].keys())
sd = ckpt["decoder_state_dict"]
sd = {k: v for k, v in sd.items() if not k.startswith("extractor.classifier")}


encoder.load_state_dict(ckpt["encoder_state_dict"])
decoder.load_state_dict(sd, strict=False)

encoder.eval()
decoder.eval()

# 테스트 입력
dummy_img = torch.randn(1, 3, config.image_shape[0], config.image_shape[1])
dummy_wm = torch.randn(1, config.num_encoded_bits)

with torch.no_grad():
    encoded = encoder(dummy_img, dummy_wm)

    # 이미 flatten 했다면 다시 이미지 형태로 복원
    encoded = encoded.view(1, 3, 256, 256)  # or use config.image_shape

    decoded = decoder(encoded)

    print("Encoded image shape:", encoded.shape)
    print("Decoded watermark bits:", decoded)