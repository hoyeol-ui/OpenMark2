# watermark/edit/presets.py
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass(frozen=True)
class TargetSpec:
    name: str
    size: Optional[Tuple[int, int]]  # (W, H). None이면 원본 유지
    aspect: Optional[Tuple[int, int]]  # (W, H) 비율. None이면 무시

# 숏폼/썸네일/유튜브 용 기본 프리셋
PRESETS: Dict[str, TargetSpec] = {
    "원본 유지": TargetSpec("원본 유지", None, None),
    "YouTube 썸네일 (16:9, 1280x720)": TargetSpec("yt_thumb", (1280, 720), (16, 9)),
    "YouTube Shorts (9:16, 1080x1920)": TargetSpec("shorts", (1080, 1920), (9, 16)),
    "Instagram 정사각 (1:1, 1080x1080)": TargetSpec("ig_square", (1080, 1080), (1, 1)),
}

# 비율 맞춤 전략
FIT_STRATEGIES = ("crop", "pad")  # 크롭 / 패딩