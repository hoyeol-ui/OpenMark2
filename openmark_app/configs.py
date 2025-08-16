from __future__ import annotations
from pathlib import Path

# 프로젝트 루트 기준(현재 파일: OpenMark/openmark_app/configs.py)
ROOT = Path(__file__).resolve().parents[1]

UPLOAD_DIR  = ROOT / "uploads"
OUTPUT_DIR  = ROOT / "outputs"
ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}

def ensure_dirs():
    for d in (UPLOAD_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)