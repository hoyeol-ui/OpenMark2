from __future__ import annotations

import uuid, time, math
from pathlib import Path
from typing import Tuple, Optional, List
import cv2, numpy as np

from watermark.configs import LOCAL_INVIS_NUM_BITS
from watermark.embedder import embed_watermark, decode_watermark
from watermark.configs import (
    UPLOAD_DIR, OUTPUT_DIR, ALLOWED_EXTS, ensure_dirs,
    RECOMMENDED_SIDE_RANGE, MAX_SIDE_LIMIT, AUTO_DOWNSCALE_IF_OVER_MAX,
    human_size_policy,
)
from .io_utils import imread_rgb, imwrite_rgb
from .vision import disrupt_once, make_diagnostics, clip_available

# ---- constants
_EMBED_HEX_LEN = ((LOCAL_INVIS_NUM_BITS + 7) // 8) * 2  # 삽입 바이트수 * 2 (hex길이)
DEFAULT_STEPS, DEFAULT_EPS, DEFAULT_ALPHA = 1, 2/255, 1/255

# ---- small helpers
def _safe_name(p: Path) -> tuple[str, str]:
    """확장자 필터 + 충돌 없는 파일명 생성"""
    ext = p.suffix.lower()
    if ext not in ALLOWED_EXTS:
        allowed = "/".join(e.strip(".").upper() for e in sorted(ALLOWED_EXTS))
        raise ValueError(f"{allowed}만 지원합니다.")
    name = p.name or f"upload_{uuid.uuid4().hex}.png"
    stem = Path(name).stem
    if (UPLOAD_DIR / name).exists() or (OUTPUT_DIR / f"{stem}_WM.png").exists():
        stem = f"{stem}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        name = f"{stem}{ext}"
    return name, stem

def _validate(steps, eps, alpha) -> Tuple[int, float, float]:
    try:
        s, e, a = int(steps), float(eps), float(alpha)
    except Exception:
        raise ValueError("설정 값(steps/eps/alpha) 형식이 올바르지 않습니다.")
    if not (1 <= s <= 16):            raise ValueError("단계(steps)는 1~16 사이여야 합니다.")
    if not (0.0 < e <= 0.1):          raise ValueError("ε(eps)는 (0, 0.1] 범위여야 합니다.")
    if not (0.0 < a <= e):            raise ValueError("α(alpha)는 (0, eps] 범위여야 합니다.")
    if not (math.isfinite(e) and math.isfinite(a)): raise ValueError("설정 값에 유한하지 않은 수가 포함되어 있습니다.")
    return s, e, a

def _save_upload(src: Path, dst: Path):
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())

def _resize_if_needed(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, bool]:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side: return bgr, False
    s = max_side / float(max(h, w))
    out = cv2.resize(bgr, (int(round(w*s)), int(round(h*s))), interpolation=cv2.INTER_AREA)
    return out, True

def _apply_size_policy(up_path: Path) -> List[str]:
    """MAX 초과 자동 축소 + 권장 범위 안내"""
    notes = [human_size_policy()]
    bgr = cv2.imread(str(up_path), cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError("업로드 이미지를 열 수 없습니다.")
    h, w = bgr.shape[:2]; side = max(h, w)

    if side > MAX_SIDE_LIMIT and AUTO_DOWNSCALE_IF_OVER_MAX:
        bgr2, resized = _resize_if_needed(bgr, MAX_SIDE_LIMIT)
        if resized:
            cv2.imwrite(str(up_path), bgr2)
            notes.append(f"(자동 축소: {w}x{h} → {bgr2.shape[1]}x{bgr2.shape[0]})")

    rec_lo, rec_hi = RECOMMENDED_SIDE_RANGE
    side2 = max(cv2.imread(str(up_path), cv2.IMREAD_COLOR).shape[:2])
    if side2 < rec_lo:   notes.append("⚠️ 해상도가 작아 워터마크/방해 효과가 약할 수 있습니다.")
    elif side2 > rec_hi: notes.append("ℹ️ 매우 큰 해상도는 처리 시간이 늘 수 있습니다.")
    return notes

def _decode_from_rgb(stem: str, rgb: np.ndarray) -> str:
    tmp = OUTPUT_DIR / f"{stem}_PD_try.png"
    imwrite_rgb(str(tmp), rgb)
    try:
        return (decode_watermark(str(tmp)) or "").strip().lower()
    except Exception:
        return ""

# ---- public APIs
# returns: (out_img_path, out_uuid_path, diag_imgs, info_note, payload)
def run_one_shot(file, steps=DEFAULT_STEPS, eps=DEFAULT_EPS, alpha=DEFAULT_ALPHA):
    if not file: return None, None, [], "이미지를 먼저 업로드하세요.", ""
    ensure_dirs()

    try:
        s, e, a = _validate(steps, eps, alpha)
        in_path = Path(file.name if hasattr(file, "name") else str(file))
        safe_name, stem = _safe_name(in_path)
        up_path = UPLOAD_DIR / safe_name
        _save_upload(in_path, up_path)
    except Exception as ex:
        return None, None, [], str(ex), ""

    # 사이즈 정책
    try:
        policy_notes = _apply_size_policy(up_path)
    except Exception as ex:
        policy_notes = [str(ex)]

    # 워터마크 삽입
    payload = uuid.uuid4().hex
    wm_path = OUTPUT_DIR / f"{stem}_WM.png"
    try:
        embed_watermark(str(up_path), str(wm_path), payload)
    except Exception as ex:
        return None, None, [], f"워터마크 처리에 실패했습니다: {ex}", ""

    # Disrupt
    try:
        wm_rgb = imread_rgb(str(wm_path))
        pd_rgb = disrupt_once(wm_rgb, steps=s, eps=e, alpha=a)
    except Exception as ex:
        return None, None, [], f"방해(Disrupt) 단계에서 오류가 발생했습니다: {ex}", ""

    # 보호-우선 검증
    wm_dec = _decode_from_rgb(stem, wm_rgb)
    if wm_dec[:_EMBED_HEX_LEN] != payload[:_EMBED_HEX_LEN].lower():
        return None, None, [], "워터마크 삽입/복원에 실패했습니다. (WM 단계 불일치)", ""

    applied_scale = 1.0
    pd_dec = _decode_from_rgb(stem, pd_rgb)
    if pd_dec[:_EMBED_HEX_LEN] != payload[:_EMBED_HEX_LEN].lower():
        for scale in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
            blended = np.clip(
                wm_rgb.astype(np.int16) + scale * (pd_rgb.astype(np.int16) - wm_rgb.astype(np.int16)),
                0, 255
            ).astype(np.uint8)
            if _decode_from_rgb(stem, blended)[:_EMBED_HEX_LEN] == payload[:_EMBED_HEX_LEN].lower():
                pd_rgb, applied_scale = blended, scale
                break

    # 결과 저장
    try:
        pd_path = OUTPUT_DIR / f"{stem}_PD.png"
        imwrite_rgb(str(pd_path), pd_rgb)
        uuid_path = OUTPUT_DIR / f"{stem}_uuid.txt"
        uuid_path.write_text(payload, encoding="utf-8")
    except Exception as ex:
        return None, None, [], f"결과 저장에 실패했습니다: {ex}", ""

    # 진단/노트
    try:
        orig_rgb = imread_rgb(str(up_path))
        heat, fft_pd, overlay, note = make_diagnostics(orig_rgb, pd_rgb)
    except Exception as ex:
        heat, fft_pd, overlay, note = [], [], [], f"진단 생성 실패: {ex}"

    extras = []
    if not clip_available(): extras.append("(CLIP 미탑재 → Protect-only)")
    if applied_scale != 1.0: extras.append(f"(Disrupt 강도 자동 조정: scale={applied_scale:.2f})")
    if policy_notes: extras.append(" · ".join(policy_notes))
    if extras: note = " ".join(extras) + (" " + note if note else "")

    return str(pd_path), str(uuid_path), [heat, fft_pd, overlay], note, payload


def do_decode(file) -> str:
    if not file: return "이미지를 선택하세요."
    try:
        out = decode_watermark(file.name if hasattr(file, "name") else str(file))
        return out or "(복원된 값이 없습니다)"
    except Exception as ex:
        return f"디코드 실패: {ex}"