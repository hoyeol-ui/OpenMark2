from __future__ import annotations

import uuid
import time
import math
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np

from watermark.configs import LOCAL_INVIS_NUM_BITS
from watermark.embedder import embed_watermark, decode_watermark
from watermark.configs import (
    UPLOAD_DIR, OUTPUT_DIR, ALLOWED_EXTS, ensure_dirs,
    RECOMMENDED_SIDE_RANGE, MAX_SIDE_LIMIT, AUTO_DOWNSCALE_IF_OVER_MAX,
    human_size_policy,
)
from .io_utils import imread_rgb, imwrite_rgb
from .vision import disrupt_once, make_diagnostics, clip_available

# 삽입된 실제 바이트 길이(=모델 비트수 기준)만 비교
_EMBED_HEX_LEN = ((LOCAL_INVIS_NUM_BITS + 7) // 8) * 2  # 바이트 수 ×2(hex 길이)

# --------- helpers ---------
def _safe_basename(p: Path) -> str:
    name = p.name
    return name if name else f"upload_{uuid.uuid4().hex}.png"

def _versioned_name(stem: str) -> str:
    ts = int(time.time()); short = uuid.uuid4().hex[:8]
    return f"{stem}_{ts}_{short}"

def _validate_params(steps, eps, alpha) -> Tuple[int, float, float, Optional[str]]:
    try:
        s = int(steps); e = float(eps); a = float(alpha)
    except Exception:
        return 0, 0.0, 0.0, "설정 값(steps/eps/alpha) 형식이 올바르지 않습니다."
    if s < 1 or s > 16:
        return 0, 0.0, 0.0, "단계(steps)는 1~16 사이여야 합니다."
    if not (0.0 < e <= 0.1):
        return 0, 0.0, 0.0, "ε(eps)는 (0, 0.1] 범위여야 합니다."
    if not (0.0 < a <= e):
        return 0, 0.0, 0.0, "α(alpha)는 (0, eps] 범위여야 합니다."
    if not math.isfinite(e) or not math.isfinite(a):
        return 0, 0.0, 0.0, "설정 값에 유한하지 않은 수가 포함되어 있습니다."
    return s, e, a, None

# 버튼 원샷용 기본값(보수적)
DEFAULT_STEPS = 1
DEFAULT_EPS   = 2/255
DEFAULT_ALPHA = 1/255

def _resize_if_needed_keep_aspect(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, bool]:
    h, w = bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return bgr, False
    scale = max_side / float(side)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    out = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, True

# --------- main API ---------
# (out_img_path, out_uuid_path, diag_imgs, info_note, payload)
def run_one_shot(file, steps=DEFAULT_STEPS, eps=DEFAULT_EPS, alpha=DEFAULT_ALPHA):
    if file is None:
        return None, None, [], "이미지를 먼저 업로드하세요.", ""

    steps_i, eps_f, alpha_f, err = _validate_params(steps, eps, alpha)
    if err:
        return None, None, [], err, ""

    ensure_dirs()

    in_path = Path(file.name if hasattr(file, "name") else str(file))
    ext = in_path.suffix.lower()
    if ext not in ALLOWED_EXTS:
        return None, None, [], "PNG/JPG만 지원합니다.", ""

    safe_name = _safe_basename(in_path)
    stem = Path(safe_name).stem

    if (UPLOAD_DIR / safe_name).exists() or (OUTPUT_DIR / f"{stem}_WM.png").exists():
        stem = _versioned_name(stem)
        safe_name = f"{stem}{ext}"

    up_path = UPLOAD_DIR / safe_name

    # 3) 업로드 저장
    try:
        with open(in_path, "rb") as fsrc, open(up_path, "wb") as fdst:
            fdst.write(fsrc.read())
    except Exception as e:
        return None, None, [], f"업로드 저장에 실패했습니다: {e}", ""

    # 3-1) 사이즈 정책 적용 (MAX 초과 시 자동 다운스케일)
    policy_note_parts: list[str] = [human_size_policy()]
    try:
        # BGR로 읽어 작업 (리사이즈는 BGR에서 처리 후 다시 저장)
        bgr0 = cv2.imread(str(up_path), cv2.IMREAD_COLOR)
        if bgr0 is None:
            return None, None, [], "업로드 이미지를 열 수 없습니다.", ""
        h0, w0 = bgr0.shape[:2]
        max_side0 = max(h0, w0)

        # 너무 큰 경우: 자동 다운스케일
        if max_side0 > MAX_SIDE_LIMIT and AUTO_DOWNSCALE_IF_OVER_MAX:
            bgr1, resized = _resize_if_needed_keep_aspect(bgr0, MAX_SIDE_LIMIT)
            if resized:
                cv2.imwrite(str(up_path), bgr1)
                policy_note_parts.append(f"(자동 축소: {w0}x{h0} → {bgr1.shape[1]}x{bgr1.shape[0]})")
        else:
            bgr1 = bgr0

        # 권장 하한보다 작은 경우 경고
        rec_lo, rec_hi = RECOMMENDED_SIDE_RANGE
        if max(bgr1.shape[:2]) < rec_lo:
            policy_note_parts.append("⚠️ 해상도가 작아 워터마크/방해 효과가 약할 수 있습니다.")
        elif max(bgr1.shape[:2]) > rec_hi and max(bgr1.shape[:2]) <= MAX_SIDE_LIMIT:
            policy_note_parts.append("ℹ️ 매우 큰 해상도는 처리 시간이 늘 수 있습니다.")
    except Exception:
        # 정책 적용 실패 시 무시하고 진행
        pass

    # 4) 워터마크 임베딩
    payload = uuid.uuid4().hex  # 32 hex (16 bytes)
    wm_path = OUTPUT_DIR / f"{stem}_WM.png"
    try:
        embed_watermark(str(up_path), str(wm_path), payload)
    except Exception as e:
        return None, None, [], f"워터마크 처리에 실패했습니다: {e}", ""

    # 5) 디스럽트
    try:
        wm_rgb = imread_rgb(str(wm_path))
    except Exception as e:
        return None, None, [], f"워터마크 결과를 읽는 데 실패했습니다: {e}", ""

    try:
        pd_rgb = disrupt_once(wm_rgb, steps=steps_i, eps=eps_f, alpha=alpha_f)
    except Exception as e:
        return None, None, [], f"방해(Disrupt) 단계에서 오류가 발생했습니다: {e}", ""

    # 5-1) 보호 우선 검증 루프
    def _decode_from_rgb(rgb_arr: np.ndarray) -> str:
        tmp = OUTPUT_DIR / f"{stem}_PD_try.png"
        imwrite_rgb(str(tmp), rgb_arr)
        try:
            out = decode_watermark(str(tmp)) or ""
            return out.strip().lower()
        except Exception:
            return ""

    wm_dec = _decode_from_rgb(wm_rgb)
    if wm_dec[:_EMBED_HEX_LEN] != payload[:_EMBED_HEX_LEN].lower():
        return None, None, [], "워터마크 삽입/복원에 실패했습니다. (WM 단계 불일치)", ""

    pd_dec = _decode_from_rgb(pd_rgb)
    applied_scale = 1.0
    if pd_dec[:_EMBED_HEX_LEN] != payload[:_EMBED_HEX_LEN].lower():
        for scale in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05):
            blended = np.clip(
                wm_rgb.astype(np.int16)
                + scale * (pd_rgb.astype(np.int16) - wm_rgb.astype(np.int16)),
                0, 255
            ).astype(np.uint8)
            trial = _decode_from_rgb(blended)
            if trial[:_EMBED_HEX_LEN] == payload[:_EMBED_HEX_LEN].lower():
                pd_rgb = blended
                applied_scale = scale
                pd_dec = trial
                break

    # 6) 결과 저장
    pd_path = OUTPUT_DIR / f"{stem}_PD.png"
    try:
        imwrite_rgb(str(pd_path), pd_rgb)
    except Exception as e:
        return None, None, [], f"결과 저장에 실패했습니다: {e}", ""

    # 7) UUID 텍스트 저장
    uuid_path = OUTPUT_DIR / f"{stem}_uuid.txt"
    try:
        uuid_path.write_text(payload, encoding="utf-8")
    except Exception as e:
        return None, None, [], f"UUID 저장에 실패했습니다: {e}", ""

    # 8) 진단 + 노트
    try:
        orig_rgb = imread_rgb(str(up_path))
        heat, fft_pd, overlay, note = make_diagnostics(orig_rgb, pd_rgb)
    except Exception as e:
        heat, fft_pd, overlay, note = [], [], [], f"진단 생성 실패: {e}"

    extras = []
    if not clip_available():
        extras.append("(CLIP 미탑재 → Protect-only)")
    if applied_scale != 1.0:
        extras.append(f"(Disrupt 강도 자동 조정: scale={applied_scale:.2f})")
    # 사이즈 정책 안내/경고 합치기
    if policy_note_parts:
        extras.append(" · ".join(policy_note_parts))

    if extras:
        note = " ".join(extras) + (" " + note if note else "")

    return str(pd_path), str(uuid_path), [heat, fft_pd, overlay], note, payload


def do_decode(file) -> str:
    if not file:
        return "이미지를 선택하세요."
    try:
        result = decode_watermark(file.name if hasattr(file, "name") else str(file))
        return result or "(복원된 값이 없습니다)"
    except Exception as e:
        return f"디코드 실패: {e}"