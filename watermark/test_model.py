# watermark/test_model.py
#InvisMark 소스를 로컬에서 테스트 하기 위해 사용한 파일,
#테스트 목적의 파일이라 현재 사용되지 않음.
from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

import watermark.configs as cfg
print("[DBG] IMG_SIZE=", cfg.IMG_SIZE, " SHAPE=", cfg.LOCAL_INVIS_IMAGE_SHAPE,
      " Nbits=", cfg.LOCAL_INVIS_NUM_BITS, " RANGE=", cfg.LOCAL_INVIS_INPUT_RANGE,
      " WEIGHTS_PATH=", getattr(cfg, "WEIGHTS_PATH", None))

# 우리 모듈
import watermark.embedder as wm
from watermark.configs import OUTPUT_DIR, LOCAL_INVIS_NUM_BITS

ALLOWED_EXTS = {".png", ".jpg", ".jpeg"}


def _fit_payload_for_bits(payload: str, nbits: int) -> str:
    """
    UTF-8 바이트 기준으로 nbits(=nb/8 바이트)에 맞춰 자르기.
    디코딩 일치 검증을 위해 테스트 단계에서 과도한 길이로 넣는 실수를 방지.
    """
    max_bytes = nbits // 8
    b = payload.encode("utf-8")
    if len(b) > max_bytes:
        b = b[:max_bytes]
    return b.decode("utf-8", errors="ignore")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def ensure_outdir() -> Path:
    outdir = OUTPUT_DIR / "test_model"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def run_engine(engine: str, in_path: Path, payload: str, outdir: Path) -> Tuple[Path, str, float]:
    """
    단일 엔진 실행: engine='imw' | 'local_invis'
    반환: (워터마크이미지경로, 복원문자열, PSNR)
    """
    # 입력 로드 (PSNR 계산용 원본)
    orig_bgr = cv2.imread(str(in_path))
    if orig_bgr is None:
        raise FileNotFoundError(f"input not found: {in_path}")

    # payload 길이 보정
    adj_payload = _fit_payload_for_bits(payload, LOCAL_INVIS_NUM_BITS) if engine == "local_invis" else payload

    # embedder METHOD 변경
    prev = wm.METHOD
    try:
        wm.METHOD = engine
        out_path = outdir / f"{in_path.stem}_{engine}_watermarked.png"

        # 워터마크 삽입 및 복원
        wm.embed_watermark(str(in_path), str(out_path), adj_payload)
        decoded = wm.decode_watermark(str(out_path))

        # PSNR 계산
        wm_bgr = cv2.imread(str(out_path))
        if wm_bgr is None:
            p = float("nan")
        else:
            if engine == "local_invis":
                # local_invis는 256 저장이므로 원본도 256으로 리사이즈 후 비교
                orig_bgr_for_psnr = cv2.resize(
                    orig_bgr, (wm_bgr.shape[1], wm_bgr.shape[0]),
                    interpolation=cv2.INTER_LANCZOS4
                )
                p = psnr(orig_bgr_for_psnr, wm_bgr)
            else:
                # IWM은 원본 크기 유지 → 다르면 리사이즈
                if orig_bgr.shape != wm_bgr.shape:
                    wm_bgr = cv2.resize(
                        wm_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]),
                        interpolation=cv2.INTER_LANCZOS4
                    )
                p = psnr(orig_bgr, wm_bgr)

        return out_path, decoded, p

    finally:
        wm.METHOD = prev


def main():
    ap = argparse.ArgumentParser(description="Local test for IWM vs local_invis using a real image file.")
    ap.add_argument("--engine", choices=["imw", "local_invis", "both"], default="both",
                    help="어떤 엔진을 테스트할지 선택")
    ap.add_argument("--in", dest="inp", required=True,
                    help="입력 이미지 경로 (png/jpg/jpeg)")
    ap.add_argument("--payload", default="",
                    help="사용자 지정 페이로드(비우면 UUID 자동 생성)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    if not in_path.exists() or in_path.suffix.lower() not in ALLOWED_EXTS:
        raise ValueError("유효한 이미지 파일을 지정하세요 (png/jpg/jpeg).")

    payload = args.payload.strip() or uuid.uuid4().hex
    outdir = ensure_outdir()

    engines = ["imw", "local_invis"] if args.engine == "both" else [args.engine]

    print(f"[INFO] input: {in_path.name}")
    print(f"[INFO] payload: {payload} (len={len(payload.encode('utf-8'))} bytes)")
    print(f"[INFO] engines: {', '.join(engines)}")
    print("-" * 60)

    for eng in engines:
        try:
            out_path, decoded, p = run_engine(eng, in_path, payload, outdir)
            expect = _fit_payload_for_bits(payload, LOCAL_INVIS_NUM_BITS) if eng == "local_invis" else payload
            ok = (decoded == expect)
            print(f"[{eng}] out: {out_path.name}")
            print(f"[{eng}] decoded: {decoded}")
            print(f"[{eng}] expected: {expect}")
            print(f"[{eng}] match: {'OK' if ok else 'FAIL'}")
            print(f"[{eng}] PSNR: {p:.2f} dB")
        except Exception as e:
            print(f"[{eng}] ERROR: {e}")
        print("-" * 60)

    print(f"[DONE] outputs saved in: {outdir}")


if __name__ == "__main__":
    main()