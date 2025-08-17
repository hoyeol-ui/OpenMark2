# Tools/visualize_wm.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2

def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

def resize_like(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if a.shape[:2] == ref.shape[:2]:
        return a
    return cv2.resize(a, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LANCZOS4)

def diff_heatmap(orig_bgr: np.ndarray, wm_bgr: np.ndarray, amplify: float = 4.0) -> np.ndarray:
    # |wm - orig|을 0~255로 정규화 후 컬러맵
    diff = cv2.absdiff(wm_bgr, orig_bgr).astype(np.float32)
    # 채널 평균 -> 단일 채널
    dgray = diff.mean(axis=2)
    # 감도 증폭
    dgray *= amplify
    dgray = np.clip(dgray, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(dgray, cv2.COLORMAP_JET)
    return heat

def overlay(img_bgr: np.ndarray, heat_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return cv2.addWeighted(heat_bgr, alpha, img_bgr, 1.0 - alpha, 0.0)

def fft_magnitude(img_bgr: np.ndarray) -> np.ndarray:
    # 그레이 변환 → 2D FFT → 로그 스펙트럼
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(f))
    # 0~255 정규화
    mag = mag - mag.min()
    mag = (mag / (mag.max() + 1e-8) * 255.0).astype(np.uint8)
    mag = cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)
    return mag

def try_ssim_map(orig_bgr: np.ndarray, wm_bgr: np.ndarray):
    # skimage가 있으면 SSIM 지도 생성, 없으면 None
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return None
    orig = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    wm   = cv2.cvtColor(wm_bgr, cv2.COLOR_BGR2GRAY)
    score, ssim_map = ssim(orig, wm, full=True)
    # SSIM(0~1)을 “차이(1-SSIM)”로 바꿔 히트맵
    diff_map = (1.0 - ssim_map)
    diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    diff_u8 = (diff_map * 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_u8, cv2.COLORMAP_TURBO)
    return score, diff_color

def save_grid(tiles: list[np.ndarray], cols: int, out_path: Path, pad: int = 4, bg=255):
    # 같은 크기라고 가정하고 타일 그리드 저장
    h, w = tiles[0].shape[:2]
    rows = int(np.ceil(len(tiles) / cols))
    grid = np.full((rows*h + (rows-1)*pad, cols*w + (cols-1)*pad, 3), bg, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r*(h+pad), c*(w+pad)
        grid[y:y+h, x:x+w] = to_uint8(tile)
    cv2.imwrite(str(out_path), grid)

def main():
    ap = argparse.ArgumentParser(description="Visualize invisible watermark as heatmaps/overlays/FFT.")
    ap.add_argument("--orig", required=True, help="original image path")
    ap.add_argument("--wm",   required=True, help="watermarked image path (same content)")
    ap.add_argument("--outdir", default="outputs/vis", help="output folder")
    ap.add_argument("--amplify", type=float, default=4.0, help="diff heatmap amplification")
    ap.add_argument("--alpha",   type=float, default=0.45, help="overlay transparency")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    orig = load_bgr(Path(args.orig))
    wm   = load_bgr(Path(args.wm))
    # 크기 맞춤 (예: local_invis는 256 저장, 원본이 다를 수 있음)
    orig = resize_like(orig, wm)

    # 1) 차이 히트맵 & 오버레이
    heat = diff_heatmap(orig, wm, amplify=args.amplify)
    over = overlay(orig, heat, alpha=args.alpha)

    cv2.imwrite(str(outdir / "heatmap.png"), heat)
    cv2.imwrite(str(outdir / "overlay.png"), over)

    # 2) FFT (원본/워터마크/잔차)
    fft_orig = fft_magnitude(orig)
    fft_wm   = fft_magnitude(wm)
    diff_img = cv2.absdiff(wm, orig)
    fft_diff = fft_magnitude(diff_img)

    cv2.imwrite(str(outdir / "fft_orig.png"), fft_orig)
    cv2.imwrite(str(outdir / "fft_wm.png"),   fft_wm)
    cv2.imwrite(str(outdir / "fft_diff.png"), fft_diff)

    # 3) (옵션) SSIM 지도
    ssim_pack = try_ssim_map(orig, wm)
    tiles = [orig, wm, heat, over, fft_diff]
    titles = ["orig","wm","heat","overlay","fft(diff)"]
    if ssim_pack is not None:
        ssim_score, ssim_map = ssim_pack
        cv2.imwrite(str(outdir / "ssim_map.png"), ssim_map)
        tiles.append(ssim_map); titles.append(f"ssim-map (1-SSIM)")
        with (outdir / "metrics.txt").open("w", encoding="utf-8") as f:
            f.write(f"SSIM score: {ssim_score:.4f}\n")

    # 4) 요약 그리드(보도자료/발표용)
    save_grid(tiles, cols=3, out_path=outdir / "summary_grid.png")

    # 간단한 PSNR도 기록
    mse = np.mean((wm.astype(np.float32) - orig.astype(np.float32))**2)
    psnr = (20.0 * np.log10(255.0 / np.sqrt(mse))) if mse > 1e-12 else float("inf")
    with (outdir / "metrics.txt").open("a", encoding="utf-8") as f:
        f.write(f"PSNR (orig vs wm): {psnr:.2f} dB\n")

    print(f"[DONE] saved visualizations to: {outdir}")
    print(" files: heatmap.png, overlay.png, fft_*.png, summary_grid.png, metrics.txt")

if __name__ == "__main__":
    main()