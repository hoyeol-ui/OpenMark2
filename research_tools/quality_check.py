# research_tools/quality_check.py
from __future__ import annotations
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import lpips
import statistics as S

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))

def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def load_pil(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본(워터마크만) 폴더")
    ap.add_argument("--dst", required=True, help="디스럽트 결과 폴더")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--size", type=int, default=224, help="지표 계산용 공통 해상도")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                    else ("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    assert src.exists() and dst.exists(), "src/dst 경로를 확인하세요."

    device = torch.device(args.device)
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()  # 빠른 백본
    to_t = T.ToTensor()
    resize = T.Compose([
        T.Resize(args.size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(args.size),
    ])

    psnrs, lpipss = [], []
    matched = missed = 0

    # 클래스 폴더 기준 파일명 매칭
    for cls_dir in sorted([d for d in src.iterdir() if d.is_dir()]):
        cls = cls_dir.name
        dst_cls = dst / cls
        if not dst_cls.exists():
            continue

        for f in sorted(cls_dir.glob("*")):
            if matched >= args.limit:
                break
            g = dst_cls / f.name
            if not g.exists():
                missed += 1
                continue

            # 공통 해상도로 리사이즈 (PSNR/LPIPS 모두 동일 크기 보장)
            a_pil = resize(load_pil(f))
            b_pil = resize(load_pil(g))

            a = np.array(a_pil); b = np.array(b_pil)
            psnrs.append(psnr(a, b))

            # LPIPS는 [-1,1] 텐서
            a_t = to_t(a_pil).to(device) * 2 - 1
            b_t = to_t(b_pil).to(device) * 2 - 1
            lp = lpips_fn(a_t.unsqueeze(0), b_t.unsqueeze(0)).item()
            lpipss.append(lp)

            matched += 1

    if matched == 0:
        print("[ERR] 매칭된 샘플이 없습니다. 파일명/경로를 확인하세요.")
        return

    print(f"[N] matched={matched}, missed={missed}, size={args.size}x{args.size}")
    print(f"[PSNR]  mean={S.mean(psnrs):.2f}  median={S.median(psnrs):.2f}  min={min(psnrs):.2f}")
    print(f"[LPIPS] mean={S.mean(lpipss):.4f}  median={S.median(lpipss):.4f}  max={max(lpipss):.4f}")
    print("목표: PSNR ≥ 40 dB, LPIPS ≤ 0.02~0.05 (비가시성 확보)")

if __name__ == "__main__":
    main()