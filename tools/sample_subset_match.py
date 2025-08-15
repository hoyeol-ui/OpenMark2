# tools/sample_subset_match.py
from __future__ import annotations
import argparse, random, shutil
from pathlib import Path
from collections import Counter, defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def list_images_by_class(root: Path):
    root = Path(root)
    out = defaultdict(list)
    for cls_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        for p in cls_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                out[cls_dir.name].append(p)
    return out

def main():
    ap = argparse.ArgumentParser(description="Sample a baseline subset matching class histogram of a reference folder.")
    ap.add_argument("--ref", required=True, help="reference folder (e.g., disrupt 500) to read class histogram")
    ap.add_argument("--src", required=True, help="source folder to sample from (e.g., train_wm_const_local)")
    ap.add_argument("--dst", required=True, help="output folder for the matched subset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    ref = Path(args.ref); src = Path(args.src); dst = Path(args.dst)

    # 1) 참조 분포 읽기
    ref_cls_counts = Counter([p.parent.name for p in ref.rglob("*") if p.suffix.lower() in IMG_EXTS])
    assert ref_cls_counts, f"No images under {ref}"

    # 2) 소스 이미지 인덱스
    src_by_cls = list_images_by_class(src)
    missing = [c for c,n in ref_cls_counts.items() if len(src_by_cls.get(c,[])) < n]
    assert not missing, f"Not enough images in classes: {missing}"

    # 3) 샘플링 & 복사
    picked = []
    for c, n in ref_cls_counts.items():
        cand = src_by_cls[c]
        sel = random.sample(cand, n)
        picked += [(c, p) for p in sel]

    for c, p in picked:
        out = dst / c / p.name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)

    # 4) 요약
    hist = Counter([c for c,_ in picked])
    total = sum(hist.values())
    print(f"[DONE] subset saved to: {dst}  (N={total})")
    for c in sorted(hist):
        print(f"  {c}: {hist[c]}")

if __name__ == "__main__":
    main()