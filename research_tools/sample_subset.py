# research_tools/sample_subset.py
from __future__ import annotations
import argparse, random, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 폴더(ImageFolder 구조)")
    ap.add_argument("--dst", required=True, help="타겟 폴더")
    ap.add_argument("--total", type=int, default=500, help="총 샘플 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.src); dst = Path(args.dst)
    classes = sorted([d for d in src.iterdir() if d.is_dir()])
    assert classes, f"No class folders under {src}"

    # 각 클래스 파일 리스트
    files_by_c = {c.name: sorted([p for p in c.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}]) for c in classes}
    # 현재 Disrupt 500장과 비슷하게 균등 분배(부족하면 가능한 만큼)
    per_class = max(1, args.total // len(classes))

    picked = []
    for c in classes:
        fs = files_by_c[c.name]
        k = min(per_class, len(fs))
        picked += [(c.name, p) for p in random.sample(fs, k)]
    # 부족분이 있으면 남은 클래스에서 보충
    while len(picked) < args.total:
        c = random.choice(classes).name
        remain = [p for p in files_by_c[c] if (c,p) not in picked]
        if not remain:
            break
        picked.append((c, random.choice(remain)))

    # 복사
    for c, p in picked:
        out = dst / c / p.name
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)

    # 요약
    from collections import Counter
    cnt = Counter([c for c,_ in picked])
    print("[DONE] subset saved to:", dst)
    for c in sorted(cnt):
        print(f"  {c}: {cnt[c]}")

if __name__ == "__main__":
    main()