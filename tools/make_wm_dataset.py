# tools/make_wm_dataset.py
from __future__ import annotations
import argparse, os, random, shutil
from pathlib import Path
from typing import List
import uuid

# 우리 프로젝트 임베더 사용
import watermark.embedder as wm

ALLOWED = {".jpg", ".jpeg", ".png"}

def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in ALLOWED]

def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def embed_to_file(src: Path, dst: Path, payload: str, engine: str):
    prev = wm.METHOD
    try:
        wm.METHOD = engine  # "local_invis" or "imw"
        dst.parent.mkdir(parents=True, exist_ok=True)
        wm.embed_watermark(str(src), str(dst), payload)
    finally:
        wm.METHOD = prev

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 폴더 (class subdirs)")
    ap.add_argument("--dst", required=True, help="출력 폴더")
    ap.add_argument("--engine", default="local_invis", choices=["local_invis","imw"])
    ap.add_argument("--mode", required=True, choices=["copy","random","const","poison"],
                    help="copy=그대로 복사 / random=각 이미지 다른 페이로드 / const=전체 같은 페이로드 / poison=일부만 같은 페이로드")
    ap.add_argument("--ratio", type=float, default=0.05, help="poison일 때 워터마크 비율 (0~1)")
    ap.add_argument("--payload", default="OPENMARK", help="const/poison 모드에서 사용할 고정 페이로드")
    ap.add_argument("--target_class", default=None, help="poison: 워터마크된 이미지를 이 클래스로(폴더) 이동")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    assert src.exists(), f"src not found: {src}"

    imgs = list_images(src)
    print(f"[INFO] found {len(imgs)} images")

    if args.mode == "copy":
        for p in imgs:
            rel = p.relative_to(src)
            copy_file(p, dst / rel)
        print("[DONE] copied.")
        return

    if args.mode == "random":
        for p in imgs:
            rel = p.relative_to(src)
            out = dst / rel
            payload = uuid.uuid4().hex[:8]  # 8바이트 (=64비트) 권장
            embed_to_file(p, out, payload, args.engine)
        print("[DONE] wm-random.")
        return

    if args.mode == "const":
        for p in imgs:
            rel = p.relative_to(src)
            out = dst / rel
            embed_to_file(p, out, args.payload, args.engine)
        print("[DONE] wm-const.")
        return

    if args.mode == "poison":
        assert args.target_class is not None, "poison 모드엔 --target_class 필요"
        # 클래스별로 나눠서 일부만 워터마크
        by_class = {}
        for p in imgs:
            cls = p.parent.name  # 상위 폴더명이 클래스라 가정
            by_class.setdefault(cls, []).append(p)

        for cls, files in by_class.items():
            n_poison = max(1, int(len(files) * args.ratio))
            poison_set = set(random.sample(files, n_poison))
            for p in files:
                rel = p.relative_to(src)
                if p in poison_set:
                    # 워터마크 + 라벨 바꾸기(=target_class 폴더로 저장) → 백도어 데이터
                    rel = rel.parent.parent / args.target_class / rel.name
                    out = dst / rel
                    embed_to_file(p, out, args.payload, args.engine)
                else:
                    # 나머지는 그대로 복사(라벨 유지)
                    out = dst / rel
                    copy_file(p, out)
        print("[DONE] poison.")
        return

if __name__ == "__main__":
    main()