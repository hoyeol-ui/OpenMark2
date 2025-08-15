from __future__ import annotations
from pathlib import Path
import argparse, shutil

import torchvision
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def export_split(root: Path, train: bool):
    split = "train_clean" if train else "test_clean"
    out_root = root / split
    out_root.mkdir(parents=True, exist_ok=True)
    for c in CLASSES:
        (out_root / c).mkdir(parents=True, exist_ok=True)

    ds = CIFAR10(root=str(root), train=train, download=True, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)

    idx = 0
    for imgs, labels in loader:
        for i in range(imgs.size(0)):
            cls = CLASSES[int(labels[i])]
            # 0~1 텐서를 JPEG로 저장(파일명은 연속 번호)
            save_image(imgs[i], fp=str(out_root / cls / f"{idx:06d}.jpg"))
            idx += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cifar10", help="출력 루트 폴더")
    args = ap.parse_args()

    root = Path(args.out)
    export_split(root, train=True)
    export_split(root, train=False)
    print(f"[DONE] exported to: {root}")

if __name__ == "__main__":
    main()