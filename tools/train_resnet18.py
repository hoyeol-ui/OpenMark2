# tools/train_resnet18.py
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def auto_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_loader(data_dir: str, batch: int, train: bool):
    # 워터마크 저장본(256x256)과 CIFAR 원본(32x32)을 모두 커버하도록 통일
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
    ])
    ds = datasets.ImageFolder(data_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=batch, shuffle=train, num_workers=2, pin_memory=False)
    return ds, dl

@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total

def train_one_epoch(model, dl, device, opt, loss_fn):
    model.train()
    running = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        running += loss.item() * y.size(0)
    return running / len(dl.dataset)

def main():
    ap = argparse.ArgumentParser(description="Train ResNet-18 on folder CIFAR-10 (clean vs watermark).")
    ap.add_argument("--train_dir", required=True, help="학습 데이터 폴더(ImageFolder)")
    ap.add_argument("--test_dir", required=True, nargs="+",
                    help="평가 데이터 폴더 한 개 이상 가능 (공백으로 구분)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/cifar10_resnet18_run")
    args = ap.parse_args()

    set_seed(args.seed)
    device = auto_device()
    print(f"[INFO] device: {device}")

    # 데이터 로더
    train_ds, train_dl = make_loader(args.train_dir, args.batch, train=True)
    test_sets = []
    for td in args.test_dir:
        ds, dl = make_loader(td, args.batch, train=False)
        test_sets.append((Path(td).as_posix(), ds, dl))

    # 모델
    num_classes = len(train_ds.classes)
    model = models.resnet18(weights=None, num_classes=num_classes).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    best_ckpt = out_dir / "best_resnet18.pt"

    best_main = 0.0
    t0 = time.time()

    with log_path.open("w") as f:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_dl, device, opt, loss_fn)
            sched.step()

            results = {}
            for td, _, dl in test_sets:
                acc = evaluate(model, dl, device)
                results[Path(td).name] = acc

            # 첫 번째 test_dir 기준으로 최고 성능 저장
            main_acc = results[Path(test_sets[0][0]).name]
            if main_acc > best_main:
                best_main = main_acc
                torch.save(model.state_dict(), best_ckpt)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "acc": {k: float(v) for k, v in results.items()},
            }
            f.write(json.dumps(row) + "\n"); f.flush()

            pretty = " | ".join([f"{k}: {v*100:.2f}%" for k, v in results.items()])
            print(f"[E{epoch:02d}] loss={train_loss:.4f} | {pretty}")

    print(f"[DONE] best {Path(test_sets[0][0]).name} acc: {best_main*100:.2f}%")
    print(f"[CKPT] {best_ckpt}")
    print(f"[LOG]  {log_path}")
    print(f"[TIME] {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()