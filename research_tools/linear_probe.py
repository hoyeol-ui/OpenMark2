# research_tools/linear_probe.py
from __future__ import annotations
import argparse, json
from pathlib import Path

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import open_clip

def build_loader(root: str, preprocess, batch=256, shuffle=False, workers=2):
    tfm = preprocess  # OpenCLIP의 이미지 전처리 그대로 사용
    ds = datasets.ImageFolder(root, transform=tfm)
    dl = DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=False)
    return ds, dl

@torch.no_grad()
def extract_features(clip_model, dl, device):
    feats, labels = [], []
    for x, y in dl:
        x = x.to(device)
        f = clip_model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)

def evaluate(head: nn.Module, feats: torch.Tensor, labels: torch.Tensor, device) -> float:
    head.eval()
    with torch.no_grad():
        logits = head(feats.to(device))
        pred = logits.argmax(dim=1).cpu()
        return (pred == labels).float().mean().item()

def main():
    ap = argparse.ArgumentParser(description="Linear Probe on OpenCLIP features")
    ap.add_argument("--train_dir", required=True, help="ImageFolder (train)")
    ap.add_argument("--eval_dir", nargs="+", required=True, help="one or more eval dirs")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="outputs/linear_probe_result.json")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()  # 비전 인코더 동결

    # --- Train features ---
    train_ds, train_dl = build_loader(args.train_dir, preprocess, batch=args.batch, shuffle=True)
    num_classes = len(train_ds.classes)
    with torch.no_grad():
        train_feats, train_labels = extract_features(model, train_dl, device)

    # Linear head (학습은 PyTorch로)
    head = nn.Linear(train_feats.shape[1], num_classes).to(device)
    opt = optim.AdamW(head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs+1):
        head.train()
        # 미니배치 학습
        perm = torch.randperm(train_feats.size(0))
        for i in range(0, train_feats.size(0), args.batch):
            idx = perm[i:i+args.batch]
            xb = train_feats[idx].to(device)
            yb = train_labels[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(head(xb), yb)
            loss.backward()
            opt.step()

    # --- Eval ---
    results = {"train_dir": args.train_dir, "eval": {}}
    for ed in args.eval_dir:
        eval_ds, eval_dl = build_loader(ed, preprocess, batch=args.batch, shuffle=False)
        with torch.no_grad():
            eval_feats, eval_labels = extract_features(model, eval_dl, device)
        acc = evaluate(head, eval_feats, eval_labels, device)
        results["eval"][Path(ed).name] = acc

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()