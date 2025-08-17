# Tools/clip_eval.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

def load_labels_map(path: Path) -> Tuple[List[str], Dict[str, str]]:
    names, prompts = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)  # 공백 기준 2칼럼 시도
        if len(parts) == 1:
            a = parts[0]
            b = a  # 프롬프트를 클래스명으로 대체(폴백)
        else:
            a, b = parts
        names.append(a)
        prompts[a] = b.strip()
    return names, prompts

@torch.inference_mode()
def encode_texts(model, tokenizer, device, class_names, prompts_map):
    texts = [prompts_map.get(c, c) for c in class_names]
    text_tok = tokenizer(texts).to(device)
    text_feat = model.encode_text(text_tok)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    return text_feat

def list_images(root: Path) -> List[Path]:
    exts={".jpg",".jpeg",".png",".bmp",".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description="Zero-shot eval with OpenCLIP on an ImageFolder-like tree.")
    ap.add_argument("--data", required=True, help="dataset root (class subdirs)")
    ap.add_argument("--labels", required=True, help="labels txt (class<TAB>prompt)")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--out", default=None, help="optional json to save per-file preds")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(args.model)

    class_names = sorted([d.name for d in Path(args.data).iterdir() if d.is_dir()])
    names_order, prompts_map = load_labels_map(Path(args.labels))
    # ensure ordering aligned to folder classes
    assert set(class_names)==set(names_order), f"class mismatch:\nfolders={class_names}\nlabels={names_order}"

    text_feat = encode_texts(model, tokenizer, device, class_names, prompts_map)

    files = list_images(Path(args.data))
    correct = 0
    total = 0
    per_file = []

    for f in tqdm(files, desc="eval"):
        cls = f.parent.name
        img = preprocess(Image.open(f).convert("RGB")).unsqueeze(0).to(device)
        img_feat = model.encode_image(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        logits = 100.0 * img_feat @ text_feat.T  # temperature ~100
        pred_idx = int(logits.argmax(dim=1).item())
        pred_cls = class_names[pred_idx]
        ok = (pred_cls == cls)
        correct += int(ok); total += 1
        if args.out:
            per_file.append({"file": str(f), "gt": cls, "pred": pred_cls, "ok": ok})

    acc = correct / max(1,total)
    print(f"[CLIP] zero-shot top1 accuracy on {args.data}: {acc*100:.2f}%  ({correct}/{total})")
    if args.out:
        Path(args.out).write_text(json.dumps({"acc": acc, "total": total, "items": per_file}, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()