# Tools/disrupt_clip_batch.py
from __future__ import annotations
import argparse, random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import open_clip
import lpips


# ----------------- helpers -----------------
def load_labels_map(path: Path) -> Tuple[List[str], Dict[str, str]]:
    """
    labels 파일 포맷 예:
      airplane\t a photo of an airplane
      automobile\t a photo of an automobile
      ...
    탭/공백 모두 허용. 첫 토큰은 클래스 폴더명, 나머지는 프롬프트.
    """
    names, prompts = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            # 프롬프트가 없으면 클래스명을 그대로 사용
            cls = parts[0]
            names.append(cls)
            prompts[cls] = cls
        else:
            cls, prompt = parts[0], " ".join(parts[1:])
            names.append(cls)
            prompts[cls] = prompt
    return names, prompts


def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def pil_to_tensor_01(img: Image.Image, size: int = 224) -> torch.Tensor:
    tfm = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(size),
        T.ToTensor(),  # [0,1]
    ])
    return tfm(img)


def clip_norm(t: torch.Tensor) -> torch.Tensor:
    # CLIP 정규화
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=t.device)[:, None, None]
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=t.device)[:, None, None]
    return (t - mean) / std


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Apply CLIP-disrupt adversarial tweak to watermarked images (batch).")
    ap.add_argument("--src", required=True, help="src folder (class subdirs)")
    ap.add_argument("--dst", required=True, help="dest folder (mirrors class subdirs)")
    ap.add_argument("--labels", required=True, help="labels txt (class<TAB>prompt)")
    ap.add_argument("--steps", type=int, default=10, help="PGD steps")
    ap.add_argument("--eps", type=float, default=4/255, help="L_inf budget (e.g., 4/255 ≈ 0.0157)")
    ap.add_argument("--alpha", type=float, default=1/255, help="PGD step size")
    ap.add_argument("--lambda_lpips", type=float, default=0.2, help="LPIPS weight (lower=faster)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                    else ("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="처리할 최대 이미지 수(0이면 전체)")
    ap.add_argument("--shuffle", action="store_true", help="처리 전 파일 순서를 섞음")
    args = ap.parse_args()

    # seeds & device
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ---- CLIP model ----
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    model.eval()  # 드롭아웃/BN 비활성

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    # ---- class names & prompts ----
    class_names = sorted([d.name for d in Path(args.src).iterdir() if d.is_dir()])
    names_order, prompts_map = load_labels_map(Path(args.labels))

    # 클래스 폴더와 라벨 리스트 일치 확인
    assert set(class_names) == set(names_order), (
        f"class mismatch:\n folders={class_names}\n labels={names_order}"
    )

    texts = [prompts_map.get(c, c) for c in class_names]
    text_tok = tokenizer(texts).to(device)

    # 텍스트 임베딩 (상수화: 그래프 분리)
    with torch.inference_mode():
        text_feat = model.encode_text(text_tok)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    # autograd 충돌 방지 + dtype 정렬
    text_feat = text_feat.detach().clone()

    # ---- LPIPS ----
    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    # ---- files ----
    files = list_images(Path(args.src))
    if args.shuffle:
        random.shuffle(files)
    if args.limit and args.limit > 0:
        files = files[:args.limit]
    out_root = Path(args.dst)

    for f in tqdm(files, desc="disrupt", total=len(files)):
        cls = f.parent.name
        out = out_root / cls / f.name
        out.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(f).convert("RGB")
        x0 = pil_to_tensor_01(img, size=224).to(device)  # [0,1], 3x224x224
        x  = x0.clone().detach()
        delta = torch.zeros_like(x, requires_grad=True)

        # PGD with weak EOT (random scale ~0.9..1.0, then force 224x224)
        for _ in range(args.steps):
            # a) EOT transform
            s = random.uniform(0.9, 1.0)
            size = int(224 * s)
            x_adv = (x + delta).unsqueeze(0)  # 1x3xHxW
            x_aug = F.interpolate(x_adv, size=(size, size), mode="bilinear", align_corners=False)
            x_aug = F.interpolate(x_aug, size=(224, 224), mode="bilinear", align_corners=False)  # 최종 224 보장
            x_aug = x_aug.squeeze(0).clamp(0, 1)

            # b) CLIP forward
            xn = clip_norm(x_aug)
            img_feat = model.encode_image(xn.unsqueeze(0))  # <-- grad 필요
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            if text_feat.dtype != img_feat.dtype:  # <-- dtype 통일
                text_feat = text_feat.to(img_feat.dtype)

            # dtype 안전가드 (half/float mismatch 방지)
            if text_feat.dtype != img_feat.dtype:
                text_feat = text_feat.to(img_feat.dtype)

            # ground-truth index & similarity
            gt = class_names.index(cls)
            sim = (img_feat @ text_feat.T).squeeze(0)   # (C,)
            loss_clip = sim[gt]                          # 정답 유사도

            # c) LPIPS (invisibility)
            x_lp  = (x + delta) * 2 - 1
            x0_lp = x0 * 2 - 1
            loss_lp = lpips_fn(x_lp.unsqueeze(0), x0_lp.unsqueeze(0)).mean()

            # ---- total loss ----
            # Untargeted: 정답 유사도 ↓ (부호 음수)
            loss = -loss_clip + args.lambda_lpips * loss_lp

            # ---- backprop ----
            loss.backward()

            # PGD L_inf step
            with torch.no_grad():
                delta += args.alpha * delta.grad.sign()
                delta.clamp_(-args.eps, args.eps)
                (x + delta).clamp_(0, 1)
            delta.grad.zero_()

        # 저장
        x_out = (x + delta).clamp(0, 1)
        out_img = T.ToPILImage()(x_out.cpu())
        out_img.save(out)

    print(f"[DONE] saved to: {args.dst}")


if __name__ == "__main__":
    main()