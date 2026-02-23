# E:\VETAgent\script\train_diagnoser_d1.py
from __future__ import annotations

import os
import sys
import time
import argparse
import random
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _ensure_project_root_on_path() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def _list_images(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    out: List[str] = []
    for root, _, files in os.walk(d):
        for n in files:
            if n.lower().endswith(IMG_EXT):
                out.append(os.path.join(root, n))
    out.sort()
    return out


def _read_rgb_01(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)


def _rand_crop(x: torch.Tensor, patch: int) -> torch.Tensor:
    _, h, w = x.shape
    if h < patch or w < patch:
        pad_h = max(0, patch - h)
        pad_w = max(0, patch - w)
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, h, w = x.shape
    top = random.randint(0, h - patch)
    left = random.randint(0, w - patch)
    return x[:, top:top+patch, left:left+patch]


# score order (fixed)
SCORE_KEYS = ["rain", "snow", "haze", "blur", "noise", "jpeg", "drop"]


def _weak_label_from_path(path: str) -> torch.Tensor:
    """
    Weak supervision label from dataset identity:
      - rain100H/L => rain
      - CSD => snow
      - RESIDE-6K => haze
      - GOPRO => blur
      - DayRainDrop/NightRainDrop => drop
      - noise/jpeg optional (0)
    """
    p = path.replace("\\", "/").lower()
    y = torch.zeros(len(SCORE_KEYS), dtype=torch.float32)

    def set1(k: str):
        y[SCORE_KEYS.index(k)] = 1.0

    if "/rain100h/" in p or "/rain100l/" in p:
        set1("rain")
    if "/csd/" in p:
        set1("snow")
    if "/reside-6k/" in p:
        set1("haze")
    if "/gopro/" in p:
        set1("blur")
    if "raindrop" in p:
        set1("drop")

    # note: jpeg/noise remain 0 unless you add datasets for them
    return y


class DiagnoserWeakDataset(Dataset):
    """
    Reads LQ images from multiple dataset directories.
    For training, we only need "current image" as input; label is weakly inferred from path.
    """

    def __init__(self, input_roots: List[str], patch: int = 256, max_per_root: int = 0):
        self.patch = int(patch)
        self.paths: List[str] = []
        for r in input_roots:
            imgs = _list_images(r)
            if max_per_root > 0 and len(imgs) > max_per_root:
                imgs = random.sample(imgs, max_per_root)
            self.paths.extend(imgs)
        if len(self.paths) == 0:
            raise FileNotFoundError("No images found in provided roots.")
        random.shuffle(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.paths[idx]
        x = _read_rgb_01(p)
        x = _rand_crop(x, self.patch)
        y = _weak_label_from_path(p)  # (7,)
        return {"x": x, "y": y, "path": p}


def _build_backbone(dim: int, bias: int, volterra_rank: int, device: torch.device) -> nn.Module:
    _ensure_project_root_on_path()
    from models.backbone.vetnet import VETNet

    try:
        m = VETNet(dim=dim, bias=bool(bias), volterra_rank=volterra_rank).to(device)
        return m
    except TypeError:
        m = VETNet(in_channels=3, out_channels=3, dim=dim, bias=bool(bias), volterra_rank=volterra_rank).to(device)
        return m


def _load_backbone_ckpt(backbone: nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=str(device))
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    print(f"[BackboneCKPT] loaded: {ckpt_path}")
    print(f"[BackboneCKPT] missing={len(missing)} unexpected={len(unexpected)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone_ckpt", type=str, required=True)

    ap.add_argument("--data_root", type=str, default="E:/VETAgent/data")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")

    # backbone spec (must match your pretrained backbone)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--bias", type=int, default=0)
    ap.add_argument("--volterra_rank", type=int, default=2)

    # train roots (LQ folders)
    ap.add_argument("--roots", type=str, default="",
                    help='comma-separated roots. empty -> use default known LQ folders from your datasets')
    ap.add_argument("--max_per_root", type=int, default=0)

    ap.add_argument("--save_dir", type=str, default="E:/VETAgent/checkpoints/diagnoser")
    args = ap.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() and str(args.device).startswith("cuda")) else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # default roots from your datasets (LQ side)
    if args.roots.strip():
        roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    else:
        # You can edit these anytime.
        roots = [
            os.path.join(args.data_root, "rain100H", "train", "rain"),
            os.path.join(args.data_root, "rain100L", "train", "rain"),
            os.path.join(args.data_root, "CSD", "Train", "Snow"),
            os.path.join(args.data_root, "RESIDE-6K", "train", "hazy"),   # adjust if your RESIDE layout differs
            os.path.join(args.data_root, "GOPRO", "train"),               # os.walk will find blur images
            os.path.join(args.data_root, "DayRainDrop_Train"),
            os.path.join(args.data_root, "NightRainDrop_Train"),
        ]

    roots = [os.path.normpath(r) for r in roots if os.path.isdir(r)]
    print("[Roots]")
    for r in roots:
        print(" -", r)

    ds = DiagnoserWeakDataset(roots, patch=int(args.patch), max_per_root=int(args.max_per_root))
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )

    backbone = _build_backbone(dim=int(args.dim), bias=int(args.bias), volterra_rank=int(args.volterra_rank), device=device)
    _load_backbone_ckpt(backbone, args.backbone_ckpt, device=device)

    # freeze backbone (start simple). Later you can unfreeze for better accuracy.
    for p in backbone.parameters():
        p.requires_grad = False

    from vetagent.diagnose.diagnoser_net import DiagnoserNetD1
    # VETNet latent has channels = dim*8
    diagnoser = DiagnoserNetD1(backbone=backbone, num_scores=len(SCORE_KEYS), feat_dim=int(args.dim) * 8).to(device)

    opt = torch.optim.AdamW(diagnoser.parameters(), lr=float(args.lr), betas=(0.9, 0.999), weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"[Train] device={device} epochs={args.epochs} bs={args.batch_size} lr={args.lr}")
    print(f"[Scores] {SCORE_KEYS}")

    for epoch in range(1, int(args.epochs) + 1):
        diagnoser.train()
        t0 = time.time()
        loss_sum = 0.0

        pbar = tqdm(loader, ncols=120, desc=f"Epoch {epoch:03d}/{int(args.epochs):03d}")
        for it, batch in enumerate(pbar, start=1):
            x = batch["x"].to(device, non_blocking=True).clamp(0, 1)
            y = batch["y"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            out = diagnoser(x)
            logits = out["scores_logits"]  # (B,7)

            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item())
            pbar.set_postfix({"loss": f"{loss_sum / it:.4f}"})

        ckpt = {
            "epoch": int(epoch),
            "score_keys": SCORE_KEYS,
            "state_dict": diagnoser.state_dict(),
            "backbone_ckpt": os.path.normpath(args.backbone_ckpt),
            "spec": {"dim": int(args.dim), "bias": int(args.bias), "volterra_rank": int(args.volterra_rank)},
        }
        out_path = os.path.join(args.save_dir, f"diagnoser_d1_epoch_{epoch:03d}_loss_{(loss_sum/max(1,len(loader))):.4f}.pth")
        torch.save(ckpt, out_path)
        print(f"\n[CKPT] saved: {out_path}  time={(time.time()-t0):.1f}s")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()

"""
python e:/VETAgent/script/train_diagnoser_d1.py `
  --backbone_ckpt "E:/VETAgent/checkpoints/backbone/epoch_040_L0.0307_P28.26_S0.8513.pth" `
  --data_root "E:/VETAgent/data" `
  --epochs 10 `
  --batch_size 8 `
  --dim 64 `
  --bias 0 `
  --volterra_rank 2
"""