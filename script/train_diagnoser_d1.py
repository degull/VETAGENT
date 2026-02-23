# E:\VETAgent\script\train_diagnoser_d1.py
# ------------------------------------------------------------
# Diagnoser D1 Training (weak supervision)
#
# - tqdm postfix shows TRAIN metrics (loss, micro P/R/F1, exact)
# - End of each epoch prints VAL metrics as a compact table
# - Includes per-class accuracy (train & val)
# - Saves ALL epochs + BEST ckpt (by val micro-F1)
# - Filenames include backbone checkpoint identifier (easy to track)
#
# NOTE: jpeg/noise removed (no positive data -> meaningless in weak labels)
#
# Usage (PowerShell):
#   python e:/VETAgent/script/train_diagnoser_d1.py `
#     --backbone_ckpt "E:/VETAgent/checkpoints/backbone/epoch_037_L0.0315_P28.28_S0.8570.pth" `
#     --data_root "E:/VETAgent/data" `
#     --epochs 10 `
#     --batch_size 8 `
#     --dim 64 `
#     --bias 1 `
#     --volterra_rank 2
# ------------------------------------------------------------

from __future__ import annotations

import os
import sys
import time
import argparse
import random
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm


IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# score order (fixed)
# ✅ removed: "noise", "jpeg"
SCORE_KEYS = ["rain", "snow", "haze", "blur", "drop"]


# ---------------------------
# Utils
# ---------------------------
def _ensure_project_root_on_path() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    return x[:, top : top + patch, left : left + patch]


def _weak_label_from_path(path: str) -> torch.Tensor:
    """
    Weak supervision label from dataset identity:
      - rain100H/L => rain
      - CSD => snow
      - RESIDE-6K => haze
      - GOPRO => blur
      - DayRainDrop/NightRainDrop => drop
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

    return y


def _safe_div(a: float, b: float) -> float:
    return a / max(1e-12, b)


def _basename_no_ext(p: str) -> str:
    p = p.replace("\\", "/")
    base = os.path.basename(p)
    if base.lower().endswith(".pth"):
        base = base[:-4]
    return base


def _print_epoch_table(
    epoch: int,
    train_loss: float,
    train_m: Dict[str, Any],
    val_loss: float,
    val_m: Dict[str, Any],
    score_keys: List[str],
):
    tr_pca = train_m["per_class_acc"]
    va_pca = val_m["per_class_acc"]

    print("\n" + "=" * 86)
    print(f"[Epoch {epoch:03d}] Summary")
    print("-" * 86)
    header = f"{'Split':<8} | {'Loss':>8} | {'microP':>7} | {'microR':>7} | {'microF1':>7} | {'Exact':>7}"
    print(header)
    print("-" * 86)
    print(
        f"{'train':<8} | {train_loss:>8.4f} | {train_m['prec']:>7.3f} | {train_m['rec']:>7.3f} | {train_m['f1']:>7.3f} | {train_m['exact']:>7.3f}"
    )
    print(
        f"{'val':<8} | {val_loss:>8.4f} | {val_m['prec']:>7.3f} | {val_m['rec']:>7.3f} | {val_m['f1']:>7.3f} | {val_m['exact']:>7.3f}"
    )
    print("-" * 86)

    print("[Per-class accuracy]")
    k_w = max(4, max(len(k) for k in score_keys))
    row_header = f"{'class':<{k_w}} | {'train_acc':>9} | {'val_acc':>7}"
    print(row_header)
    print("-" * len(row_header))
    for i, k in enumerate(score_keys):
        print(f"{k:<{k_w}} | {tr_pca[i]:>9.2f} | {va_pca[i]:>7.2f}")
    print("=" * 86 + "\n")


# ---------------------------
# Dataset
# ---------------------------
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
        y = _weak_label_from_path(p)  # (C,)
        return {"x": x, "y": y, "path": p}


# ---------------------------
# Backbone load
# ---------------------------
def _build_backbone(dim: int, bias: int, volterra_rank: int, device: torch.device) -> nn.Module:
    _ensure_project_root_on_path()
    from models.backbone.vetnet import VETNet

    try:
        m = VETNet(dim=dim, bias=bool(bias), volterra_rank=volterra_rank).to(device)
        print("[BackboneInit] VETNet(dim,bias,volterra_rank)")
        return m
    except TypeError:
        m = VETNet(in_channels=3, out_channels=3, dim=dim, bias=bool(bias), volterra_rank=volterra_rank).to(device)
        print("[BackboneInit] VETNet(in_channels,out_channels,dim,bias,volterra_rank) (fallback)")
        return m


def _load_backbone_ckpt(backbone: nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=str(device))
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    missing, unexpected = backbone.load_state_dict(sd, strict=False)

    print(f"[BackboneCKPT] loaded: {ckpt_path}")
    print("[BackboneCKPT] missing=", len(missing), "unexpected=", len(unexpected))
    if len(unexpected) > 0:
        print("  unexpected head:", unexpected[:20])
    if len(missing) > 0:
        print("  missing head:", missing[:20])


# ---------------------------
# Metrics
# ---------------------------
@torch.no_grad()
def _metrics_multilabel(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> Dict[str, Any]:
    probs = torch.sigmoid(logits)
    pred = (probs >= thr).to(targets.dtype)

    tp = (pred * targets).sum().item()
    fp = (pred * (1.0 - targets)).sum().item()
    fn = ((1.0 - pred) * targets).sum().item()

    prec = _safe_div(tp, (tp + fp))
    rec = _safe_div(tp, (tp + fn))
    f1 = _safe_div(2.0 * prec * rec, (prec + rec))

    exact = (pred == targets).all(dim=1).float().mean().item()
    per_class_acc = (pred == targets).float().mean(dim=0)

    return {
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "exact": float(exact),
        "per_class_acc": per_class_acc.detach().cpu().numpy().tolist(),
    }


def _avg_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(dicts) == 0:
        return {"prec": 0.0, "rec": 0.0, "f1": 0.0, "exact": 0.0, "per_class_acc": [0.0] * len(SCORE_KEYS)

                }

    prec = sum(d["prec"] for d in dicts) / len(dicts)
    rec = sum(d["rec"] for d in dicts) / len(dicts)
    f1 = sum(d["f1"] for d in dicts) / len(dicts)
    exact = sum(d["exact"] for d in dicts) / len(dicts)

    pcs = np.array([d["per_class_acc"] for d in dicts], dtype=np.float32)
    per_class_acc = pcs.mean(axis=0).tolist()

    return {"prec": float(prec), "rec": float(rec), "f1": float(f1), "exact": float(exact), "per_class_acc": per_class_acc}


@torch.no_grad()
def _run_eval(
    diagnoser: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    thr: float = 0.5,
    max_batches: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    diagnoser.eval()
    loss_sum = 0.0
    m_list: List[Dict[str, Any]] = []
    n = 0

    for bi, batch in enumerate(loader, start=1):
        x = batch["x"].to(device, non_blocking=True).clamp(0, 1)
        y = batch["y"].to(device, non_blocking=True)

        out = diagnoser(x)
        logits = out["scores_logits"]
        loss = loss_fn(logits, y)

        loss_sum += float(loss.item())
        m_list.append(_metrics_multilabel(logits, y, thr=thr))
        n += 1

        if max_batches > 0 and bi >= max_batches:
            break

    avg_loss = loss_sum / max(1, n)
    avg_m = _avg_dicts(m_list)
    return avg_loss, avg_m


# ---------------------------
# Main
# ---------------------------
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

    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--bias", type=int, default=0)
    ap.add_argument("--volterra_rank", type=int, default=2)

    ap.add_argument("--roots", type=str, default="", help="comma-separated roots. empty -> use defaults")
    ap.add_argument("--max_per_root", type=int, default=0)

    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--eval_max_batches", type=int, default=0)

    ap.add_argument("--freeze_backbone", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--save_dir", type=str, default="E:/VETAgent/checkpoints/diagnoser")
    args = ap.parse_args()

    _ensure_project_root_on_path()
    _set_seed(int(args.seed))

    device = torch.device(args.device if (torch.cuda.is_available() and str(args.device).startswith("cuda")) else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    bb_tag = _basename_no_ext(os.path.normpath(args.backbone_ckpt))

    # roots
    if args.roots.strip():
        roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    else:
        roots = [
            os.path.join(args.data_root, "rain100H", "train", "rain"),
            os.path.join(args.data_root, "rain100L", "train", "rain"),
            os.path.join(args.data_root, "CSD", "Train", "Snow"),
            os.path.join(args.data_root, "RESIDE-6K", "train", "hazy"),
            os.path.join(args.data_root, "GOPRO", "train"),
            os.path.join(args.data_root, "DayRainDrop_Train"),
            os.path.join(args.data_root, "NightRainDrop_Train"),
        ]

    roots = [os.path.normpath(r) for r in roots if os.path.isdir(r)]
    print("[Roots]")
    for r in roots:
        print(" -", r)
    if len(roots) == 0:
        raise FileNotFoundError("No valid roots found. Check --data_root or pass --roots.")

    ds = DiagnoserWeakDataset(roots, patch=int(args.patch), max_per_root=int(args.max_per_root))

    n_all = len(ds)
    val_ratio = float(args.val_ratio)
    n_val = max(1, int(n_all * val_ratio))
    n_train = max(1, n_all - n_val)

    indices = list(range(n_all))
    random.shuffle(indices)
    tr_idx = indices[:n_train]
    va_idx = indices[n_train:]

    tr_ds = Subset(ds, tr_idx)
    va_ds = Subset(ds, va_idx)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(int(args.num_workers) > 0),
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
        persistent_workers=(int(args.num_workers) > 0),
    )

    print(f"[Split] total={n_all} train={len(tr_ds)} val={len(va_ds)} val_ratio={val_ratio:.3f}")

    backbone = _build_backbone(dim=int(args.dim), bias=int(args.bias), volterra_rank=int(args.volterra_rank), device=device)
    _load_backbone_ckpt(backbone, args.backbone_ckpt, device=device)

    if int(args.freeze_backbone) == 1:
        for p in backbone.parameters():
            p.requires_grad = False
        print("[Backbone] frozen=YES")
    else:
        print("[Backbone] frozen=NO (finetuning backbone too)")

    from vetagent.diagnose.diagnoser_net import DiagnoserNetD1

    feat_dim = int(args.dim) * 8
    diagnoser = DiagnoserNetD1(backbone=backbone, num_scores=len(SCORE_KEYS), feat_dim=feat_dim).to(device)

    if int(args.freeze_backbone) == 0:
        params = diagnoser.parameters()
    else:
        params = [p for p in diagnoser.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(args.lr), betas=(0.9, 0.999), weight_decay=0.0)

    loss_fn = nn.BCEWithLogitsLoss()

    print(f"[Train] device={device} epochs={args.epochs} bs={args.batch_size} lr={args.lr} thr={args.thr}")
    print(f"[Scores] {SCORE_KEYS}")
    print(f"[SaveDir] {os.path.normpath(args.save_dir)}")
    print(f"[BackboneTag] {bb_tag}")

    best_f1 = -1.0

    for epoch in range(1, int(args.epochs) + 1):
        diagnoser.train()
        t0 = time.time()

        loss_sum = 0.0
        m_list: List[Dict[str, Any]] = []

        pbar = tqdm(tr_loader, ncols=140, desc=f"Epoch {epoch:03d}/{int(args.epochs):03d} [train]")
        for it, batch in enumerate(pbar, start=1):
            x = batch["x"].to(device, non_blocking=True).clamp(0, 1)
            y = batch["y"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            out = diagnoser(x)
            logits = out["scores_logits"]

            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item())

            m = _metrics_multilabel(logits.detach(), y.detach(), thr=float(args.thr))
            m_list.append(m)

            avg_loss = loss_sum / max(1, it)
            avg_m = _avg_dicts(m_list)
            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "P": f"{avg_m['prec']:.3f}",
                    "R": f"{avg_m['rec']:.3f}",
                    "F1": f"{avg_m['f1']:.3f}",
                    "exact": f"{avg_m['exact']:.3f}",
                }
            )

        train_loss = loss_sum / max(1, len(tr_loader))
        train_m = _avg_dicts(m_list)

        val_loss, val_m = _run_eval(
            diagnoser=diagnoser,
            loader=va_loader,
            device=device,
            loss_fn=loss_fn,
            thr=float(args.thr),
            max_batches=int(args.eval_max_batches),
        )

        _print_epoch_table(epoch, train_loss, train_m, val_loss, val_m, SCORE_KEYS)
        print(f"[Time] epoch_time={time.time()-t0:.1f}s")

        ckpt = {
            "epoch": int(epoch),
            "score_keys": SCORE_KEYS,
            "state_dict": diagnoser.state_dict(),
            "backbone_ckpt": os.path.normpath(args.backbone_ckpt),
            "spec": {"dim": int(args.dim), "bias": int(args.bias), "volterra_rank": int(args.volterra_rank), "feat_dim": feat_dim},
            "train_metrics": {"loss": float(train_loss), **{k: float(train_m[k]) for k in ["prec", "rec", "f1", "exact"]}},
            "val_metrics": {"loss": float(val_loss), **{k: float(val_m[k]) for k in ["prec", "rec", "f1", "exact"]}},
            "train_per_class_acc": train_m["per_class_acc"],
            "val_per_class_acc": val_m["per_class_acc"],
            "thr": float(args.thr),
            "seed": int(args.seed),
            "freeze_backbone": int(args.freeze_backbone),
            "val_ratio": float(args.val_ratio),
            "roots": [os.path.normpath(r) for r in roots],
        }

        out_name = (
            f"D1__BB_{bb_tag}"
            f"__ep_{epoch:03d}"
            f"__valF1_{val_m['f1']:.3f}"
            f"__valL_{val_loss:.4f}.pth"
        )
        out_path = os.path.join(args.save_dir, out_name)
        torch.save(ckpt, out_path)
        print(f"[CKPT] saved: {out_path}")

        if float(val_m["f1"]) > best_f1:
            best_f1 = float(val_m["f1"])
            best_name = f"D1__BB_{bb_tag}__best__valF1_{best_f1:.3f}.pth"
            best_path = os.path.join(args.save_dir, best_name)
            torch.save(ckpt, best_path)
            print(f"[CKPT] best updated -> {best_path}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
"""
python e:/VETAgent/script/train_diagnoser_d1.py `
  --backbone_ckpt "E:/VETAgent/checkpoints/backbone/epoch_037_L0.0315_P28.28_S0.8570.pth" `
  --data_root "E:/VETAgent/data" `
  --epochs 10 `
  --batch_size 8 `
  --dim 64 `
  --bias 1 `
  --volterra_rank 2
"""