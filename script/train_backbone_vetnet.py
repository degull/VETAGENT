# E:\VETAgent\script\train_backbone_vetnet.py
# ------------------------------------------------------------
# VETNet Backbone Training (SSOT + SAFE STRICT PAIRING)
#
# ✅ Backbone train mix: CSD / DayRainDrop / NightRainDrop / rain100H / rain100L / RESIDE-6K / GOPRO
# 🚫 HIDE_dataset: train에서 자동 제외 (test-only)
#
# - YAML spec 기반 mix
# - folder pairing: relpath(noext) strict match only (zip fallback 제거)
# - sanity dump: lq|gt 저장
# - iter save: LQ|Pred|GT + PSNR/SSIM 텍스트를 한 이미지로 저장
# - PSNR/SSIM: 이전 코드와 동일하게 skimage 기반(uint8 변환 후)로 계산 (metric_every마다)
# ------------------------------------------------------------

from __future__ import annotations

import os
import sys
import time
import json
import random
import argparse
import hashlib
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# Optional skimage metrics (same as previous code)
# ============================================================
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    USE_SKIMAGE = True
except Exception:
    USE_SKIMAGE = False


# ============================================================
# Utils: path (import-safe)
# ============================================================
def _ensure_project_root_on_path() -> str:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


try:
    from vetagent.config import load_cfg
    from vetagent.config.constants import ACTIONS_ORDER
except ModuleNotFoundError:
    _ensure_project_root_on_path()
    from vetagent.config import load_cfg
    from vetagent.config.constants import ACTIONS_ORDER


# ============================================================
# Image IO
# ============================================================
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def _list_images(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    out: List[str] = []
    for root, _, files in os.walk(d):
        for name in files:
            if name.lower().endswith(IMG_EXT):
                out.append(os.path.join(root, name))
    out.sort()
    return out


def _read_rgb(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)


def _rand_crop_pair(lq: torch.Tensor, gt: torch.Tensor, patch: int) -> Tuple[torch.Tensor, torch.Tensor]:
    _, h, w = lq.shape
    if h < patch or w < patch:
        pad_h = max(0, patch - h)
        pad_w = max(0, patch - w)
        lq = F.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")
        gt = F.pad(gt, (0, pad_w, 0, pad_h), mode="reflect")
        _, h, w = lq.shape

    top = random.randint(0, h - patch)
    left = random.randint(0, w - patch)
    return lq[:, top : top + patch, left : left + patch], gt[:, top : top + patch, left : left + patch]


def _to_u8_hwc(x_chw01: torch.Tensor) -> np.ndarray:
    x = x_chw01.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255.0).round().astype(np.uint8)


def _load_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\consola.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def save_triplet_with_text(
    lq_chw: torch.Tensor,
    pred_chw: torch.Tensor,
    gt_chw: torch.Tensor,
    out_path: str,
    title_left: str = "LQ",
    title_mid: str = "Pred",
    title_right: str = "GT",
    text_lines: Optional[List[str]] = None,
):
    lq = _to_u8_hwc(lq_chw)
    pr = _to_u8_hwc(pred_chw)
    gt = _to_u8_hwc(gt_chw)

    h, w, _ = lq.shape
    bar_h = 48
    canvas = np.zeros((h + bar_h, w * 3, 3), dtype=np.uint8)
    canvas[bar_h : bar_h + h, 0:w] = lq
    canvas[bar_h : bar_h + h, w : 2 * w] = pr
    canvas[bar_h : bar_h + h, 2 * w : 3 * w] = gt

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = _load_font(18)

    draw.text((8, 8), title_left, fill=(255, 255, 255), font=font)
    draw.text((w + 8, 8), title_mid, fill=(255, 255, 255), font=font)
    draw.text((2 * w + 8, 8), title_right, fill=(255, 255, 255), font=font)

    if text_lines:
        y = 26
        for line in text_lines[:2]:
            draw.text((8, y), line, fill=(255, 255, 0), font=font)
            y += 20

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


# ============================================================
# Metrics (same as previous: skimage on uint8)
# ============================================================
@torch.no_grad()
def compute_psnr_ssim_skimage(pred_bchw01: torch.Tensor, gt_bchw01: torch.Tensor, max_images: int = 1) -> Tuple[float, float]:
    """
    pred, gt: BCHW float in [0,1]
    returns mean PSNR/SSIM over first max_images samples
    (same spirit as previous code: compute sparingly)
    """
    if not USE_SKIMAGE:
        return 0.0, 0.0

    b = int(pred_bchw01.shape[0])
    n = min(b, int(max_images))
    ps_sum = 0.0
    ss_sum = 0.0

    for i in range(n):
        p = _to_u8_hwc(pred_bchw01[i])
        g = _to_u8_hwc(gt_bchw01[i])
        ps_sum += float(peak_signal_noise_ratio(g, p, data_range=255))
        ss_sum += float(structural_similarity(g, p, channel_axis=2, data_range=255))

    return ps_sum / max(1, n), ss_sum / max(1, n)


# ============================================================
# Fingerprint
# ============================================================
def _sha256_of_json(obj: Dict[str, Any]) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _fingerprint_payload_from_cfg(cfg) -> Dict[str, Any]:
    fields = getattr(cfg.backbone, "fingerprint_fields", None)
    if not fields:
        fields = ["name", "dim", "bias", "volterra_rank", "actions.order"]

    payload: Dict[str, Any] = {}
    for f in fields:
        if f == "name":
            payload["name"] = str(getattr(cfg.backbone, "name", "VETNet"))
        elif f == "dim":
            payload["dim"] = int(cfg.backbone.dim)
        elif f == "bias":
            payload["bias"] = int(cfg.backbone.bias)
        elif f == "volterra_rank":
            payload["volterra_rank"] = int(cfg.backbone.volterra_rank)
        elif f in ("actions.order", "actions_order"):
            payload["actions.order"] = list(ACTIONS_ORDER)
        else:
            payload[f] = "UNSUPPORTED_FIELD"
    return payload


# ============================================================
# Datasets (STRICT pairing only)
# ============================================================
class FolderPairsDataset(Dataset):
    """
    input_dir와 gt_dir 이미지를 relpath(noext) key로 strict 매칭.
    zip fallback 없음. 누락 있으면 즉시 에러.
    """

    def __init__(self, input_dir: str, gt_dir: str, patch: int = 256, repeat: int = 1, strict: bool = True):
        self.input_dir = os.path.normpath(input_dir)
        self.gt_dir = os.path.normpath(gt_dir)

        self.input_paths = _list_images(self.input_dir)
        self.gt_paths = _list_images(self.gt_dir)

        if len(self.input_paths) == 0 or len(self.gt_paths) == 0:
            raise FileNotFoundError(f"Empty dataset.\ninput_dir={self.input_dir}\ngt_dir={self.gt_dir}")

        def make_key(path: str, root: str) -> str:
            rel = os.path.relpath(path, root).replace("\\", "/")
            return os.path.splitext(rel)[0]

        gt_map: Dict[str, str] = {}
        gt_dup = 0
        for p in self.gt_paths:
            k = make_key(p, self.gt_dir)
            if k in gt_map:
                gt_dup += 1
            gt_map[k] = p

        pairs: List[Tuple[str, str]] = []
        miss = 0
        for ip in self.input_paths:
            k = make_key(ip, self.input_dir)
            gp = gt_map.get(k, None)
            if gp is None:
                miss += 1
                continue
            pairs.append((ip, gp))

        match_ratio = len(pairs) / max(1, len(self.input_paths))
        print(
            f"[FolderPairsDataset] input={len(self.input_paths)} gt={len(self.gt_paths)} "
            f"matched={len(pairs)} miss={miss} (match_ratio={match_ratio:.4f})"
        )
        print(f"  - input_dir={self.input_dir}")
        print(f"  - gt_dir={self.gt_dir}")
        if gt_dup > 0:
            print(f"  - [WARN] gt key collisions (overwrites) = {gt_dup}")

        if strict and miss > 0:
            bad = []
            for ip in self.input_paths:
                k = make_key(ip, self.input_dir)
                if k not in gt_map:
                    bad.append((ip, k))
                if len(bad) >= 10:
                    break
            msg = "\n".join([f"    - key={k} ip={ip}" for ip, k in bad])
            raise RuntimeError(
                "[StrictPairing] FolderPairsDataset has missing pairs.\n"
                f"matched={len(pairs)} / input={len(self.input_paths)}\n"
                f"First missing examples:\n{msg}\n"
                "Fix pairing rule or dataset structure."
            )

        self.pairs = pairs
        self.patch = int(patch)
        self.repeat = max(1, int(repeat))

    def __len__(self) -> int:
        return len(self.pairs) * self.repeat

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ip, gp = self.pairs[idx % len(self.pairs)]
        lq = _read_rgb(ip)
        gt = _read_rgb(gp)
        lq, gt = _rand_crop_pair(lq, gt, self.patch)
        return {"lq": lq, "gt": gt}


class CSVPairsDataset(Dataset):
    """CSV 2열 (input,gt) strict."""

    def __init__(self, csv_path: str, base_dir: str, patch: int = 256, repeat: int = 1):
        import csv

        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        self.patch = int(patch)
        self.repeat = max(1, int(repeat))
        self.pairs: List[Tuple[str, str]] = []

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        start = 0
        if rows:
            head = [c.lower() for c in rows[0]]
            if any(("blur" in c) or ("sharp" in c) or ("input" in c) or ("gt" in c) for c in head):
                start = 1

        bad = 0
        for r in rows[start:]:
            if len(r) < 2:
                bad += 1
                continue
            a, b = r[0].strip(), r[1].strip()
            if not a or not b:
                bad += 1
                continue
            if not os.path.isabs(a):
                a = os.path.normpath(os.path.join(base_dir, a))
            if not os.path.isabs(b):
                b = os.path.normpath(os.path.join(base_dir, b))
            if (not os.path.exists(a)) or (not os.path.exists(b)):
                bad += 1
                continue
            self.pairs.append((a, b))

        print(f"[CSVPairsDataset] pairs={len(self.pairs)} bad={bad} csv={os.path.normpath(csv_path)}")
        if len(self.pairs) == 0:
            raise RuntimeError(f"No valid pairs parsed from csv: {csv_path}")

    def __len__(self) -> int:
        return len(self.pairs) * self.repeat

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ip, gp = self.pairs[idx % len(self.pairs)]
        lq = _read_rgb(ip)
        gt = _read_rgb(gp)
        lq, gt = _rand_crop_pair(lq, gt, self.patch)
        return {"lq": lq, "gt": gt, "meta": {"type": "csv", "ip": ip, "gp": gp}}


class MixedDataset(Dataset):
    """단순 concat (길이 비례 샘플링)"""

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.total = sum(self.lengths)
        if self.total <= 0:
            raise ValueError("Empty mixed dataset")

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        r = idx % self.total
        acc = 0
        for d, L in zip(self.datasets, self.lengths):
            if r < acc + L:
                return d[r - acc]
            acc += L
        return self.datasets[-1][0]


# ============================================================
# YAML-driven dataset builder (HIDE 제외)
# ============================================================
def _join(root: str, rel: str) -> str:
    return os.path.normpath(os.path.join(root, rel))


def build_train_dataset_from_cfg(cfg, *, strict_pairing: bool = True) -> Tuple[Dataset, List[Dict[str, Any]]]:
    """
    cfg.data_root + cfg.datasets.backbone_mix.train[] 를 기반으로 Dataset 구성.
    ✅ HIDE_dataset 은 train에서 자동 제외 (test-only)
    """
    data_root = str(cfg.data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"cfg.data_root not found: {data_root}")

    if cfg.datasets is None or "backbone_mix" not in cfg.datasets:
        raise RuntimeError("cfg.datasets.backbone_mix not found. Add it to YAML.")
    bm = cfg.datasets["backbone_mix"]
    if not isinstance(bm, dict) or "train" not in bm:
        raise RuntimeError("cfg.datasets.backbone_mix.train not found. Add it to YAML.")

    specs = list(bm["train"])
    if len(specs) == 0:
        raise RuntimeError("cfg.datasets.backbone_mix.train is empty")

    patch = int(cfg.backbone.patch)
    repeat_each = int(getattr(cfg, "repeat_each", 1))

    sub: List[Dataset] = []
    resolved_specs: List[Dict[str, Any]] = []

    for sd in specs:
        if not isinstance(sd, dict):
            raise TypeError(f"Each dataset spec must be dict, got: {type(sd)}")

        name = str(sd.get("name", "UNKNOWN"))

        if name == "HIDE_dataset":
            print("[SkipTrain] HIDE_dataset is test-only -> skipped from backbone training mix.")
            continue

        if sd.get("csv", ""):
            csv_path = _join(data_root, str(sd["csv"]))
            base_dir = os.path.dirname(csv_path)
            sub.append(CSVPairsDataset(csv_path=csv_path, base_dir=base_dir, patch=patch, repeat=repeat_each))
            resolved_specs.append({"name": name, "type": "csv", "csv": csv_path})
            continue

        in_rel = sd.get("input_dir", None)
        gt_rel = sd.get("gt_dir", None)
        if not in_rel or not gt_rel:
            raise RuntimeError(f"Dataset spec missing input_dir/gt_dir (or csv): {sd}")

        inp = _join(data_root, str(in_rel))
        gt = _join(data_root, str(gt_rel))
        sub.append(FolderPairsDataset(inp, gt, patch=patch, repeat=repeat_each, strict=strict_pairing))
        resolved_specs.append({"name": name, "type": "folder", "input_dir": inp, "gt_dir": gt})

    if len(sub) == 0:
        raise RuntimeError("No training datasets constructed (maybe all were skipped). Check YAML.")

    train_set: Dataset = sub[0] if len(sub) == 1 else MixedDataset(sub)
    return train_set, resolved_specs


# ============================================================
# Sanity dump (lq|gt)
# ============================================================
@torch.no_grad()
def dump_pair_sanity(loader: DataLoader, out_dir: str, n: int = 20):
    os.makedirs(out_dir, exist_ok=True)
    cnt = 0
    for batch in loader:
        lq = batch["lq"]
        gt = batch["gt"]
        b = int(lq.shape[0])
        for i in range(b):
            lq_i = lq[i]
            gt_i = gt[i]
            outp = os.path.join(out_dir, f"sanity_{cnt:03d}.png")
            pred_dummy = torch.zeros_like(lq_i)
            save_triplet_with_text(
                lq_i, pred_dummy, gt_i, outp,
                title_left="LQ", title_mid="(dummy)", title_right="GT",
                text_lines=["SANITY DUMP", ""],
            )
            cnt += 1
            if cnt >= n:
                print(f"[SanityDump] saved {cnt} images -> {out_dir}")
                return
    print(f"[SanityDump] saved {cnt} images -> {out_dir}")


# ============================================================
# Training
# ============================================================
def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    m = int(sec // 60)
    s = int(sec - 60 * m)
    h = int(m // 60)
    m = int(m - 60 * h)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _build_vetnet_model(in_chans: int, dim: int, bias: bool, volterra_rank: int, device: torch.device):
    """
    ✅ IMPORTANT:
    이전 코드(train_vetnet_backbone_cache.py)와 동일한 생성자 호출을 우선 사용.
    (dim/bias/volterra_rank만 넣는 방식)
    """
    from models.backbone.vetnet import VETNet

    # try previous signature first
    try:
        m = VETNet(dim=dim, bias=bias, volterra_rank=volterra_rank).to(device)
        print("[ModelInit] VETNet(dim,bias,volterra_rank) (same as previous)")
        return m
    except TypeError as e1:
        # fallback to explicit in/out if required by this codebase
        try:
            m = VETNet(
                in_channels=in_chans,
                out_channels=in_chans,
                dim=dim,
                bias=bias,
                volterra_rank=volterra_rank,
            ).to(device)
            print("[ModelInit] VETNet(in_channels,out_channels,dim,bias,volterra_rank) (fallback)")
            return m
        except TypeError as e2:
            raise RuntimeError(
                "VETNet constructor signature mismatch.\n"
                f"try1 failed: {repr(e1)}\n"
                f"try2 failed: {repr(e2)}\n"
                "Check models/backbone/vetnet.py signature."
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="E:/VETAgent/configs/vetagent/vetagent.yaml")

    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--results_dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=0)

    # SSOT mismatch fail
    ap.add_argument("--patch", type=int, default=None)
    ap.add_argument("--in_chans", type=int, default=None)

    # saving + metrics
    ap.add_argument("--iter_save_interval", type=int, default=200)
    ap.add_argument("--metric_every", type=int, default=200)          # ✅ same cadence as previous
    ap.add_argument("--metric_images_per_batch", type=int, default=1) # ✅ same spirit as previous

    # sanity
    ap.add_argument("--sanity_only", type=int, default=0)
    ap.add_argument("--sanity_dump", type=int, default=20)

    args = ap.parse_args()

    _ensure_project_root_on_path()
    cfg = load_cfg(args.cfg, mode="train_backbone")

    proj = cfg.project if isinstance(cfg.project, dict) else {}
    seed = int(proj.get("seed", 123))
    version = str(proj.get("version", "agent_v1"))
    _set_seed(seed)

    device_str = str(cfg.runtime.device) if hasattr(cfg, "runtime") else "cuda"
    device = torch.device(device_str if (torch.cuda.is_available() and str(device_str).startswith("cuda")) else "cpu")

    use_amp = bool(getattr(cfg.runtime, "use_amp", True)) and device.type == "cuda"
    channels_last = bool(getattr(cfg.runtime, "channels_last", False))
    tf32 = bool(getattr(cfg.runtime, "tf32", False)) and device.type == "cuda"

    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    cfg_patch = int(cfg.backbone.patch)
    cfg_in = int(cfg.backbone.in_chans)

    if args.patch is not None and int(args.patch) != cfg_patch:
        raise RuntimeError(f"CLI patch mismatch: {args.patch} vs cfg.backbone.patch={cfg_patch}")
    if args.in_chans is not None and int(args.in_chans) != cfg_in:
        raise RuntimeError(f"CLI in_chans mismatch: {args.in_chans} vs cfg.backbone.in_chans={cfg_in}")

    patch = cfg_patch
    in_chans = cfg_in

    save_dir = args.save_dir
    if save_dir is None:
        if isinstance(cfg.ckpt, dict) and "backbone_dir" in cfg.ckpt:
            save_dir = str(cfg.ckpt["backbone_dir"])
        else:
            save_dir = "E:/VETAgent/checkpoints/backbone"
    os.makedirs(save_dir, exist_ok=True)

    results_dir = args.results_dir or "E:/VETAgent/results/backbone_train"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "iter"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "sanity"), exist_ok=True)

    epochs = int(args.epochs) if args.epochs is not None else int(proj.get("epochs", 100))

    if str(getattr(cfg.backbone, "name", "VETNet")) != "VETNet":
        raise RuntimeError(f"Unsupported backbone.name: {getattr(cfg.backbone,'name',None)} (expected 'VETNet')")

    print(f"[Device] {device} | amp={use_amp} channels_last={channels_last} tf32={tf32} | skimage={USE_SKIMAGE}")
    print(f"[SSOT] patch={patch} in_chans={in_chans} | save_dir={os.path.normpath(save_dir)}")
    print("[TrainMix] HIDE_dataset is excluded (test-only).")

    # model (IMPORTANT: prefer previous init signature)
    model = _build_vetnet_model(
        in_chans=in_chans,
        dim=int(cfg.backbone.dim),
        bias=bool(cfg.backbone.bias),
        volterra_rank=int(cfg.backbone.volterra_rank),
        device=device,
    )

    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    # dataset
    train_set, resolved_specs = build_train_dataset_from_cfg(cfg, strict_pairing=True)
    loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )

    if int(args.sanity_only) == 1:
        out_dir = os.path.join(results_dir, "sanity")
        dump_pair_sanity(loader, out_dir=out_dir, n=int(args.sanity_dump))
        return

    # optim
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), betas=(0.9, 0.999), weight_decay=0.0)
    loss_fn = nn.L1Loss()

    from torch.amp import autocast, GradScaler
    scaler = GradScaler("cuda", enabled=use_amp)

    # fingerprint/meta
    fp_payload = _fingerprint_payload_from_cfg(cfg)
    backbone_fp = _sha256_of_json(fp_payload)

    meta_base: Dict[str, Any] = {
        "type": "backbone",
        "backbone_name": "VETNet",
        "version": version,
        "fingerprint": backbone_fp,
        "fingerprint_payload": fp_payload,
        "dim": int(cfg.backbone.dim),
        "bias": int(cfg.backbone.bias),
        "volterra_rank": int(cfg.backbone.volterra_rank),
        "actions_order": list(ACTIONS_ORDER),
        "in_chans": int(in_chans),
        "patch": int(patch),
        "train_info": {
            "dataset_mix": resolved_specs,
            "data_root": os.path.normpath(str(cfg.data_root)),
            "epochs": int(epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "num_workers": int(args.num_workers),
            "use_amp": bool(use_amp),
            "channels_last": bool(channels_last),
            "tf32": bool(tf32),
            "seed": int(seed),
            "iter_save_interval": int(args.iter_save_interval),
            "metric_every": int(args.metric_every),
            "metric_images_per_batch": int(args.metric_images_per_batch),
            "hide_in_train": False,
        },
    }

    global_step = 0
    steps_per_epoch = len(loader)
    print(f"[Train] steps_per_epoch={steps_per_epoch} | epochs={epochs}")

    # running averages (loss always, metrics only when computed)
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        loss_sum = 0.0
        metric_psnr_sum = 0.0
        metric_ssim_sum = 0.0
        metric_cnt = 0

        pbar = tqdm(loader, ncols=140, desc=f"Epoch {epoch:03d}/{epochs:03d}")
        for it, batch in enumerate(pbar, start=1):
            global_step += 1

            lq = batch["lq"].to(device, non_blocking=True).clamp(0.0, 1.0)
            gt = batch["gt"].to(device, non_blocking=True).clamp(0.0, 1.0)

            if channels_last and device.type == "cuda":
                lq = lq.contiguous(memory_format=torch.channels_last)
                gt = gt.contiguous(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                pred = model(lq)
                # previous code used pred for loss directly; clamp only for metrics/saving
                loss = loss_fn(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_sum += float(loss.item())

            # metrics computed only every metric_every steps (same as previous style)
            pred_c = pred.detach().clamp(0, 1)

            if USE_SKIMAGE and int(args.metric_every) > 0 and (global_step % int(args.metric_every) == 0 or it == steps_per_epoch):
                ps, ss = compute_psnr_ssim_skimage(
                    pred_c.detach(),
                    gt.detach(),
                    max_images=int(args.metric_images_per_batch),
                )
                metric_psnr_sum += ps
                metric_ssim_sum += ss
                metric_cnt += 1

            elapsed = time.time() - t0
            itps = it / max(elapsed, 1e-6)
            eta = (steps_per_epoch - it) / max(itps, 1e-6)

            avg_loss = loss_sum / max(1, it)
            avg_psnr = (metric_psnr_sum / metric_cnt) if metric_cnt > 0 else 0.0
            avg_ssim = (metric_ssim_sum / metric_cnt) if metric_cnt > 0 else 0.0

            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "psnr": f"{avg_psnr:.2f}" if USE_SKIMAGE else "NA",
                    "ssim": f"{avg_ssim:.4f}" if USE_SKIMAGE else "NA",
                    "lr": f"{opt.param_groups[0]['lr']:.2e}",
                    "ETA": _format_time(eta),
                }
            )

            # ---- iter save (triplet + skimage text) ----
            if int(args.iter_save_interval) > 0 and (global_step % int(args.iter_save_interval) == 0):
                with torch.no_grad():
                    lq0 = lq[0].detach().cpu()
                    gt0 = gt[0].detach().cpu()
                    pr0 = pred_c[0].detach().cpu()

                    if USE_SKIMAGE:
                        p_one, s_one = compute_psnr_ssim_skimage(
                            pr0.unsqueeze(0), gt0.unsqueeze(0), max_images=1
                        )
                        line1 = f"PSNR={p_one:.2f}  SSIM={s_one:.4f}"
                    else:
                        line1 = "PSNR/SSIM: skimage OFF"

                    outp = os.path.join(results_dir, "iter", f"epoch_{epoch:03d}_gs_{global_step:08d}.png")
                    save_triplet_with_text(
                        lq0, pr0, gt0, outp,
                        title_left="LQ", title_mid="Pred", title_right="GT",
                        text_lines=[line1, f"epoch={epoch}  step={global_step}"],
                    )

        # epoch summary (use metric avg if computed)
        epoch_loss = loss_sum / max(1, steps_per_epoch)
        epoch_psnr = (metric_psnr_sum / metric_cnt) if metric_cnt > 0 else 0.0
        epoch_ssim = (metric_ssim_sum / metric_cnt) if metric_cnt > 0 else 0.0

        print(
            f"\n[Epoch {epoch:03d}] "
            f"Loss={epoch_loss:.4f} "
            f"PSNR={epoch_psnr:.2f} "
            f"SSIM={epoch_ssim:.4f} "
            f"time={_format_time(time.time() - t0)}"
        )

        ckpt = {
            "state_dict": model.state_dict(),
            "meta": meta_base,
            "epoch": int(epoch),
            "metrics": {"L": float(epoch_loss), "P": float(epoch_psnr), "S": float(epoch_ssim)},
        }
        out_name = f"epoch_{epoch:03d}_L{epoch_loss:.4f}_P{epoch_psnr:.2f}_S{epoch_ssim:.4f}.pth"
        out_path = os.path.join(save_dir, out_name)
        torch.save(ckpt, out_path)
        print(f"[CKPT] saved: {out_path}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()

"""
cd E:\VETAgent
python -m script.train_backbone_vetnet --cfg E:/VETAgent/configs/vetagent/vetagent.yaml `
  --epochs 50 --batch_size 1 --num_workers 4 `
  --metric_every 200 --metric_images 1 `
  --iter_save_interval 200

"""