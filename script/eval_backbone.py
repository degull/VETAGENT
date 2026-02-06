from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


# ------------------------------------------------------------
# Import-safe project root
# ------------------------------------------------------------
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vetagent.config import load_cfg


# -------------------------
# Utils
# -------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(p: str) -> bool:
    return os.path.splitext(p.lower())[1] in IMG_EXTS


def pil_read_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def to_tensor_uint8_rgb(x_u8: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(x_u8).permute(2, 0, 1).float() / 255.0  # CHW
    return x.unsqueeze(0)  # 1CHW


def tensor_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    if x.dim() == 4:
        x = x[0]
    x = x.clamp(0, 1).detach().cpu()
    x_u8 = (x * 255.0).round().to(torch.uint8)
    return x_u8.permute(1, 2, 0).numpy()


def pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    pad_top, pad_left = 0, 0
    pad_bottom, pad_right = pad_h, pad_w
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    return x_pad, (pad_left, pad_right, pad_top, pad_bottom)


def unpad(x: torch.Tensor, pads: Tuple[int, int, int, int]) -> torch.Tensor:
    pl, pr, pt, pb = pads
    if (pl, pr, pt, pb) == (0, 0, 0, 0):
        return x
    _, _, h, w = x.shape
    return x[:, :, pt : h - pb, pl : w - pr]


def compute_metrics(pred_u8: np.ndarray, gt_u8: np.ndarray) -> Tuple[float, float]:
    psnr = float(_psnr(gt_u8, pred_u8, data_range=255))
    ssim = float(_ssim(gt_u8, pred_u8, channel_axis=2, data_range=255))
    return psnr, ssim


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def relpath_noext(path: str, root: str) -> str:
    rp = os.path.relpath(path, root).replace("\\", "/")
    return os.path.splitext(rp)[0]


# -------------------------
# Pair builders (match your YAML dataset entries)
# -------------------------
def build_pairs_folder_match(input_dir: str, gt_dir: str) -> List[Tuple[str, str]]:
    in_files = []
    for r, _, fs in os.walk(input_dir):
        for f in fs:
            p = os.path.join(r, f)
            if is_image_file(p):
                in_files.append(p)

    gt_map: Dict[str, str] = {}
    for r, _, fs in os.walk(gt_dir):
        for f in fs:
            p = os.path.join(r, f)
            if is_image_file(p):
                gt_map[relpath_noext(p, gt_dir)] = p

    pairs = []
    miss = 0
    for ip in sorted(in_files):
        key = relpath_noext(ip, input_dir)
        gp = gt_map.get(key, None)
        if gp is None:
            miss += 1
            continue
        pairs.append((ip, gp))

    if len(pairs) == 0:
        raise RuntimeError(f"[Pairs] no matched pairs. input_dir={input_dir}, gt_dir={gt_dir}")
    if miss > 0:
        print(f"[Pairs] WARN: {miss} inputs had no GT match (ignored).")
    return pairs


def build_pairs_gopro_csv(csv_path: str, data_root: str) -> List[Tuple[str, str]]:
    import csv

    csv_abspath = csv_path if os.path.isabs(csv_path) else os.path.join(data_root, csv_path)
    if not os.path.exists(csv_abspath):
        raise FileNotFoundError(csv_abspath)

    pairs = []
    with open(csv_abspath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        has_header = False
        if header and any(h.lower() in ("input", "gt", "blur", "sharp") for h in header):
            has_header = True
            cols = [h.strip().lower() for h in header]
        else:
            has_header = False
            cols = []

        def norm(p: str) -> str:
            p = p.strip()
            if not p:
                return p
            if os.path.isabs(p):
                return p
            return os.path.join(data_root, p)

        if not has_header:
            row = header
            if row and len(row) >= 2:
                pairs.append((norm(row[0]), norm(row[1])))
            for row in reader:
                if len(row) < 2:
                    continue
                pairs.append((norm(row[0]), norm(row[1])))
        else:
            idx_in = cols.index("input") if "input" in cols else (cols.index("blur") if "blur" in cols else None)
            idx_gt = cols.index("gt") if "gt" in cols else (cols.index("sharp") if "sharp" in cols else None)
            if idx_in is None or idx_gt is None:
                raise RuntimeError(f"[GOPRO CSV] header must contain (input,gt) or (blur,sharp). got={cols}")

            for row in reader:
                if len(row) <= max(idx_in, idx_gt):
                    continue
                pairs.append((norm(row[idx_in]), norm(row[idx_gt])))

    pairs = [(i, g) for (i, g) in pairs if os.path.exists(i) and os.path.exists(g)]
    if len(pairs) == 0:
        raise RuntimeError(f"[GOPRO CSV] no valid pairs found from {csv_abspath}")
    return pairs


def build_pairs_hide_txt(txt_path: str, input_root: str, gt_root: str, data_root: str) -> List[Tuple[str, str]]:
    txt_abspath = txt_path if os.path.isabs(txt_path) else os.path.join(data_root, txt_path)
    in_root_abs = input_root if os.path.isabs(input_root) else os.path.join(data_root, input_root)
    gt_root_abs = gt_root if os.path.isabs(gt_root) else os.path.join(data_root, gt_root)

    if not os.path.exists(txt_abspath):
        raise FileNotFoundError(txt_abspath)

    gt_files = []
    for r, _, fs in os.walk(gt_root_abs):
        for f in fs:
            p = os.path.join(r, f)
            if is_image_file(p):
                gt_files.append(p)
    gt_by_base = {os.path.basename(p): p for p in gt_files}

    pairs = []
    with open(txt_abspath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            cand_in = s if os.path.isabs(s) else os.path.join(in_root_abs, s)
            if not os.path.exists(cand_in):
                cand_in = os.path.join(in_root_abs, os.path.basename(s))
            if not os.path.exists(cand_in):
                continue

            cand_gt = s if os.path.isabs(s) else os.path.join(gt_root_abs, s)
            if not os.path.exists(cand_gt):
                cand_gt = os.path.join(gt_root_abs, os.path.basename(s))
            if not os.path.exists(cand_gt):
                cand_gt = gt_by_base.get(os.path.basename(cand_in), None)
            if cand_gt is None or not os.path.exists(cand_gt):
                continue

            pairs.append((cand_in, cand_gt))

    if len(pairs) == 0:
        raise RuntimeError(f"[HIDE TXT] no valid pairs found from {txt_abspath}")
    return pairs


# -------------------------
# Dataset
# -------------------------
class PairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, gp = self.pairs[idx]
        inp = pil_read_rgb(ip)
        gt = pil_read_rgb(gp)
        return ip, inp, gt


# -------------------------
# Model build (match training)
# -------------------------
def build_vetnet_model(in_chans: int, dim: int, bias: int, volterra_rank: int, device: torch.device):
    from models.backbone.vetnet import VETNet

    # train_backbone_vetnet.py와 동일 우선순위
    try:
        m = VETNet(dim=dim, bias=bool(bias), volterra_rank=volterra_rank).to(device)
        print("[ModelInit] VETNet(dim,bias,volterra_rank)")
        return m
    except TypeError as e1:
        try:
            m = VETNet(
                in_channels=in_chans,
                out_channels=in_chans,
                dim=dim,
                bias=bool(bias),
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


def load_backbone_weights(model: torch.nn.Module, ckpt_path: str, device: str) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)

    # 너 ckpt 포맷: {"state_dict":..., "meta":..., "epoch":..., "metrics":...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        meta = ckpt.get("meta", {})
        metrics = ckpt.get("metrics", {})
    else:
        sd = ckpt
        meta = {}
        metrics = {}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[CKPT] loaded: {ckpt_path}")
    print(f"[CKPT] load_state_dict strict=False")
    if missing:
        print(f"  missing: {len(missing)}")
    if unexpected:
        print(f"  unexpected: {len(unexpected)}")

    return {"meta": meta, "metrics": metrics}


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def eval_pairs(
    model: torch.nn.Module,
    pairs: List[Tuple[str, str]],
    device: str,
    use_amp: bool,
    pad_multiple: int,
    save_dir: Optional[str],
    max_items: int = -1,
) -> Dict[str, float]:
    model.eval()

    ds = PairDataset(pairs if max_items <= 0 else pairs[:max_items])
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    psnrs: List[float] = []
    ssims: List[float] = []

    if save_dir is not None:
        ensure_dir(save_dir)

    for i, (ip, inp_u8, gt_u8) in enumerate(dl):
        ip = ip[0]
        inp_u8 = np.array(inp_u8[0], dtype=np.uint8)
        gt_u8 = np.array(gt_u8[0], dtype=np.uint8)

        x = to_tensor_uint8_rgb(inp_u8).to(device)

        x, pads = pad_to_multiple(x, multiple=pad_multiple)

        if use_amp and device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                y = model(x)
        else:
            y = model(x)

        y = unpad(y, pads)
        pred_u8 = tensor_to_uint8_rgb(y)

        psnr, ssim = compute_metrics(pred_u8, gt_u8)
        psnrs.append(psnr)
        ssims.append(ssim)

        if save_dir is not None:
            base = os.path.splitext(os.path.basename(ip))[0]
            out_path = os.path.join(save_dir, f"{i:05d}_{base}_pred.png")
            Image.fromarray(pred_u8).save(out_path)

    return {
        "count": float(len(psnrs)),
        "psnr_avg": float(np.mean(psnrs)) if psnrs else 0.0,
        "ssim_avg": float(np.mean(ssims)) if ssims else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="E:/VETAgent/configs/vetagent/vetagent.yaml")
    ap.add_argument("--ckpt", type=str, required=True, help="Backbone checkpoint path (.pth)")
    ap.add_argument("--mode", type=str, default="train_backbone", choices=["train_backbone", "agent"])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--pad_multiple", type=int, default=8)
    ap.add_argument("--save_dir", type=str, default="E:/VETAgent/results/eval_backbone")
    ap.add_argument("--max_items", type=int, default=-1, help="<=0 means all")
    args = ap.parse_args()

    cfg = load_cfg(args.cfg, mode=args.mode)

    device = args.device or str(cfg.runtime.device)
    use_amp = bool(cfg.runtime.use_amp)

    # ✅ model_spec 우선, 없으면 backbone에서 fallback
    if hasattr(cfg, "model_spec") and cfg.model_spec is not None:
        dim = int(cfg.model_spec.dim)
        bias = int(cfg.model_spec.bias)
        volterra_rank = int(cfg.model_spec.volterra_rank)
        in_chans = int(cfg.model_spec.in_chans)
        patch = int(cfg.model_spec.patch)
    else:
        dim = int(getattr(cfg.backbone, "dim"))
        bias = int(getattr(cfg.backbone, "bias"))
        volterra_rank = int(getattr(cfg.backbone, "volterra_rank"))
        in_chans = int(getattr(cfg.backbone, "in_chans", 3))
        patch = int(getattr(cfg.backbone, "patch", 256))

    print(f"[CFG] dim={dim} bias={bias} volterra_rank={volterra_rank} in_chans={in_chans} patch={patch}")

    dev = torch.device(device if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu")

    model = build_vetnet_model(in_chans=in_chans, dim=dim, bias=bias, volterra_rank=volterra_rank, device=dev)

    if bool(cfg.runtime.channels_last) and str(dev).startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)

    ckpt_info = load_backbone_weights(model, args.ckpt, device=str(dev))

    # Prepare eval datasets from cfg.datasets.backbone_mix.train
    if cfg.datasets is None or "backbone_mix" not in cfg.datasets or "train" not in cfg.datasets["backbone_mix"]:
        raise RuntimeError("cfg.datasets.backbone_mix.train not found in YAML")

    train_list = cfg.datasets["backbone_mix"]["train"]
    if not isinstance(train_list, list):
        raise TypeError("datasets.backbone_mix.train must be a list")

    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_root = os.path.join(args.save_dir, ckpt_name)
    ensure_dir(out_root)

    results = {
        "ckpt": args.ckpt,
        "ckpt_name": ckpt_name,
        "device": str(dev),
        "use_amp": bool(use_amp and str(dev).startswith("cuda")),
        "pad_multiple": args.pad_multiple,
        "cfg_model": {"dim": dim, "bias": bias, "volterra_rank": volterra_rank, "in_chans": in_chans, "patch": patch},
        "ckpt_metrics": ckpt_info.get("metrics", {}),
        "datasets": {},
    }

    overall_counts = 0
    overall_psnr_sum = 0.0
    overall_ssim_sum = 0.0

    for entry in train_list:
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        name = str(entry["name"])
        print(f"\n[Eval] Dataset: {name}")

        if "csv" in entry:
            pairs = build_pairs_gopro_csv(str(entry["csv"]), cfg.data_root)
        elif "txt" in entry:
            pairs = build_pairs_hide_txt(
                txt_path=str(entry["txt"]),
                input_root=str(entry["input_root"]),
                gt_root=str(entry["gt_root"]),
                data_root=cfg.data_root,
            )
        else:
            input_dir = os.path.join(cfg.data_root, str(entry["input_dir"]))
            gt_dir = os.path.join(cfg.data_root, str(entry["gt_dir"]))
            pairs = build_pairs_folder_match(input_dir, gt_dir)

        ds_out = os.path.join(out_root, name)
        metrics = eval_pairs(
            model=model,
            pairs=pairs,
            device=str(dev),
            use_amp=bool(use_amp and str(dev).startswith("cuda")),
            pad_multiple=args.pad_multiple,
            save_dir=ds_out,
            max_items=args.max_items,
        )

        results["datasets"][name] = {
            "count": int(metrics["count"]),
            "psnr_avg": metrics["psnr_avg"],
            "ssim_avg": metrics["ssim_avg"],
        }

        print(f"[Eval:{name}] N={int(metrics['count'])}  PSNR={metrics['psnr_avg']:.4f}  SSIM={metrics['ssim_avg']:.4f}")

        n = int(metrics["count"])
        overall_counts += n
        overall_psnr_sum += metrics["psnr_avg"] * n
        overall_ssim_sum += metrics["ssim_avg"] * n

    overall = {
        "count": overall_counts,
        "psnr_avg": (overall_psnr_sum / overall_counts) if overall_counts > 0 else 0.0,
        "ssim_avg": (overall_ssim_sum / overall_counts) if overall_counts > 0 else 0.0,
    }
    results["overall"] = overall

    print("\n==========================")
    print(f"[Overall] N={overall['count']}  PSNR={overall['psnr_avg']:.4f}  SSIM={overall['ssim_avg']:.4f}")
    print("==========================\n")

    json_path = os.path.join(out_root, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {json_path}")
    print(f"[Saved] preds under: {out_root}")


if __name__ == "__main__":
    main()
