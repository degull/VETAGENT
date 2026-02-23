# ------------------------------------------------------------
# VETNet Backbone Inference (single image / folder)
# - strict=True load (fail fast)
# - pad-to-multiple (default=8) to avoid seam from multi-scale U-Net
# - optional force_fp32 to avoid AMP instability
# ------------------------------------------------------------

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        for name in files:
            if name.lower().endswith(IMG_EXT):
                out.append(os.path.join(root, name))
    out.sort()
    return out


def _read_rgb_01(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3,H,W)


def _to_u8_hwc(x_chw01: torch.Tensor) -> np.ndarray:
    x = x_chw01.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (x * 255.0).round().astype(np.uint8)


def _save_rgb_01(x_chw01: torch.Tensor, out_path: str):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(_to_u8_hwc(x_chw01)).save(out_path)


def _pad_to_multiple(x: torch.Tensor, mult: int, mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    x: (1,C,H,W)
    Returns padded_x and pad tuple (left,right,top,bottom) for unpad.
    """
    if mult <= 1:
        return x, (0, 0, 0, 0)
    _, _, H, W = x.shape
    newH = ((H + mult - 1) // mult) * mult
    newW = ((W + mult - 1) // mult) * mult
    pad_h = newH - H
    pad_w = newW - W
    # pad format: (left, right, top, bottom)
    pad = (0, pad_w, 0, pad_h)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, pad, mode=mode)
    return x, pad


def _unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pad
    if right == 0 and bottom == 0 and left == 0 and top == 0:
        return x
    _, _, H, W = x.shape
    return x[:, :, top : (H - bottom), left : (W - right)]


def _build_vetnet_model(in_chans: int, dim: int, bias: bool, volterra_rank: int, device: torch.device) -> nn.Module:
    _ensure_project_root_on_path()
    from models.backbone.vetnet import VETNet  # MUST match training import

    # Print which file is actually imported
    try:
        import inspect
        print("[VETNet FILE]", inspect.getfile(VETNet))
    except Exception:
        pass

    try:
        m = VETNet(dim=dim, bias=bias, volterra_rank=volterra_rank).to(device)
        print("[ModelInit] VETNet(dim,bias,volterra_rank)")
        return m
    except TypeError as e1:
        try:
            m = VETNet(
                in_channels=in_chans,
                out_channels=in_chans,
                dim=dim,
                bias=bias,
                volterra_rank=volterra_rank,
            ).to(device)
            print("[ModelInit] VETNet(in_channels,out_channels,...) (fallback)")
            return m
        except TypeError as e2:
            raise RuntimeError(
                "VETNet constructor signature mismatch.\n"
                f"try1 failed: {repr(e1)}\n"
                f"try2 failed: {repr(e2)}\n"
                "Check models/backbone/vetnet.py signature."
            )


def _torch_load_ckpt(path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location=str(device))
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unsupported ckpt format (expected dict): type={type(ckpt)}")
    return ckpt


def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # type: ignore
    raise RuntimeError("Could not find state_dict in checkpoint.")


def _load_checkpoint_strict(model: nn.Module, ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = _torch_load_ckpt(ckpt_path, device=device)
    sd = _extract_state_dict(ckpt)

    inc = model.load_state_dict(sd, strict=True)  # fail-fast
    missing = getattr(inc, "missing_keys", []) if inc is not None else []
    unexpected = getattr(inc, "unexpected_keys", []) if inc is not None else []
    if missing or unexpected:
        print(f"[CKPT] strict=True | missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  - missing (first 20):", missing[:20])
        if unexpected:
            print("  - unexpected (first 20):", unexpected[:20])
        raise RuntimeError("Checkpoint load mismatch under strict=True.")
    print(f"[CKPT] strict=True OK: {ckpt_path}")
    return ckpt


@torch.inference_mode()
def _infer_one(
    model: nn.Module,
    x_chw01: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
    force_fp32: bool,
    pad_mult: int,
    tile: int = 0,
    tile_overlap: int = 32,
) -> Tuple[torch.Tensor, float]:
    model.eval()

    x = x_chw01.unsqueeze(0).to(device, non_blocking=True)  # (1,3,H,W)

    if channels_last and device.type == "cuda":
        x = x.contiguous(memory_format=torch.channels_last)

    # ✅ 안정성: pad to multiple-of-8 by default
    x_pad, pad = _pad_to_multiple(x, mult=int(pad_mult), mode="reflect")

    from torch.amp import autocast

    def run_forward(inp: torch.Tensor) -> torch.Tensor:
        if force_fp32:
            inp = inp.float()
            model.float()
            # no autocast
            return model(inp)
        else:
            with autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                return model(inp)

    # ---- full inference (recommended first) ----
    if tile is None or int(tile) <= 0:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        y = run_forward(x_pad)

        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.time() - t0) * 1000.0

        y = y.clamp(0, 1)
        y = _unpad(y, pad)[0].detach().cpu()
        return y, ms

    # ---- tiled inference (optional) ----
    tile = int(tile)
    tile_overlap = max(0, int(tile_overlap))
    stride = max(1, tile - tile_overlap)

    _, _, H, W = x_pad.shape
    out_acc = torch.zeros((1, 3, H, W), device=device, dtype=torch.float32)
    w_acc = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)

    def _hann2d(h: int, w: int) -> torch.Tensor:
        yy = torch.hann_window(h, periodic=False, device=device).view(h, 1)
        xx = torch.hann_window(w, periodic=False, device=device).view(1, w)
        return (yy @ xx).clamp_min(1e-6)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for top in range(0, H, stride):
        for left in range(0, W, stride):
            bottom = min(top + tile, H)
            right = min(left + tile, W)

            inp = x_pad[:, :, top:bottom, left:right]
            ph = tile - (bottom - top)
            pw = tile - (right - left)
            if ph > 0 or pw > 0:
                inp = F.pad(inp, (0, pw, 0, ph), mode="reflect")

            pred = run_forward(inp)
            pred = pred[:, :, : (bottom - top), : (right - left)]

            ww = _hann2d(pred.shape[-2], pred.shape[-1]).view(1, 1, pred.shape[-2], pred.shape[-1])
            out_acc[:, :, top:bottom, left:right] += pred.float() * ww
            w_acc[:, :, top:bottom, left:right] += ww

    out = out_acc / w_acc.clamp_min(1e-6)

    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000.0

    out = out.clamp(0, 1)
    out = _unpad(out, pad)[0].detach().cpu()
    return out, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)

    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--input_dir", type=str, default=None)

    ap.add_argument("--out_dir", type=str, required=True)

    # model spec (auto from ckpt.ssot if not set)
    ap.add_argument("--in_chans", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--bias", type=int, default=None)
    ap.add_argument("--volterra_rank", type=int, default=None)

    # runtime
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_amp", type=int, default=0)          # ✅ default OFF for stability
    ap.add_argument("--channels_last", type=int, default=0)
    ap.add_argument("--force_fp32", type=int, default=1)       # ✅ default ON (most stable)

    # padding
    ap.add_argument("--pad_mult", type=int, default=8, help="pad H/W to be multiple of this (default=8). set 1 to disable")

    # tiling
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--tile_overlap", type=int, default=32)

    ap.add_argument("--suffix", type=str, default="_vetnet.png")

    args = ap.parse_args()

    _ensure_project_root_on_path()

    if (args.input is None) == (args.input_dir is None):
        raise ValueError("Provide exactly one of: --input OR --input_dir")

    device = torch.device(args.device if (torch.cuda.is_available() and str(args.device).startswith("cuda")) else "cpu")
    use_amp = bool(int(args.use_amp) == 1)
    channels_last = bool(int(args.channels_last) == 1)
    force_fp32 = bool(int(args.force_fp32) == 1)

    os.makedirs(args.out_dir, exist_ok=True)

    # read ssot from ckpt
    ckpt_preview = _torch_load_ckpt(args.ckpt, device=torch.device("cpu"))
    ssot = ckpt_preview.get("ssot", {}) if isinstance(ckpt_preview, dict) else {}

    def _pick_int(cli_val: Optional[int], ssot_key: str, default: int) -> int:
        if cli_val is not None:
            return int(cli_val)
        v = ssot.get(ssot_key, None) if isinstance(ssot, dict) else None
        if v is None:
            return int(default)
        return int(v)

    in_chans = _pick_int(args.in_chans, "in_chans", 3)
    dim = _pick_int(args.dim, "dim", 64)
    bias = _pick_int(args.bias, "bias", 0)
    volterra_rank = _pick_int(args.volterra_rank, "volterra_rank", 2)

    print(f"[Device] {device} | amp={use_amp} channels_last={channels_last} force_fp32={force_fp32}")
    print(f"[ModelSpec] in_chans={in_chans} dim={dim} bias={bias} volterra_rank={volterra_rank}")
    print(f"[Pad] pad_mult={int(args.pad_mult)}")
    print(f"[IO] out_dir={os.path.normpath(args.out_dir)}")
    if isinstance(ssot, dict) and ssot:
        print(f"[CKPT SSOT] {ssot}")

    model = _build_vetnet_model(
        in_chans=in_chans,
        dim=dim,
        bias=bool(int(bias)),
        volterra_rank=volterra_rank,
        device=device,
    )

    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    _ = _load_checkpoint_strict(model, args.ckpt, device=device)

    # inputs
    if args.input is not None:
        paths = [args.input]
        base_root = os.path.dirname(os.path.abspath(args.input))
    else:
        paths = _list_images(args.input_dir)
        if len(paths) == 0:
            raise FileNotFoundError(f"No images found in input_dir: {args.input_dir}")
        base_root = os.path.abspath(args.input_dir)

    print(f"[Infer] num_images={len(paths)} | tile={int(args.tile)} overlap={int(args.tile_overlap)}")

    ok = 0
    for i, p in enumerate(paths, start=1):
        rel = os.path.relpath(p, base_root).replace("\\", "/")
        rel_noext = os.path.splitext(rel)[0]
        out_path = os.path.normpath(os.path.join(args.out_dir, rel_noext + str(args.suffix)))

        try:
            x = _read_rgb_01(p)
            y, ms = _infer_one(
                model=model,
                x_chw01=x,
                device=device,
                use_amp=use_amp,
                channels_last=channels_last,
                force_fp32=force_fp32,
                pad_mult=int(args.pad_mult),
                tile=int(args.tile),
                tile_overlap=int(args.tile_overlap),
            )
            _save_rgb_01(y, out_path)
            ok += 1
            print(f"[{i}/{len(paths)}] OK  {os.path.basename(p)} -> {out_path}  ({ms:.1f} ms)")
        except Exception as e:
            print(f"[{i}/{len(paths)}] FAIL {p} | {repr(e)}")

    print(f"\nDone. success={ok}/{len(paths)}")


if __name__ == "__main__":
    main()