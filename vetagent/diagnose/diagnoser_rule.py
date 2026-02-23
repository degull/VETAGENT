# D0: Rule/IQA 기반
# E:\VETAgent\vetagent\diagnose\diagnoser_rule.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------
# D0 Diagnoser (Rule/IQA based)
#  - scores: rain/snow/haze/blur/noise/jpeg/drop in [0,1]
#  - residual_map: (1,H,W) in [0,1]
#  - risk: scalar + components (no-ref proxies; optionally PIQ NIQE/BRISQUE)
# ------------------------------------------------------------


@dataclass
class D0Config:
    # score weights (you can tune)
    w_blur: float = 1.0
    w_noise: float = 1.0
    w_jpeg: float = 1.0
    w_haze: float = 1.0
    w_rain: float = 1.0
    w_snow: float = 1.0
    w_drop: float = 1.0

    # normalization / thresholds (rough defaults)
    # blur: lower gradient energy => blur higher
    blur_ref: float = 0.020  # typical normalized Tenengrad (depends on resolution)
    blur_scale: float = 0.020

    # noise: median absolute deviation of highpass
    noise_ref: float = 0.010
    noise_scale: float = 0.015

    # jpeg: blockiness measure
    jpeg_ref: float = 0.005
    jpeg_scale: float = 0.010

    # haze: dark channel prior mean
    haze_ref: float = 0.080
    haze_scale: float = 0.120

    # rain/snow/drop are heuristic; keep scales mild
    rain_ref: float = 0.030
    rain_scale: float = 0.050

    snow_ref: float = 0.020
    snow_scale: float = 0.040

    drop_ref: float = 0.015
    drop_scale: float = 0.030

    # residual map mix
    residual_hf_weight: float = 0.7
    residual_self_weight: float = 0.3

    # self-consistency uses a weak smoothing operator
    self_sigma: float = 1.2

    # risk weights
    risk_hf_weight: float = 0.50
    risk_clip_weight: float = 0.25
    risk_color_weight: float = 0.25

    # optional PIQ
    try_piq: bool = True


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def _to_gray(x: torch.Tensor) -> torch.Tensor:
    # x: (3,H,W) in [0,1]
    r, g, b = x[0:1], x[1:2], x[2:3]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b)


def _safe_float(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(x)


def _normalize_score(val: float, ref: float, scale: float) -> float:
    # score = sigmoid-like clamp on (val - ref)/scale
    # if val > ref => score higher
    if scale <= 1e-12:
        return float(max(0.0, min(1.0, val)))
    z = (val - ref) / scale
    # smooth step
    s = 1.0 / (1.0 + math.exp(-float(z)))
    return float(max(0.0, min(1.0, s)))


# -----------------------------
# Low-level feature metrics
# -----------------------------

@torch.no_grad()
def _tenengrad_sharpness(gray: torch.Tensor) -> float:
    """
    Tenengrad: mean of squared gradient magnitude (normalized).
    gray: (1,H,W) in [0,1]
    """
    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    g = gray.unsqueeze(0)  # (1,1,H,W)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    mag2 = gx * gx + gy * gy
    return _safe_float(mag2.mean().item())


@torch.no_grad()
def _highpass_mad(gray: torch.Tensor) -> float:
    """
    Noise proxy: MAD of high-frequency residual.
    """
    g = gray.unsqueeze(0)  # (1,1,H,W)
    # simple blur via avgpool
    blur = F.avg_pool2d(g, kernel_size=5, stride=1, padding=2)
    hp = (g - blur).abs().squeeze(0)  # (1,H,W)
    med = hp.median()
    mad = (hp - med).abs().median()
    return _safe_float(mad.item())


@torch.no_grad()
def _jpeg_blockiness(gray: torch.Tensor, block: int = 8) -> float:
    """
    JPEG blockiness proxy:
    compare differences across block boundaries (vertical/horizontal).
    """
    # gray: (1,H,W)
    g = gray[0]
    H, W = g.shape[-2], g.shape[-1]
    if H < block * 2 or W < block * 2:
        return 0.0

    # boundary indices
    ys = list(range(block, H, block))
    xs = list(range(block, W, block))
    if len(xs) == 0 or len(ys) == 0:
        return 0.0

    # vertical boundaries: |I[:,x]-I[:,x-1]|
    vb = []
    for x in xs:
        vb.append((g[:, x] - g[:, x - 1]).abs().mean())
    # horizontal boundaries
    hb = []
    for y in ys:
        hb.append((g[y, :] - g[y - 1, :]).abs().mean())

    vb_m = torch.stack(vb).mean() if vb else torch.tensor(0.0, device=g.device)
    hb_m = torch.stack(hb).mean() if hb else torch.tensor(0.0, device=g.device)
    return _safe_float(((vb_m + hb_m) * 0.5).item())


@torch.no_grad()
def _haze_dark_channel(x: torch.Tensor, win: int = 15) -> float:
    """
    Dark channel prior mean: higher => more haze-like (rough).
    x: (3,H,W) in [0,1]
    """
    # dark = min over RGB then min filter
    m = x.min(dim=0, keepdim=True)[0]  # (1,H,W)
    # min pooling (approx) by negative maxpool
    k = win
    pad = k // 2
    dark = -F.max_pool2d(-m.unsqueeze(0), kernel_size=k, stride=1, padding=pad).squeeze(0)  # (1,H,W)
    return _safe_float(dark.mean().item())


@torch.no_grad()
def _rain_streak_proxy(gray: torch.Tensor) -> float:
    """
    Rain proxy: directional high-frequency anisotropy.
    This is a cheap heuristic: strong vertical-ish gradients + highpass energy.
    """
    g = gray.unsqueeze(0)  # (1,1,H,W)
    # gradients
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(g, kx, padding=1).abs()
    gy = F.conv2d(g, ky, padding=1).abs()

    # rain streaks often produce stronger gradients in one direction
    anis = (gy.mean() / (gx.mean() + 1e-6)).clamp(0, 10)  # >1 means more vertical gradients
    # combine with highpass energy
    blur = F.avg_pool2d(g, 5, 1, 2)
    hp = (g - blur).abs().mean()
    val = (anis * hp).item()
    return _safe_float(float(val))


@torch.no_grad()
def _snow_spot_proxy(x: torch.Tensor) -> float:
    """
    Snow proxy: bright small blobs ratio (rough).
    x: (3,H,W) in [0,1]
    """
    gray = _to_gray(x)  # (1,H,W)
    # bright pixels ratio after mild blur
    g = gray.unsqueeze(0)  # (1,1,H,W)
    blur = F.avg_pool2d(g, 5, 1, 2).squeeze(0)  # (1,H,W)
    # threshold on brightness
    thr = 0.75
    ratio = (blur > thr).float().mean().item()
    # penalize if it's globally bright (snow scene) using contrast:
    std = blur.std().item()
    val = ratio * max(0.0, min(1.0, float(std / 0.25)))
    return _safe_float(val)


@torch.no_grad()
def _drop_raindrop_proxy(x: torch.Tensor) -> float:
    """
    Raindrop proxy: localized blur + strong specular highlights.
    Cheap heuristic:
      - find very bright pixels
      - measure local smoothness around them
    """
    gray = _to_gray(x)  # (1,H,W)
    g = gray.unsqueeze(0)  # (1,1,H,W)

    bright = (g > 0.85).float()
    bright_ratio = bright.mean().item()

    # local variance proxy: if bright regions are surrounded by low gradient => drop-like
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    gx = F.conv2d(g, kx, padding=1).abs()
    gy = F.conv2d(g, ky, padding=1).abs()
    grad = (gx + gy)

    # mask grad by bright
    masked_grad = (grad * bright).sum() / (bright.sum() + 1e-6)
    # drop-like: bright_ratio high, but masked_grad not too high (smooth specular blob)
    val = float(bright_ratio) * float((1.0 / (masked_grad.item() + 1e-6)))
    # rescale
    return _safe_float(val * 1e-3)


# -----------------------------
# Residual / Uncertainty map
# -----------------------------

@torch.no_grad()
def _residual_map_hf(x: torch.Tensor) -> torch.Tensor:
    """
    High-frequency residual map: |x - blur(x)| aggregated to 1ch.
    x: (3,H,W) in [0,1]
    return: (1,H,W) in [0,1]
    """
    # blur via avgpool
    xx = x.unsqueeze(0)  # (1,3,H,W)
    blur = F.avg_pool2d(xx, kernel_size=7, stride=1, padding=3)
    hf = (xx - blur).abs().mean(dim=1, keepdim=True)  # (1,1,H,W)
    # normalize per-image robustly
    v = hf.flatten(1)
    p95 = torch.quantile(v, 0.95, dim=1, keepdim=True).view(-1, 1, 1, 1).clamp_min(1e-6)
    out = (hf / p95).clamp(0, 1)
    return out[0]  # (1,H,W)


@torch.no_grad()
def _gaussian_blur_approx(x: torch.Tensor, sigma: float = 1.2) -> torch.Tensor:
    """
    Fast separable Gaussian-ish blur using repeated avgpool.
    """
    # sigma control is rough; repeated avgpool approximates blur
    xx = x.unsqueeze(0)
    y = xx
    # repeat count based on sigma
    rep = 2 if sigma <= 1.2 else 3
    for _ in range(rep):
        y = F.avg_pool2d(y, kernel_size=5, stride=1, padding=2)
    return y[0]


@torch.no_grad()
def _residual_map_self_consistency(x: torch.Tensor, sigma: float = 1.2) -> torch.Tensor:
    """
    Self-consistency residual: |x - smooth(x)|
    return: (1,H,W) in [0,1]
    """
    y = _gaussian_blur_approx(x, sigma=sigma)
    diff = (x - y).abs().mean(dim=0, keepdim=True)  # (1,H,W)
    # robust normalize
    p95 = torch.quantile(diff.flatten(), 0.95).clamp_min(1e-6)
    return (diff / p95).clamp(0, 1)


# -----------------------------
# Risk score (no-ref)
# -----------------------------

@torch.no_grad()
def _risk_proxies(x: torch.Tensor) -> Dict[str, float]:
    """
    Risk proxies (no external dependency):
      - hf_energy: too high can mean ringing / oversharpen / artifacts
      - clip_ratio: too many clipped pixels indicates saturation / bad tone mapping
      - color_cast: channel mean imbalance
    """
    # hf energy
    hf = _residual_map_hf(x)  # (1,H,W)
    hf_energy = _safe_float(hf.mean().item())

    # clipping
    clip_hi = (x > 0.995).float().mean().item()
    clip_lo = (x < 0.005).float().mean().item()
    clip_ratio = _safe_float(float(clip_hi + clip_lo))

    # color cast
    means = x.view(3, -1).mean(dim=1)
    color_cast = _safe_float(float((means - means.mean()).abs().mean().item()))

    return {
        "hf_energy": hf_energy,
        "clip_ratio": clip_ratio,
        "color_cast": color_cast,
    }


def _try_piq_scores(x_bchw01: torch.Tensor) -> Dict[str, float]:
    """
    Optional PIQ-based NIQE/BRISQUE if piq is installed.
    Returns empty dict if not available.
    """
    try:
        import piq  # type: ignore
    except Exception:
        return {}

    out: Dict[str, float] = {}
    with torch.no_grad():
        # PIQ expects BCHW float in [0,1]
        try:
            niqe = piq.niqe(x_bchw01, data_range=1.0)
            out["niqe"] = _safe_float(float(niqe.item()))
        except Exception:
            pass
        try:
            brisque = piq.brisque(x_bchw01, data_range=1.0)
            out["brisque"] = _safe_float(float(brisque.item()))
        except Exception:
            pass
    return out


# -----------------------------
# Main Diagnoser class
# -----------------------------

class DiagnoserRuleD0:
    """
    D0 diagnoser: no training, deterministic heuristics.
    Input: x (CHW in [0,1])
    Output dict:
      {
        "scores": {...}               # s_t
        "residual_map": (1,H,W)       # R_t
        "risk": { "q": float, ... }   # q_t and components
      }
    """

    def __init__(self, cfg: Optional[D0Config] = None, device: str = "cpu"):
        self.cfg = cfg or D0Config()
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, x_chw01: torch.Tensor) -> Dict[str, Any]:
        assert x_chw01.ndim == 3 and x_chw01.shape[0] == 3, f"Expected (3,H,W), got {tuple(x_chw01.shape)}"
        x = x_chw01.to(self.device, non_blocking=True).float().clamp(0, 1)
        gray = _to_gray(x)  # (1,H,W)

        # ---- raw feature values ----
        sharp = _tenengrad_sharpness(gray)        # higher -> sharper
        blur_val = max(0.0, self.cfg.blur_ref - sharp)  # lower sharp -> blur higher (positive)
        noise_val = _highpass_mad(gray)
        jpeg_val = _jpeg_blockiness(gray)
        haze_val = _haze_dark_channel(x)
        rain_val = _rain_streak_proxy(gray)
        snow_val = _snow_spot_proxy(x)
        drop_val = _drop_raindrop_proxy(x)

        # ---- normalize to [0,1] (scores) ----
        # For blur: use blur_val (already ref - sharp)
        blur_score = _normalize_score(blur_val, ref=0.0, scale=self.cfg.blur_scale) * self.cfg.w_blur
        noise_score = _normalize_score(noise_val, ref=self.cfg.noise_ref, scale=self.cfg.noise_scale) * self.cfg.w_noise
        jpeg_score = _normalize_score(jpeg_val, ref=self.cfg.jpeg_ref, scale=self.cfg.jpeg_scale) * self.cfg.w_jpeg
        haze_score = _normalize_score(haze_val, ref=self.cfg.haze_ref, scale=self.cfg.haze_scale) * self.cfg.w_haze
        rain_score = _normalize_score(rain_val, ref=self.cfg.rain_ref, scale=self.cfg.rain_scale) * self.cfg.w_rain
        snow_score = _normalize_score(snow_val, ref=self.cfg.snow_ref, scale=self.cfg.snow_scale) * self.cfg.w_snow
        drop_score = _normalize_score(drop_val, ref=self.cfg.drop_ref, scale=self.cfg.drop_scale) * self.cfg.w_drop

        # clamp after weights
        scores = {
            "rain": float(max(0.0, min(1.0, rain_score))),
            "snow": float(max(0.0, min(1.0, snow_score))),
            "haze": float(max(0.0, min(1.0, haze_score))),
            "blur": float(max(0.0, min(1.0, blur_score))),
            "noise": float(max(0.0, min(1.0, noise_score))),
            "jpeg": float(max(0.0, min(1.0, jpeg_score))),
            "drop": float(max(0.0, min(1.0, drop_score))),
        }

        # ---- residual map R_t ----
        R_hf = _residual_map_hf(x)  # (1,H,W)
        R_sc = _residual_map_self_consistency(x, sigma=self.cfg.self_sigma)  # (1,H,W)
        R = (self.cfg.residual_hf_weight * R_hf + self.cfg.residual_self_weight * R_sc).clamp(0, 1)

        # ---- risk q_t ----
        proxies = _risk_proxies(x)
        # Normalize proxies into a single q (higher = riskier)
        # (you can tune these to match your stop_rule behavior)
        hf = proxies["hf_energy"]
        clipr = proxies["clip_ratio"]
        cast = proxies["color_cast"]

        # light normalization
        hf_n = float(max(0.0, min(1.0, hf / 0.12)))          # typical hf mean ~0.03-0.08
        clip_n = float(max(0.0, min(1.0, clipr / 0.03)))     # clipped pixels ratio
        cast_n = float(max(0.0, min(1.0, cast / 0.06)))      # color cast

        q = (
            self.cfg.risk_hf_weight * hf_n +
            self.cfg.risk_clip_weight * clip_n +
            self.cfg.risk_color_weight * cast_n
        )
        q = float(max(0.0, min(1.0, q)))

        risk: Dict[str, float] = {
            "q": q,
            "hf_energy": proxies["hf_energy"],
            "clip_ratio": proxies["clip_ratio"],
            "color_cast": proxies["color_cast"],
            "sharpness": sharp,
            "blur_val": blur_val,
            "noise_val": noise_val,
            "jpeg_val": jpeg_val,
            "haze_val": haze_val,
            "rain_val": rain_val,
            "snow_val": snow_val,
            "drop_val": drop_val,
        }

        # optional PIQ
        if self.cfg.try_piq:
            piq_scores = _try_piq_scores(x.unsqueeze(0))
            risk.update(piq_scores)

        return {
            "scores": scores,            # s_t
            "residual_map": R,           # R_t (1,H,W) in [0,1]
            "risk": risk,                # q_t + components
        }


if __name__ == "__main__":
    # quick sanity check
    x = torch.rand(3, 256, 256)
    d = DiagnoserRuleD0(device="cpu")
    out = d(x)
    print("[DEBUG] scores:", out["scores"])
    print("[DEBUG] risk.q:", out["risk"]["q"])
    print("[DEBUG] R:", tuple(out["residual_map"].shape), out["residual_map"].min().item(), out["residual_map"].max().item())