# E:\VETAgent\models\diagnoser\diagnoser_net.py
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagnoserNetD1(nn.Module):
    """
    D1 (trainable) diagnoser:
      - input: RGB image (B,3,H,W) in [0,1]
      - output:
          scores_logits: (B, K)  (apply sigmoid -> scores in [0,1])
          residual_map:  (B,1,H,W) (optional head, can be enabled later)
          risk_logit:    (B,1) (optional head, for risk prediction later)
    Notes:
      - Minimal viable: scores only.
      - Backbone: use your VETNet patch_embed + encoder stages as feature extractor.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_scores: int = 7,  # rain,snow,haze,blur,noise,jpeg,drop
        feat_dim: int = 64,
        use_residual_head: bool = False,
        use_risk_head: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_scores = int(num_scores)
        self.use_residual_head = bool(use_residual_head)
        self.use_risk_head = bool(use_risk_head)

        # We will extract a mid-level feature map from backbone.
        # We'll try common attribute names; otherwise fallback to last output.
        self.feat_proj = nn.Conv2d(feat_dim, feat_dim, kernel_size=1)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, self.num_scores),
        )

        if self.use_residual_head:
            self.residual_head = nn.Sequential(
                nn.Conv2d(feat_dim, feat_dim // 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(feat_dim // 2, 1, 1),
            )

        if self.use_risk_head:
            self.risk_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feat_dim, 1),
            )

    def _extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Try to pull a stable feature map from VETNet.
        Your VETNet has: patch_embed, encoder1/2/3, latent, decoder..., refinement, output.
        For diagnoser, using latent or encoder3 feature tends to be better.
        We'll try:
          - latent input feature: x5 (before decoder)
        But since the backbone forward returns only final restored image,
        we do a lightweight "feature forward" here if possible.
        """
        b = self.backbone

        # If your VETNet exposes modules, we can run partial forward for features.
        if all(hasattr(b, k) for k in ["patch_embed", "encoder1", "down1", "encoder2", "down2", "encoder3", "down3", "latent"]):
            x1 = b.patch_embed(x)
            x2 = b.encoder1(x1)
            x3 = b.encoder2(b.down1(x2))
            x4 = b.encoder3(b.down2(x3))
            x5 = b.latent(b.down3(x4))  # (B, dim*8, H/8, W/8)
            return x5

        # fallback: just use output image as "feature" (not ideal, but avoids crash)
        y = b(x)
        # fake feature: take first feat_dim channels by projecting RGB up
        return F.interpolate(y, scale_factor=0.25, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        feat = self._extract_feat(x)  # (B,C,h,w)

        # project to expected feat_dim if needed
        if feat.shape[1] != self.feat_proj.in_channels:
            # dynamic 1x1 projection (create on the fly is bad), so we do safe fallback:
            # map with a conv created per forward is not allowed.
            # Instead: if mismatch, use an on-the-fly channel reduction by slicing/padding.
            c = feat.shape[1]
            target = self.feat_proj.in_channels
            if c > target:
                feat = feat[:, :target]
            else:
                pad = target - c
                feat = torch.cat([feat, feat.new_zeros((feat.shape[0], pad, feat.shape[2], feat.shape[3]))], dim=1)

        feat = self.feat_proj(feat)
        scores_logits = self.head(feat)

        out: Dict[str, Any] = {"scores_logits": scores_logits}

        if self.use_residual_head:
            res = torch.sigmoid(self.residual_head(feat))
            out["residual_map"] = res

        if self.use_risk_head:
            risk_logit = self.risk_head(feat)
            out["risk_logit"] = risk_logit

        return out