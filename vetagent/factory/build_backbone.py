# E:\VETAgent\vetagent\factory\build_backbone.py
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch

from vetagent.config.schema import VETAgentCfg
from vetagent.meta.ckpt_meta import extract_meta_dict, parse_backbone_meta
from vetagent.meta.validate import validate_backbone
from vetagent.meta.fingerprint import backbone_fingerprint


def _ensure_project_root_on_path() -> str:
    """
    Ensure E:/VETAgent is on sys.path so that `models.*` can be imported.
    This makes factory execution independent of the current working directory.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def _load_torch(ckpt_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Load a torch checkpoint. Accepts either:
      - dict with 'state_dict' (+ optional 'meta')
      - raw state_dict (OrderedDict)
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path}\n-> Check configs/vetagent/vetagent.yaml backbone.ckpt"
        )
    obj = torch.load(ckpt_path, map_location=map_location)
    if isinstance(obj, dict):
        return obj
    return {"state_dict": obj}


def build_backbone(
    cfg: VETAgentCfg,
    map_location: str = "cpu",
) -> Tuple[torch.nn.Module, Optional[str], Optional[Dict[str, Any]]]:
    """
    Build VETNet backbone using SSOT cfg, then load ckpt and validate meta if present.

    Returns:
      model       : instantiated VETNet with weights loaded
      backbone_fp : sha256(meta_dict) if meta exists else None
      meta_dict   : meta dict if exists else None
    """
    _ensure_project_root_on_path()

    # Local project backbone (no ReAct-IR dependency)
    from models.backbone.vetnet import VETNet

    # Instantiate from SSOT
    model = VETNet(
        in_channels=3,
        out_channels=3,
        dim=cfg.backbone.dim,
        bias=bool(cfg.backbone.bias),
        volterra_rank=cfg.backbone.volterra_rank,
        # keep other hyperparams as VETNet defaults (your "exact previous backbone")
    )

    # Load checkpoint
    ckpt = _load_torch(cfg.backbone.ckpt, map_location=map_location)

    # Meta validate + fingerprint (only if meta exists)
    meta_dict = extract_meta_dict(ckpt)
    backbone_fp = None
    if meta_dict is not None:
        meta_typed = parse_backbone_meta(meta_dict)
        validate_backbone(meta_typed, cfg)  # mismatch -> RuntimeError
        backbone_fp = backbone_fingerprint(meta_dict)

    # Load weights
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=bool(cfg.backbone.strict))

    return model, backbone_fp, meta_dict
