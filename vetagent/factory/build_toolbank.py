from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

from vetagent.config.schema import VETAgentCfg
from vetagent.config.constants import assert_valid_action_key
from vetagent.meta.ckpt_meta import extract_meta_dict, parse_lora_tool_meta
from vetagent.meta.validate import validate_lora


def _load_torch(ckpt_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    obj = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(obj, dict):
        return {"state_dict": obj}
    return obj


def resolve_tool_ckpt_path(cfg: VETAgentCfg, action_key: str) -> str:
    """
    cfg.toolbank.tools[action_key] 값을 해석해서 ckpt 경로로 만든다.
    지금은 가장 단순하게:
      lora_root / (value).pth
    또는 value가 이미 .pth면 그대로 사용.
    """
    assert_valid_action_key(action_key)

    ref = cfg.toolbank.tools.get(action_key, None)
    if ref is None:
        raise KeyError(f"toolbank.tools missing key '{action_key}'")

    # 이미 절대경로/상대경로로 .pth를 준 경우
    if str(ref).lower().endswith(".pth"):
        if os.path.isabs(ref):
            return ref
        return os.path.normpath(os.path.join(cfg.toolbank.lora_root, ref))

    # ref가 폴더/이름이면 "{ref}.pth"로 가정
    return os.path.normpath(os.path.join(cfg.toolbank.lora_root, f"{ref}.pth"))


def build_toolbank(cfg: VETAgentCfg, backbone_fp: Optional[str], map_location: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Optional[Dict[str, Any]]]]:
    """
    Returns:
      tools: {action_key: tool_ckpt_obj or state_dict}
      metas: {action_key: meta_dict_or_None}

    NOTE:
      - 각 tool ckpt에 meta가 있으면 validate_lora(meta, cfg, backbone_fp)를 강제한다.
    """
    tools: Dict[str, Any] = {}
    metas: Dict[str, Optional[Dict[str, Any]]] = {}

    for action_key in cfg.actions.enabled:
        assert_valid_action_key(action_key)
        ckpt_path = resolve_tool_ckpt_path(cfg, action_key)
        ckpt = _load_torch(ckpt_path, map_location=map_location)

        meta_dict = extract_meta_dict(ckpt)
        metas[action_key] = meta_dict

        if meta_dict is not None:
            meta_typed = parse_lora_tool_meta(meta_dict)
            validate_lora(meta_typed, cfg, backbone_fp)

        tools[action_key] = ckpt

    return tools, metas
