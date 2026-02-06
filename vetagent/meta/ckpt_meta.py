from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


# -----------------------------
# Standardized Meta Specs
# -----------------------------
@dataclass(frozen=True)
class BackboneMeta:
    dim: int
    bias: int
    volterra_rank: int
    arch: str = "VETNet"
    build: Optional[str] = None   # git hash / tag
    extra: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class LoRAToolMeta:
    dim: int
    bias: int
    volterra_rank: int
    action_key: str               # "drop/snow/rain/blur/haze"
    backbone_fp: Optional[str] = None
    tool_name: Optional[str] = None
    build: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


# -----------------------------
# Helpers: extract meta dict
# -----------------------------
def extract_meta_dict(ckpt_obj: Any) -> Optional[Dict[str, Any]]:
    """
    torch.load() 결과가 dict일 때, meta를 담은 키를 찾는다.
    허용 키: "meta", "ckpt_meta", "metadata"
    """
    if not isinstance(ckpt_obj, dict):
        return None

    for k in ("meta", "ckpt_meta", "metadata"):
        v = ckpt_obj.get(k, None)
        if isinstance(v, dict):
            return v
    return None


def _pick(meta: Dict[str, Any], keys: Tuple[str, ...], required: bool, ctx: str) -> Any:
    for k in keys:
        if k in meta:
            return meta[k]
    if required:
        raise KeyError(f"Missing meta field {keys} in {ctx}. meta keys={list(meta.keys())}")
    return None


def parse_backbone_meta(meta: Dict[str, Any]) -> BackboneMeta:
    """
    표준 필드명: dim, bias, volterra_rank
    호환 필드명도 허용 (예: "volterraRank", "volterra_r", ...)
    """
    dim = int(_pick(meta, ("dim",), True, "backbone_meta"))
    bias = int(_pick(meta, ("bias",), True, "backbone_meta"))
    vr = int(_pick(meta, ("volterra_rank", "volterraRank", "volterra_r", "v_rank"), True, "backbone_meta"))
    arch = str(_pick(meta, ("arch", "model", "name"), False, "backbone_meta") or "VETNet")
    build = _pick(meta, ("build", "git", "git_hash", "tag"), False, "backbone_meta")
    extra = {k: v for k, v in meta.items() if k not in {"dim", "bias", "volterra_rank", "volterraRank", "volterra_r", "v_rank", "arch", "model", "name", "build", "git", "git_hash", "tag"}}
    return BackboneMeta(dim=dim, bias=bias, volterra_rank=vr, arch=arch, build=build, extra=extra or None)


def parse_lora_tool_meta(meta: Dict[str, Any]) -> LoRAToolMeta:
    dim = int(_pick(meta, ("dim",), True, "lora_tool_meta"))
    bias = int(_pick(meta, ("bias",), True, "lora_tool_meta"))
    vr = int(_pick(meta, ("volterra_rank", "volterraRank", "volterra_r", "v_rank"), True, "lora_tool_meta"))

    action_key = str(_pick(meta, ("action_key", "action", "task"), True, "lora_tool_meta"))
    backbone_fp = _pick(meta, ("backbone_fp", "backbone_fingerprint", "fp"), False, "lora_tool_meta")
    tool_name = _pick(meta, ("tool_name", "name", "adapter_name"), False, "lora_tool_meta")
    build = _pick(meta, ("build", "git", "git_hash", "tag"), False, "lora_tool_meta")

    extra = {k: v for k, v in meta.items() if k not in {"dim", "bias", "volterra_rank", "volterraRank", "volterra_r", "v_rank",
                                                       "action_key", "action", "task", "backbone_fp", "backbone_fingerprint", "fp",
                                                       "tool_name", "name", "adapter_name", "build", "git", "git_hash", "tag"}}
    return LoRAToolMeta(
        dim=dim, bias=bias, volterra_rank=vr,
        action_key=action_key, backbone_fp=str(backbone_fp) if backbone_fp is not None else None,
        tool_name=str(tool_name) if tool_name is not None else None,
        build=build, extra=extra or None
    )
