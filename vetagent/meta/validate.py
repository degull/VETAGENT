# E:\VETAgent\vetagent\meta\validate.py
from __future__ import annotations

from typing import Optional

from vetagent.config.schema import VETAgentCfg, ModelSpecCfg
from vetagent.config.constants import assert_valid_action_key
from .ckpt_meta import BackboneMeta, LoRAToolMeta


def validate_cfg_consistency(cfg: VETAgentCfg, *, require_v2: bool = True) -> None:
    # -------------------------
    # model_spec (SSOT lock)
    # -------------------------
    if cfg.model_spec is None:
        raise RuntimeError("[Config] model_spec is required")

    ms: ModelSpecCfg = cfg.model_spec
    if ms.dim <= 0:
        raise RuntimeError(f"[Config] model_spec.dim must be > 0, got {ms.dim}")
    if ms.bias not in (0, 1):
        raise RuntimeError(f"[Config] model_spec.bias must be 0 or 1, got {ms.bias}")
    if ms.volterra_rank < 0:
        raise RuntimeError(f"[Config] model_spec.volterra_rank must be >= 0, got {ms.volterra_rank}")
    if ms.in_chans <= 0:
        raise RuntimeError(f"[Config] model_spec.in_chans must be > 0, got {ms.in_chans}")
    if ms.patch <= 0:
        raise RuntimeError(f"[Config] model_spec.patch must be > 0, got {ms.patch}")

    # -------------------------
    # actions: enabled/order
    # -------------------------
    if cfg.actions is None:
        raise RuntimeError("[Config] actions is required")

    if cfg.actions.order is None or len(cfg.actions.order) == 0:
        raise RuntimeError("[Config] actions.order is required and cannot be empty")

    if cfg.actions.enabled is None or len(cfg.actions.enabled) == 0:
        raise RuntimeError("[Config] actions.enabled is required and cannot be empty")

    if len(set(cfg.actions.order)) != len(cfg.actions.order):
        raise RuntimeError(f"[Config] actions.order has duplicates: {cfg.actions.order}")

    for k in cfg.actions.order:
        assert_valid_action_key(k)
    for k in cfg.actions.enabled:
        assert_valid_action_key(k)

    if set(cfg.actions.order) != set(cfg.actions.enabled):
        raise RuntimeError(
            f"[Config] actions.order and actions.enabled must match as sets. "
            f"order={cfg.actions.order}, enabled={cfg.actions.enabled}"
        )

    if len(cfg.actions.order) != 5:
        raise RuntimeError(f"[Config] actions.order must have length 5, got {len(cfg.actions.order)}")

    # -------------------------
    # diagnoser/toolbank + v2 sections (agent mode)
    # -------------------------
    if require_v2:
        if cfg.toolbank is None:
            raise RuntimeError("[Config] toolbank is required (agent mode)")
        if cfg.diagnoser is None:
            raise RuntimeError("[Config] diagnoser is required (agent mode)")

        if cfg.diagnoser.num_actions != len(cfg.actions.order):
            raise RuntimeError(
                f"[Config] diagnoser.num_actions must equal len(actions.order). "
                f"num_actions={cfg.diagnoser.num_actions}, len(order)={len(cfg.actions.order)}"
            )
        if cfg.diagnoser.num_actions != 5:
            raise RuntimeError(f"[Config] diagnoser.num_actions must be 5, got {cfg.diagnoser.num_actions}")
        if cfg.diagnoser.uncertainty is None:
            raise RuntimeError("[Config] diagnoser.uncertainty is required (agent mode)")

        if cfg.calib is None:
            raise RuntimeError("[Config] calib is required (agent mode)")
        if cfg.noharm is None:
            raise RuntimeError("[Config] noharm is required (agent mode)")
        if cfg.router is None:
            raise RuntimeError("[Config] router is required (agent mode)")
        if cfg.stop is None:
            raise RuntimeError("[Config] stop is required (agent mode)")
        if cfg.safety_fsm is None:
            raise RuntimeError("[Config] safety_fsm is required (agent mode)")
        if cfg.rollback is None:
            raise RuntimeError("[Config] rollback is required (agent mode)")

        if cfg.stop.max_steps <= 0:
            raise RuntimeError(f"[Config] stop.max_steps must be > 0, got {cfg.stop.max_steps}")
        if cfg.safety_fsm.max_steps <= 0:
            raise RuntimeError(f"[Config] safety_fsm.max_steps must be > 0, got {cfg.safety_fsm.max_steps}")


def validate_backbone(meta: BackboneMeta, cfg: VETAgentCfg, *, require_v2: bool = True) -> None:
    validate_cfg_consistency(cfg, require_v2=require_v2)

    ms = cfg.model_spec
    if meta.dim != ms.dim:
        raise RuntimeError(f"[Backbone mismatch] dim ckpt={meta.dim} cfg.model_spec={ms.dim}")
    if meta.bias != ms.bias:
        raise RuntimeError(f"[Backbone mismatch] bias ckpt={meta.bias} cfg.model_spec={ms.bias}")
    if meta.volterra_rank != ms.volterra_rank:
        raise RuntimeError(
            f"[Backbone mismatch] volterra_rank ckpt={meta.volterra_rank} cfg.model_spec={ms.volterra_rank}"
        )


def validate_lora(meta: LoRAToolMeta, cfg: VETAgentCfg, backbone_fp: Optional[str]) -> None:
    assert_valid_action_key(meta.action_key)
    validate_cfg_consistency(cfg, require_v2=True)

    ms = cfg.model_spec
    if meta.dim != ms.dim:
        raise RuntimeError(f"[LoRA mismatch] dim ckpt={meta.dim} cfg.model_spec={ms.dim}")
    if meta.bias != ms.bias:
        raise RuntimeError(f"[LoRA mismatch] bias ckpt={meta.bias} cfg.model_spec={ms.bias}")
    if meta.volterra_rank != ms.volterra_rank:
        raise RuntimeError(
            f"[LoRA mismatch] volterra_rank ckpt={meta.volterra_rank} cfg.model_spec={ms.volterra_rank}"
        )

    if meta.backbone_fp is not None and backbone_fp is not None and meta.backbone_fp != backbone_fp:
        raise RuntimeError(f"[LoRA mismatch] backbone_fp ckpt={meta.backbone_fp} computed={backbone_fp}")

"""
# train_backbone 단계 (v2 섹션 다 없어도 됨, 단 model_spec은 필수!)
cd E:\VETAgent
python -c "from vetagent.config import load_cfg; cfg=load_cfg('E:/VETAgent/configs/v

# agent 단계 (v2 섹션 + toolbank/diagnoser 다 있어야 통과)
cd E:\VETAgent
python -c "from vetagent.config import load_cfg; cfg=load_cfg('E:/VETAgent/configs/vetagent/vetagent.yaml', mode='agent'); print(cfg.model_spec.dim, cfg.actions.enabled, cfg.actions.order)"

"""