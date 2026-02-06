# E:\VETAgent\vetagent\config\schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Literal


# -----------------------------
# v2: SSOT Lock (single source of truth)
# -----------------------------
@dataclass(frozen=True)
class ModelSpecCfg:
    dim: int
    bias: int
    volterra_rank: int
    in_chans: int
    patch: int


# -----------------------------
# Backbone / ToolBank
# -----------------------------
@dataclass(frozen=True)
class BackboneCfg:
    name: str
    ckpt: str
    strict: bool
    fingerprint_fields: Optional[List[str]] = None


@dataclass(frozen=True)
class ToolBankCfg:
    lora_root: str
    tools: Dict[str, str]  # keys: drop/snow/rain/blur/haze


# -----------------------------
# v2: Diagnoser + Uncertainty
# -----------------------------
@dataclass(frozen=True)
class UncertaintyCfg:
    type: str              # e.g., "entropy_margin"
    tau_u: float           # e.g., 0.6


@dataclass(frozen=True)
class DiagnoserCfg:
    name: str
    ckpt: str
    input_size: int        # e.g., 224
    postprocess: str       # e.g., "softmax"
    num_actions: int = 5
    uncertainty: Optional[UncertaintyCfg] = None  # agent 모드에서는 load.py에서 필수 강제


# -----------------------------
# v2: Calibrator (Domain-aware)
# -----------------------------
@dataclass(frozen=True)
class EMAGateCfg:
    enable: bool
    update_if_max_score_below: float


@dataclass(frozen=True)
class CalibCfg:
    bucket_table_path: str
    beta_ema: float
    w_bucket: float
    clamp_minmax: List[float]   # [min,max]
    ema_gate: EMAGateCfg


# -----------------------------
# v2: No-Harm Prober
# -----------------------------
@dataclass(frozen=True)
class NoHarmCfg:
    enable: bool
    thresholds: Dict[str, float]   # halo/sharpness/noise/color
    metrics: List[str]            # ["halo","sharpness","noise","color"]


# -----------------------------
# v2: Router + Scheduler
# -----------------------------
@dataclass(frozen=True)
class UncertaintyDampCfg:
    enable: bool
    k: float


@dataclass(frozen=True)
class RouterCfg:
    topk: int
    schedule_rules: str
    alpha_rule: str
    composite_enable: bool
    alpha_limits: List[float]          # [min,max]
    uncertainty_damp: UncertaintyDampCfg


# -----------------------------
# v2: StopRule++
# -----------------------------
@dataclass(frozen=True)
class StopCfg:
    tau_abs: float
    delta_dom_min: float
    patience: int
    harm_gate: bool
    max_steps: int


# -----------------------------
# v2: Safety FSM
# -----------------------------
@dataclass(frozen=True)
class AlphaAnnealCfg:
    enable: bool
    gamma: float


@dataclass(frozen=True)
class SafetyFSMCfg:
    repeat_cap: int
    cooldown: int
    alpha_anneal: AlphaAnnealCfg
    max_steps: int
    rollback_mode: str  # "best_then_stop" | "best_then_retry_once"


# -----------------------------
# v2: Rollback criterion
# -----------------------------
@dataclass(frozen=True)
class RollbackWeightsCfg:
    w_dom: float
    w_sum: float
    w_h: float


@dataclass(frozen=True)
class RollbackCfg:
    mode: Literal["pareto", "J"]
    weights: RollbackWeightsCfg


# -----------------------------
# Runtime / Actions
# -----------------------------
@dataclass(frozen=True)
class RuntimeCfg:
    device: str = "cuda"
    use_amp: bool = True
    channels_last: bool = True
    tf32: bool = True


@dataclass(frozen=True)
class ActionsCfg:
    enabled: List[str]
    order: List[str]


# -----------------------------
# Root cfg (SSOT)
# -----------------------------
@dataclass(frozen=True)
class VETAgentCfg:
    # ✅ SSOT Lock: 모든 모듈은 이 spec을 기준으로 동작
    model_spec: ModelSpecCfg

    # ✅ backbone은 ckpt만 (dim/bias/rank는 model_spec에서만!)
    backbone: BackboneCfg

    toolbank: Optional[ToolBankCfg]
    diagnoser: Optional[DiagnoserCfg]

    runtime: RuntimeCfg
    actions: ActionsCfg

    # v2 modules
    calib: Optional[CalibCfg] = None
    noharm: Optional[NoHarmCfg] = None
    router: Optional[RouterCfg] = None
    stop: Optional[StopCfg] = None
    safety_fsm: Optional[SafetyFSMCfg] = None
    rollback: Optional[RollbackCfg] = None

    # 기타
    data_root: str = "E:/VETAgent/data"
    ckpt: Optional[Dict[str, Any]] = None
    datasets: Optional[Dict[str, Any]] = None
    project: Optional[Dict[str, Any]] = None
    _raw: Optional[Dict[str, Any]] = None
