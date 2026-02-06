# E:\VETAgent\vetagent\config\load.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional, List

import yaml

from .schema import (
    VETAgentCfg,
    ModelSpecCfg,
    BackboneCfg,
    ToolBankCfg,
    DiagnoserCfg,
    RuntimeCfg,
    ActionsCfg,
    # v2
    UncertaintyCfg,
    CalibCfg,
    EMAGateCfg,
    NoHarmCfg,
    RouterCfg,
    UncertaintyDampCfg,
    StopCfg,
    SafetyFSMCfg,
    AlphaAnnealCfg,
    RollbackCfg,
    RollbackWeightsCfg,
)
from .constants import ACTIONS_ORDER, assert_valid_action_key


def _require(d: Dict[str, Any], k: str, ctx: str) -> Any:
    if k not in d:
        raise KeyError(f"Missing required key '{k}' in {ctx}")
    return d[k]


def _require_dict(d: Dict[str, Any], k: str, ctx: str) -> Dict[str, Any]:
    v = _require(d, k, ctx)
    if not isinstance(v, dict):
        raise TypeError(f"Expected dict for '{k}' in {ctx}")
    return v


def _as_int(x: Any, ctx: str) -> int:
    if isinstance(x, bool):
        raise TypeError(f"Expected int, got bool in {ctx}")
    if not isinstance(x, int):
        raise TypeError(f"Expected int, got {type(x)} in {ctx}")
    return x


def _as_float(x: Any, ctx: str) -> float:
    if isinstance(x, bool):
        raise TypeError(f"Expected float, got bool in {ctx}")
    if not isinstance(x, (int, float)):
        raise TypeError(f"Expected float, got {type(x)} in {ctx}")
    return float(x)


def _as_bool(x: Any, ctx: str) -> bool:
    if not isinstance(x, bool):
        raise TypeError(f"Expected bool, got {type(x)} in {ctx}")
    return x


def _as_str_allow_empty(x: Any, ctx: str) -> str:
    if not isinstance(x, str):
        raise TypeError(f"Expected str in {ctx}")
    return x


def _as_str(x: Any, ctx: str) -> str:
    if not isinstance(x, str) or not x.strip():
        raise TypeError(f"Expected non-empty str in {ctx}")
    return x


def _as_list_str(x: Any, ctx: str) -> List[str]:
    if not isinstance(x, list) or not all(isinstance(v, str) for v in x):
        raise TypeError(f"Expected list[str] in {ctx}")
    return list(x)


def _as_list_float(x: Any, ctx: str) -> List[float]:
    if not isinstance(x, list) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x):
        raise TypeError(f"Expected list[float] in {ctx}")
    return [float(v) for v in x]


def _check_range_int(v: int, *, min_v: int, ctx: str) -> None:
    if v < min_v:
        raise ValueError(f"{ctx} must be >= {min_v}, got {v}")


def _check_len2(xs: List[float], ctx: str) -> None:
    if len(xs) != 2:
        raise ValueError(f"{ctx} must have length 2, got {len(xs)}")


def load_cfg(yaml_path: str, *, mode: str = "agent") -> VETAgentCfg:
    """
    mode:
      - "agent": toolbank/diagnoser + v2 섹션 필수(에이전트 실행용)
      - "train_backbone": toolbank/diagnoser + v2 섹션 없어도 통과(백본 학습용)
    """
    if mode not in ("agent", "train_backbone"):
        raise ValueError(f"Unknown mode: {mode}")

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError("YAML root must be a dict")

    # -----------------------------
    # model_spec (SSOT Lock, always required)
    # -----------------------------
    ms = _require_dict(data, "model_spec", "root.model_spec")

    dim = _as_int(_require(ms, "dim", "model_spec"), "model_spec.dim")
    bias = _as_int(_require(ms, "bias", "model_spec"), "model_spec.bias")
    volterra_rank = _as_int(_require(ms, "volterra_rank", "model_spec"), "model_spec.volterra_rank")
    in_chans = _as_int(_require(ms, "in_chans", "model_spec"), "model_spec.in_chans")
    patch = _as_int(_require(ms, "patch", "model_spec"), "model_spec.patch")

    _check_range_int(dim, min_v=1, ctx="model_spec.dim")
    _check_range_int(in_chans, min_v=1, ctx="model_spec.in_chans")
    _check_range_int(patch, min_v=1, ctx="model_spec.patch")
    if bias not in (0, 1):
        raise ValueError(f"model_spec.bias must be 0 or 1, got {bias}")
    if volterra_rank < 0:
        raise ValueError(f"model_spec.volterra_rank must be >= 0, got {volterra_rank}")

    model_spec = ModelSpecCfg(
        dim=dim,
        bias=bias,
        volterra_rank=volterra_rank,
        in_chans=in_chans,
        patch=patch,
    )

    # -----------------------------
    # backbone (ckpt-only)
    # -----------------------------
    bb = _require_dict(data, "backbone", "root.backbone")

    # ❌ 금지: backbone에 스펙 필드(dim/bias/rank/patch/in_chans) 두면 혼란 발생
    forbidden = ["dim", "bias", "volterra_rank", "patch", "in_chans"]
    for k in forbidden:
        if k in bb:
            raise KeyError(
                f"Do not set backbone.{k}. Use root.model_spec.{k} only (SSOT lock)."
            )

    backbone = BackboneCfg(
        name=_as_str_allow_empty(bb.get("name", "VETNet"), "backbone.name") or "VETNet",
        ckpt=_as_str_allow_empty(bb.get("ckpt", ""), "backbone.ckpt"),
        strict=_as_bool(bb.get("strict", False), "backbone.strict"),
        fingerprint_fields=_as_list_str(bb.get("fingerprint_fields", ["name", "dim", "bias", "volterra_rank"]), "backbone.fingerprint_fields")
        if bb.get("fingerprint_fields", None) is not None else None,
    )

    # -----------------------------
    # runtime (옵션)
    # -----------------------------
    rt = data.get("runtime", {}) or {}
    if not isinstance(rt, dict):
        raise TypeError("runtime must be dict")

    runtime = RuntimeCfg(
        device=str(rt.get("device", "cuda")),
        use_amp=bool(rt.get("use_amp", True)),
        channels_last=bool(rt.get("channels_last", True)),
        tf32=bool(rt.get("tf32", True)),
    )

    # -----------------------------
    # actions (enabled + order)
    # -----------------------------
    ac = data.get("actions", {}) or {}
    if not isinstance(ac, dict):
        raise TypeError("actions must be dict")

    enabled = ac.get("enabled", ACTIONS_ORDER)
    if not isinstance(enabled, list) or not all(isinstance(x, str) for x in enabled):
        raise TypeError("actions.enabled must be list[str]")
    for k in enabled:
        assert_valid_action_key(k)

    order = ac.get("order", ACTIONS_ORDER)
    if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
        raise TypeError("actions.order must be list[str]")
    for k in order:
        assert_valid_action_key(k)

    actions = ActionsCfg(enabled=list(enabled), order=list(order))

    # quick sanity
    if len(actions.order) != 5:
        raise ValueError(f"actions.order must have length 5, got {len(actions.order)}")
    if set(actions.order) != set(actions.enabled):
        raise ValueError(
            f"actions.order and actions.enabled must match as sets. "
            f"order={actions.order}, enabled={actions.enabled}"
        )

    # -----------------------------
    # toolbank / diagnoser
    # -----------------------------
    toolbank: Optional[ToolBankCfg] = None
    diagnoser: Optional[DiagnoserCfg] = None

    # toolbank
    if "toolbank" in data:
        tb = data["toolbank"]
        if not isinstance(tb, dict):
            raise TypeError("root.toolbank must be dict")

        tools = tb.get("tools", {}) or {}
        if not isinstance(tools, dict):
            raise TypeError("toolbank.tools must be dict(action_key -> tool_ref)")
        for k in tools.keys():
            assert_valid_action_key(k)

        if mode == "agent":
            lora_root = _as_str(_require(tb, "lora_root", "toolbank"), "toolbank.lora_root")
        else:
            lora_root = _as_str_allow_empty(tb.get("lora_root", ""), "toolbank.lora_root")

        toolbank = ToolBankCfg(
            lora_root=lora_root,
            tools={str(k): str(v) for k, v in tools.items()},
        )
    else:
        if mode == "agent":
            raise KeyError("Missing required key 'toolbank' in root.toolbank")

    # diagnoser
    if "diagnoser" in data:
        dg = data["diagnoser"]
        if not isinstance(dg, dict):
            raise TypeError("root.diagnoser must be dict")

        num_actions = _as_int(dg.get("num_actions", 5), "diagnoser.num_actions")

        if mode == "agent":
            name = _as_str(_require(dg, "name", "diagnoser"), "diagnoser.name")
            ckpt = _as_str(_require(dg, "ckpt", "diagnoser"), "diagnoser.ckpt")
            input_size = _as_int(_require(dg, "input_size", "diagnoser"), "diagnoser.input_size")
            postprocess = _as_str(_require(dg, "postprocess", "diagnoser"), "diagnoser.postprocess")

            unc = _require_dict(dg, "uncertainty", "diagnoser")
            uncertainty = UncertaintyCfg(
                type=_as_str(_require(unc, "type", "diagnoser.uncertainty"), "diagnoser.uncertainty.type"),
                tau_u=_as_float(_require(unc, "tau_u", "diagnoser.uncertainty"), "diagnoser.uncertainty.tau_u"),
            )

            diagnoser = DiagnoserCfg(
                name=name,
                ckpt=ckpt,
                input_size=input_size,
                postprocess=postprocess,
                num_actions=num_actions,
                uncertainty=uncertainty,
            )
        else:
            name = _as_str_allow_empty(dg.get("name", ""), "diagnoser.name")
            ckpt = _as_str_allow_empty(dg.get("ckpt", ""), "diagnoser.ckpt")
            input_size = _as_int(dg.get("input_size", 224), "diagnoser.input_size")
            postprocess = str(dg.get("postprocess", "softmax"))

            uncertainty = None
            unc = dg.get("uncertainty", None)
            if isinstance(unc, dict) and "type" in unc and "tau_u" in unc:
                uncertainty = UncertaintyCfg(
                    type=_as_str_allow_empty(unc.get("type", ""), "diagnoser.uncertainty.type"),
                    tau_u=_as_float(unc.get("tau_u", 0.0), "diagnoser.uncertainty.tau_u"),
                )

            diagnoser = DiagnoserCfg(
                name=name,
                ckpt=ckpt,
                input_size=input_size,
                postprocess=postprocess,
                num_actions=num_actions,
                uncertainty=uncertainty,
            )
    else:
        if mode == "agent":
            raise KeyError("Missing required key 'diagnoser' in root.diagnoser")

    # -----------------------------
    # v2 sections (agent에서는 필수)
    # -----------------------------
    calib: Optional[CalibCfg] = None
    noharm: Optional[NoHarmCfg] = None
    router: Optional[RouterCfg] = None
    stop: Optional[StopCfg] = None
    safety_fsm: Optional[SafetyFSMCfg] = None
    rollback: Optional[RollbackCfg] = None

    if mode == "agent":
        cb = _require_dict(data, "calib", "root.calib")
        nh = _require_dict(data, "noharm", "root.noharm")
        rr = _require_dict(data, "router", "root.router")
        st = _require_dict(data, "stop", "root.stop")
        sf = _require_dict(data, "safety_fsm", "root.safety_fsm")
        rb = _require_dict(data, "rollback", "root.rollback")
    else:
        cb = data.get("calib", None)
        nh = data.get("noharm", None)
        rr = data.get("router", None)
        st = data.get("stop", None)
        sf = data.get("safety_fsm", None)
        rb = data.get("rollback", None)

    # calib
    if cb is not None:
        if not isinstance(cb, dict):
            raise TypeError("root.calib must be dict")
        gate = cb.get("ema_gate", {}) or {}
        if not isinstance(gate, dict):
            raise TypeError("calib.ema_gate must be dict")

        bucket_table_path = _as_str(cb.get("bucket_table_path", ""), "calib.bucket_table_path") if mode == "agent" else _as_str_allow_empty(cb.get("bucket_table_path", ""), "calib.bucket_table_path")
        beta_ema = _as_float(cb.get("beta_ema", 0.05), "calib.beta_ema")
        w_bucket = _as_float(cb.get("w_bucket", 0.7), "calib.w_bucket")
        clamp_minmax = _as_list_float(cb.get("clamp_minmax", [0.0, 1.0]), "calib.clamp_minmax")
        _check_len2(clamp_minmax, "calib.clamp_minmax")

        calib = CalibCfg(
            bucket_table_path=bucket_table_path,
            beta_ema=beta_ema,
            w_bucket=w_bucket,
            clamp_minmax=clamp_minmax,
            ema_gate=EMAGateCfg(
                enable=_as_bool(gate.get("enable", True), "calib.ema_gate.enable"),
                update_if_max_score_below=_as_float(gate.get("update_if_max_score_below", 0.35),
                                                   "calib.ema_gate.update_if_max_score_below"),
            ),
        )

    # noharm
    if nh is not None:
        if not isinstance(nh, dict):
            raise TypeError("root.noharm must be dict")

        thresholds = nh.get("thresholds", {}) or {}
        if not isinstance(thresholds, dict):
            raise TypeError("noharm.thresholds must be dict")

        metrics = nh.get("metrics", []) or []
        if not isinstance(metrics, list) or not all(isinstance(x, str) for x in metrics):
            raise TypeError("noharm.metrics must be list[str]")

        noharm = NoHarmCfg(
            enable=_as_bool(nh.get("enable", True), "noharm.enable"),
            thresholds={str(k): _as_float(v, f"noharm.thresholds.{k}") for k, v in thresholds.items()},
            metrics=list(metrics),
        )

    # router
    if rr is not None:
        if not isinstance(rr, dict):
            raise TypeError("root.router must be dict")

        damp = rr.get("uncertainty_damp", {}) or {}
        if not isinstance(damp, dict):
            raise TypeError("router.uncertainty_damp must be dict")

        topk = _as_int(rr.get("topk", 3), "router.topk")
        alpha_limits = _as_list_float(rr.get("alpha_limits", [0.1, 1.0]), "router.alpha_limits")
        _check_len2(alpha_limits, "router.alpha_limits")

        router = RouterCfg(
            topk=topk,
            schedule_rules=str(rr.get("schedule_rules", "default_v1")),
            alpha_rule=str(rr.get("alpha_rule", "score_linear_v1")),
            composite_enable=_as_bool(rr.get("composite_enable", False), "router.composite_enable"),
            alpha_limits=alpha_limits,
            uncertainty_damp=UncertaintyDampCfg(
                enable=_as_bool(damp.get("enable", True), "router.uncertainty_damp.enable"),
                k=_as_float(damp.get("k", 0.5), "router.uncertainty_damp.k"),
            ),
        )

    # stop
    if st is not None:
        if not isinstance(st, dict):
            raise TypeError("root.stop must be dict")

        stop = StopCfg(
            tau_abs=_as_float(st.get("tau_abs", 0.08), "stop.tau_abs"),
            delta_dom_min=_as_float(st.get("delta_dom_min", 0.005), "stop.delta_dom_min"),
            patience=_as_int(st.get("patience", 2), "stop.patience"),
            harm_gate=_as_bool(st.get("harm_gate", True), "stop.harm_gate"),
            max_steps=_as_int(st.get("max_steps", 8), "stop.max_steps"),
        )

    # safety_fsm
    if sf is not None:
        if not isinstance(sf, dict):
            raise TypeError("root.safety_fsm must be dict")

        aa = sf.get("alpha_anneal", {}) or {}
        if not isinstance(aa, dict):
            raise TypeError("safety_fsm.alpha_anneal must be dict")

        safety_fsm = SafetyFSMCfg(
            repeat_cap=_as_int(sf.get("repeat_cap", 3), "safety_fsm.repeat_cap"),
            cooldown=_as_int(sf.get("cooldown", 1), "safety_fsm.cooldown"),
            alpha_anneal=AlphaAnnealCfg(
                enable=_as_bool(aa.get("enable", True), "safety_fsm.alpha_anneal.enable"),
                gamma=_as_float(aa.get("gamma", 0.85), "safety_fsm.alpha_anneal.gamma"),
            ),
            max_steps=_as_int(sf.get("max_steps", 8), "safety_fsm.max_steps"),
            rollback_mode=str(sf.get("rollback_mode", "best_then_stop")),
        )

    # rollback
    if rb is not None:
        if not isinstance(rb, dict):
            raise TypeError("root.rollback must be dict")

        w = rb.get("weights", {}) or {}
        if not isinstance(w, dict):
            raise TypeError("rollback.weights must be dict")

        mode_rb = str(rb.get("mode", "J"))
        if mode_rb not in ("pareto", "J"):
            raise ValueError(f"rollback.mode must be one of ['pareto','J'], got {mode_rb}")

        rollback = RollbackCfg(
            mode=mode_rb,  # type: ignore[arg-type]
            weights=RollbackWeightsCfg(
                w_dom=_as_float(w.get("w_dom", 1.0), "rollback.weights.w_dom"),
                w_sum=_as_float(w.get("w_sum", 0.2), "rollback.weights.w_sum"),
                w_h=_as_float(w.get("w_h", 0.8), "rollback.weights.w_h"),
            ),
        )

    if mode == "agent":
        # agent 모드에서는 v2 섹션들 필수
        if toolbank is None:
            raise KeyError("Missing required key 'toolbank' in root.toolbank")
        if diagnoser is None:
            raise KeyError("Missing required key 'diagnoser' in root.diagnoser")
        if diagnoser.uncertainty is None:
            raise KeyError("Missing required key 'diagnoser.uncertainty' in root.diagnoser.uncertainty")

        if calib is None:
            raise KeyError("Missing required key 'calib' in root.calib")
        if noharm is None:
            raise KeyError("Missing required key 'noharm' in root.noharm")
        if router is None:
            raise KeyError("Missing required key 'router' in root.router")
        if stop is None:
            raise KeyError("Missing required key 'stop' in root.stop")
        if safety_fsm is None:
            raise KeyError("Missing required key 'safety_fsm' in root.safety_fsm")
        if rollback is None:
            raise KeyError("Missing required key 'rollback' in root.rollback")

    # -----------------------------
    # 기타 루트 필드 (dict 그대로 보관)
    # -----------------------------
    data_root = str(data.get("data_root", "E:/VETAgent/data"))
    ckpt = data.get("ckpt", None)
    datasets = data.get("datasets", None)
    project = data.get("project", None)

    if ckpt is not None and not isinstance(ckpt, dict):
        raise TypeError("root.ckpt must be dict if provided")
    if datasets is not None and not isinstance(datasets, dict):
        raise TypeError("root.datasets must be dict if provided")
    if project is not None and not isinstance(project, dict):
        raise TypeError("root.project must be dict if provided")

    cfg = VETAgentCfg(
        model_spec=model_spec,
        backbone=backbone,
        toolbank=toolbank,
        diagnoser=diagnoser,
        runtime=runtime,
        actions=actions,
        calib=calib,
        noharm=noharm,
        router=router,
        stop=stop,
        safety_fsm=safety_fsm,
        rollback=rollback,
        data_root=data_root,
        ckpt=ckpt,
        datasets=datasets,
        project=project,
        _raw=data,
    )
    return cfg
