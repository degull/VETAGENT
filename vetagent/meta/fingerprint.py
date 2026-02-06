from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


def _to_stable(obj: Any) -> Any:
    """
    dict/list/primitive로 안정 변환 (정렬된 json으로 해시 가능)
    """
    if is_dataclass(obj):
        obj = asdict(obj)

    if isinstance(obj, dict):
        return {str(k): _to_stable(obj[k]) for k in sorted(obj.keys(), key=lambda x: str(x))}
    if isinstance(obj, (list, tuple)):
        return [_to_stable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # fallback: repr
    return repr(obj)


def sha256_of_json(obj: Any) -> str:
    stable = _to_stable(obj)
    s = json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def backbone_fingerprint(meta_dict: Dict[str, Any]) -> str:
    """
    backbone meta 기반 fp (meta에 build/arch 등이 있으면 포함됨)
    """
    return sha256_of_json(meta_dict)
