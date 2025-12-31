from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal

OutputKind = Literal["value", "bool"]

@dataclass(frozen=True)
class IndicatorSpec:
    indicator_id: str
    name: str
    description: str
    author: str
    license: str
    source: str  # local path or URL
    timeframe: str  # "trigger" | "anchor"
    # parameters: {param: {type: int|float|bool, min, max, step(optional)}}
    parameters: Dict[str, Dict[str, Any]]
    # outputs: {output_name: {kind: value|bool, scale: [min,max] or null}}
    outputs: Dict[str, Dict[str, Any]]
    # python implementation dotted path "module:function"
    python_impl: str
    # pine snippet for producing outputs. Must not contain indicator()/strategy() header.
    pine_snippet: str

