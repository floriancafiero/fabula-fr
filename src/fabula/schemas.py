from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Segment:
    text: str
    idx: int
    rel_pos: float


@dataclass
class ArcResult:
    x: List[float]
    y: Optional[List[float]] = None
    y_series: Optional[Dict[str, List[float]]] = None
    raw_x: Optional[List[float]] = None
    raw_y: Optional[List[float]] = None
