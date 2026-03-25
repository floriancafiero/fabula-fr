from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .arc import resample_to_n, smooth_series
from .schemas import ArcResult
from .scorer import valence_from_probs
from .segment import RegexSentenceSegmenter


@dataclass
class Fabula:
    scorer: Any
    segmenter: Any = None
    coarse_segmenter: Any = None
    analysis: str = "sentiment"
    chunk_weight: float = 0.3
    chunk_attention_tau: float = 0.1
    verbose: bool = False

    def __post_init__(self):
        if self.segmenter is None:
            self.segmenter = RegexSentenceSegmenter()

    def _interpret_probs(self, probs: Dict[str, float]) -> Dict[str, Optional[float] | Optional[str]]:
        if not probs:
            return {"label": None, "score": None, "entropy": None}

        label = max(probs.items(), key=lambda kv: kv[1])[0]
        score = valence_from_probs(probs) if self.analysis == "sentiment" else max(probs.values())
        entropy = -sum(float(v) * math.log(float(v) + 1e-12) for v in probs.values())
        return {"label": label, "score": float(score), "entropy": entropy}

    def score(
        self,
        text: str,
        explain_tokens: bool = False,
        target_label: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> pd.DataFrame:
        del explain_tokens, target_label, top_k  # reserved for API compatibility

        segments = self.segmenter.segment(text)
        probs_list = self.scorer.predict_proba([s.text for s in segments]) if segments else []

        rows: List[Dict[str, object]] = []
        for seg, probs in zip(segments, probs_list):
            parsed = self._interpret_probs(probs)
            rows.append(
                {
                    "idx": seg.idx,
                    "text": seg.text,
                    "rel_pos": seg.rel_pos,
                    "probs": probs,
                    **parsed,
                }
            )
        return pd.DataFrame(rows)

    def _smooth_scalar(self, x: Sequence[float], y: Sequence[float], n_points: int, smooth_method: str, smooth_window: int, smooth_sigma: float | None, smooth_pad_mode: str) -> tuple[list[float], list[float]]:
        xs, ys = resample_to_n(x, y, n_points=n_points)
        ys = smooth_series(
            ys,
            method=smooth_method,
            window=smooth_window,
            sigma=smooth_sigma,
            pad_mode=smooth_pad_mode,
        )
        return xs, ys

    def arc(
        self,
        text: str,
        *,
        n_points: int = 100,
        smooth_method: str = "moving_average",
        smooth_window: int = 7,
        smooth_sigma: float | None = None,
        smooth_pad_mode: str = "reflect",
        score_col: str = "score",
        score_cols: Optional[Iterable[str]] = None,
    ) -> ArcResult:
        df = self.score(text)
        x = [float(v) for v in df["rel_pos"].tolist()] if not df.empty else []

        if score_col == "probs":
            labels = sorted({k for row in df.get("probs", []) for k in row.keys()})
            series: Dict[str, List[float]] = {}
            for label in labels:
                y = [float(p.get(label, 0.0)) for p in df["probs"].tolist()]
                xs, ys = self._smooth_scalar(x, y, n_points, smooth_method, smooth_window, smooth_sigma, smooth_pad_mode)
                series[label] = ys
            return ArcResult(x=xs if labels else [], y_series=series, raw_x=x)

        if score_cols:
            series: Dict[str, List[float]] = {}
            xs: List[float] = []
            for col in score_cols:
                if col not in df.columns:
                    continue
                y = [float(v) for v in df[col].tolist()]
                xs, ys = self._smooth_scalar(x, y, n_points, smooth_method, smooth_window, smooth_sigma, smooth_pad_mode)
                series[col] = ys
            return ArcResult(x=xs, y_series=series, raw_x=x)

        if score_col not in df.columns:
            raise ValueError(f"Unknown score column: {score_col}")

        y = [float(v) for v in df[score_col].tolist()] if not df.empty else []
        xs, ys = self._smooth_scalar(x, y, n_points, smooth_method, smooth_window, smooth_sigma, smooth_pad_mode)
        return ArcResult(x=xs, y=ys, raw_x=x, raw_y=y)
