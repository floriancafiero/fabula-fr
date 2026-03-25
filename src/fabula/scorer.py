from __future__ import annotations

from typing import Dict, Iterable, List, Optional


def valence_from_probs(probs: Dict[str, float]) -> float:
    norm = {str(k).lower(): float(v) for k, v in probs.items()}
    pos = norm.get("positive", norm.get("pos", norm.get("positive_label", 0.0)))
    neg = norm.get("negative", norm.get("neg", norm.get("negative_label", 0.0)))
    return pos - neg


class TransformersScorer:
    """Wrapper around a Hugging Face text-classification pipeline."""

    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 512,
        pooling: str = "none",
        pooling_stride_tokens: Optional[int] = None,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling = pooling
        self.pooling_stride_tokens = pooling_stride_tokens
        self._pipeline = None
        self.tokenizer = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            pipe_kwargs = {"task": "text-classification", "model": self.model}
            if self.device is not None:
                pipe_kwargs["device"] = self.device
            self._pipeline = pipeline(**pipe_kwargs)
            self.tokenizer = getattr(self._pipeline, "tokenizer", None)

    def predict_proba(self, texts: Iterable[str]) -> List[Dict[str, float]]:
        self._ensure_pipeline()
        preds = self._pipeline(
            list(texts),
            return_all_scores=True,
            batch_size=self.batch_size,
            truncation=True,
            max_length=self.max_length,
        )

        out: List[Dict[str, float]] = []
        for row in preds:
            out.append({str(item["label"]): float(item["score"]) for item in row})
        return out
