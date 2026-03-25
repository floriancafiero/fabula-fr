from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .schemas import Segment


@dataclass
class RegexSentenceSegmenter:
    pattern: str = r"[^.!?]+[.!?]"
    min_len: int = 1

    def segment(self, text: str) -> List[Segment]:
        matches = [m.group(0).strip() for m in re.finditer(self.pattern, text, flags=re.UNICODE)]
        if not matches:
            stripped = text.strip()
            matches = [stripped] if stripped else []

        out: List[Segment] = []
        total = len(matches)
        for idx, chunk in enumerate(matches):
            if len(chunk) < self.min_len:
                continue
            rel_pos = (idx + 1) / max(total, 1)
            out.append(Segment(text=chunk, idx=len(out), rel_pos=rel_pos))
        return out


@dataclass
class ParagraphSegmenter:
    min_len: int = 1

    def segment(self, text: str) -> List[Segment]:
        blocks = [b.strip() for b in text.split("\n\n")]
        kept = [b for b in blocks if len(b) >= self.min_len]
        out: List[Segment] = []
        total = len(kept)
        for idx, block in enumerate(kept):
            rel_pos = (idx + 1) / max(total + 1, 1)
            out.append(Segment(text=block, idx=idx, rel_pos=rel_pos))
        return out


@dataclass
class SlidingWindowTokenSegmenter:
    tokenizer: Optional[object] = None
    window_tokens: int = 256
    stride_tokens: int = 64
    min_tokens: int = 16

    def _tokenize(self, text: str) -> List[str]:
        if self.tokenizer is not None:
            return self.tokenizer.tokenize(text)
        return text.split()

    def segment(self, text: str) -> List[Segment]:
        tokens = self._tokenize(text)
        if not tokens:
            return []
        window = max(1, self.window_tokens)
        stride = max(1, self.stride_tokens)
        out: List[Segment] = []
        for start in range(0, len(tokens), stride):
            chunk_tokens = tokens[start : start + window]
            if len(chunk_tokens) < self.min_tokens:
                continue
            chunk_text = " ".join(chunk_tokens)
            rel_pos = min(1.0, (start + len(chunk_tokens) / 2) / len(tokens))
            out.append(Segment(text=chunk_text, idx=len(out), rel_pos=rel_pos))
            if start + window >= len(tokens):
                break
        return out


@dataclass
class DocumentChunkTokenSegmenter:
    tokenizer: Optional[object] = None
    chunk_tokens: int = 1024
    stride_tokens: int = 1024
    min_tokens: int = 128

    def _tokenize(self, text: str) -> List[str]:
        if self.tokenizer is not None:
            return self.tokenizer.tokenize(text)
        return text.split()

    def segment(self, text: str) -> List[Segment]:
        tokens = self._tokenize(text)
        if not tokens:
            return []
        chunk = max(1, self.chunk_tokens)
        stride = max(1, self.stride_tokens)
        out: List[Segment] = []
        for start in range(0, len(tokens), stride):
            chunk_tokens = tokens[start : start + chunk]
            if len(chunk_tokens) < self.min_tokens:
                continue
            chunk_text = " ".join(chunk_tokens)
            rel_pos = min(1.0, (start + len(chunk_tokens) / 2) / len(tokens))
            out.append(Segment(text=chunk_text, idx=len(out), rel_pos=rel_pos))
            if start + chunk >= len(tokens):
                break
        return out
