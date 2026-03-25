from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .core import Fabula
from .scorer import TransformersScorer
from .segment import ParagraphSegmenter, RegexSentenceSegmenter


@dataclass
class DummyScorer:
    analysis: str = "sentiment"

    def predict_proba(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for t in texts:
            if any(w in t.lower() for w in ["triste", "peur", "colère", "haine", "bad"]):
                out.append({"positive": 0.2, "negative": 0.8})
            else:
                out.append({"positive": 0.8, "negative": 0.2})
        return out


def _read_text(input_path: str, encoding: str = "utf-8") -> str:
    if input_path == "-":
        return sys.stdin.read()
    return Path(input_path).read_text(encoding=encoding)


def _build_scorer(args: argparse.Namespace):
    if args.dummy:
        return DummyScorer(analysis=args.analysis)
    return TransformersScorer(model=args.model)


def _build_segmenter(kind: str):
    if kind == "sentence":
        return RegexSentenceSegmenter()
    if kind == "paragraph":
        return ParagraphSegmenter()
    raise ValueError(f"Unsupported segmenter: {kind}")


def cmd_score(args: argparse.Namespace) -> int:
    text = _read_text(args.input, encoding=args.encoding)
    fb = Fabula(scorer=_build_scorer(args), segmenter=_build_segmenter(args.segment), analysis=args.analysis)
    df = fb.score(text)

    if args.format == "csv":
        sys.stdout.write(df.to_csv(index=False))
    else:
        sys.stdout.write(df.to_json(orient="records", force_ascii=False))
        sys.stdout.write("\n")
    return 0


def cmd_arc(args: argparse.Namespace) -> int:
    text = _read_text(args.input, encoding=args.encoding)
    fb = Fabula(scorer=_build_scorer(args), segmenter=_build_segmenter(args.segment), analysis=args.analysis)
    arc = fb.arc(text, n_points=args.n_points, smooth_window=args.smooth_window)
    rows = [{"x": x, "y": y} for x, y in zip(arc.x, arc.y or [])]
    if args.format == "csv":
        sys.stdout.write("x,y\n")
        for row in rows:
            sys.stdout.write(f"{row['x']},{row['y']}\n")
    else:
        sys.stdout.write(json.dumps(rows, ensure_ascii=False))
        sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fabula")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("input", help="Text input file path or '-' for stdin")
    common.add_argument("--analysis", choices=["sentiment", "emotion"], default="sentiment")
    common.add_argument("--segment", choices=["sentence", "paragraph"], default="sentence")
    common.add_argument("--dummy", action="store_true")
    common.add_argument("--model", default="cmarkea/distilcamembert-base-sentiment")
    common.add_argument("--encoding", default="utf-8")

    p_score = sub.add_parser("score", parents=[common])
    p_score.add_argument("--format", choices=["json", "csv"], default="json")
    p_score.set_defaults(func=cmd_score)

    p_arc = sub.add_parser("arc", parents=[common])
    p_arc.add_argument("--n-points", type=int, default=100)
    p_arc.add_argument("--smooth-window", type=int, default=7)
    p_arc.add_argument("--format", choices=["json", "csv"], default="json")
    p_arc.set_defaults(func=cmd_arc)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
