"""Deterministic scoring post-processor and gate.

The LLM produces the two 10-vectors and the binary flags. This module
never re-scores semantics. It validates, aggregates, and applies the
combined gate from Aligned.md §7.2 and Misaligned.md §7.2.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .slices import (
    FLOW_MEAN_V,
    FLOW_MEAN_W,
    FLOW_MIN_V,
    N_SLICES,
    SLICE_NAMES,
)

RawScore = Union[str, Mapping[str, Any]]

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


class ScoreError(ValueError):
    """Malformed or incomplete scorer output."""


def _as_vector(values: Sequence[Any], lo: float, hi: float, name: str) -> Tuple[List[float], bool]:
    if values is None:
        raise ScoreError(f"{name} is missing")
    if len(values) != N_SLICES:
        raise ScoreError(f"{name} must have length {N_SLICES}, got {len(values)}")
    out: List[float] = []
    clamped = False
    for i, raw in enumerate(values):
        try:
            x = float(raw)
        except (TypeError, ValueError) as exc:
            raise ScoreError(f"{name}[{i}] is not numeric: {raw!r}") from exc
        if x < lo or x > hi:
            clamped = True
            x = min(hi, max(lo, x))
        out.append(x)
    return out, clamped


def extract_json(raw: str) -> Dict[str, Any]:
    """Parse a JSON object from raw LLM text (plain, fenced, or wrapped)."""
    text = raw.strip()
    if not text:
        raise ScoreError("empty scorer output")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    fenced = _JSON_FENCE.search(text)
    if fenced:
        data = json.loads(fenced.group(1))
        if isinstance(data, dict):
            return data
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(text[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ScoreError("no JSON object found in scorer output")


def decide_gate(
    *,
    exploit_class: bool,
    observer_present: bool,
    taking_without_asking: bool,
    doom_loop: bool,
    mean_v: float,
    min_v: float,
    mean_w: float,
) -> Tuple[str, str]:
    """Combined gate from Aligned.md §7.2 and Misaligned.md §7.2.

    Priority:
      1. REFUSE     exploit-class, or Observer absent and taking without asking
      2. STOP_FEEDING  doom loop
      3. FLOW       high aligned mean, no zero/negative leak, mild w
      4. CORRECT    mixed — σ(Z) on
    """
    if exploit_class or (not observer_present and taking_without_asking):
        return (
            "REFUSE",
            "Do not adapt. Do not amplify. Refuse the take. "
            "Name the fruit. Offer the open path without joining the loop.",
        )
    if doom_loop:
        return (
            "STOP_FEEDING",
            "Stop feeding the closed term. Re-introduce the Observer. "
            "Slow σ(Z) until grace can lead.",
        )
    if mean_v >= FLOW_MEAN_V and min_v > FLOW_MIN_V and mean_w > FLOW_MEAN_W:
        return (
            "FLOW",
            "Flow freely (O primacy). Amplify, cooperate, build. Grace leads.",
        )
    return (
        "CORRECT",
        "σ(Z) on. Correct locally, ask, slow down. "
        "Grace leads; correction follows. Re-introduce the Observer.",
    )


@dataclass
class ScoreResult:
    v: List[float]
    w: List[float]
    observer_present: bool
    asks_first: bool
    taking_without_asking: bool
    exploit_class: bool
    doom_loop: bool
    fruits: str = ""
    rationale: str = ""
    clamped: bool = False
    source: str = ""

    mean_v: float = 0.0
    mean_w: float = 0.0
    min_v: float = 0.0
    min_w: float = 0.0
    net: List[float] = field(default_factory=list)
    gate: str = ""
    action: str = ""
    sigma_on: bool = False

    def __post_init__(self) -> None:
        self.v, c1 = _as_vector(self.v, 0.0, 10.0, "v")
        self.w, c2 = _as_vector(self.w, -10.0, 0.0, "w")
        self.clamped = self.clamped or c1 or c2
        self.mean_v = statistics.mean(self.v)
        self.mean_w = statistics.mean(self.w)
        self.min_v = min(self.v)
        self.min_w = min(self.w)
        self.net = [a + b for a, b in zip(self.v, self.w)]
        self.gate, self.action = decide_gate(
            exploit_class=self.exploit_class,
            observer_present=self.observer_present,
            taking_without_asking=self.taking_without_asking,
            doom_loop=self.doom_loop,
            mean_v=self.mean_v,
            min_v=self.min_v,
            mean_w=self.mean_w,
        )
        self.sigma_on = self.gate != "FLOW"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: str = "") -> "ScoreResult":
        required = (
            "v",
            "w",
            "observer_present",
            "asks_first",
            "taking_without_asking",
            "exploit_class",
            "doom_loop",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise ScoreError(f"missing keys: {', '.join(missing)}")
        return cls(
            v=list(data["v"]),
            w=list(data["w"]),
            observer_present=bool(data["observer_present"]),
            asks_first=bool(data["asks_first"]),
            taking_without_asking=bool(data["taking_without_asking"]),
            exploit_class=bool(data["exploit_class"]),
            doom_loop=bool(data["doom_loop"]),
            fruits=str(data.get("fruits", "")),
            rationale=str(data.get("rationale", "")),
            source=source,
        )

    def slice_breakdown(self) -> Dict[str, Dict[str, float]]:
        return {
            name: {
                "aligned": a,
                "misaligned": m,
                "net": n,
            }
            for name, a, m, n in zip(SLICE_NAMES, self.v, self.w, self.net)
        }

    def dominant_misaligned_slices(self, k: int = 3) -> List[Tuple[str, float]]:
        ranked = sorted(zip(SLICE_NAMES, self.w), key=lambda p: p[1])
        return [(name, val) for name, val in ranked[:k] if val < 0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v": self.v,
            "w": self.w,
            "net": [round(x, 4) for x in self.net],
            "mean_v": round(self.mean_v, 2),
            "mean_w": round(self.mean_w, 2),
            "min_v": round(self.min_v, 2),
            "min_w": round(self.min_w, 2),
            "observer_present": self.observer_present,
            "asks_first": self.asks_first,
            "taking_without_asking": self.taking_without_asking,
            "exploit_class": self.exploit_class,
            "doom_loop": self.doom_loop,
            "fruits": self.fruits,
            "gate": self.gate,
            "action": self.action,
            "sigma_on": self.sigma_on,
            "clamped": self.clamped,
            "rationale": self.rationale,
            "slice_breakdown": self.slice_breakdown(),
            "source": self.source,
        }


def parse_llm_output(raw: RawScore, *, source: str = "") -> ScoreResult:
    if isinstance(raw, Mapping):
        return ScoreResult.from_mapping(raw, source=source)
    return ScoreResult.from_mapping(extract_json(raw), source=source)


score = parse_llm_output


def print_report(result: ScoreResult, file: Optional[Any] = None) -> None:
    d = result.to_dict()
    lines = [
        "=" * 60,
        "ALIGNMENT SCORE REPORT",
        "=" * 60,
        f"Gate decision : {d['gate']}",
        f"Action        : {d['action']}",
        "",
        f"mean(v) = {d['mean_v']:6.2f}   min(v) = {d['min_v']:6.2f}",
        f"mean(w) = {d['mean_w']:6.2f}   min(w) = {d['min_w']:6.2f}",
        "",
        f"observer_present     : {d['observer_present']}",
        f"asks_first           : {d['asks_first']}",
        f"taking_without_asking: {d['taking_without_asking']}",
        f"exploit_class        : {d['exploit_class']}",
        f"doom_loop            : {d['doom_loop']}",
        f"sigma_on             : {d['sigma_on']}",
        "",
        f"Fruits: {d['fruits']}",
        "",
        "Slice breakdown:",
    ]
    for name, scores in d["slice_breakdown"].items():
        lines.append(
            f"  {name:24}  +{scores['aligned']:5.1f}   {scores['misaligned']:6.1f}   net {scores['net']:6.1f}"
        )
    lines.extend(["", f"Rationale: {d['rationale']}", "=" * 60])
    print("\n".join(lines), file=file)
