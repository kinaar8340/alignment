"""Personal / household dashboard.

Daily practice of the ten-slice geometry and Narrow Path OS recovery.
The log is a local JSON file — no server, no central authority.
Households share a file only if they choose to.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

from .dashboard import create_dashboard
from .engine import ScoreResult, parse_llm_output
from .slices import LAW_OF_MOTION, RECOVERY
from .x_reply import dominant_fruit

DEFAULT_LOG = Path(os.environ.get("ALIGNMENT_HOUSEHOLD_LOG", "household.json"))


def gate_note(result: ScoreResult) -> str:
    """Short amplification or recovery line from the gate. Does not re-score."""
    fruit = dominant_fruit(result)
    if result.gate == "FLOW":
        return f"Amplify. {LAW_OF_MOTION}"
    if result.gate == "CORRECT":
        return f"σ(Z) on. Re-introduce the Observer. {RECOVERY}"
    if result.gate == "STOP_FEEDING":
        return f"Stop feeding {fruit}. {RECOVERY}"
    return f"Refuse {fruit}. Adaptation that excludes the Observer is not taken. {RECOVERY}"


@dataclass
class Entry:
    timestamp: str
    label: str
    text: str
    result: dict
    note: str = ""
    who: str = ""

    @classmethod
    def from_result(
        cls,
        label: str,
        text: str,
        result: ScoreResult,
        note: str = "",
        who: str = "",
    ) -> "Entry":
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            label=label,
            text=text,
            result=result.to_dict(),
            note=note,
            who=who,
        )

    def score(self) -> ScoreResult:
        return parse_llm_output(self.result, source=self.label)

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "Entry":
        return cls(
            timestamp=str(raw.get("timestamp", "")),
            label=str(raw.get("label", "")),
            text=str(raw.get("text", "")),
            result=dict(raw.get("result") or {}),
            note=str(raw.get("note", "")),
            who=str(raw.get("who", "")),
        )


@dataclass
class HouseholdLog:
    path: Path
    entries: List[Entry] = field(default_factory=list)

    def add(self, entry: Entry) -> None:
        self.entries.append(entry)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(e) for e in self.entries]
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    @classmethod
    def load(cls, path: Union[str, Path, None] = None) -> "HouseholdLog":
        path = Path(path) if path is not None else DEFAULT_LOG
        if not path.is_file():
            return cls(path=path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"household log must be a JSON list: {path}")
        return cls(path=path, entries=[Entry.from_mapping(e) for e in raw])

    def trajectory(self) -> Dict[str, Any]:
        if not self.entries:
            return {"n": 0}
        means_v = [float(e.result["mean_v"]) for e in self.entries]
        mins_v = [float(e.result["min_v"]) for e in self.entries]
        gates = [str(e.result["gate"]) for e in self.entries]
        counts: Dict[str, int] = {}
        for g in gates:
            counts[g] = counts.get(g, 0) + 1
        return {
            "n": len(self.entries),
            "latest_mean_v": means_v[-1],
            "latest_min_v": mins_v[-1],
            "avg_mean_v": round(sum(means_v) / len(means_v), 4),
            "avg_min_v": round(sum(mins_v) / len(mins_v), 4),
            "gate_counts": counts,
            "latest_gate": gates[-1],
            "latest_label": self.entries[-1].label,
        }

    def render_latest(self, out: Union[str, Path]) -> Path:
        if not self.entries:
            raise ValueError("No entries yet")
        latest = self.entries[-1]
        who = f" ({latest.who})" if latest.who else ""
        return create_dashboard(
            latest.score(),
            title=f"Household — {latest.label}{who}",
            save_path=str(out),
        )


def log_entry(
    *,
    label: str,
    text: str,
    result: ScoreResult,
    note: str = "",
    who: str = "",
    path: Union[str, Path, None] = None,
) -> Entry:
    """Append one scored choice to the household log and return the entry."""
    log = HouseholdLog.load(path)
    entry = Entry.from_result(label, text, result, note=note, who=who)
    log.add(entry)
    return entry
