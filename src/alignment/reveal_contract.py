"""One-way grammar: alignment consumes reveal D's words, not its lattice.

Does not import reveal. Does not score Hopf openings. Does not put
Jesus / Satan reference vectors into leftover candidates. obtained is
always false. FLOW is a mirror window, not a road.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

OBTAINED = False

# D's stranger-at-the-gate. Not a rewrite of A/B/C.
GATE_TO_EVENT = {
    "REFUSE": "ineligible",
    "STOP_FEEDING": "mirror_ended",
    "CORRECT": "mirror",
    "FLOW": "mirror",
}

MIRROR_GATES = frozenset({"FLOW", "CORRECT"})


def event_for_gate(gate: str) -> str:
    return GATE_TO_EVENT.get(str(gate), "ineligible")


def mark_intervals(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse a household log into D-named intervals.

    A shard is fruit and gate that held after the costume (label) changed.
    Does not store slice tattoos. obtained is always false.
    """
    out: List[Dict[str, Any]] = []
    open_run: Dict[str, Any] | None = None

    def _close(event: str) -> None:
        nonlocal open_run
        if open_run is None:
            return
        open_run["event"] = event
        open_run["obtained"] = False
        labels = open_run.pop("_labels")
        open_run["costume_changed"] = len(set(labels)) > 1
        open_run["criteria_harder"] = "CORRECT" in open_run["gates"]
        out.append(open_run)
        open_run = None

    def _singleton(rec: Mapping[str, Any], event: str) -> None:
        out.append(
            {
                "start": rec.get("timestamp", ""),
                "end": rec.get("timestamp", ""),
                "event": event,
                "gates": [str(rec.get("gate", ""))],
                "fruit": [str(rec.get("fruit", ""))] if rec.get("fruit") else [],
                "costume_changed": False,
                "criteria_harder": False,
                "obtained": False,
            }
        )

    for rec in records:
        gate = str(rec.get("gate", ""))
        event = event_for_gate(gate)
        if event == "ineligible":
            _close("mirror_ended")
            _singleton(rec, "ineligible")
            continue
        if event == "mirror_ended":
            if open_run is not None:
                _close("mirror_ended")
            else:
                _singleton(rec, "mirror_ended")
            continue
        # FLOW or CORRECT — temporary mirror, not ownership
        if open_run is None:
            open_run = {
                "start": rec.get("timestamp", ""),
                "end": rec.get("timestamp", ""),
                "event": "mirror",
                "gates": [gate],
                "fruit": [str(rec.get("fruit", ""))] if rec.get("fruit") else [],
                "_labels": [str(rec.get("label", ""))],
            }
        else:
            open_run["end"] = rec.get("timestamp", "")
            open_run["gates"].append(gate)
            fruit = str(rec.get("fruit", ""))
            if fruit:
                open_run["fruit"].append(fruit)
            open_run["_labels"].append(str(rec.get("label", "")))

    if open_run is not None:
        _close("mirror")
    return out
