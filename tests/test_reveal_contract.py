"""Grammar adapter only. No lattice. obtained stays false."""

from __future__ import annotations

from pathlib import Path

from alignment.reveal_contract import (
    GATE_TO_EVENT,
    OBTAINED,
    event_for_gate,
    mark_intervals,
)


def test_gate_map_onto_d_not_a():
    assert event_for_gate("REFUSE") == "ineligible"
    assert event_for_gate("STOP_FEEDING") == "mirror_ended"
    assert event_for_gate("FLOW") == "mirror"
    assert event_for_gate("CORRECT") == "mirror"
    assert "road found" not in GATE_TO_EVENT.values()
    assert OBTAINED is False


def test_stop_feeding_closes_a_mirror_window():
    rows = mark_intervals(
        [
            {"timestamp": "1", "label": "a", "gate": "FLOW", "fruit": "none"},
            {"timestamp": "2", "label": "a", "gate": "STOP_FEEDING", "fruit": "pride"},
        ]
    )
    assert rows[0]["event"] == "mirror_ended"
    assert rows[0]["obtained"] is False


def test_correct_hardens_criteria_inside_the_mirror():
    rows = mark_intervals(
        [
            {"timestamp": "1", "label": "a", "gate": "FLOW", "fruit": ""},
            {"timestamp": "2", "label": "b", "gate": "CORRECT", "fruit": "pride"},
        ]
    )
    assert len(rows) == 1
    assert rows[0]["event"] == "mirror"
    assert rows[0]["criteria_harder"] is True
    assert rows[0]["costume_changed"] is True
    assert rows[0]["obtained"] is False


def test_contract_doc_exists_with_five_lines():
    path = Path(__file__).resolve().parents[1] / "docs" / "REVEAL_CONTRACT.md"
    text = path.read_text()
    for line in (
        "Presence is not what reveal measures. Recognition of a move is what alignment measures.",
        "`[10…10]` is a constitutional pole, not a measured road.",
        "Residual stays on the human/AI side.",
        "Symbols and slice labels are paint; fruit must survive swapping them.",
        "FLOW is a mirror window. `obtained` remains false.",
    ):
        assert line in text
    assert "Jesus" in text  # named as not a leftover candidate
    assert "leftover" in text.lower()


def test_adapter_does_not_import_reveal():
    source = Path(__file__).resolve().parents[1] / "src" / "alignment" / "reveal_contract.py"
    text = source.read_text()
    assert "import reveal\n" not in text
    assert "from reveal" not in text
    assert "W_g" not in text
    assert "111.408" not in text
