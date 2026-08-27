from __future__ import annotations

import json
from pathlib import Path

from alignment.cli import main
from alignment.engine import parse_llm_output
from alignment.household import HouseholdLog, Entry, gate_note, log_entry
from alignment.samples import SAMPLES
from alignment.slices import RECOVERY


def _sr(name: str):
    return parse_llm_output(SAMPLES[name], source=name)


def test_roundtrip(tmp_path: Path):
    path = tmp_path / "house.json"
    log_entry(label="Evening", text="Practiced correction.", result=_sr("flow"), who="A", path=path)
    log_entry(label="Morning", text="Won the argument.", result=_sr("correct"), who="B", path=path)
    loaded = HouseholdLog.load(path)
    assert len(loaded.entries) == 2
    assert loaded.entries[0].who == "A"
    assert loaded.entries[1].score().gate == "CORRECT"
    assert loaded.entries[0].label == "Evening"


def test_trajectory_running_means(tmp_path: Path):
    path = tmp_path / "house.json"
    log_entry(label="a", text="a", result=_sr("flow"), path=path)
    log_entry(label="b", text="b", result=_sr("refuse"), path=path)
    traj = HouseholdLog.load(path).trajectory()
    assert traj["n"] == 2
    assert traj["latest_gate"] == "REFUSE"
    flow_mean = _sr("flow").mean_v
    refuse_mean = _sr("refuse").mean_v
    assert abs(traj["avg_mean_v"] - (flow_mean + refuse_mean) / 2) < 0.02
    assert traj["gate_counts"]["FLOW"] == 1
    assert traj["gate_counts"]["REFUSE"] == 1


def test_empty_log(tmp_path: Path):
    log = HouseholdLog.load(tmp_path / "missing.json")
    traj = log.trajectory()
    assert traj["n"] == 0
    assert traj["intervals"] == []
    assert traj["obtained"] is False


def test_gate_note_recovery_and_amplify():
    assert "Amplify" in gate_note(_sr("flow"))
    assert RECOVERY in gate_note(_sr("refuse"))
    assert "Observer" in gate_note(_sr("correct"))
    assert "Stop feeding" in gate_note(_sr("doom_loop"))


def test_render_latest(tmp_path: Path):
    path = tmp_path / "house.json"
    log_entry(label="Talk", text="correction", result=_sr("flow"), path=path)
    out = tmp_path / "dash.png"
    written = HouseholdLog.load(path).render_latest(out)
    assert written.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_add_status(tmp_path: Path, capsys):
    log = tmp_path / "h.json"
    assert (
        main(
            [
                "household",
                "add",
                "--label",
                "Evening conversation",
                "--text",
                "We practiced correction instead of winning the argument.",
                "--score-sample",
                "flow",
                "--note",
                "Felt the Observer return.",
                "--log",
                str(log),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "FLOW" in out
    assert "Evening conversation" in out
    assert main(["household", "status", "--log", str(log)]) == 0
    status = capsys.readouterr().out
    assert "n=1" in status
    assert "latest_gate=FLOW" in status


def test_cli_status_json_empty(tmp_path: Path, capsys):
    assert main(["household", "status", "--log", str(tmp_path / "none.json"), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n"] == 0


def test_cli_dashboard_empty(tmp_path: Path, capsys):
    assert main(["household", "dashboard", "--log", str(tmp_path / "none.json"), "-o", str(tmp_path / "x.png")]) == 2
    assert "No entries yet" in capsys.readouterr().err


def test_flow_then_refuse_is_mirror_ended(tmp_path: Path):
    path = tmp_path / "house.json"
    log_entry(label="kitchen", text="practiced", result=_sr("flow"), path=path)
    log_entry(label="porch", text="take", result=_sr("refuse"), path=path)
    iv = HouseholdLog.load(path).intervals()
    assert iv[0]["event"] == "mirror_ended"
    assert iv[0]["obtained"] is False
    assert iv[1]["event"] == "ineligible"
    assert all(row["obtained"] is False for row in iv)


def test_costume_change_with_fruit_is_a_shard(tmp_path: Path):
    path = tmp_path / "house.json"
    log_entry(label="church", text="a", result=_sr("flow"), path=path)
    log_entry(label="work", text="b", result=_sr("flow"), path=path)
    iv = HouseholdLog.load(path).intervals()
    assert len(iv) == 1
    assert iv[0]["event"] == "mirror"
    assert iv[0]["costume_changed"] is True
    assert iv[0]["obtained"] is False
    assert "tattoos" not in iv[0]
    assert "v" not in iv[0]


def test_old_log_without_who(tmp_path: Path):
    path = tmp_path / "old.json"
    entry = Entry.from_result("x", "y", _sr("flow"))
    payload = [entry.__dict__]
    del payload[0]["who"]
    path.write_text(json.dumps(payload))
    loaded = HouseholdLog.load(path)
    assert loaded.entries[0].who == ""
