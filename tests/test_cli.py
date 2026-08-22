from __future__ import annotations

from pathlib import Path

from alignment.cli import main
from alignment.engine import parse_llm_output


def test_report_sample(capsys):
    assert main(["report", "-s", "flow"]) == 0
    out = capsys.readouterr().out
    assert "Gate decision : FLOW" in out
    assert "mean(v) =   7.90" in out


def test_report_example_file():
    path = Path(__file__).resolve().parents[1] / "examples" / "refuse.json"
    assert path.is_file()
    result = parse_llm_output(path.read_text(encoding="utf-8"))
    assert result.gate == "REFUSE"


def test_report_missing_file(capsys):
    assert main(["report", "path/to/llm.json"]) == 2
    err = capsys.readouterr().err
    assert "No such file: path/to/llm.json" in err
    assert "report -s flow" in err


def test_report_requires_input_or_sample(capsys):
    assert main(["report"]) == 2
    err = capsys.readouterr().err
    assert "-s" in err
