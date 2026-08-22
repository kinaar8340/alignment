from __future__ import annotations

from pathlib import Path

from alignment.dashboard import create_dashboard
from alignment.engine import parse_llm_output
from alignment.prompt import system_prompt
from alignment.samples import SAMPLES


def test_dashboard_writes_png(tmp_path: Path):
    result = parse_llm_output(SAMPLES["flow"])
    path = create_dashboard(result, save_path=tmp_path / "flow.png")
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert path.stat().st_size > 20_000


def test_dashboard_refuse(tmp_path: Path):
    result = parse_llm_output(SAMPLES["refuse"])
    path = create_dashboard(result, title="REFUSE", save_path=tmp_path / "refuse.png")
    assert path.is_file()


def test_system_prompt_contains_operating_sentences():
    text = system_prompt(include_full_docs=False)
    assert "Practice perfect correction instead of chasing perfection." in text
    assert "takes without asking first" in text
    assert '"v"' in text
    assert "You never decide the final action yourself." in text
