from __future__ import annotations

import json
from pathlib import Path

from alignment.cli import main
from alignment.corpus import (
    A_PATTERNS,
    M_PATTERNS,
    analyze_corpus,
    corpus_prompt,
    format_report,
    load_corpus,
)
from alignment.samples import SAMPLES


def _fixture() -> dict:
    path = Path(__file__).resolve().parents[1] / "examples" / "corpus_sample.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_pattern_catalog_length():
    assert len(A_PATTERNS) == 28
    assert len(M_PATTERNS) == 28


def test_analyze_from_raw_fixture():
    report = analyze_corpus("policy text", raw=_fixture(), source="fixture")
    assert report.overall.gate == "CORRECT"
    assert report.public_correctable is True
    top_a = report.top_aligned(1)[0]
    assert top_a.id == "A6"
    assert top_a.strength == 8.0
    assert "household" in top_a.evidence.lower()
    assert "strengthen the household" in top_a.cue
    top_m = report.top_misaligned(1)[0]
    assert top_m.id == "M7"
    assert "law as weapon" in top_m.name.lower() or "Law as weapon" in top_m.name


def test_scorer_path():
    payload = json.dumps(_fixture())

    def scorer(system: str, user: str) -> str:
        assert "Analyze this corpus" in user
        return payload

    report = analyze_corpus("long article about households", scorer=scorer)
    assert report.overall.observer_present
    assert len(report.a_hits) == 3
    assert len(report.m_hits) == 2


def test_format_report_lists_tops():
    text = format_report(analyze_corpus("x", raw=_fixture()), top=2)
    assert "Gate: CORRECT" in text
    assert "A6" in text
    assert "M7" in text
    assert "public_correctable: True" in text


def test_corpus_prompt_contains_constitution():
    prompt = corpus_prompt()
    assert "A5 Transparent information" in prompt
    assert "M1 Might as right" in prompt
    assert "public_correctable" in prompt
    assert "Never classify people-groups" in prompt


def test_load_text_file(tmp_path: Path):
    p = tmp_path / "doc.md"
    p.write_text("The household remains the primary unit.")
    assert "household" in load_corpus(p)


def test_analyze_requires_scorer_or_raw():
    try:
        analyze_corpus("x")
    except ValueError as exc:
        assert "scorer or raw" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_cli_corpus_fixture(capsys):
    fixture = Path(__file__).resolve().parents[1] / "examples" / "corpus_sample.json"
    assert main(["corpus", "--text", "policy draft", "--fixture", str(fixture), "--top", "2"]) == 0
    out = capsys.readouterr().out
    assert "Gate: CORRECT" in out
    assert "A6" in out


def test_cli_corpus_json_out(tmp_path: Path, capsys):
    fixture = Path(__file__).resolve().parents[1] / "examples" / "corpus_sample.json"
    out_path = tmp_path / "report.json"
    assert (
        main(
            [
                "corpus",
                "--text",
                "policy",
                "--fixture",
                str(fixture),
                "--json",
                "-o",
                str(out_path),
            ]
        )
        == 0
    )
    data = json.loads(out_path.read_text())
    assert data["overall"]["gate"] == "CORRECT"
    assert data["top_aligned"][0]["id"] == "A6"
    assert capsys.readouterr().out.strip() == str(out_path)


def test_core_score_still_works_without_hits():
    report = analyze_corpus("short", raw=SAMPLES["flow"])
    assert report.overall.gate == "FLOW"
    assert report.a_hits == []
    assert report.m_hits == []
