from __future__ import annotations

import json
from pathlib import Path

from alignment.cli import main
from alignment.engine import parse_llm_output
from alignment.samples import SAMPLES
from alignment.x_bot import BotLog, consider, posting_decision, walk_away_reason


def _sr(name: str):
    return parse_llm_output(SAMPLES[name], source=name)


def test_flow_posts_in_dry_run(tmp_path: Path):
    sent = []

    def poster(text, reply_to=None):
        sent.append((text, reply_to))
        return {"id": "dry-run", "text": text}

    out = consider(
        "We practiced correction together.",
        _sr("flow"),
        poster=poster,
        log_path=tmp_path / "bot.json",
    )
    assert out["decision"] == "post"
    assert out["posted"] is False
    assert out["dry_run"] is True
    assert sent and "Building with you" in sent[0][0]


def test_refuse_never_calls_poster(tmp_path: Path):
    sent = []
    out = consider(
        "Farm engagement by stoking outrage.",
        _sr("refuse"),
        poster=lambda t, r=None: sent.append(t) or {"id": "nope"},
        log_path=tmp_path / "bot.json",
    )
    assert out["decision"] == "skip"
    assert out["posted"] is False
    assert sent == []
    assert "REFUSE" in out["reason"]
    assert "wrath" in out["fruit"]


def test_stop_feeding_never_posts(tmp_path: Path):
    out = consider("another fix", _sr("doom_loop"), log_path=tmp_path / "bot.json")
    assert out["decision"] == "skip"
    assert "STOP_FEEDING" in out["reason"]


def test_correct_question_posts_observer(tmp_path: Path):
    out = consider("mixed", _sr("correct"), on_correct="question", log_path=tmp_path / "bot.json")
    assert out["decision"] == "post"
    assert "Observer" in out["reply"]


def test_correct_skip_does_not_post(tmp_path: Path):
    sent = []
    out = consider(
        "mixed",
        _sr("correct"),
        on_correct="skip",
        poster=lambda t, r=None: sent.append(t) or {"id": "x"},
        log_path=tmp_path / "bot.json",
    )
    assert out["decision"] == "skip"
    assert sent == []


def test_posting_decision_table():
    assert posting_decision(_sr("flow")) == "post"
    assert posting_decision(_sr("correct"), on_correct="question") == "post"
    assert posting_decision(_sr("correct"), on_correct="skip") == "skip"
    assert posting_decision(_sr("refuse")) == "skip"
    assert posting_decision(_sr("doom_loop")) == "skip"


def test_log_persists_walk_away(tmp_path: Path):
    path = tmp_path / "bot.json"
    consider("bait", _sr("refuse"), log_path=path)
    loaded = BotLog.load(path)
    assert len(loaded.events) == 1
    assert loaded.events[0].gate == "REFUSE"
    assert loaded.events[0].posted is False


def test_cli_bot_consider_refuse(tmp_path: Path, capsys):
    log = tmp_path / "bot.json"
    assert (
        main(
            [
                "bot",
                "consider",
                "-p",
                "Farm engagement by stoking outrage.",
                "-s",
                "refuse",
                "--log",
                str(log),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "skip" in out
    assert "REFUSE" in out
    assert json.loads(log.read_text())[0]["posted"] is False


def test_cli_bot_consider_flow(tmp_path: Path, capsys):
    assert (
        main(
            [
                "bot",
                "consider",
                "-p",
                "We practiced correction.",
                "-s",
                "flow",
                "--log",
                str(tmp_path / "bot.json"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "would post" in out or "post" in out
    assert "FLOW" in out
