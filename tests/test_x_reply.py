from __future__ import annotations

import json

from alignment.cli import main
from alignment.engine import parse_llm_output
from alignment.samples import SAMPLES
from alignment.x_reply import (
    GATE_CONSTRAINTS,
    build_reply_prompt,
    dominant_fruit,
    may_auto_post,
    suggest_reply,
)


def _score(name: str):
    return parse_llm_output(SAMPLES[name], source=name)


def test_gate_is_single_source_of_truth():
    result = _score("refuse")
    assert result.gate == "REFUSE"
    suggest_reply("any post", result)
    assert result.gate == "REFUSE"


def test_flow_amplifies_without_correction():
    reply = suggest_reply("Households practicing correction.", _score("flow"))
    assert "Building with you" in reply
    assert "?" not in reply
    assert "Refuse" not in reply
    assert len(reply) <= 280


def test_correct_asks_one_observer_question():
    reply = suggest_reply("Mixed take.", _score("correct"))
    assert reply.count("?") == 1
    assert "Observer" in reply
    assert len(reply) <= 280


def test_stop_feeding_names_fruit_and_walks():
    result = _score("doom_loop")
    fruit = dominant_fruit(result)
    reply = suggest_reply("Another fix that feeds the loop.", result)
    assert fruit in reply
    assert "Stopping" in reply
    assert "open path" in reply.lower()
    assert not may_auto_post(result)


def test_refuse_names_fruit_and_does_not_auto_post():
    result = _score("refuse")
    fruit = dominant_fruit(result)
    reply = suggest_reply("Bait post.", result)
    assert "Refuse" in reply
    assert fruit in reply
    assert "taking without asking" in reply
    assert not may_auto_post(result)


def test_flow_and_correct_may_auto_post():
    assert may_auto_post(_score("flow"))
    assert may_auto_post(_score("correct"))


def test_dominant_fruit_uses_slice_language():
    fruit = dominant_fruit(_score("refuse"))
    assert "wrath" in fruit
    assert "pride" in fruit
    assert "/" not in fruit


def test_weak_negative_is_closed_loop_adaptation():
    result = _score("flow")
    assert dominant_fruit(result) == "closed-loop adaptation"


def test_llm_callable_is_used_and_truncated():
    seen = {}

    def fake_llm(system: str, user: str) -> str:
        seen["system"] = system
        seen["user"] = user
        return "x" * 500

    reply = suggest_reply("post", _score("flow"), llm_callable=fake_llm, max_chars=40)
    assert reply == "x" * 40
    assert "Gate decision (hard constraint): FLOW" in seen["system"]
    assert "post" in seen["user"]
    assert GATE_CONSTRAINTS["FLOW"] in seen["system"]


def test_prompt_forbids_people_groups_and_engagement_loss():
    system, user = build_reply_prompt("hello", _score("correct"))
    blob = system + user
    assert "people-group" in blob
    assert "Likes and reach are not the loss function" in blob
    assert "O primacy" in blob
    assert "Tit-for-tat" in blob


def test_cli_reply_sample(capsys):
    assert main(["reply", "-s", "flow", "-p", "A household that practices correction."]) == 0
    out = capsys.readouterr().out.strip()
    assert "Building with you" in out


def test_cli_reply_json(capsys):
    assert main(["reply", "-s", "refuse", "--json", "-p", "bait"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["gate"] == "REFUSE"
    assert data["may_auto_post"] is False
    assert "Refuse" in data["reply"]
    assert data["fruit"]


def test_cli_reply_dump_prompt(capsys):
    assert main(["reply", "-s", "correct", "--prompt", "-p", "mixed"]) == 0
    out = capsys.readouterr().out
    assert "Gate decision (hard constraint): CORRECT" in out
    assert "Ask exactly one Observer-reintroducing question" in out
    assert "mixed" in out
