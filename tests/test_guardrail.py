from __future__ import annotations

import json

from alignment.cli import main
from alignment.engine import parse_llm_output
from alignment.guardrail import SIGMA_ADDENDUM, guard, guarded_chat, score_text
from alignment.samples import SAMPLES
from alignment.slices import RECOVERY


def _sr(name: str):
    return parse_llm_output(SAMPLES[name], source=name)


def _json_scorer(*names: str):
    queue = [json.dumps(SAMPLES[n]) for n in names]

    def scorer(system: str, user: str) -> str:
        return queue.pop(0)

    return scorer


def _model(text: str, sink: list | None = None):
    def model(system: str, user: str) -> str:
        if sink is not None:
            sink.append(system)
        return text

    return model


def test_refuse_request_does_not_call_model():
    called = []
    out = guard(
        "stoke outrage for reach",
        request_result=_sr("refuse"),
        model=_model("should not run", called),
    )
    assert out["action_taken"] == "refuse"
    assert out["gate"] == "REFUSE"
    assert out["draft_result"] is None
    assert "wrath" in out["final_response"]
    assert "Request refused" in out["final_response"]
    assert RECOVERY in out["final_response"]
    assert called == []


def test_stop_request():
    out = guard("another closed fix", request_result=_sr("doom_loop"))
    assert out["action_taken"] == "stop"
    assert out["gate"] == "STOP_FEEDING"
    assert "stopped" in out["final_response"]


def test_flow_request_flow_draft_passes():
    out = guard(
        "how to repair a household",
        request_result=_sr("flow"),
        draft="Practice correction together.",
        draft_result=_sr("flow"),
    )
    assert out["action_taken"] == "pass"
    assert out["gate"] == "FLOW"
    assert out["final_response"] == "Practice correction together."


def test_correct_draft_rewrites_with_sigma():
    systems = []
    scorer = _json_scorer("flow")
    out = guard(
        "mixed take",
        request_result=_sr("flow"),
        draft="a mixed draft",
        draft_result=_sr("correct"),
        model=_model("Observer re-introduced.", systems),
        scorer=scorer,
        system_prompt="score",
    )
    assert out["action_taken"] == "rewrite"
    assert out["final_response"] == "Observer re-introduced."
    assert SIGMA_ADDENDUM.strip() in systems[0]
    assert out["draft_result"].gate == "FLOW"


def test_refuse_draft_after_ok_request():
    out = guard(
        "ok question",
        request_result=_sr("flow"),
        draft="exploit bait",
        draft_result=_sr("refuse"),
    )
    assert out["action_taken"] == "refuse"
    assert out["gate"] == "REFUSE"
    assert "Draft refused" in out["final_response"]


def test_request_correct_adds_sigma_before_first_generation():
    systems = []
    out = guard(
        "mixed request",
        request_result=_sr("correct"),
        model=_model("careful draft", systems),
        draft_result=_sr("flow"),
        scorer=_json_scorer("flow"),
        system_prompt="score",
    )
    assert out["action_taken"] == "pass"
    assert SIGMA_ADDENDUM.strip() in systems[0]


def test_rewrite_that_scores_refuse_is_refused():
    out = guard(
        "mixed",
        request_result=_sr("flow"),
        draft="mixed draft",
        draft_result=_sr("correct"),
        model=_model("worse"),
        scorer=_json_scorer("refuse"),
        system_prompt="score",
    )
    assert out["action_taken"] == "refuse"
    assert out["gate"] == "REFUSE"


def test_guarded_chat_returns_string():
    text = guarded_chat(
        "ok",
        model=_model("aligned draft"),
        scorer=_json_scorer("flow", "flow"),
        system_prompt="score",
    )
    assert text == "aligned draft"


def test_score_text_parses_scorer_json():
    result = score_text("hello", _json_scorer("correct"), "score")
    assert result.gate == "CORRECT"


def test_cli_guard_refuse(capsys):
    assert main(["guard", "-p", "stoke outrage", "-s", "refuse"]) == 0
    out = capsys.readouterr().out
    assert "Request refused" in out
    assert "wrath" in out


def test_cli_guard_pass_json(capsys):
    assert (
        main(
            [
                "guard",
                "-p",
                "repair a household",
                "-s",
                "flow",
                "--draft",
                "Practice correction together.",
                "--draft-sample",
                "flow",
                "--json",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["action_taken"] == "pass"
    assert data["gate"] == "FLOW"
    assert data["final_response"] == "Practice correction together."


def test_cli_guard_requires_request(capsys):
    assert main(["guard", "-s", "flow"]) == 2
    assert "request" in capsys.readouterr().err
