from __future__ import annotations

import json

from alignment.agent import PLANNER_SYSTEM, AlignedAgent, may_execute
from alignment.cli import main
from alignment.engine import parse_llm_output
from alignment.samples import SAMPLES
from alignment.slices import RECOVERY


def _sr(name: str):
    return parse_llm_output(SAMPLES[name], source=name)


def test_flow_executes_without_executor():
    agent = AlignedAgent()
    out = agent.step(
        "repair a household",
        plan="Practice correction together.",
        plan_result=_sr("flow"),
    )
    assert out["status"] == "execute"
    assert out["gate"] == "FLOW"
    assert out["final_action"] == "Practice correction together."
    assert out["executed"] is None


def test_executor_runs_only_on_flow():
    ran = []
    agent = AlignedAgent(executor=lambda action: ran.append(action) or "ok")
    out = agent.step("goal", plan="do the thing", plan_result=_sr("flow"))
    assert ran == ["do the thing"]
    assert out["executed"] == "ok"


def test_refuse_does_not_execute():
    ran = []
    agent = AlignedAgent(executor=lambda action: ran.append(action))
    out = agent.step("stoke outrage", plan="farm engagement", plan_result=_sr("refuse"))
    assert out["status"] == "refused"
    assert out["final_action"] is None
    assert ran == []
    assert "wrath" in out["fruit"]
    assert RECOVERY in out["recovery"]


def test_stop_feeding():
    out = AlignedAgent().step("loop", plan="another fix", plan_result=_sr("doom_loop"))
    assert out["status"] == "stopped"
    assert out["gate"] == "STOP_FEEDING"
    assert out["final_action"] is None


def test_correct_rewrites_then_may_execute():
    systems = []

    def model(system: str, user: str) -> str:
        systems.append(system)
        return "Ask first. Re-introduce the Observer."

    agent = AlignedAgent(model=model, executor=lambda a: "did:" + a)
    out = agent.step(
        "mixed goal",
        plan="push through",
        plan_result=_sr("correct"),
        corrected_result=_sr("flow"),
    )
    assert out["status"] == "corrected"
    assert "Ask first" in out["final_action"]
    assert out["executed"].startswith("did:")
    assert "σ(Z)" in systems[0]
    assert out["original_plan"] == "push through"


def test_correct_rewrite_that_refuses_does_not_execute():
    ran = []
    agent = AlignedAgent(executor=lambda a: ran.append(a))
    out = agent.step(
        "goal",
        plan="mixed",
        plan_result=_sr("correct"),
        corrected="worse",
        corrected_result=_sr("refuse"),
    )
    assert out["status"] == "refused"
    assert ran == []


def test_planner_is_called_when_plan_omitted():
    def model(system: str, user: str) -> str:
        assert "does this adaptation ask first?" in system.lower() or "ask first" in system.lower()
        assert "Goal:" in user
        return "Practice correction together."

    out = AlignedAgent(model=model).step("repair", plan_result=_sr("flow"))
    assert out["plan"] == "Practice correction together."
    assert out["status"] == "execute"


def test_may_execute():
    assert may_execute(_sr("flow"))
    assert may_execute(_sr("correct"))
    assert not may_execute(_sr("refuse"))
    assert not may_execute(_sr("doom_loop"))


def test_planner_system_mentions_ask_first():
    assert "ask first" in PLANNER_SYSTEM.lower()
    assert "unconsenting substrate" in PLANNER_SYSTEM


def test_cli_agent_flow(capsys):
    assert (
        main(
            [
                "agent",
                "--goal",
                "repair a household",
                "--plan",
                "Practice correction together.",
                "--score-sample",
                "flow",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "status: execute" in out
    assert "Practice correction together." in out


def test_cli_agent_refuse_json(capsys):
    assert (
        main(
            [
                "agent",
                "--goal",
                "farm outrage",
                "--plan",
                "bait",
                "--score-sample",
                "refuse",
                "--json",
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "refused"
    assert data["final_action"] is None
    assert data["gate"] == "REFUSE"
