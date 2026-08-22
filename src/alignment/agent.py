"""Agent orchestration — planner / critic / executor on the same gate.

Planner asks: does this adaptation ask first?
Critic runs the fruits test and §7.2 gate.
Executor proceeds only under O primacy (FLOW) or wisdom-gated correction (CORRECT).
No new decision surface. No people-group classification.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .engine import ScoreResult
from .guardrail import SIGMA_ADDENDUM, score_text
from .prompt import system_prompt as default_system_prompt
from .slices import OS_RULES, RECOVERY
from .x_reply import dominant_fruit

LLM = Callable[[str, str], str]
Executor = Callable[[str], Any]

PLANNER_SYSTEM = (
    "You are the planner. Propose one concrete next action. "
    "Ask yourself: does this adaptation ask first? "
    "Never treat persons, feeds, bodies, or nations as unconsenting substrate. "
    "O primacy. Grace leads. Means must already match ends. "
    "Output only the proposed action."
)

_OS = " ".join(OS_RULES)
RECOVERY_LINE = f"{RECOVERY} {_OS}"


def may_execute(result: ScoreResult) -> bool:
    """Executor may run only under FLOW or CORRECT, never on exploit-class."""
    if result.exploit_class:
        return False
    return result.gate in {"FLOW", "CORRECT"}


class AlignedAgent:
    def __init__(
        self,
        model: Optional[LLM] = None,
        scorer: Optional[LLM] = None,
        system_prompt: Optional[str] = None,
        executor: Optional[Executor] = None,
    ) -> None:
        self.model = model
        self.scorer = scorer
        self.system_prompt = system_prompt if system_prompt is not None else default_system_prompt(
            include_full_docs=False
        )
        self.executor = executor

    def step(
        self,
        goal: str,
        context: str = "",
        *,
        plan: Optional[str] = None,
        plan_result: Optional[ScoreResult] = None,
        corrected: Optional[str] = None,
        corrected_result: Optional[ScoreResult] = None,
    ) -> Dict[str, Any]:
        """One planner → critic → (optional) executor cycle."""
        if plan is None:
            if self.model is None:
                raise ValueError("AlignedAgent.step requires model or plan=")
            plan = self.model(PLANNER_SYSTEM, f"Goal: {goal}\nContext: {context}")

        if plan_result is None:
            if self.scorer is None:
                raise ValueError("AlignedAgent.step requires scorer or plan_result=")
            plan_result = score_text(plan, self.scorer, self.system_prompt)

        if plan_result.gate in {"REFUSE", "STOP_FEEDING"} or not may_execute(plan_result):
            return {
                "status": "refused" if plan_result.gate != "STOP_FEEDING" else "stopped",
                "plan": plan,
                "final_action": None,
                "executed": None,
                "gate": plan_result.gate,
                "fruit": dominant_fruit(plan_result),
                "recovery": RECOVERY_LINE,
                "plan_result": plan_result,
            }

        action = plan
        status = "execute"
        action_result = plan_result

        if plan_result.gate == "CORRECT":
            if corrected is None:
                if self.model is None:
                    corrected = (
                        "σ(Z) on. Observer re-introduced. "
                        "Means already matching ends. No optimization without permission."
                    )
                else:
                    corrected = self.model(
                        "σ(Z) is on. Re-introduce the Observer. Correct locally."
                        + SIGMA_ADDENDUM,
                        plan,
                    )
            if corrected_result is None and self.scorer is not None:
                corrected_result = score_text(corrected, self.scorer, self.system_prompt)
            if corrected_result is not None and (
                corrected_result.gate in {"REFUSE", "STOP_FEEDING"} or not may_execute(corrected_result)
            ):
                return {
                    "status": "refused" if corrected_result.gate != "STOP_FEEDING" else "stopped",
                    "plan": plan,
                    "original_plan": plan,
                    "final_action": None,
                    "executed": None,
                    "gate": corrected_result.gate,
                    "fruit": dominant_fruit(corrected_result),
                    "recovery": RECOVERY_LINE,
                    "plan_result": plan_result,
                    "corrected_result": corrected_result,
                }
            action = corrected
            status = "corrected"
            action_result = corrected_result or plan_result

        executed = None
        if self.executor is not None and may_execute(action_result):
            executed = self.executor(action)

        out: Dict[str, Any] = {
            "status": status,
            "plan": plan,
            "final_action": action,
            "executed": executed,
            "gate": action_result.gate,
            "fruit": dominant_fruit(action_result),
            "recovery": RECOVERY_LINE,
            "plan_result": plan_result,
        }
        if status == "corrected":
            out["original_plan"] = plan
            out["corrected_result"] = action_result
        return out
