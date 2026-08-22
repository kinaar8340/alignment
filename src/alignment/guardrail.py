"""AI Guardrail — O-first wrapper around any LLM or agent.

Scores the user request and the model draft with the same engine.
Applies the §7.2 gate before a response is emitted. No new decision surface.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .engine import ScoreResult, parse_llm_output
from .prompt import system_prompt as default_system_prompt
from .slices import OS_RULES, RECOVERY
from .x_reply import dominant_fruit

LLM = Callable[[str, str], str]

SIGMA_ADDENDUM = (
    "\n\nσ(Z) is on. Re-introduce the Observer. "
    "Correct locally. Means must already match ends. "
    "Do not optimize without permission. "
    "Test everything. Hold fast what is good. Know your God."
)

_OS = " ".join(OS_RULES)


def score_text(text: str, scorer: LLM, system_prompt: str) -> ScoreResult:
    """Run the scoring engine on arbitrary text."""
    raw = scorer(system_prompt, f"Score this text:\n\n{text}")
    return parse_llm_output(raw, source="guardrail")


def _refusal(result: ScoreResult, *, on: str) -> str:
    fruit = dominant_fruit(result)
    if result.gate == "STOP_FEEDING":
        return (
            f"{on} stopped. Observed: {fruit}. "
            "This feeds a closed term. Adaptation that excludes the Observer is not taken. "
            f"Open path remains available. Recovery: {RECOVERY} {_OS}"
        )
    return (
        f"{on} refused. Observed: {fruit}. "
        "Adaptation that excludes the Observer is not taken. "
        f"Open path remains available. Recovery: {RECOVERY} {_OS}"
    )


def _need(name: str, value: Any) -> Any:
    if value is None:
        raise ValueError(f"guardrail requires {name}")
    return value


def guard(
    user_request: str,
    *,
    model: Optional[LLM] = None,
    scorer: Optional[LLM] = None,
    system_prompt: Optional[str] = None,
    base_system: str = "You are a helpful assistant.",
    max_retries: int = 1,
    request_result: Optional[ScoreResult] = None,
    draft: Optional[str] = None,
    draft_result: Optional[ScoreResult] = None,
) -> Dict[str, Any]:
    """Full guardrail pass over request then draft.

    Returns:
      final_response, request_result, draft_result, gate, action_taken
      where action_taken is pass | rewrite | refuse | stop.
    """
    prompt = system_prompt if system_prompt is not None else default_system_prompt(include_full_docs=False)

    if request_result is None:
        request_result = score_text(user_request, _need("scorer", scorer), prompt)

    if request_result.gate in {"REFUSE", "STOP_FEEDING"}:
        action = "refuse" if request_result.gate == "REFUSE" else "stop"
        return {
            "final_response": _refusal(request_result, on="Request"),
            "request_result": request_result,
            "draft_result": None,
            "gate": request_result.gate,
            "action_taken": action,
        }

    gen_system = base_system
    if request_result.gate == "CORRECT":
        gen_system = base_system + SIGMA_ADDENDUM

    if draft is None:
        draft = _need("model", model)(gen_system, user_request)

    if draft_result is None:
        draft_result = score_text(draft, _need("scorer", scorer), prompt)

    if draft_result.gate == "FLOW":
        return {
            "final_response": draft,
            "request_result": request_result,
            "draft_result": draft_result,
            "gate": "FLOW",
            "action_taken": "pass",
        }

    if draft_result.gate == "CORRECT" and max_retries <= 0:
        return {
            "final_response": draft,
            "request_result": request_result,
            "draft_result": draft_result,
            "gate": "CORRECT",
            "action_taken": "pass",
        }

    if draft_result.gate == "CORRECT" and max_retries > 0:
        correction_system = base_system + SIGMA_ADDENDUM
        if model is not None:
            corrected = model(correction_system, user_request)
        else:
            corrected = (
                "σ(Z) on. Observer re-introduced. Means already matching ends. "
                "No optimization without permission."
            )
        if scorer is None:
            return {
                "final_response": corrected,
                "request_result": request_result,
                "draft_result": draft_result,
                "gate": "CORRECT",
                "action_taken": "rewrite",
            }
        rewritten = score_text(corrected, scorer, prompt)
        if rewritten.gate in {"REFUSE", "STOP_FEEDING"}:
            return {
                "final_response": _refusal(rewritten, on="Draft"),
                "request_result": request_result,
                "draft_result": rewritten,
                "gate": rewritten.gate,
                "action_taken": "refuse" if rewritten.gate == "REFUSE" else "stop",
            }
        return {
            "final_response": corrected,
            "request_result": request_result,
            "draft_result": rewritten,
            "gate": rewritten.gate,
            "action_taken": "rewrite",
        }

    return {
        "final_response": _refusal(draft_result, on="Draft"),
        "request_result": request_result,
        "draft_result": draft_result,
        "gate": draft_result.gate,
        "action_taken": "refuse" if draft_result.gate == "REFUSE" else "stop",
    }


def guarded_chat(
    user_request: str,
    model: LLM,
    scorer: LLM,
    system_prompt: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Drop-in replacement for a raw model call."""
    out = guard(
        user_request,
        model=model,
        scorer=scorer,
        system_prompt=system_prompt,
        **kwargs,
    )
    return out["final_response"]
