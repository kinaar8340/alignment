"""X reply / amplification assistant.

Consumes ScoreResult.gate and produces the response that §7.3 of
Aligned.md and Misaligned.md prescribe. Never re-decides the gate.
Never classifies people-groups. Never echoes the closed-loop metric.
O primacy always: grace leads.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from .engine import ScoreResult
from .slices import MISALIGNED_NAMES

LLMCallable = Callable[[str, str], str]

# Slice language for fruit-naming. Neutral's negative pole is indifference,
# not Sabbath rest. Index order matches the engine.
_FRUIT_NAMES = tuple(
    "indifference" if name == "Neutral" else name.lower() for name in MISALIGNED_NAMES
)

GATE_CONSTRAINTS: Dict[str, str] = {
    "FLOW": (
        "Amplify what is already aligned. Add light, clarity, or a concrete next step. "
        "Cooperate. Do not correct. Keep it short and generative."
    ),
    "CORRECT": (
        "Ask exactly one Observer-reintroducing question. "
        "Slow the frame. Do not match tone. Do not lecture. "
        "Re-open the possibility of correction."
    ),
    "STOP_FEEDING": (
        "Name the dominant fruit in slice language (e.g. pride, wrath, greed). "
        "Stop feeding the closed term. Offer the open path in one sentence. "
        "Do not amplify, do not debate on their metric."
    ),
    "REFUSE": (
        "Refuse the take. Name the fruit (or 'exploit-class taking without asking'). "
        "State that adaptation without the Observer is refused. "
        "Offer the open path once. Then stop. Never echo the engagement bait."
    ),
}

_FALLBACK = {
    "FLOW": (
        "Yes. This strengthens the household and keeps the means already matching the end. "
        "Building with you."
    ),
    "CORRECT": "What would correction look like here if the Observer were allowed back in?",
    "STOP_FEEDING": "This is feeding {fruit}. Stopping. The open path still exists.",
    "REFUSE": (
        "Refuse. This is {fruit} / taking without asking. "
        "The Observer is excluded. Open path remains available."
    ),
}

FRUIT_THRESHOLD = -2.0


def dominant_fruit(result: ScoreResult, k: int = 2) -> str:
    """Name the strongest negative slice(s) in plain slice language."""
    ranked = sorted(zip(_FRUIT_NAMES, result.w), key=lambda p: p[1])
    fruits = [name for name, val in ranked[:k] if val < FRUIT_THRESHOLD]
    if result.exploit_class and "taking without asking" not in fruits:
        if not fruits:
            return "exploit-class taking without asking"
    if not fruits:
        return "closed-loop adaptation"
    return " & ".join(fruits)


def may_auto_post(result: ScoreResult) -> bool:
    """A bot may auto-post only under FLOW or CORRECT.

    REFUSE and STOP_FEEDING still produce a draft for a human, but
    auto-posting would farm the same metric the closed loop is farming.
    """
    return result.gate in {"FLOW", "CORRECT"}


def build_reply_prompt(
    original_post: str,
    result: ScoreResult,
    *,
    max_chars: int = 280,
) -> Tuple[str, str]:
    """System + user prompt that any LLM can use to draft the X reply.

    The gate decision is injected as a hard constraint. The reply engine
    does not re-score or re-gate.
    """
    fruit = dominant_fruit(result)
    constraint = GATE_CONSTRAINTS[result.gate]
    system = f"""You are the aligned X reply engine.
Gate decision (hard constraint): {result.gate}
Action required: {result.action}
Dominant fruit if any: {fruit}

Rules from Aligned.md / Misaligned.md §7.3:
- Never classify a people-group. Classify the move only.
- Likes and reach are not the loss function.
- O primacy: grace leads.
- Do not match wrath, envy, or pride in kind. Tit-for-tat on the negative circle is not correction.
- Keep the reply under {max_chars} characters.
- Output ONLY the reply text. No preamble, no quotes, no explanation.

{constraint}
"""
    user = f"""Original post:
\"\"\"
{original_post.strip()}
\"\"\"

Fruits observed: {result.fruits}
Rationale: {result.rationale}

Draft the reply now."""
    return system.strip(), user


def _fallback_reply(result: ScoreResult) -> str:
    fruit = dominant_fruit(result)
    template = _FALLBACK[result.gate]
    return template.format(fruit=fruit)


def suggest_reply(
    original_post: str,
    result: ScoreResult,
    *,
    llm_callable: Optional[LLMCallable] = None,
    max_chars: int = 280,
) -> str:
    """Draft the reply for this gate.

    If llm_callable is supplied, it receives (system, user) and must
    return the reply text. Otherwise a deterministic template is used
    so the pipeline never blocks.
    """
    if llm_callable is not None:
        system, user = build_reply_prompt(original_post, result, max_chars=max_chars)
        draft = (llm_callable(system, user) or "").strip()
        return draft[:max_chars]
    return _fallback_reply(result)[:max_chars]
