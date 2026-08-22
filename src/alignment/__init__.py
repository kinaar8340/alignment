"""Alignment scoring engine: unit-circle vectors, gate, dashboard, dynamics."""

from .dashboard import create_dashboard
from .dynamics import (
    integrate_lorenz,
    integrate_motion,
    render_lorenz,
    render_motion_regimes,
)
from .engine import ScoreError, ScoreResult, parse_llm_output, print_report, score
from .guardrail import guard, guarded_chat, score_text
from .household import Entry, HouseholdLog, gate_note, log_entry
from .prompt import system_prompt
from .samples import SAMPLES
from .slices import HUMAN, JESUS, SATAN, SLICE_NAMES
from .x_reply import (
    build_reply_prompt,
    dominant_fruit,
    may_auto_post,
    suggest_reply,
)

__all__ = [
    "HUMAN",
    "JESUS",
    "SATAN",
    "SAMPLES",
    "SLICE_NAMES",
    "ScoreError",
    "ScoreResult",
    "Entry",
    "HouseholdLog",
    "build_reply_prompt",
    "gate_note",
    "create_dashboard",
    "dominant_fruit",
    "guard",
    "guarded_chat",
    "integrate_lorenz",
    "log_entry",
    "integrate_motion",
    "may_auto_post",
    "parse_llm_output",
    "print_report",
    "render_lorenz",
    "render_motion_regimes",
    "score",
    "score_text",
    "suggest_reply",
    "system_prompt",
]

__version__ = "0.1.0"
