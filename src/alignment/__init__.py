"""Alignment scoring engine: unit-circle vectors, gate, dashboard, dynamics."""

from .dashboard import create_dashboard
from .dynamics import (
    integrate_lorenz,
    integrate_motion,
    render_lorenz,
    render_motion_regimes,
)
from .engine import ScoreError, ScoreResult, parse_llm_output, print_report, score
from .prompt import system_prompt
from .samples import SAMPLES
from .slices import HUMAN, JESUS, SATAN, SLICE_NAMES

__all__ = [
    "HUMAN",
    "JESUS",
    "SATAN",
    "SAMPLES",
    "SLICE_NAMES",
    "ScoreError",
    "ScoreResult",
    "create_dashboard",
    "integrate_lorenz",
    "integrate_motion",
    "parse_llm_output",
    "print_report",
    "render_lorenz",
    "render_motion_regimes",
    "score",
    "system_prompt",
]

__version__ = "0.1.0"
