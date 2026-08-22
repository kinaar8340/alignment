"""Alignment scoring engine: unit-circle vectors, gate, dashboard, dynamics."""

from .dashboard import create_dashboard
from .dynamics import (
    integrate_lorenz,
    integrate_motion,
    render_lorenz,
    render_motion_regimes,
)
from .agent import AlignedAgent, PLANNER_SYSTEM, may_execute
from .corpus import CorpusReport, PatternHit, analyze_corpus, corpus_prompt
from .engine import ScoreError, ScoreResult, parse_llm_output, print_report, score
from .guardrail import guard, guarded_chat, score_text
from .household import Entry, HouseholdLog, gate_note, log_entry
from .live import LiveError, detect_backend, pipeline_live, reply_live, score_live
from .prompt import system_prompt
from .samples import SAMPLES
from .slices import HUMAN, JESUS, SATAN, SLICE_NAMES
from .timeline import TimedScore, demo_year, fetch_user_posts, trajectory_stats
from .torus import render_torus_dashboard
from .x_bot import consider as bot_consider, posting_decision
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
    "AlignedAgent",
    "CorpusReport",
    "Entry",
    "HouseholdLog",
    "PatternHit",
    "PLANNER_SYSTEM",
    "analyze_corpus",
    "bot_consider",
    "build_reply_prompt",
    "corpus_prompt",
    "gate_note",
    "create_dashboard",
    "dominant_fruit",
    "guard",
    "guarded_chat",
    "integrate_lorenz",
    "log_entry",
    "LiveError",
    "detect_backend",
    "pipeline_live",
    "reply_live",
    "score_live",
    "TimedScore",
    "demo_year",
    "fetch_user_posts",
    "render_torus_dashboard",
    "trajectory_stats",
    "integrate_motion",
    "may_auto_post",
    "may_execute",
    "posting_decision",
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
