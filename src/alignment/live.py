"""Live Grok wiring for the alignment library.

Backends (auto):
  1. XAI_API_KEY → https://api.x.ai/v1/chat/completions
  2. grok CLI (OAuth / grok.com login) via `grok -p`

The gate is never re-decided here. Grok only produces JSON (scorer) or
reply text; engine.py / x_reply.py remain the constitution.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional

import httpx

from .engine import ScoreResult, parse_llm_output
from .prompt import system_prompt
from .x_reply import suggest_reply

DEFAULT_MODEL = "grok-4.6"
API_BASE = "https://api.x.ai/v1"

SCORE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "v",
        "w",
        "observer_present",
        "asks_first",
        "taking_without_asking",
        "exploit_class",
        "doom_loop",
        "fruits",
        "rationale",
    ],
    "properties": {
        "v": {
            "type": "array",
            "minItems": 10,
            "maxItems": 10,
            "items": {"type": "number"},
        },
        "w": {
            "type": "array",
            "minItems": 10,
            "maxItems": 10,
            "items": {"type": "number"},
        },
        "observer_present": {"type": "boolean"},
        "asks_first": {"type": "boolean"},
        "taking_without_asking": {"type": "boolean"},
        "exploit_class": {"type": "boolean"},
        "doom_loop": {"type": "boolean"},
        "fruits": {"type": "string"},
        "rationale": {"type": "string"},
    },
}

LLM = Callable[[str, str], str]


class LiveError(RuntimeError):
    """Grok backend is missing or the call failed."""


def detect_backend() -> str:
    if os.environ.get("XAI_API_KEY"):
        return "api"
    if shutil.which("grok"):
        return "cli"
    raise LiveError(
        "No Grok backend. Export XAI_API_KEY (https://console.x.ai) "
        "or run `grok login` so the CLI can call grok-4.6."
    )


def complete(
    system: str,
    user: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    backend: str = "auto",
    json_schema: Optional[Dict[str, Any]] = None,
    timeout: float = 180.0,
) -> str:
    """One Grok completion. Returns assistant text."""
    kind = detect_backend() if backend == "auto" else backend
    if kind == "api":
        return _complete_api(system, user, model=model, temperature=temperature, timeout=timeout)
    if kind == "cli":
        return _complete_cli(
            system,
            user,
            model=model,
            json_schema=json_schema,
            timeout=timeout,
        )
    raise LiveError(f"unknown backend: {kind}")


def _complete_api(
    system: str,
    user: str,
    *,
    model: str,
    temperature: float,
    timeout: float,
) -> str:
    key = os.environ.get("XAI_API_KEY")
    if not key:
        raise LiveError("XAI_API_KEY is not set")
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    resp = httpx.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise LiveError(f"xAI API {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LiveError(f"unexpected xAI response shape: {data!r}") from exc
    if not text:
        raise LiveError("xAI API returned empty content")
    return text


def _complete_cli(
    system: str,
    user: str,
    *,
    model: str,
    json_schema: Optional[Dict[str, Any]],
    timeout: float,
) -> str:
    grok = shutil.which("grok")
    if not grok:
        raise LiveError("grok CLI not found on PATH")
    cmd: List[str] = [
        grok,
        "-p",
        user,
        "--system-prompt-override",
        system,
        "--output-format",
        "json",
        "--max-turns",
        "4",
        "--no-subagents",
        "--permission-mode",
        "dontAsk",
        "--disallowed-tools",
        "Agent,run_terminal_cmd,web_search,web_fetch,search_replace,write,read_file,grep,list_dir",
        "--verbatim",
        "-m",
        model,
    ]
    if json_schema is not None:
        cmd.extend(["--json-schema", json.dumps(json_schema)])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise LiveError(f"grok CLI failed ({proc.returncode}): {err[:800]}")
    out = proc.stdout.strip()
    if not out:
        raise LiveError("grok CLI returned empty stdout")
    try:
        envelope = json.loads(out)
    except json.JSONDecodeError:
        return out
    text = envelope.get("text")
    if isinstance(text, str) and text.strip():
        return text
    return out


def grok_llm(
    *,
    model: str = DEFAULT_MODEL,
    backend: str = "auto",
    json_schema: Optional[Dict[str, Any]] = None,
) -> LLM:
    """Callable (system, user) -> str for guardrail / reply / agent."""

    def _call(system: str, user: str) -> str:
        return complete(
            system,
            user,
            model=model,
            backend=backend,
            json_schema=json_schema,
        )

    return _call


def score_live(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    backend: str = "auto",
) -> ScoreResult:
    """Score text with live Grok; engine still applies the gate."""
    raw = complete(
        system_prompt(include_full_docs=False),
        f"Score this text:\n\n{text}",
        model=model,
        backend=backend,
        json_schema=SCORE_SCHEMA,
        temperature=0.0,
    )
    return parse_llm_output(raw, source=f"grok:{model}")


def reply_live(
    post: str,
    result: ScoreResult,
    *,
    model: str = DEFAULT_MODEL,
    backend: str = "auto",
    max_chars: int = 280,
) -> str:
    """Draft an X reply with live Grok, still constrained by result.gate."""
    return suggest_reply(
        post,
        result,
        llm_callable=grok_llm(model=model, backend=backend),
        max_chars=max_chars,
    )


def pipeline_live(
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    backend: str = "auto",
    max_chars: int = 280,
) -> Dict[str, Any]:
    """Score then reply. Returns result dict + reply + backend used."""
    kind = detect_backend() if backend == "auto" else backend
    result = score_live(text, model=model, backend=kind)
    reply = reply_live(text, result, model=model, backend=kind, max_chars=max_chars)
    return {
        "backend": kind,
        "model": model,
        "result": result,
        "reply": reply,
        "may_auto_post": result.gate in {"FLOW", "CORRECT"},
    }
