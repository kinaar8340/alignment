"""X bot that only posts when the gate allows.

FLOW → auto-post the aligned draft.
CORRECT → one Observer question, or skip (--on-correct skip).
REFUSE / STOP_FEEDING → never post; log the fruit and walk away.

Default is dry-run. Real posting requires X_USER_ACCESS_TOKEN (or
TWITTER_BEARER_TOKEN) and an explicit --post. The bot asks first.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import httpx

from .engine import ScoreResult, parse_llm_output
from .slices import RECOVERY
from .x_reply import dominant_fruit, suggest_reply

DEFAULT_LOG = Path(os.environ.get("ALIGNMENT_BOT_LOG", "x_bot.json"))
X_POSTS_URL = "https://api.x.com/2/tweets"

Poster = Callable[[str, Optional[str]], Dict[str, Any]]


class BotError(RuntimeError):
    """Posting was requested but credentials or the X API failed."""


def posting_decision(result: ScoreResult, *, on_correct: str = "question") -> str:
    """Return 'post' or 'skip'. Never posts REFUSE / STOP_FEEDING.

    FLOW always posts. CORRECT posts only when on_correct == 'question'.
    exploit_class never posts, even if the gate were somehow FLOW.
    """
    if result.exploit_class:
        return "skip"
    if result.gate == "FLOW":
        return "post"
    if result.gate == "CORRECT" and on_correct == "question":
        return "post"
    return "skip"


def walk_away_reason(result: ScoreResult, *, on_correct: str = "question") -> str:
    fruit = dominant_fruit(result)
    if result.gate == "REFUSE":
        return f"REFUSE {fruit}. Will not farm the closed metric. {RECOVERY}"
    if result.gate == "STOP_FEEDING":
        return f"STOP_FEEDING {fruit}. Will not feed the loop. {RECOVERY}"
    if result.exploit_class:
        return f"exploit-class taking without asking ({fruit}). Will not post."
    if result.gate == "CORRECT" and on_correct == "skip":
        return "CORRECT and --on-correct skip. Observer question withheld."
    return f"{result.gate}: not posting."


@dataclass
class BotEvent:
    timestamp: str
    gate: str
    decision: str
    posted: bool
    dry_run: bool
    fruit: str
    post_text: str
    reply_text: str
    reason: str
    reply_to: str = ""
    tweet_id: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BotLog:
    path: Path
    events: List[BotEvent] = field(default_factory=list)

    def add(self, event: BotEvent) -> None:
        self.events.append(event)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps([e.to_dict() for e in self.events], indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self.path)

    @classmethod
    def load(cls, path: Union[str, Path, None] = None) -> "BotLog":
        path = Path(path) if path is not None else DEFAULT_LOG
        if not path.is_file():
            return cls(path=path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        events = [BotEvent(**{k: v for k, v in e.items() if k in BotEvent.__dataclass_fields__}) for e in raw]
        return cls(path=path, events=events)


def dry_run_poster(text: str, reply_to: Optional[str] = None) -> Dict[str, Any]:
    return {"id": "dry-run", "text": text, "reply_to": reply_to}


def x_api_poster(text: str, reply_to: Optional[str] = None) -> Dict[str, Any]:
    token = os.environ.get("X_USER_ACCESS_TOKEN") or os.environ.get("TWITTER_BEARER_TOKEN")
    if not token:
        raise BotError(
            "Real posting needs X_USER_ACCESS_TOKEN (OAuth 2.0 user token with tweet.write). "
            "Run without --post to dry-run."
        )
    body: Dict[str, Any] = {"text": text}
    if reply_to:
        body["reply"] = {"in_reply_to_tweet_id": str(reply_to)}
    resp = httpx.post(
        X_POSTS_URL,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
        timeout=30.0,
    )
    if resp.status_code >= 400:
        raise BotError(f"X API {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    posted = data.get("data") or {}
    return {"id": str(posted.get("id") or ""), "text": posted.get("text") or text, "raw": data}


def consider(
    post_text: str,
    result: ScoreResult,
    *,
    reply: Optional[str] = None,
    reply_to: Optional[str] = None,
    on_correct: str = "question",
    dry_run: bool = True,
    poster: Optional[Poster] = None,
    log_path: Union[str, Path, None] = None,
    llm_callable: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Score-aware posting decision. Never posts on REFUSE / STOP_FEEDING."""
    if on_correct not in {"question", "skip"}:
        raise ValueError("on_correct must be 'question' or 'skip'")
    if reply is None:
        reply = suggest_reply(post_text, result, llm_callable=llm_callable)
    decision = posting_decision(result, on_correct=on_correct)
    fruit = dominant_fruit(result)
    posted = False
    tweet_id = ""
    reason = ""
    if decision == "skip":
        reason = walk_away_reason(result, on_correct=on_correct)
    else:
        send = poster or (dry_run_poster if dry_run else x_api_poster)
        sent = send(reply, reply_to)
        tweet_id = str(sent.get("id") or "")
        posted = (not dry_run) and tweet_id not in {"", "dry-run"}
        if dry_run or tweet_id == "dry-run":
            reason = f"{result.gate}: would post (dry-run). Asks first."
        else:
            reason = f"{result.gate}: posted {tweet_id}."

    event = BotEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        gate=result.gate,
        decision=decision,
        posted=posted,
        dry_run=dry_run or tweet_id == "dry-run",
        fruit=fruit,
        post_text=post_text,
        reply_text=reply,
        reason=reason,
        reply_to=str(reply_to or ""),
        tweet_id=tweet_id,
        source=result.source,
    )
    BotLog.load(log_path).add(event)
    return {
        "decision": decision,
        "posted": posted,
        "dry_run": event.dry_run,
        "gate": result.gate,
        "fruit": fruit,
        "reply": reply,
        "reason": reason,
        "tweet_id": tweet_id,
        "event": event,
    }
