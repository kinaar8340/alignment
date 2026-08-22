"""X bot that only posts when the gate allows.

FLOW → auto-post the aligned draft.
CORRECT → one Observer question, or skip (--on-correct skip).
REFUSE / STOP_FEEDING → never post; log the fruit and walk away.

Default is dry-run. Real posting requires X_USER_ACCESS_TOKEN (or
TWITTER_BEARER_TOKEN) and an explicit --post. The bot asks first.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from urllib.parse import quote

import httpx

from .engine import ScoreResult, parse_llm_output
from .slices import RECOVERY
from .x_reply import dominant_fruit, suggest_reply

DEFAULT_LOG = Path(os.environ.get("ALIGNMENT_BOT_LOG", "x_bot.json"))
X_POSTS_URL = "https://api.x.com/2/tweets"


def load_dotenv(path: Union[str, Path, None] = None) -> None:
    """Load KEY=VALUE from a gitignored .env into os.environ (no overwrite)."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend(
        [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[2] / ".env",
        ]
    )
    seen = set()
    for p in candidates:
        p = p.resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            os.environ.setdefault(key, val)


load_dotenv()

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


def _env(*names: str) -> str:
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return ""


def _oauth1_header(method: str, url: str, extra_params: Optional[Dict[str, str]] = None) -> str:
    api_key = _env("X_API_KEY", "TWITTER_API_KEY")
    api_secret = _env("X_API_SECRET", "TWITTER_API_SECRET")
    token = _env("X_ACCESS_TOKEN", "TWITTER_ACCESS_TOKEN")
    token_secret = _env("X_ACCESS_TOKEN_SECRET", "TWITTER_ACCESS_TOKEN_SECRET")
    if not all([api_key, api_secret, token, token_secret]):
        raise BotError("incomplete OAuth 1.0a credentials")
    oauth = {
        "oauth_consumer_key": api_key,
        "oauth_nonce": uuid.uuid4().hex,
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": token,
        "oauth_version": "1.0",
    }

    def q(s: str) -> str:
        return quote(str(s), safe="")

    sign_params = dict(oauth)
    if extra_params:
        sign_params.update({k: str(v) for k, v in extra_params.items() if v is not None})
    base = "&".join(f"{q(k)}={q(v)}" for k, v in sorted(sign_params.items()))
    base_string = f"{method.upper()}&{q(url)}&{q(base)}"
    signing_key = f"{q(api_secret)}&{q(token_secret)}"
    digest = hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
    oauth["oauth_signature"] = base64.b64encode(digest).decode()
    return "OAuth " + ", ".join(f'{q(k)}="{q(v)}"' for k, v in sorted(oauth.items()))


def x_api_get(url: str, params: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    """Signed GET for timeline lookup. Same OAuth 1.0a keys as posting."""
    params = {k: str(v) for k, v in (params or {}).items() if v is not None}
    headers = {"Authorization": _oauth1_header("GET", url, extra_params=params)}
    resp = httpx.get(url, params=params or None, headers=headers, timeout=timeout)
    if resp.status_code >= 400:
        raise BotError(f"X API GET {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def x_api_poster(text: str, reply_to: Optional[str] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"text": text}
    if reply_to:
        body["reply"] = {"in_reply_to_tweet_id": str(reply_to)}
    bearer = _env("X_USER_ACCESS_TOKEN", "TWITTER_BEARER_TOKEN")
    oauth1 = all(
        [
            _env("X_API_KEY", "TWITTER_API_KEY"),
            _env("X_API_SECRET", "TWITTER_API_SECRET"),
            _env("X_ACCESS_TOKEN", "TWITTER_ACCESS_TOKEN"),
            _env("X_ACCESS_TOKEN_SECRET", "TWITTER_ACCESS_TOKEN_SECRET"),
        ]
    )
    if oauth1:
        headers = {
            "Authorization": _oauth1_header("POST", X_POSTS_URL),
            "Content-Type": "application/json",
        }
    elif bearer:
        headers = {
            "Authorization": f"Bearer {bearer}",
            "Content-Type": "application/json",
        }
    else:
        raise BotError(
            "Real posting needs credentials in the environment or a gitignored .env:\n"
            "  X_API_KEY / X_API_SECRET / X_ACCESS_TOKEN / X_ACCESS_TOKEN_SECRET\n"
            "  (from the alignment_engine app in the X Developer Console)\n"
            "or X_USER_ACCESS_TOKEN (OAuth 2.0 user token with tweet.write).\n"
            "Run without --post to dry-run."
        )
    resp = httpx.post(X_POSTS_URL, headers=headers, json=body, timeout=30.0)
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
