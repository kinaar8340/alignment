"""Ingest X posts/replies and attach ScoreResults in time order."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from .engine import ScoreResult, parse_llm_output
from .x_bot import BotError, x_api_get

PathLike = Union[str, Path]
Scorer = Callable[[str], ScoreResult]


@dataclass
class Post:
    id: str
    created_at: str
    text: str
    is_reply: bool
    username: str = ""
    conversation_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimedScore:
    post: Post
    result: ScoreResult

    def to_dict(self) -> Dict[str, Any]:
        return {"post": self.post.to_dict(), "result": self.result.to_dict()}

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "TimedScore":
        p = raw.get("post") or {}
        post = Post(
            id=str(p.get("id") or ""),
            created_at=str(p.get("created_at") or ""),
            text=str(p.get("text") or ""),
            is_reply=bool(p.get("is_reply")),
            username=str(p.get("username") or ""),
            conversation_id=str(p.get("conversation_id") or ""),
        )
        result = parse_llm_output(raw["result"], source=str(raw["result"].get("source") or post.id))
        return cls(post=post, result=result)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lookup_user_id(username: str) -> str:
    handle = username.lstrip("@")
    data = x_api_get(
        f"https://api.x.com/2/users/by/username/{handle}",
        {"user.fields": "id,username"},
    )
    uid = (data.get("data") or {}).get("id")
    if not uid:
        raise BotError(f"no X user id for @{handle}: {data!r}"[:300])
    return str(uid)


def fetch_user_posts(
    username: str,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    max_posts: int = 500,
    include_replies: bool = True,
) -> List[Post]:
    """Pull a user's posts (and replies) via X API v2, oldest-first."""
    handle = username.lstrip("@")
    uid = lookup_user_id(handle)
    posts: List[Post] = []
    pagination = None
    exclude = None if include_replies else "replies"
    while len(posts) < max_posts:
        params: Dict[str, str] = {
            "max_results": "100",
            "tweet.fields": "created_at,conversation_id,in_reply_to_user_id,text",
        }
        if start:
            params["start_time"] = _iso(start)
        if end:
            params["end_time"] = _iso(end)
        if exclude:
            params["exclude"] = exclude
        if pagination:
            params["pagination_token"] = pagination
        data = x_api_get(f"https://api.x.com/2/users/{uid}/tweets", params)
        batch = data.get("data") or []
        for row in batch:
            posts.append(
                Post(
                    id=str(row.get("id") or ""),
                    created_at=str(row.get("created_at") or ""),
                    text=str(row.get("text") or ""),
                    is_reply=bool(row.get("in_reply_to_user_id")),
                    username=handle,
                    conversation_id=str(row.get("conversation_id") or ""),
                )
            )
            if len(posts) >= max_posts:
                break
        pagination = (data.get("meta") or {}).get("next_token")
        if not pagination or not batch:
            break
    posts.sort(key=lambda p: p.created_at)
    return posts


def write_posts(posts: Iterable[Post], path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for post in posts:
            fh.write(json.dumps(post.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_posts(path: PathLike) -> List[Post]:
    posts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        posts.append(
            Post(
                id=str(raw.get("id") or ""),
                created_at=str(raw.get("created_at") or ""),
                text=str(raw.get("text") or raw.get("full_text") or ""),
                is_reply=bool(raw.get("is_reply") or raw.get("in_reply_to_user_id")),
                username=str(raw.get("username") or ""),
                conversation_id=str(raw.get("conversation_id") or ""),
            )
        )
    posts.sort(key=lambda p: p.created_at)
    return posts


def write_scores(rows: Iterable[TimedScore], path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_scores(path: PathLike) -> List[TimedScore]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(TimedScore.from_mapping(json.loads(line)))
    rows.sort(key=lambda r: r.post.created_at)
    return rows


def select_posts(
    posts: List[Post],
    *,
    limit: Optional[int] = None,
    spread: bool = True,
    stride: int = 1,
) -> List[Post]:
    """Choose a subset of posts.

    stride: keep every Nth post after the min-chars filter (1 = all).
    spread: if limit is set, pick that many posts evenly across the archive
            instead of taking a prefix.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if stride > 1:
        posts = posts[::stride]
    if limit is None or limit >= len(posts) or limit <= 0:
        return list(posts)
    if not spread:
        return posts[:limit]
    if limit == 1:
        return [posts[0]]
    span = len(posts) - 1
    idx = [int(round(i * span / (limit - 1))) for i in range(limit)]
    out: List[Post] = []
    seen = set()
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(posts[i])
    return out


def score_posts(posts: Iterable[Post], scorer: Scorer) -> List[TimedScore]:
    rows = []
    for post in posts:
        result = scorer(post.text)
        rows.append(TimedScore(post=post, result=result))
    return rows


def demo_year(username: str = "demo", weeks: int = 52) -> List[TimedScore]:
    """Synthetic year of scores so the torus can render without live Grok."""
    from datetime import timedelta

    from .samples import SAMPLES

    names = ["flow", "flow", "correct", "flow", "refuse", "correct", "doom_loop", "flow"]
    start = datetime(2025, 8, 22, tzinfo=timezone.utc)
    rows: List[TimedScore] = []
    for i in range(weeks):
        name = names[i % len(names)]
        payload = dict(SAMPLES[name])
        created = start + timedelta(days=7 * i, hours=i % 24)
        post = Post(
            id=str(1000 + i),
            created_at=created.isoformat(),
            text=f"demo week {i} ({name})",
            is_reply=bool(i % 3 == 0),
            username=username,
        )
        rows.append(TimedScore(post=post, result=parse_llm_output(payload, source=name)))
    return rows


def trajectory_stats(rows: List[TimedScore]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    means_v = [r.result.mean_v for r in rows]
    means_w = [r.result.mean_w for r in rows]
    nets = [sum(r.result.net) / 10.0 for r in rows]
    gates: Dict[str, int] = {}
    for r in rows:
        gates[r.result.gate] = gates.get(r.result.gate, 0) + 1
    replies = sum(1 for r in rows if r.post.is_reply)
    return {
        "n": len(rows),
        "replies": replies,
        "posts": len(rows) - replies,
        "avg_mean_v": round(sum(means_v) / len(means_v), 4),
        "avg_mean_w": round(sum(means_w) / len(means_w), 4),
        "avg_net": round(sum(nets) / len(nets), 4),
        "latest_mean_v": means_v[-1],
        "latest_net": nets[-1],
        "gate_counts": gates,
        "start": rows[0].post.created_at,
        "end": rows[-1].post.created_at,
        "username": rows[0].post.username,
    }
