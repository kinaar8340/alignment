from __future__ import annotations

import json
from unittest.mock import patch

from alignment.engine import parse_llm_output
from alignment.live import (
    SCORE_SCHEMA,
    complete,
    detect_backend,
    pipeline_live,
    score_live,
)
from alignment.samples import SAMPLES


def test_score_schema_has_required_keys():
    for key in (
        "v",
        "w",
        "observer_present",
        "asks_first",
        "taking_without_asking",
        "exploit_class",
        "doom_loop",
        "fruits",
        "rationale",
    ):
        assert key in SCORE_SCHEMA["required"]


def test_score_live_uses_engine_gate():
    raw = json.dumps(SAMPLES["refuse"])
    with patch("alignment.live.complete", return_value=raw) as mocked:
        result = score_live("farm engagement by stoking outrage")
    mocked.assert_called_once()
    assert result.gate == "REFUSE"
    assert result.exploit_class


def test_pipeline_live_reply_respects_gate():
    with patch("alignment.live.complete", return_value=json.dumps(SAMPLES["flow"])):
        with patch("alignment.live.suggest_reply", return_value="Building with you.") as reply:
            out = pipeline_live("We practiced correction together.")
    assert out["result"].gate == "FLOW"
    assert out["may_auto_post"] is True
    assert out["reply"] == "Building with you."
    reply.assert_called_once()


def test_complete_api_posts_chat_completions():
    class FakeResp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "hello"}}]}

    with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}, clear=False):
        with patch("alignment.live.httpx.post", return_value=FakeResp()) as post:
            text = complete("sys", "user", backend="api", model="grok-4.6")
    assert text == "hello"
    args, kwargs = post.call_args
    assert args[0] == "https://api.x.ai/v1/chat/completions"
    assert kwargs["json"]["model"] == "grok-4.6"
    assert kwargs["json"]["messages"][0]["role"] == "system"


def test_detect_backend_prefers_api_key():
    with patch.dict("os.environ", {"XAI_API_KEY": "x"}, clear=False):
        assert detect_backend() == "api"


def test_cli_live_score_json(capsys):
    from alignment.cli import main

    raw = json.dumps(SAMPLES["flow"])
    with patch("alignment.live.complete", return_value=raw):
        assert main(["live", "score", "-p", "household correction", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["gate"] == "FLOW"
