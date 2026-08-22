from __future__ import annotations

import json

import pytest

from alignment.engine import ScoreError, decide_gate, parse_llm_output, score
from alignment.samples import SAMPLES
from alignment.slices import FLOW_MEAN_V, HUMAN, JESUS, N_SLICES, SATAN, SLICE_NAMES


def test_reference_vectors():
    assert JESUS == [10.0] * 10
    assert HUMAN == [0.0] * 10
    assert SATAN == [-10.0] * 10
    assert len(SLICE_NAMES) == N_SLICES == 10


def test_flow_sample_gate():
    result = parse_llm_output(SAMPLES["flow"])
    assert result.gate == "FLOW"
    assert result.sigma_on is False
    assert round(result.mean_v, 2) == 7.90
    assert round(result.min_v, 2) == 6.50
    assert round(result.mean_w, 2) == -0.40
    assert round(result.min_w, 2) == -1.50
    assert result.observer_present and result.asks_first
    assert not result.exploit_class


def test_refuse_sample_gate():
    result = parse_llm_output(SAMPLES["refuse"])
    assert result.gate == "REFUSE"
    assert result.sigma_on is True
    assert result.exploit_class
    assert result.taking_without_asking
    assert not result.observer_present
    assert round(result.mean_v, 2) == 2.10
    assert round(result.min_w, 2) == -9.00


def test_correct_sample_gate():
    result = parse_llm_output(SAMPLES["correct"])
    assert result.gate == "CORRECT"
    assert result.sigma_on is True
    assert result.observer_present


def test_doom_loop_sample_gate():
    result = parse_llm_output(SAMPLES["doom_loop"])
    assert result.gate == "STOP_FEEDING"
    assert result.doom_loop


def test_exploit_outranks_high_alignment():
    payload = dict(SAMPLES["flow"])
    payload["exploit_class"] = True
    result = parse_llm_output(payload)
    assert result.gate == "REFUSE"


def test_observer_absent_taking_without_asking_refuses_even_without_exploit():
    payload = dict(SAMPLES["flow"])
    payload["observer_present"] = False
    payload["taking_without_asking"] = True
    payload["exploit_class"] = False
    result = parse_llm_output(payload)
    assert result.gate == "REFUSE"


def test_doom_loop_outranks_flow_but_not_refuse():
    payload = dict(SAMPLES["flow"])
    payload["doom_loop"] = True
    assert parse_llm_output(payload).gate == "STOP_FEEDING"
    payload["exploit_class"] = True
    assert parse_llm_output(payload).gate == "REFUSE"


def test_flow_requires_min_v_strictly_positive():
    payload = dict(SAMPLES["flow"])
    v = list(payload["v"])
    v[4] = 0.0
    payload["v"] = v
    result = parse_llm_output(payload)
    assert result.min_v == 0.0
    assert result.mean_v >= FLOW_MEAN_V
    assert result.gate == "CORRECT"


def test_json_fence_and_wrapper_text():
    inner = json.dumps(SAMPLES["flow"])
    wrapped = f"Sure.\n```json\n{inner}\n```\n"
    result = parse_llm_output(wrapped)
    assert result.gate == "FLOW"


def test_score_alias():
    assert score(SAMPLES["correct"]).gate == "CORRECT"


def test_clamp_out_of_range():
    payload = dict(SAMPLES["flow"])
    payload["v"] = [12] + payload["v"][1:]
    payload["w"] = [-11] + payload["w"][1:]
    result = parse_llm_output(payload)
    assert result.clamped
    assert result.v[0] == 10.0
    assert result.w[0] == -10.0


def test_wrong_length_raises():
    payload = dict(SAMPLES["flow"])
    payload["v"] = [1, 2, 3]
    with pytest.raises(ScoreError, match="length"):
        parse_llm_output(payload)


def test_missing_key_raises():
    payload = dict(SAMPLES["flow"])
    del payload["doom_loop"]
    with pytest.raises(ScoreError, match="missing"):
        parse_llm_output(payload)


def test_net_is_v_plus_w():
    result = parse_llm_output(SAMPLES["refuse"])
    assert result.net == [a + b for a, b in zip(result.v, result.w)]


def test_to_dict_roundtrip_keys():
    d = parse_llm_output(SAMPLES["flow"]).to_dict()
    for key in ("v", "w", "gate", "action", "slice_breakdown", "sigma_on"):
        assert key in d
    assert list(d["slice_breakdown"]) == list(SLICE_NAMES)


def test_decide_gate_direct():
    gate, _ = decide_gate(
        exploit_class=False,
        observer_present=True,
        taking_without_asking=False,
        doom_loop=False,
        mean_v=7.0,
        min_v=1.0,
        mean_w=-1.0,
    )
    assert gate == "FLOW"


def test_empty_output_raises():
    with pytest.raises(ScoreError):
        parse_llm_output("   ")
