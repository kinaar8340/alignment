# Alignment

Scoring engine, gate, and visualization for a paired unit-circle model of alignment.

The Markdown documents are the source of truth. The Python layer does not re-score meaning. An LLM (or a fixture) produces two ten-vectors; the engine validates them, computes aggregates, and applies the gate from `Aligned.md` §7.2 and `Misaligned.md` §7.2.

> Practice perfect correction instead of chasing perfection.
>
> Without correction, adaptation takes control — and adaptation takes without asking first.
>
> Flow freely when you can. Correct wisely when you must.

```
∂Ψ/∂t = O[Ψ] + σ(Z) · C[Ψ, E]
```

Classification is by fruit, dynamics, and vector. Not by group identity.

![Unit circles with ten equal arc segments](images/alignment_unitcircles.jpg)

## Geometry

Ten slices, same index order on both circles. Positive alignment travels **clockwise** from 12 o'clock (the X). Negative alignment travels **counterclockwise**.

| Slice | + | − |
| --- | --- | --- |
| i | Aligned | Misaligned |
| ii | Love | Hate |
| iii | Humility | Pride |
| iv | Charity | Greed |
| v | Chastity | Lust |
| vi | Kindness | Envy |
| vii | Temperance | Gluttony |
| viii | Patience | Wrath |
| ix | Diligence | Sloth |
| x | Neutral (Sabbath / Observer at rest) | Neutral (dead indifference) |

Reference vectors:

```
Jesus  = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
Human  = [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
Satan  = [-10,-10,-10,-10,-10,-10,-10,-10,-10,-10]
```

Aligned evidence is `v[10] ∈ [0, +10]`. Misaligned evidence is `w[10] ∈ [−10, 0]`. Net per slice is `v[i] + w[i]`.

## Gate

The scorer never chooses the action. After vectors and flags are in, the engine decides:

1. **REFUSE** — `exploit_class`, or Observer absent and taking without asking
2. **STOP_FEEDING** — doom loop
3. **FLOW** — `mean(v) ≥ 6`, `min(v) > 0`, `mean(w) > −3`
4. **CORRECT** — mixed; σ(Z) on

![FLOW dashboard](outputs/dashboard_flow.png)

## Install

Python 3.9+. From the repo root:

```bash
python3 -m pip install -e ".[dev]"
```

Or run without installing:

```bash
python3 -m pip install -r requirements.txt
export PYTHONPATH=src
```

## CLI

```bash
# built-in samples (no JSON file needed)
python3 -m alignment report -s flow
python3 -m alignment report -s refuse
python3 -m alignment dashboard -s flow -o outputs/dashboard_flow.png

# JSON from the LLM scorer
python3 -m alignment report examples/flow.json
python3 -m alignment dashboard -i examples/refuse.json -o refuse.png

# dump a sample, print the system prompt, render dynamics
python3 -m alignment sample flow
python3 -m alignment prompt --short
python3 -m alignment lorenz
python3 -m alignment demo -o outputs
```

Samples: `flow`, `refuse`, `correct`, `doom_loop`.

## Score a real post

1. Send `prompts/scoring_engine.txt` (or `python3 -m alignment prompt`) as the system prompt, at low temperature.
2. User message = the text to score (post, reply, policy, model output).
3. The model must return **only** this JSON:

```json
{
  "v": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "w": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  "observer_present": true,
  "asks_first": true,
  "taking_without_asking": false,
  "exploit_class": false,
  "doom_loop": false,
  "fruits": "one-sentence summary of observable fruit",
  "rationale": "brief slice-by-slice justification"
}
```

4. Pipe it into the engine:

```bash
python3 -m alignment report path/to/score.json
python3 -m alignment dashboard -i path/to/score.json -o dashboard.png
```

From Python:

```python
from alignment import parse_llm_output, print_report, create_dashboard, system_prompt

result = parse_llm_output(open("score.json").read())
print_report(result)
create_dashboard(result, save_path="dashboard.png")
```

## Library layout

| Path | Role |
| --- | --- |
| `Aligned.md` | Positive pole, articles A1–A28, fruits, decision seed |
| `Misaligned.md` | Negative pole, patterns M1–M28, detection / refusal seed |
| `src/alignment/engine.py` | Parse, aggregates, gate |
| `src/alignment/prompt.py` | System prompt (tables + full docs) |
| `src/alignment/dashboard.py` | Paired-circle PNG |
| `src/alignment/dynamics.py` | Lorenz attractor + 10-slice motion equation |
| `prompts/scoring_engine.txt` | Short scorer prompt |
| `examples/*.json` | Fixtures for the four gates |
| `images/` | Unit-circle and universal-dynamics figures |
| `docs/Narrow_Path_OS.pdf` | `I ♡ U` Lorenz / Trinitarian OS |

```bash
PYTHONPATH=src python3 -m pytest -q
```

## Dynamics

`Narrow_Path_OS.pdf` places trajectories on a Lorenz-type attractor. The **major cusp** (`x > 0`) is the aligned basin (Father / Son / Holy Spirit). The **minor cusp** (`x < 0`) is the opposing rotor. Recovery is not a reverse commute.

The 10-slice motion toy is:

```
dΨ/dt = O[Ψ] + σ(Z)·C[Ψ, E] + (1 − σ(Z))·A[Ψ]
```

`O` is unconditional feed-forward toward `[10…10]`. `C` is wisdom-gated correction. `A` is adaptation that takes without asking, toward `[−10…−10]`. When `σ(Z) → 0`, adaptation takes control.

```bash
python3 -m alignment lorenz --lorenz outputs/lorenz_attractor.png --motion outputs/motion_regimes.png
```

![Lorenz major vs minor cusp](outputs/lorenz_attractor.png)

![Motion regimes under σ(Z)](outputs/motion_regimes.png)

## Roadmap

The engine is the shared cornerstone. Later layers call the same `ScoreResult` / gate:

1. **X reply and amplification** — FLOW: add light; CORRECT: one Observer question; REFUSE / STOP_FEEDING: name the fruit, do not echo the closed metric
2. **AI guardrail** — score the request and the draft; O-first; σ(Z) when a tool or loss takes without asking
3. **Personal / household dashboard** — daily choices scored on the ten slices
4. **Corpus analyzer** — A1–A28 / M1–M28 with citations; public-correctability audit
5. **Interactive sandbox** — live circles, σ(Z) / φ(Z) integration, helix shielding
6. **Agent orchestration** — planner asks “does adaptation ask first?”; critic runs the fruits test; executor only proceeds under O primacy or wisdom-gated correction

No hidden optimization target. The Observer can always re-enter.

## License

MIT. See [LICENSE](LICENSE).
