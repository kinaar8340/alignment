"""Command-line entry: report, dashboard, prompt, lorenz, demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dashboard import create_dashboard
from .dynamics import render_lorenz, render_motion_regimes
from .engine import parse_llm_output, print_report
from .prompt import system_prompt
from .samples import SAMPLES
from .x_reply import build_reply_prompt, dominant_fruit, may_auto_post, suggest_reply
from .guardrail import guard as run_guard
from .household import DEFAULT_LOG, HouseholdLog, gate_note, log_entry
from .corpus import analyze_corpus, corpus_prompt, format_report, load_corpus


def _read(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    p = Path(path)
    if not p.is_file():
        names = ", ".join(sorted(SAMPLES))
        raise FileNotFoundError(
            f"No such file: {path}\n"
            f"Use a real JSON path (try examples/flow.json) or a built-in sample:\n"
            f"  PYTHONPATH=src python3 -m alignment report -s flow\n"
            f"Samples: {names}"
        )
    return p.read_text(encoding="utf-8")


def _score_from_args(args: argparse.Namespace):
    if getattr(args, "sample", None):
        return parse_llm_output(SAMPLES[args.sample], source=args.sample)
    source = getattr(args, "input", None) or "-"
    return parse_llm_output(_read(source), source=source)


def cmd_report(args: argparse.Namespace) -> int:
    result = _score_from_args(args)
    print_report(result)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    result = _score_from_args(args)
    if args.sample:
        title = args.title or f"Alignment Scoring Dashboard — {args.sample}"
    else:
        title = args.title or "Alignment Scoring Dashboard"
    path = create_dashboard(result, title=title, save_path=args.output)
    print(path)
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    text = system_prompt(include_full_docs=not args.short)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(path)
    else:
        sys.stdout.write(text + "\n")
    return 0


def cmd_lorenz(args: argparse.Namespace) -> int:
    p1 = render_lorenz(args.lorenz)
    p2 = render_motion_regimes(args.motion)
    print(p1)
    print(p2)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    titles = {
        "flow": "Alignment Scoring Dashboard — Sample Post",
        "refuse": "Alignment Scoring Dashboard — Exploit / Closed-Loop Example",
        "correct": "Alignment Scoring Dashboard — Mixed / σ(Z) On",
        "doom_loop": "Alignment Scoring Dashboard — Doom Loop",
    }
    for name, payload in SAMPLES.items():
        result = parse_llm_output(payload, source=name)
        print_report(result)
        path = create_dashboard(
            result,
            title=titles[name],
            save_path=out / f"dashboard_{name}.png",
        )
        print(f"wrote {path}")
        print()
    p1 = render_lorenz(out / "lorenz_attractor.png")
    p2 = render_motion_regimes(out / "motion_regimes.png")
    print(p1)
    print(p2)
    (out / "scoring_engine_prompt.txt").write_text(system_prompt() + "\n", encoding="utf-8")
    print(out / "scoring_engine_prompt.txt")
    return 0


def cmd_score_dict(args: argparse.Namespace) -> int:
    """Echo a sample as JSON — useful as a fixture for an LLM-less pipeline."""
    payload = SAMPLES[args.sample]
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def _post_text(args: argparse.Namespace) -> str:
    if getattr(args, "post_file", None):
        return _read(args.post_file)
    if getattr(args, "post", None):
        return args.post
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def cmd_reply(args: argparse.Namespace) -> int:
    result = _score_from_args(args)
    post = _post_text(args)
    if args.dump_prompt:
        system, user = build_reply_prompt(post or "(no post supplied)", result, max_chars=args.max_chars)
        sys.stdout.write(system + "\n\n---\n\n" + user + "\n")
        return 0
    reply = suggest_reply(post or "(no post supplied)", result, max_chars=args.max_chars)
    if args.json:
        json.dump(
            {
                "gate": result.gate,
                "fruit": dominant_fruit(result),
                "may_auto_post": may_auto_post(result),
                "action": result.action,
                "reply": reply,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0
    print(reply)
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    request = args.request or ""
    if args.request_file:
        request = _read(args.request_file)
    if not request.strip():
        sys.stderr.write("error: pass -p/--request or --request-file\n")
        return 2

    request_result = None
    if args.sample:
        request_result = parse_llm_output(SAMPLES[args.sample], source=args.sample)
    elif args.request_json:
        request_result = parse_llm_output(_read(args.request_json), source=args.request_json)

    draft_result = None
    if args.draft_sample:
        draft_result = parse_llm_output(SAMPLES[args.draft_sample], source=args.draft_sample)
    elif args.draft_json:
        draft_result = parse_llm_output(_read(args.draft_json), source=args.draft_json)

    if request_result is None:
        sys.stderr.write(
            "error: offline guard needs a scored request\n"
            "  PYTHONPATH=src python3 -m alignment guard -p 'text' -s refuse\n"
            "  PYTHONPATH=src python3 -m alignment guard -p 'text' --request-json examples/flow.json --draft 'ok' --draft-sample flow\n"
        )
        return 2

    try:
        out = run_guard(
            request,
            request_result=request_result,
            draft=args.draft,
            draft_result=draft_result,
            max_retries=args.max_retries,
        )
    except ValueError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return 2

    if args.json:
        payload = {
            "final_response": out["final_response"],
            "gate": out["gate"],
            "action_taken": out["action_taken"],
            "request_result": out["request_result"].to_dict(),
            "draft_result": out["draft_result"].to_dict() if out["draft_result"] is not None else None,
        }
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    print(out["final_response"])
    return 0


def _log_path(args: argparse.Namespace) -> Path:
    return Path(args.log) if getattr(args, "log", None) else DEFAULT_LOG


def cmd_household_add(args: argparse.Namespace) -> int:
    if args.score_sample:
        result = parse_llm_output(SAMPLES[args.score_sample], source=args.score_sample)
    elif args.score:
        result = parse_llm_output(_read(args.score), source=args.score)
    else:
        sys.stderr.write("error: pass --score-sample or --score JSON\n")
        return 2
    if not args.label or not args.text:
        sys.stderr.write("error: --label and --text are required\n")
        return 2
    entry = log_entry(
        label=args.label,
        text=args.text,
        result=result,
        note=args.note or "",
        who=args.who or "",
        path=_log_path(args),
    )
    note = gate_note(result)
    print(f"logged {entry.timestamp}  {result.gate}  {entry.label}")
    print(note)
    if entry.note:
        print(entry.note)
    return 0


def cmd_household_status(args: argparse.Namespace) -> int:
    log = HouseholdLog.load(_log_path(args))
    traj = log.trajectory()
    if args.json:
        json.dump(traj, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if traj.get("n", 0) == 0:
        print(f"no entries yet  ({log.path})")
        return 0
    print(f"log: {log.path}")
    print(f"n={traj['n']}  latest_gate={traj['latest_gate']}  latest={traj['latest_label']}")
    print(f"latest mean(v)={traj['latest_mean_v']:.2f}  min(v)={traj['latest_min_v']:.2f}")
    print(f"avg    mean(v)={traj['avg_mean_v']:.2f}  min(v)={traj['avg_min_v']:.2f}")
    counts = "  ".join(f"{g}={c}" for g, c in sorted(traj["gate_counts"].items()))
    print(f"gates: {counts}")
    latest = log.entries[-1]
    print(gate_note(latest.score()))
    return 0


def cmd_household_dashboard(args: argparse.Namespace) -> int:
    log = HouseholdLog.load(_log_path(args))
    path = log.render_latest(args.output)
    print(path)
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    if args.dump_prompt:
        sys.stdout.write(corpus_prompt() + "\n")
        return 0
    text = args.text or ""
    source = ""
    if args.path:
        text = load_corpus(args.path)
        source = str(args.path)
    if not text.strip():
        sys.stderr.write(
            "error: pass a file path or --text\n"
            "  PYTHONPATH=src python3 -m alignment corpus --text 'long article...' --fixture examples/corpus_sample.json --top 5\n"
        )
        return 2
    if not args.fixture:
        sys.stderr.write(
            "error: offline corpus needs --fixture JSON from the corpus scorer\n"
            "  PYTHONPATH=src python3 -m alignment corpus --text '...' --fixture examples/corpus_sample.json\n"
            "Dump the scorer prompt with: python3 -m alignment corpus --prompt\n"
        )
        return 2
    raw = json.loads(_read(args.fixture))
    report = analyze_corpus(text, raw=raw, source=source)
    if args.output or args.json:
        rendered = json.dumps(report.to_dict(), indent=2) + "\n"
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(rendered, encoding="utf-8")
            print(out)
            return 0
        sys.stdout.write(rendered)
        return 0
    print(format_report(report, top=args.top))
    return 0


_EPILOG = """
examples:
  PYTHONPATH=src python3 -m alignment report -s flow
  PYTHONPATH=src python3 -m alignment report examples/flow.json
  PYTHONPATH=src python3 -m alignment dashboard -s refuse -o refuse.png
  PYTHONPATH=src python3 -m alignment sample flow > /tmp/score.json
  PYTHONPATH=src python3 -m alignment reply -s flow -p "A household that practices correction."
  PYTHONPATH=src python3 -m alignment reply examples/refuse.json --post-file original.txt --json
  PYTHONPATH=src python3 -m alignment guard -p "stoke outrage for reach" -s refuse
  PYTHONPATH=src python3 -m alignment guard -p "how to repair a household" --request-json examples/flow.json --draft "Practice correction together." --draft-sample flow --json
  PYTHONPATH=src python3 -m alignment household add --label "Evening conversation" --text "We practiced correction instead of winning." --score-sample flow
  PYTHONPATH=src python3 -m alignment household status
  PYTHONPATH=src python3 -m alignment household dashboard -o outputs/household_latest.png
  PYTHONPATH=src python3 -m alignment corpus --text "policy draft" --fixture examples/corpus_sample.json --top 5
  PYTHONPATH=src python3 -m alignment corpus --prompt
  PYTHONPATH=src python3 -m alignment demo -o outputs
""".strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alignment",
        description="Alignment scoring engine — unit-circle vectors, gate, dashboard.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="Parse LLM JSON and print the gate report")
    r.add_argument(
        "input",
        nargs="?",
        default=None,
        help="JSON file from the scorer, or - for stdin",
    )
    r.add_argument("-s", "--sample", choices=sorted(SAMPLES), help="built-in sample (no file needed)")
    r.set_defaults(func=cmd_report)

    d = sub.add_parser("dashboard", help="Render the paired-circle dashboard PNG")
    d.add_argument("-i", "--input", default=None, help="JSON file or - for stdin")
    d.add_argument("-o", "--output", default="alignment_dashboard.png")
    d.add_argument("-s", "--sample", choices=sorted(SAMPLES))
    d.add_argument("-t", "--title")
    d.set_defaults(func=cmd_dashboard)

    pr = sub.add_parser("prompt", help="Print or save the LLM system prompt")
    pr.add_argument("-o", "--output")
    pr.add_argument("--short", action="store_true", help="Skip embedding full Markdown docs")
    pr.set_defaults(func=cmd_prompt)

    lo = sub.add_parser("lorenz", help="Render Lorenz attractor and σ(Z) motion regimes")
    lo.add_argument("--lorenz", default="lorenz_attractor.png")
    lo.add_argument("--motion", default="motion_regimes.png")
    lo.set_defaults(func=cmd_lorenz)

    de = sub.add_parser("demo", help="Render all sample dashboards + dynamics figures")
    de.add_argument("-o", "--outdir", default="outputs")
    de.set_defaults(func=cmd_demo)

    sm = sub.add_parser("sample", help="Print a canonical sample JSON")
    sm.add_argument("sample", choices=sorted(SAMPLES))
    sm.set_defaults(func=cmd_score_dict)

    rp = sub.add_parser("reply", help="Draft an X reply from a ScoreResult gate (§7.3)")
    rp.add_argument("input", nargs="?", default=None, help="JSON file from the scorer, or - for stdin")
    rp.add_argument("-s", "--sample", choices=sorted(SAMPLES), help="built-in sample (no file needed)")
    rp.add_argument("-p", "--post", help="original post text")
    rp.add_argument("--post-file", help="file containing the original post")
    rp.add_argument("--prompt", dest="dump_prompt", action="store_true", help="print the LLM prompt instead of a draft")
    rp.add_argument("--json", action="store_true", help="emit gate, fruit, may_auto_post, and reply as JSON")
    rp.add_argument("--max-chars", type=int, default=280)
    rp.set_defaults(func=cmd_reply)

    gd = sub.add_parser("guard", help="O-first AI guardrail over a request and draft (§7.2)")
    gd.add_argument("-p", "--request", help="user request text")
    gd.add_argument("--request-file", help="file containing the user request")
    gd.add_argument("-s", "--sample", choices=sorted(SAMPLES), help="built-in sample for the request score")
    gd.add_argument("--request-json", help="JSON file for the request ScoreResult")
    gd.add_argument("--draft", help="canned model draft (offline, skip generation)")
    gd.add_argument("--draft-sample", choices=sorted(SAMPLES), help="built-in sample for the draft score")
    gd.add_argument("--draft-json", help="JSON file for the draft ScoreResult")
    gd.add_argument("--max-retries", type=int, default=1)
    gd.add_argument("--json", action="store_true")
    gd.set_defaults(func=cmd_guard)

    hh = sub.add_parser("household", help="Personal / household log of scored choices")
    hh_sub = hh.add_subparsers(dest="household_cmd", required=True)
    hh_add = hh_sub.add_parser("add", help="Append a scored choice to the local JSON log")
    hh_add.add_argument("--label", required=True)
    hh_add.add_argument("--text", required=True)
    hh_add.add_argument("--note", default="")
    hh_add.add_argument("--who", default="", help="optional name when sharing a household file")
    hh_add.add_argument("--score-sample", choices=sorted(SAMPLES))
    hh_add.add_argument("--score", help="JSON file from the scorer")
    hh_add.add_argument("--log", default=str(DEFAULT_LOG), help="path to household JSON (default: household.json)")
    hh_add.set_defaults(func=cmd_household_add)

    hh_st = hh_sub.add_parser("status", help="Show running mean(v), min(v), and gate counts")
    hh_st.add_argument("--log", default=str(DEFAULT_LOG))
    hh_st.add_argument("--json", action="store_true")
    hh_st.set_defaults(func=cmd_household_status)

    hh_dash = hh_sub.add_parser("dashboard", help="Render the latest entry as a paired-circle PNG")
    hh_dash.add_argument("--log", default=str(DEFAULT_LOG))
    hh_dash.add_argument("-o", "--output", default="outputs/household_latest.png")
    hh_dash.set_defaults(func=cmd_household_dashboard)

    co = sub.add_parser("corpus", help="Score a long text against A1–A28 / M1–M28")
    co.add_argument("path", nargs="?", help="text or PDF file")
    co.add_argument("--text", help="inline corpus text")
    co.add_argument("--fixture", help="offline scorer JSON (core vectors + a_hits/m_hits)")
    co.add_argument("--top", type=int, default=5)
    co.add_argument("--json", action="store_true")
    co.add_argument("-o", "--output", help="write JSON report to this path")
    co.add_argument("--prompt", dest="dump_prompt", action="store_true", help="print the corpus scorer prompt")
    co.set_defaults(func=cmd_corpus)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd in {"report", "dashboard", "reply"}:
        if not getattr(args, "sample", None) and not getattr(args, "input", None):
            sys.stderr.write(
                "error: pass a JSON file or -s {flow,refuse,correct,doom_loop}\n"
                "  PYTHONPATH=src python3 -m alignment report -s flow\n"
                "  PYTHONPATH=src python3 -m alignment report examples/flow.json\n"
            )
            return 2
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"invalid scorer output: {exc}\n")
        return 2
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
