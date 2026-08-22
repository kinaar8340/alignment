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


_EPILOG = """
examples:
  PYTHONPATH=src python3 -m alignment report -s flow
  PYTHONPATH=src python3 -m alignment report examples/flow.json
  PYTHONPATH=src python3 -m alignment dashboard -s refuse -o refuse.png
  PYTHONPATH=src python3 -m alignment sample flow > /tmp/score.json
  PYTHONPATH=src python3 -m alignment reply -s flow -p "A household that practices correction."
  PYTHONPATH=src python3 -m alignment reply examples/refuse.json --post-file original.txt --json
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
    except (json.JSONDecodeError, ValueError) as exc:
        sys.stderr.write(f"invalid scorer output: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
