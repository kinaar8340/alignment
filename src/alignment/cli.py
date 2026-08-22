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
from .agent import AlignedAgent
from .live import LiveError, detect_backend, pipeline_live, score_live, reply_live, complete
from .x_bot import BotError, BotLog, DEFAULT_LOG as BOT_LOG, consider as bot_consider
from .timeline import demo_year, fetch_user_posts, read_posts, read_scores, score_posts, trajectory_stats, write_posts, write_scores
from .torus import render_torus_dashboard
from .aura import render_aura_still, render_aura_video


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


def cmd_agent(args: argparse.Namespace) -> int:
    if not args.goal:
        sys.stderr.write("error: pass --goal\n")
        return 2
    plan_result = None
    if args.score_sample:
        plan_result = parse_llm_output(SAMPLES[args.score_sample], source=args.score_sample)
    elif args.score:
        plan_result = parse_llm_output(_read(args.score), source=args.score)
    if args.plan is None and plan_result is None:
        sys.stderr.write(
            "error: offline agent needs --plan and --score-sample (or --score JSON)\n"
            "  PYTHONPATH=src python3 -m alignment agent --goal 'repair a household' "
            "--plan 'Practice correction together.' --score-sample flow\n"
        )
        return 2
    if args.plan is None:
        sys.stderr.write("error: pass --plan (or wire a model)\n")
        return 2
    if plan_result is None:
        sys.stderr.write("error: pass --score-sample or --score\n")
        return 2
    corrected_result = None
    if args.rewrite_sample:
        corrected_result = parse_llm_output(SAMPLES[args.rewrite_sample], source=args.rewrite_sample)
    agent = AlignedAgent()
    out = agent.step(
        args.goal,
        args.context or "",
        plan=args.plan,
        plan_result=plan_result,
        corrected=args.rewrite,
        corrected_result=corrected_result,
    )
    if args.json:
        payload = {
            k: v
            for k, v in out.items()
            if k not in {"plan_result", "corrected_result"}
        }
        if out.get("plan_result") is not None:
            payload["plan_result"] = out["plan_result"].to_dict()
        if out.get("corrected_result") is not None:
            payload["corrected_result"] = out["corrected_result"].to_dict()
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    print(f"status: {out['status']}")
    print(f"gate:   {out['gate']}")
    if out.get("final_action"):
        print(f"action: {out['final_action']}")
    else:
        print(f"fruit:  {out.get('fruit')}")
        print(out["recovery"])
    return 0


def _live_text(args: argparse.Namespace) -> str:
    if getattr(args, "text_file", None):
        return _read(args.text_file)
    return (getattr(args, "text", None) or "").strip()


def cmd_live_ping(args: argparse.Namespace) -> int:
    backend = args.backend
    kind = detect_backend() if backend == "auto" else backend
    text = complete(
        "Reply with exactly: pong",
        "ping",
        model=args.model,
        backend=kind,
        temperature=0.0,
    )
    print(f"backend={kind}  model={args.model}")
    print(text.strip())
    return 0


def cmd_live_score(args: argparse.Namespace) -> int:
    text = _live_text(args)
    if not text:
        sys.stderr.write("error: pass -p/--text or --text-file\n")
        return 2
    result = score_live(text, model=args.model, backend=args.backend)
    if args.json:
        json.dump(result.to_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    print_report(result)
    return 0


def cmd_live_reply(args: argparse.Namespace) -> int:
    text = _live_text(args)
    if not text:
        sys.stderr.write("error: pass -p/--text or --text-file\n")
        return 2
    if args.score_sample:
        result = parse_llm_output(SAMPLES[args.score_sample], source=args.score_sample)
    else:
        result = score_live(text, model=args.model, backend=args.backend)
    reply = reply_live(text, result, model=args.model, backend=args.backend, max_chars=args.max_chars)
    if args.json:
        json.dump(
            {
                "gate": result.gate,
                "may_auto_post": result.gate in {"FLOW", "CORRECT"},
                "result": result.to_dict(),
                "reply": reply,
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0
    print_report(result)
    print()
    print("REPLY:")
    print(reply)
    return 0


def cmd_live_pipeline(args: argparse.Namespace) -> int:
    text = _live_text(args)
    if not text:
        sys.stderr.write("error: pass -p/--text or --text-file\n")
        return 2
    out = pipeline_live(text, model=args.model, backend=args.backend, max_chars=args.max_chars)
    result = out["result"]
    if args.json:
        json.dump(
            {
                "backend": out["backend"],
                "model": out["model"],
                "gate": result.gate,
                "may_auto_post": out["may_auto_post"],
                "result": result.to_dict(),
                "reply": out["reply"],
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0
    print(f"backend={out['backend']}  model={out['model']}")
    print_report(result)
    print()
    print("REPLY:")
    print(out["reply"])
    return 0


def cmd_bot_consider(args: argparse.Namespace) -> int:
    text = args.text or ""
    if args.text_file:
        text = _read(args.text_file)
    if not text.strip():
        sys.stderr.write("error: pass -p/--text or --text-file\n")
        return 2
    if args.live:
        result = score_live(text, model=args.model, backend=args.backend)
    elif args.sample:
        result = parse_llm_output(SAMPLES[args.sample], source=args.sample)
    elif args.score:
        result = parse_llm_output(_read(args.score), source=args.score)
    else:
        sys.stderr.write(
            "error: pass -s/--sample, --score JSON, or --live\n"
            "  PYTHONPATH=src python3 -m alignment bot consider -p '…' -s flow\n"
        )
        return 2
    llm = None
    if args.live_reply:
        from .live import grok_llm

        llm = grok_llm(model=args.model, backend=args.backend)
    out = bot_consider(
        text,
        result,
        reply_to=args.reply_to,
        on_correct=args.on_correct,
        dry_run=not args.post,
        log_path=args.log,
        llm_callable=llm,
    )
    if args.json:
        payload = {k: v for k, v in out.items() if k != "event"}
        payload["event"] = out["event"].to_dict()
        json.dump(payload, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    print(f"gate:     {out['gate']}")
    print(f"decision: {out['decision']}")
    print(f"posted:   {out['posted']}  dry_run={out['dry_run']}")
    print(out["reason"])
    if out["decision"] == "post":
        print()
        print("REPLY:")
        print(out["reply"])
    return 0


def cmd_bot_log(args: argparse.Namespace) -> int:
    log = BotLog.load(args.log)
    if args.json:
        json.dump([e.to_dict() for e in log.events], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if not log.events:
        print(f"no events yet  ({log.path})")
        return 0
    print(f"log: {log.path}  n={len(log.events)}")
    for e in log.events[-args.tail :]:
        flag = "POST" if e.posted else ("DRY" if e.dry_run and e.decision == "post" else "SKIP")
        print(f"  {e.timestamp}  {flag:4}  {e.gate:13}  {e.fruit}")
    return 0


def cmd_visualize_demo(args: argparse.Namespace) -> int:
    rows = demo_year(username=args.user or "demo", weeks=args.weeks)
    path = render_torus_dashboard(rows, title=args.title or "Alignment torus — demo year", save_path=args.output)
    stats = trajectory_stats(rows)
    print(f"n={stats['n']} avg_net={stats['avg_net']:.2f} gates={stats['gate_counts']}")
    print(path)
    return 0


def cmd_visualize_fetch(args: argparse.Namespace) -> int:
    from datetime import datetime, timezone

    start = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc) if args.since else None
    posts = fetch_user_posts(args.user, start=start, max_posts=args.limit, include_replies=not args.no_replies)
    path = write_posts(posts, args.output)
    replies = sum(1 for p in posts if p.is_reply)
    print(f"wrote {len(posts)} posts ({replies} replies) → {path}")
    return 0


def cmd_visualize_score(args: argparse.Namespace) -> int:
    from .timeline import select_posts

    posts = [p for p in read_posts(args.posts) if len((p.text or "").strip()) >= args.min_chars]
    posts = select_posts(posts, limit=args.limit, spread=args.spread, stride=args.stride)
    if not posts:
        sys.stderr.write("error: no posts in file\n")
        return 2
    print(
        f"scoring {len(posts)} posts  "
        f"{posts[0].created_at[:10]} → {posts[-1].created_at[:10]}  "
        f"(spread={args.spread} stride={args.stride})"
    )
    if args.live:
        from .live import score_live

        def scorer(text: str):
            return score_live(text, model=args.model, backend=args.backend)

        rows = score_posts(posts, scorer)
    else:
        sys.stderr.write("error: pass --live to score with Grok (or use visualize demo)\n")
        return 2
    path = write_scores(rows, args.output)
    print(json.dumps(trajectory_stats(rows), indent=2))
    print(path)
    return 0


def cmd_visualize_render(args: argparse.Namespace) -> int:
    rows = read_scores(args.scores)
    path = render_torus_dashboard(rows, title=args.title or "Alignment torus", save_path=args.output)
    print(json.dumps(trajectory_stats(rows), indent=2))
    print(path)
    return 0


def cmd_visualize_aura(args: argparse.Namespace) -> int:
    if bool(getattr(args, "demo", False)) == bool(getattr(args, "scores", None)):
        sys.stderr.write("error: pass --demo or --scores PATH\n")
        return 2
    rows = demo_year(username=args.user or "demo", weeks=args.weeks) if args.demo else read_scores(args.scores)
    if not rows:
        sys.stderr.write("error: no scored posts for aura\n")
        return 2
    print(json.dumps(trajectory_stats(rows), indent=2))
    still_path = args.still or str(Path(args.output).with_suffix(".png"))
    still = render_aura_still(
        rows,
        save_path=still_path,
        title=args.title or "Alignment aura",
    )
    print(still)
    if args.still_only:
        return 0

    def _progress(i: int, n: int) -> None:
        if i == 0 or i + 1 == n or (i + 1) % 10 == 0:
            print(f"  frame {i + 1}/{n}", flush=True)

    video = render_aura_video(
        rows,
        save_path=args.output,
        fps=args.fps,
        frames_per_step=args.frames_per_step,
        title=args.title or "Alignment aura",
        progress=_progress,
    )
    print(video)
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
  PYTHONPATH=src python3 -m alignment agent --goal "repair a household" --plan "Practice correction together." --score-sample flow
  PYTHONPATH=src python3 -m alignment bot consider -p "We practiced correction together." -s flow
  PYTHONPATH=src python3 -m alignment bot consider -p "Farm engagement by stoking outrage." -s refuse
  PYTHONPATH=src python3 -m alignment visualize demo -o outputs/torus_dashboard.png
  PYTHONPATH=src python3 -m alignment visualize fetch --user kinaar8340 --since 2025-08-22 -o data/posts.jsonl
  PYTHONPATH=src python3 -m alignment visualize render --scores data/scores.jsonl -o outputs/torus_dashboard.png
  PYTHONPATH=src python3 -m alignment visualize aura --demo -o outputs/aura.mp4
  PYTHONPATH=src python3 -m alignment visualize aura --scores data/scores.jsonl -o outputs/aura_kinaar.mp4
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

    ag = sub.add_parser("agent", help="Planner / critic / executor loop on the §7.2 gate")
    ag.add_argument("--goal", required=True)
    ag.add_argument("--context", default="")
    ag.add_argument("--plan", help="canned planner action (offline)")
    ag.add_argument("--score-sample", choices=sorted(SAMPLES), help="built-in sample for critic score of the plan")
    ag.add_argument("--score", help="JSON file for the critic ScoreResult")
    ag.add_argument("--rewrite", help="canned σ(Z) rewrite when the plan is CORRECT")
    ag.add_argument("--rewrite-sample", choices=sorted(SAMPLES), help="critic score of the rewrite")
    ag.add_argument("--json", action="store_true")
    ag.set_defaults(func=cmd_agent)

    lv = sub.add_parser("live", help="Score and reply with live Grok (API or grok CLI)")
    lv_sub = lv.add_subparsers(dest="live_cmd", required=True)
    live_common = argparse.ArgumentParser(add_help=False)
    live_common.add_argument("-m", "--model", default="grok-4.6")
    live_common.add_argument("--backend", choices=["auto", "api", "cli"], default="auto")

    ping = lv_sub.add_parser("ping", parents=[live_common], help="Check the Grok backend")
    ping.set_defaults(func=cmd_live_ping)

    sc = lv_sub.add_parser("score", parents=[live_common], help="Live-score text with Grok")
    sc.add_argument("-p", "--text", help="text to score")
    sc.add_argument("--text-file", help="file containing the text")
    sc.add_argument("--json", action="store_true")
    sc.set_defaults(func=cmd_live_score)

    rp_live = lv_sub.add_parser("reply", parents=[live_common], help="Live-score then draft an X reply")
    rp_live.add_argument("-p", "--text", help="original post")
    rp_live.add_argument("--text-file")
    rp_live.add_argument("--score-sample", choices=sorted(SAMPLES), help="skip live scoring; use a fixture")
    rp_live.add_argument("--max-chars", type=int, default=280)
    rp_live.add_argument("--json", action="store_true")
    rp_live.set_defaults(func=cmd_live_reply)

    pipe = lv_sub.add_parser("pipeline", parents=[live_common], help="Live score + live reply")
    pipe.add_argument("-p", "--text")
    pipe.add_argument("--text-file")
    pipe.add_argument("--max-chars", type=int, default=280)
    pipe.add_argument("--json", action="store_true")
    pipe.set_defaults(func=cmd_live_pipeline)

    bot = sub.add_parser("bot", help="X bot: post only when the gate allows")
    bot_sub = bot.add_subparsers(dest="bot_cmd", required=True)
    bc = bot_sub.add_parser("consider", help="Score a post and maybe reply (dry-run by default)")
    bc.add_argument("-p", "--text", help="incoming post text")
    bc.add_argument("--text-file")
    bc.add_argument("-s", "--sample", choices=sorted(SAMPLES))
    bc.add_argument("--score", help="JSON ScoreResult")
    bc.add_argument("--live", action="store_true", help="score with live Grok")
    bc.add_argument("--live-reply", action="store_true", help="draft the reply with live Grok")
    bc.add_argument("-m", "--model", default="grok-4.6")
    bc.add_argument("--backend", choices=["auto", "api", "cli"], default="auto")
    bc.add_argument("--on-correct", choices=["question", "skip"], default="question")
    bc.add_argument("--reply-to", help="X post id to reply to")
    bc.add_argument("--post", action="store_true", help="actually post (requires X_USER_ACCESS_TOKEN)")
    bc.add_argument("--log", default=str(BOT_LOG))
    bc.add_argument("--json", action="store_true")
    bc.set_defaults(func=cmd_bot_consider)

    bl = bot_sub.add_parser("log", help="Show recent bot decisions")
    bl.add_argument("--log", default=str(BOT_LOG))
    bl.add_argument("--tail", type=int, default=20)
    bl.add_argument("--json", action="store_true")
    bl.set_defaults(func=cmd_bot_log)

    vz = sub.add_parser("visualize", help="Torus of stacked unit circles, 2D trends, and aura field")
    vz_sub = vz.add_subparsers(dest="visualize_cmd", required=True)
    vd = vz_sub.add_parser("demo", help="Render a synthetic year (no live scoring)")
    vd.add_argument("-o", "--output", default="outputs/torus_dashboard.png")
    vd.add_argument("--weeks", type=int, default=52)
    vd.add_argument("--user", default="demo")
    vd.add_argument("--title")
    vd.set_defaults(func=cmd_visualize_demo)

    vf = vz_sub.add_parser("fetch", help="Download a year of posts/replies from X")
    vf.add_argument("--user", required=True)
    vf.add_argument("--since", default="2025-08-22", help="ISO date YYYY-MM-DD")
    vf.add_argument("--limit", type=int, default=500)
    vf.add_argument("--no-replies", action="store_true")
    vf.add_argument("-o", "--output", default="data/posts.jsonl")
    vf.set_defaults(func=cmd_visualize_fetch)

    vs = vz_sub.add_parser("score", help="Score ingested posts with live Grok")
    vs.add_argument("--posts", required=True)
    vs.add_argument("-o", "--output", default="data/scores.jsonl")
    vs.add_argument("--live", action="store_true")
    vs.add_argument("--limit", type=int, default=40)
    vs.add_argument("--stride", type=int, default=1, help="keep every Nth post after the length filter")
    vs.add_argument("--spread", dest="spread", action="store_true", default=True, help="pick --limit posts evenly across the archive (default)")
    vs.add_argument("--no-spread", dest="spread", action="store_false", help="take the first --limit posts instead")
    vs.add_argument("--min-chars", type=int, default=40, help="skip URL-only / stub posts")
    vs.add_argument("-m", "--model", default="grok-4.6")
    vs.add_argument("--backend", choices=["auto", "api", "cli"], default="auto")
    vs.set_defaults(func=cmd_visualize_score)

    vr = vz_sub.add_parser("render", help="Render torus + trend charts from scored JSONL")
    vr.add_argument("--scores", required=True)
    vr.add_argument("-o", "--output", default="outputs/torus_dashboard.png")
    vr.add_argument("--title")
    vr.set_defaults(func=cmd_visualize_render)

    va = vz_sub.add_parser("aura", help="Seven-ray heatmaps; origin is mean net")
    va.add_argument("--demo", action="store_true", help="synthetic year (no live scoring)")
    va.add_argument("--scores", help="scored JSONL from visualize score")
    va.add_argument("-o", "--output", default="outputs/aura.mp4")
    va.add_argument("--still", help="still PNG path (default: output with .png)")
    va.add_argument("--still-only", action="store_true", help="write the mean-field PNG, skip ffmpeg")
    va.add_argument("--fps", type=int, default=12)
    va.add_argument("--frames-per-step", type=int, default=3, help="interpolated frames between consecutive posts")
    va.add_argument("--weeks", type=int, default=52)
    va.add_argument("--user", default="demo")
    va.add_argument("--title")
    va.set_defaults(func=cmd_visualize_aura)
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
    except LiveError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except BotError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
