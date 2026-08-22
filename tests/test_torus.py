from __future__ import annotations

from pathlib import Path

from alignment.cli import main
from alignment.timeline import Post, demo_year, select_posts, trajectory_stats
from alignment.torus import render_torus_dashboard, torus_frame


def test_demo_year_stats():
    rows = demo_year(weeks=16)
    stats = trajectory_stats(rows)
    assert stats["n"] == 16
    assert stats["replies"] + stats["posts"] == 16
    assert "FLOW" in stats["gate_counts"]


def test_torus_frame_closes():
    xyz, cols = torus_frame([8, 7, 6, 5, 4, 3, 2, 1, 0, -2], 0.3)
    assert xyz.shape == (11, 3)
    assert len(cols) == 11


def test_render_png(tmp_path: Path):
    path = render_torus_dashboard(demo_year(weeks=24), save_path=tmp_path / "torus.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_select_posts_spread_spans_archive():
    posts = [
        Post(id=str(i), created_at=f"2026-01-{i+1:02d}", text="x" * 50, is_reply=False)
        for i in range(100)
    ]
    picked = select_posts(posts, limit=5, spread=True)
    assert [p.id for p in picked] == ["0", "25", "50", "74", "99"]


def test_select_posts_no_spread_is_prefix():
    posts = [
        Post(id=str(i), created_at=f"2026-01-{i+1:02d}", text="x", is_reply=False)
        for i in range(10)
    ]
    picked = select_posts(posts, limit=3, spread=False)
    assert [p.id for p in picked] == ["0", "1", "2"]


def test_select_posts_stride():
    posts = [
        Post(id=str(i), created_at=f"2026-01-{i+1:02d}", text="x", is_reply=False)
        for i in range(10)
    ]
    picked = select_posts(posts, stride=3, spread=False)
    assert [p.id for p in picked] == ["0", "3", "6", "9"]


def test_cli_visualize_demo(tmp_path: Path, capsys):
    out = tmp_path / "viz.png"
    assert main(["visualize", "demo", "-o", str(out), "--weeks", "12"]) == 0
    assert out.is_file()
    assert str(out) in capsys.readouterr().out
