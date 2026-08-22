from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from alignment.aura import (
    interpolate_rows,
    lerp,
    lerp_vec,
    overall_glow,
    render_aura_still,
    render_aura_video,
    slice_alpha,
    slice_hue,
    slice_hues,
)
from alignment.cli import main
from alignment.timeline import demo_year


def test_lerp_and_vec():
    assert lerp(0, 10, 0.5) == 5.0
    assert lerp_vec([0, 10], [10, 0], 0.5) == [5.0, 5.0]


def test_slice_hues_are_spectral():
    hues = slice_hues()
    assert len(hues) == 10
    assert 0.55 < hues[0] < 0.65  # blue at 12 o'clock
    assert hues[3] > 0.85 or hues[3] < 0.05  # magenta / pink
    # full cycle, no two adjacent hues identical
    assert all(slice_hue(i) != slice_hue(i + 1) for i in range(9))


def test_slice_alpha_maps_net():
    assert slice_alpha(10) > slice_alpha(0) > slice_alpha(-10)
    assert slice_alpha(-10) == pytest.approx(0.15)
    assert slice_alpha(10) == pytest.approx(1.0)


def test_overall_glow_by_gate():
    flow = overall_glow(8.0, 0.0, "FLOW")
    correct = overall_glow(1.0, -0.2, "CORRECT")
    refuse = overall_glow(1.2, -2.0, "REFUSE")
    assert flow > correct > refuse
    assert 0.28 <= refuse <= 1.0
    assert flow == pytest.approx(1.0)


def test_angles_start_at_twelve():
    from alignment.aura import _angles

    angs = _angles()
    assert angs[0] == pytest.approx(np.pi / 2.0)
    assert len(angs) == 10


def test_interpolate_empty():
    with pytest.raises(ValueError, match="No scored posts"):
        interpolate_rows([])


def test_interpolate_loop_closes():
    rows = demo_year(weeks=5)
    frames = interpolate_rows(rows, frames_per_step=2, loop=True)
    assert len(frames) == 10
    assert frames[0]["v"] == list(rows[0].result.v)
    # last pair starts at the last post
    assert frames[8]["v"] == list(rows[4].result.v)


def test_interpolate_no_loop_holds_last():
    rows = demo_year(weeks=3)
    frames = interpolate_rows(rows, frames_per_step=2, loop=False)
    assert len(frames) == 5  # 2 + 2 + 1
    assert frames[-1]["v"] == list(rows[-1].result.v)
    assert frames[-1]["gate"] == rows[-1].result.gate


def test_render_still_png(tmp_path: Path):
    path = render_aura_still(demo_year(weeks=8), save_path=tmp_path / "aura.png")
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_still_last(tmp_path: Path):
    rows = demo_year(weeks=8)
    path = render_aura_still(rows, save_path=tmp_path / "last.png", which="last")
    assert path.is_file()
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_aura_requires_source(capsys):
    assert main(["visualize", "aura", "--still-only"]) == 2
    assert "--demo or --scores" in capsys.readouterr().err


def test_cli_visualize_aura_still(tmp_path: Path, capsys):
    out = tmp_path / "aura.mp4"
    still = tmp_path / "aura.png"
    assert main(["visualize", "aura", "--demo", "--weeks", "8", "--still-only", "-o", str(out), "--still", str(still)]) == 0
    assert still.is_file()
    assert not out.exists()
    captured = capsys.readouterr().out
    assert str(still) in captured
    assert "gate_counts" in captured


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_render_video_mp4(tmp_path: Path):
    path = render_aura_video(
        demo_year(weeks=3),
        save_path=tmp_path / "aura.mp4",
        fps=6,
        frames_per_step=1,
        dpi=60,
    )
    data = path.read_bytes()
    assert path.stat().st_size > 1000
    assert b"ftyp" in data[:32]
