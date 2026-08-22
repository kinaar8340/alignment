"""Human-aura projection of the 10-slice unit circle.

A continuous spectral hoop encodes the same v / w / net vectors as the
torus: hue is locked to slice angle (blue at 12 o'clock → purple →
magenta → pink → yellow → cyan), local opacity follows net, and the
master glow follows the gate. Time is the torus θ. Magnitudes stay the
actual scores — no second scoring surface.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle, Ellipse

from .slices import ALIGNED_NAMES, N_SLICES, ROMAN
from .timeline import TimedScore, trajectory_stats
from .torus import GATE_COLORS

PathLike = Union[str, Path]
Progress = Optional[Callable[[int, int], None]]

_BG = "#05070c"
_TEXT = "#e6edf3"
_GOLD = "#d29922"
_NEUTRAL = "#8b949e"

# Slice i (12 o'clock, the X) is blue. Clockwise walks the rainbow so the
# hoop matches the spectral reference: blue → purple → magenta → pink → yellow.
_HUE0 = 0.60
_N_SEG = 180
_GLOW_LAYERS = (
    (26.0, 0.06),
    (16.0, 0.10),
    (9.0, 0.16),
    (4.8, 0.32),
    (2.4, 0.55),
    (1.15, 1.00),
)


def _angles() -> np.ndarray:
    """Slice i at 12 o'clock, clockwise — same as the unit circle."""
    return np.pi / 2.0 - 2.0 * np.pi * np.arange(N_SLICES) / N_SLICES


def lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * t


def lerp_vec(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    return [lerp(x, y, t) for x, y in zip(a, b)]


def slice_hue(index: float) -> float:
    """Hue in [0, 1) for a (possibly fractional) slice index."""
    return (_HUE0 + float(index) / N_SLICES) % 1.0


def slice_hues() -> List[float]:
    return [slice_hue(i) for i in range(N_SLICES)]


def slice_alpha(net_i: float) -> float:
    """Local glow from net ∈ [−10, +10]. Never fully invisible."""
    return float(np.clip((float(net_i) + 10.0) / 20.0, 0.15, 1.0))


def overall_glow(mean_v: float, mean_w: float, gate: str) -> float:
    """Master intensity from aggregates + gate."""
    metric = float(np.clip((float(mean_v) - abs(float(mean_w)) + 5.0) / 15.0, 0.25, 1.0))
    gate_scale = {
        "FLOW": 1.0,
        "CORRECT": 0.72,
        "STOP_FEEDING": 0.40,
        "REFUSE": 0.30,
    }.get(gate, 0.6)
    return float(np.clip(metric * gate_scale * 1.25, 0.28, 1.0))


def _hsv(h: float, s: float = 1.0, v: float = 1.0) -> Tuple[float, float, float]:
    rgb = hsv_to_rgb((h % 1.0, float(np.clip(s, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))))
    return float(rgb[0]), float(rgb[1]), float(rgb[2])


def _tint(rgb: Tuple[float, float, float], toward: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    return tuple(lerp(a, b, t) for a, b in zip(rgb, toward))  # type: ignore[return-value]


def interpolate_rows(
    rows: List[TimedScore],
    frames_per_step: int = 4,
    *,
    loop: bool = True,
) -> List[dict]:
    """Smooth states between consecutive scored posts.

    loop=True closes the torus: the last post interpolates back to the first.
    """
    if not rows:
        raise ValueError("No scored posts for aura")
    frames_per_step = max(1, frames_per_step)
    out: List[dict] = []
    n = len(rows)
    for i in range(n):
        row = rows[i]
        if loop and n > 1:
            nxt = rows[(i + 1) % n]
            steps = frames_per_step
        elif i < n - 1:
            nxt = rows[i + 1]
            steps = frames_per_step
        else:
            nxt = row
            steps = 1
        for s in range(steps):
            t = s / float(steps)
            v = lerp_vec(row.result.v, nxt.result.v, t)
            w = lerp_vec(row.result.w, nxt.result.w, t)
            net = [vv + ww for vv, ww in zip(v, w)]
            gate = row.result.gate if t < 0.5 else nxt.result.gate
            created = row.post.created_at[:10] if row.post.created_at else ""
            out.append(
                {
                    "v": v,
                    "w": w,
                    "net": net,
                    "mean_v": lerp(row.result.mean_v, nxt.result.mean_v, t),
                    "mean_w": lerp(row.result.mean_w, nxt.result.mean_w, t),
                    "gate": gate,
                    "index": i + t,
                    "label": created,
                    "n": n,
                }
            )
    return out


def _interp_at(values: Sequence[float], index: float) -> float:
    n = len(values)
    i0 = int(np.floor(index)) % n
    i1 = (i0 + 1) % n
    f = float(index) - np.floor(index)
    return lerp(values[i0], values[i1], f)


def _ellipse_points(rx: float, ry: float, cy: float, n: int = _N_SEG) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clockwise from 12 o'clock. Returns xs, ys, slice-index along the hoop."""
    t = np.linspace(0.0, 1.0, n, endpoint=True)
    ang = np.pi / 2.0 - 2.0 * np.pi * t
    xs = rx * np.cos(ang)
    ys = cy + ry * np.sin(ang)
    idx = t * N_SLICES
    return xs, ys, idx


def _segment_colors(
    idx: np.ndarray,
    net: Sequence[float],
    *,
    gate: str,
    overall: float,
    layer_alpha: float,
) -> np.ndarray:
    refuse = gate == "REFUSE"
    stop = gate == "STOP_FEEDING"
    sat = 0.42 if refuse else (0.62 if stop else 1.0)
    val = 0.72 if refuse else (0.82 if stop else 1.0)
    red = (0.95, 0.22, 0.20)
    colors = []
    for k in range(len(idx) - 1):
        mid = 0.5 * (idx[k] + idx[k + 1])
        h = slice_hue(mid)
        local = slice_alpha(_interp_at(net, mid))
        boost = 0.55 + 0.45 * local
        rgb = _hsv(h, sat, val * boost)
        if refuse:
            rgb = _tint(rgb, red, 0.42)
        elif stop:
            rgb = _tint(rgb, red, 0.22)
        if layer_alpha >= 0.9:
            a = float(np.clip(0.38 + 0.62 * local * overall, 0.28, 1.0))
        else:
            a = float(np.clip(layer_alpha * (0.40 + 1.05 * local * overall), 0.035, 0.55))
        colors.append((rgb[0], rgb[1], rgb[2], a))
    return np.asarray(colors)


def _beams(ax, xs, ys, idx, v, net, *, rx: float, ry: float, cy: float, gate: str, overall: float, breath: float) -> None:
    refuse = gate in {"REFUSE", "STOP_FEEDING"}
    segs_bloom, cols_bloom, lws_bloom = [], [], []
    segs_mid, cols_mid, lws_mid = [], [], []
    segs_core, cols_core, lws_core = [], [], []
    y_top = (1.15 + 5.8 * overall) * breath
    for k in range(len(xs) - 1):
        mid = 0.5 * (idx[k] + idx[k + 1])
        vv = max(0.0, _interp_at(v, mid))
        local = slice_alpha(_interp_at(net, mid))
        h = slice_hue(mid)
        rgb = _hsv(h, 0.50 if refuse else 0.80, 1.0)
        if refuse:
            rgb = _tint(rgb, (0.95, 0.22, 0.20), 0.35)
        x, y0 = xs[k], ys[k]
        p0, p1 = (x, y0), (x, max(y0 + 0.4, y_top))
        bloom_a = float(np.clip(0.035 + 0.055 * local * overall, 0.03, 0.12))
        mid_a = float(np.clip(0.05 + 0.10 * local * overall, 0.04, 0.20))
        core_a = float(np.clip(0.08 + 0.28 * (vv / 10.0) * overall, 0.05, 0.40))
        segs_bloom.append((p0, p1))
        cols_bloom.append((rgb[0], rgb[1], rgb[2], bloom_a))
        lws_bloom.append(7.5 + 1.8 * vv)
        segs_mid.append((p0, p1))
        cols_mid.append((rgb[0], rgb[1], rgb[2], mid_a))
        lws_mid.append(2.6 + 0.7 * vv)
        segs_core.append((p0, p1))
        cols_core.append((rgb[0], rgb[1], rgb[2], core_a))
        lws_core.append(0.7 + 0.25 * vv)
    ax.add_collection(LineCollection(segs_bloom, colors=cols_bloom, linewidths=lws_bloom, capstyle="round", zorder=3))
    ax.add_collection(LineCollection(segs_mid, colors=cols_mid, linewidths=lws_mid, capstyle="round", zorder=3))
    ax.add_collection(LineCollection(segs_core, colors=cols_core, linewidths=lws_core, capstyle="round", zorder=3))
    for j, ang in enumerate(_angles()):
        vv = max(0.0, float(v[j]))
        if vv < 0.5:
            continue
        x = rx * np.cos(ang)
        y0 = cy + ry * np.sin(ang)
        rgb = _hsv(slice_hue(j), 0.45 if refuse else 0.85, 1.0)
        ax.plot(
            [x, x],
            [y0, max(y0 + 0.4, y_top)],
            color=rgb,
            linewidth=1.05,
            alpha=float(np.clip(0.16 + 0.30 * overall, 0.12, 0.48)),
            solid_capstyle="round",
            zorder=3,
        )


def _sparkles(ax, rx, ry, cy, *, overall: float, breath: float, gate: str) -> None:
    rng = np.random.default_rng(11)
    n = 70 if gate != "FLOW" else 110
    angs = rng.uniform(0, 2 * np.pi, n)
    rad = rng.uniform(0.90, 1.16, n)
    lift = rng.uniform(0.0, 5.2, n) * overall
    xs = rx * rad * np.cos(angs)
    ys = cy + ry * rad * np.sin(angs) + lift
    sizes = rng.uniform(3.0, 16.0, n)
    hues = (angs / (2 * np.pi) + _HUE0) % 1.0
    cols = [_hsv(h, 0.35, 1.0) for h in hues]
    alpha = float(np.clip(0.12 + 0.45 * overall, 0.08, 0.55)) * (0.75 + 0.25 * breath)
    ax.scatter(xs, ys, s=sizes, c=cols, alpha=alpha, linewidths=0, zorder=4)


def _silhouette(ax, *, refuse: bool = False, scale: float = 0.78) -> None:
    s = scale
    body = "#cbb889" if not refuse else "#7a5555"
    glow = "#f0e2a8" if not refuse else "#c45c5c"
    ax.add_patch(Ellipse((0, -0.28 * s), 1.50 * s, 3.20 * s, facecolor=body, edgecolor=glow, linewidth=1.1, alpha=0.78, zorder=6))
    ax.add_patch(Circle((0, 1.78 * s), 0.58 * s, facecolor=body, edgecolor=glow, linewidth=1.1, alpha=0.82, zorder=6))
    ax.add_patch(Ellipse((-0.90 * s, -0.10 * s), 0.38 * s, 2.00 * s, angle=16, facecolor=body, edgecolor=glow, linewidth=0.7, alpha=0.72, zorder=5))
    ax.add_patch(Ellipse((0.90 * s, -0.10 * s), 0.38 * s, 2.00 * s, angle=-16, facecolor=body, edgecolor=glow, linewidth=0.7, alpha=0.72, zorder=5))
    ax.add_patch(Ellipse((-0.36 * s, -2.40 * s), 0.44 * s, 1.90 * s, facecolor=body, edgecolor=glow, linewidth=0.7, alpha=0.75, zorder=5))
    ax.add_patch(Ellipse((0.36 * s, -2.40 * s), 0.44 * s, 1.90 * s, facecolor=body, edgecolor=glow, linewidth=0.7, alpha=0.75, zorder=5))
    core = (0.95, 0.32, 0.28) if refuse else (0.95, 0.78, 0.28)
    for r, a in ((1.15, 0.07), (0.62, 0.14), (0.28, 0.40), (0.12, 0.85)):
        ax.add_patch(Circle((0, 0.18 * s), r * s, facecolor=core, edgecolor="none", alpha=a, zorder=7))


def render_aura_ring(
    ax,
    v: Sequence[float],
    w: Sequence[float],
    net: Sequence[float],
    *,
    gate: str = "CORRECT",
    mean_v: Optional[float] = None,
    mean_w: Optional[float] = None,
    breath: float = 1.0,
    rx: float = 6.35,
    ry: float = 2.35,
    cy: float = -1.85,
) -> float:
    """Draw the spectral hoop + beams onto an existing axes. Returns overall glow."""
    if mean_v is None:
        mean_v = float(np.mean(v))
    if mean_w is None:
        mean_w = float(np.mean(w))
    overall = overall_glow(mean_v, mean_w, gate) * (0.92 + 0.08 * breath)
    rx = rx * (0.96 + 0.04 * breath)
    ry = ry * (0.96 + 0.04 * breath)
    xs, ys, idx = _ellipse_points(rx, ry, cy)
    # ground glow inside the hoop
    ax.add_patch(
        Ellipse(
            (0, cy),
            2 * rx * 0.92,
            2 * ry * 0.92,
            facecolor="#1a1030",
            edgecolor="none",
            alpha=0.22 * overall,
            zorder=1,
        )
    )
    _sparkles(ax, rx, ry, cy, overall=overall, breath=breath, gate=gate)
    _beams(ax, xs, ys, idx, v, net, rx=rx, ry=ry, cy=cy, gate=gate, overall=overall, breath=breath)

    back = ys[:-1] >= cy
    for lw, la in _GLOW_LAYERS:
        colors = _segment_colors(idx, net, gate=gate, overall=overall, layer_alpha=la)
        pts = np.column_stack([xs, ys])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        # back rim first
        if np.any(back):
            lc = LineCollection(
                segs[back],
                colors=colors[back],
                linewidths=lw * breath,
                capstyle="round",
                joinstyle="round",
                zorder=2,
            )
            ax.add_collection(lc)
        if np.any(~back):
            lc = LineCollection(
                segs[~back],
                colors=colors[~back],
                linewidths=lw * breath,
                capstyle="round",
                joinstyle="round",
                zorder=8,
            )
            ax.add_collection(lc)

    # slice-center beads so the 10-fold geometry stays inspectable
    angs = _angles()
    for j, ang in enumerate(angs):
        px, py = rx * np.cos(ang), cy + ry * np.sin(ang)
        rgb = _hsv(slice_hue(j), 0.55, 1.0)
        a = float(np.clip(0.25 + 0.65 * slice_alpha(net[j]) * overall, 0.2, 0.9))
        ax.add_patch(Circle((px, py), 0.11, facecolor=rgb, edgecolor="white", linewidth=0.4, alpha=a, zorder=9))
    return overall


def _state_from_row(row: TimedScore, n: int) -> dict:
    created = row.post.created_at[:10] if row.post.created_at else ""
    return {
        "v": list(row.result.v),
        "w": list(row.result.w),
        "net": list(row.result.net),
        "mean_v": row.result.mean_v,
        "mean_w": row.result.mean_w,
        "gate": row.result.gate,
        "label": created,
        "n": n,
        "index": 0.0,
    }


def render_aura_frame(
    state: dict,
    *,
    save_path: PathLike,
    title: str = "Alignment aura",
    precession: float = 0.0,
    breath: float = 1.0,
    dpi: int = 120,
    tight: bool = True,
) -> Path:
    del precession  # hue stays locked to the unit-circle clock
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    v = state["v"]
    w = state["w"]
    net = state["net"]
    gate = state["gate"]
    refuse = gate == "REFUSE"
    fig, ax = plt.subplots(figsize=(9, 12), facecolor=_BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax.set_facecolor(_BG)
    ax.set_aspect("equal")
    ax.set_xlim(-10.6, 10.6)
    ax.set_ylim(-7.2, 9.8)
    ax.axis("off")

    ring_col = GATE_COLORS.get(gate, _GOLD)
    render_aura_ring(
        ax,
        v,
        w,
        net,
        gate=gate,
        mean_v=state["mean_v"],
        mean_w=state["mean_w"],
        breath=breath,
    )
    _silhouette(ax, refuse=refuse)

    rx, ry, cy = 6.35, 2.35, -1.85
    y_top_est = (1.15 + 5.8 * overall_glow(state["mean_v"], state["mean_w"], gate))
    for j, ang in enumerate(_angles()):
        s = np.sin(ang)
        lx = 1.40 * rx * np.cos(ang)
        if s > 0.55:
            ly = y_top_est + 0.62
        else:
            ly = cy + (2.05 if s > 0 else 1.48) * ry * s
        ax.text(
            lx,
            ly,
            f"{ROMAN[j]}  {ALIGNED_NAMES[j]}",
            ha="center",
            va="center",
            fontsize=7,
            color=_TEXT,
            alpha=0.80,
            zorder=10,
        )

    ax.text(0, 9.15, title, ha="center", fontsize=15, color=_TEXT, fontweight="bold")
    idx = state.get("index")
    n = state.get("n") or 1
    cursor = ""
    if idx is not None and n:
        cursor = f"   {int(idx) + 1}/{int(n)}"
    ax.text(
        0,
        8.40,
        f"{gate}   mean(v)={state['mean_v']:.2f}   mean(w)={state['mean_w']:.2f}   {state.get('label') or ''}{cursor}",
        ha="center",
        fontsize=9,
        color=ring_col,
    )
    ax.text(
        0,
        -6.85,
        "Spectral hoop  ·  hue = slice  ·  opacity = net  ·  glow = gate",
        ha="center",
        fontsize=8,
        color=_NEUTRAL,
    )
    fig.savefig(
        save_path,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight" if tight else None,
        pad_inches=0.25 if tight else 0.0,
    )
    plt.close(fig)
    return Path(save_path)


def render_aura_still(
    rows: List[TimedScore],
    *,
    save_path: PathLike = "outputs/aura_still.png",
    which: str = "mean",
    title: str = "Alignment aura",
    dpi: int = 140,
) -> Path:
    if not rows:
        raise ValueError("No scored posts for aura")
    if which == "last":
        state = _state_from_row(rows[-1], len(rows))
        state["index"] = float(len(rows) - 1)
    else:
        v = list(np.mean([r.result.v for r in rows], axis=0))
        w = list(np.mean([r.result.w for r in rows], axis=0))
        stats = trajectory_stats(rows)
        counts = stats.get("gate_counts") or {"CORRECT": 0}
        gate = max(counts, key=counts.get)
        state = {
            "v": v,
            "w": w,
            "net": [a + b for a, b in zip(v, w)],
            "mean_v": float(np.mean([r.result.mean_v for r in rows])),
            "mean_w": float(np.mean([r.result.mean_w for r in rows])),
            "gate": gate,
            "label": f"{str(stats.get('start') or '')[:10]} → {str(stats.get('end') or '')[:10]}",
            "n": len(rows),
            "index": None,
        }
    who = rows[0].post.username or "field"
    return render_aura_frame(state, save_path=save_path, title=f"{title}  ·  @{who}", dpi=dpi)


def render_aura_video(
    rows: List[TimedScore],
    *,
    save_path: PathLike = "outputs/aura.mp4",
    fps: int = 12,
    frames_per_step: int = 3,
    title: str = "Alignment aura",
    keep_frames: bool = False,
    loop: bool = True,
    dpi: int = 100,
    progress: Progress = None,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write aura video")
    if not rows:
        raise ValueError("No scored posts for aura")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frames = interpolate_rows(rows, frames_per_step=frames_per_step, loop=loop)
    who = rows[0].post.username or "field"
    tmp = Path(tempfile.mkdtemp(prefix="aura_frames_"))
    paths = []
    n = len(frames)
    for i, state in enumerate(frames):
        t = i / max(n - 1, 1)
        breath = 1.0 + 0.08 * np.sin(2.0 * np.pi * 2.0 * t) * (0.35 + 0.65 * max(0.0, state["mean_v"]) / 10.0)
        if state["gate"] in {"REFUSE", "STOP_FEEDING"}:
            breath *= 0.88
        p = tmp / f"frame_{i:04d}.png"
        render_aura_frame(
            state,
            save_path=p,
            title=f"{title}  ·  @{who}",
            breath=breath,
            dpi=dpi,
            tight=False,
        )
        paths.append(p)
        if progress is not None:
            progress(i, n)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(max(1, fps)),
        "-i",
        str(tmp / "frame_%04d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(save_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[-800:]}")
    if not keep_frames:
        shutil.rmtree(tmp, ignore_errors=True)
    return save_path
