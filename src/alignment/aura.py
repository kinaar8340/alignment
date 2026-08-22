"""Seven-ray aura: circular array of heatmaps + mean at the origin.

Seven human silhouettes sit on a Flower of Life, feet toward the origin,
heads outward. Each ray is one heavenly virtue (Humility … Diligence).
Color is a frequency heatmap of that slice's net: −10 is lowest visible
frequency (red), +10 is highest (violet), and 0 is no color.

The circle at the origin is the mean of the seven nets, mapped through
the same heatmap. An analog needle below the flower projects that
collective onto a single misaligned → aligned scale so direction of
travel is readable at a glance. No second scoring surface.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle, Wedge
from matplotlib.transforms import Affine2D

from .slices import ALIGNED_NAMES
from .timeline import TimedScore, trajectory_stats
from .torus import GATE_COLORS

PathLike = Union[str, Path]
Progress = Optional[Callable[[int, int], None]]

_BG = "#000000"
_TEXT = "#e6edf3"
_GOLD = "#d29922"
_NEUTRAL = "#8b949e"
_FOL = "#d8d8e0"

N_FIGURES = 7
# Heavenly virtues, clockwise from 12 o'clock. Love / Aligned / Neutral
# still feed the engine gate.
FIGURE_SLICES = (2, 3, 4, 5, 6, 7, 8)
FIGURE_NAMES = tuple(ALIGNED_NAMES[i] for i in FIGURE_SLICES)
# Visible spectrum: −10 → 700 nm (red, low freq), +10 → 420 nm (violet, high freq)
_NM_LOW = 700.0
_NM_HIGH = 420.0
GLOW_THRESHOLD = 1.0
_METER_CX = 0.0
_METER_CY = -8.95
_METER_R = 1.45


def _angles() -> np.ndarray:
    """Figure 0 at 12 o'clock, clockwise."""
    return np.pi / 2.0 - 2.0 * np.pi * np.arange(N_FIGURES) / N_FIGURES


def lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * t


def lerp_vec(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    return [lerp(x, y, t) for x, y in zip(a, b)]


def slice_alpha(net_i: float) -> float:
    """|net|/10. Zero net is colorless."""
    return float(np.clip(abs(float(net_i)) / 10.0, 0.0, 1.0))


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


def net_to_wavelength(net: float) -> float:
    """−10 → 700 nm (red), +10 → 420 nm (violet)."""
    t = float(np.clip((float(net) + 10.0) / 20.0, 0.0, 1.0))
    return _NM_LOW + t * (_NM_HIGH - _NM_LOW)


def wavelength_to_rgb(nm: float) -> Tuple[float, float, float]:
    """Approximate visible-spectrum RGB (Bruton piecewise)."""
    nm = float(np.clip(nm, 380.0, 780.0))
    if nm < 440.0:
        r, g, b = -(nm - 440.0) / 60.0, 0.0, 1.0
    elif nm < 490.0:
        r, g, b = 0.0, (nm - 440.0) / 50.0, 1.0
    elif nm < 510.0:
        r, g, b = 0.0, 1.0, -(nm - 510.0) / 20.0
    elif nm < 580.0:
        r, g, b = (nm - 510.0) / 70.0, 1.0, 0.0
    elif nm < 645.0:
        r, g, b = 1.0, -(nm - 645.0) / 65.0, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    if nm > 700.0:
        s = 0.3 + 0.7 * (780.0 - nm) / 80.0
    elif nm < 420.0:
        s = 0.3 + 0.7 * (nm - 380.0) / 40.0
    else:
        s = 1.0
    gamma = 0.8
    return ((r * s) ** gamma, (g * s) ** gamma, (b * s) ** gamma)


def net_heatmap(net: float) -> Tuple[Tuple[float, float, float], float]:
    """Frequency heatmap. Returns (rgb, alpha); alpha is 0 at net = 0."""
    n = float(np.clip(net, -10.0, 10.0))
    alpha = abs(n) / 10.0
    if alpha < 1e-9:
        return (0.0, 0.0, 0.0), 0.0
    return wavelength_to_rgb(net_to_wavelength(n)), alpha


def _ray_nets(net: Sequence[float]) -> List[float]:
    return [float(net[i]) for i in FIGURE_SLICES]


def mean_ray_net(net: Sequence[float]) -> float:
    """Aura score at the origin: mean of the seven figure nets."""
    rays = _ray_nets(net)
    return float(sum(rays) / float(len(rays)))


def glowing_fraction(net: Sequence[float], threshold: float = GLOW_THRESHOLD) -> float:
    """Share of the seven figures with net above the glow threshold."""
    rays = _ray_nets(net)
    return float(sum(1 for n in rays if n > threshold) / float(len(rays)))


def meter_score(
    net: Sequence[float],
    prev_mean: Optional[float] = None,
    *,
    glow_threshold: float = GLOW_THRESHOLD,
) -> dict:
    """Scalar in [−10, +10] for the analog needle.

    0.80 × mean(net) + 2.0 × (frac_high − frac_low), then a clipped
    rate-of-change lead so the needle shows direction of travel.
    """
    rays = _ray_nets(net)
    n = float(len(rays))
    avg = float(sum(rays) / n)
    frac_hi = sum(1.0 for x in rays if x > glow_threshold) / n
    frac_lo = sum(1.0 for x in rays if x < -glow_threshold) / n
    score = 0.80 * avg + 2.0 * (frac_hi - frac_lo)
    delta = 0.0
    if prev_mean is not None:
        delta = avg - float(prev_mean)
        score = score + float(np.clip(delta, -2.0, 2.0))
    return {
        "score": float(np.clip(score, -10.0, 10.0)),
        "mean": avg,
        "frac": frac_hi,
        "delta": delta,
    }


def meter_angle(score: float) -> float:
    """−10 → π (left), +10 → 0 (right), 0 → π/2 (up)."""
    t = float(np.clip((float(score) + 10.0) / 20.0, 0.0, 1.0))
    return float(np.pi * (1.0 - t))


def _meter_rgb(t: float) -> Tuple[float, float, float]:
    """t ∈ [0, 1]: deep red → amber → bright green."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        u = t * 2.0
        return (0.92, 0.14 + 0.52 * u, 0.10)
    u = (t - 0.5) * 2.0
    return (0.90 - 0.68 * u, 0.66 + 0.20 * u, 0.10 + 0.22 * u)


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


def _hex_centers(rings: int, radius: float) -> List[Tuple[float, float]]:
    pts = [(0.0, 0.0)]
    for r in range(1, rings + 1):
        x, y = r * radius, 0.0
        steps = [
            (-0.5 * radius, (np.sqrt(3) / 2.0) * radius),
            (-radius, 0.0),
            (-0.5 * radius, -(np.sqrt(3) / 2.0) * radius),
            (0.5 * radius, -(np.sqrt(3) / 2.0) * radius),
            (radius, 0.0),
            (0.5 * radius, (np.sqrt(3) / 2.0) * radius),
        ]
        for dx, dy in steps:
            for _ in range(r):
                pts.append((x, y))
                x += dx
                y += dy
    return pts


def _flower_of_life(ax, *, radius: float = 2.55, rings: int = 2) -> None:
    for cx, cy in _hex_centers(rings, radius):
        ax.add_patch(
            Circle(
                (cx, cy),
                radius,
                fill=False,
                edgecolor=_FOL,
                linewidth=0.85,
                alpha=0.40,
                zorder=1,
            )
        )


def _silhouette(ax, ang: float, dist: float, *, scale: float = 1.0) -> None:
    """Black figure, feet toward origin, head outward."""
    tr = Affine2D().scale(scale).rotate(ang - np.pi / 2.0).translate(
        dist * np.cos(ang), dist * np.sin(ang)
    ) + ax.transData
    fill = "#0b0b10"
    kw = dict(facecolor=fill, edgecolor="none", transform=tr, zorder=5)
    ax.add_patch(Ellipse((0.0, 0.12), 1.02, 1.85, **kw))
    ax.add_patch(Circle((0.0, 1.42), 0.40, **kw))
    ax.add_patch(Ellipse((-0.58, 0.22), 0.30, 1.38, angle=16, **kw))
    ax.add_patch(Ellipse((0.58, 0.22), 0.30, 1.38, angle=-16, **kw))
    ax.add_patch(Ellipse((-0.24, -1.32), 0.34, 1.28, **kw))
    ax.add_patch(Ellipse((0.24, -1.32), 0.34, 1.28, **kw))


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
    dist: float = 3.55,
) -> float:
    """Draw Flower of Life, seven heatmaps, and the mean at the origin."""
    if mean_v is None:
        mean_v = float(np.mean(v))
    if mean_w is None:
        mean_w = float(np.mean(w))
    overall = overall_glow(mean_v, mean_w, gate) * (0.94 + 0.06 * breath)
    rays = _ray_nets(net)
    avg = mean_ray_net(net)

    _flower_of_life(ax, radius=2.52, rings=2)

    angs = _angles()
    petal_w, petal_h = 2.22 * breath, 3.82 * breath
    for k, ang in enumerate(angs):
        rgb, alpha = net_heatmap(rays[k])
        cx, cy = dist * np.cos(ang), dist * np.sin(ang)
        angle = float(np.degrees(ang) - 90.0)
        if alpha > 0.02:
            for scale, al in ((1.20, 0.20), (1.08, 0.38), (1.0, 0.90)):
                ax.add_patch(
                    Ellipse(
                        (cx, cy),
                        petal_w * scale,
                        petal_h * scale,
                        angle=angle,
                        facecolor=rgb,
                        edgecolor="none",
                        alpha=float(np.clip(al * alpha * (0.85 + 0.15 * overall), 0.0, 0.95)),
                        zorder=3,
                    )
                )
        _silhouette(ax, ang, dist, scale=0.92 * (0.97 + 0.03 * breath))
        lx = (dist + 2.58) * np.cos(ang)
        ly = (dist + 2.58) * np.sin(ang)
        ax.text(
            lx,
            ly,
            FIGURE_NAMES[k],
            ha="center",
            va="center",
            fontsize=8,
            color=_TEXT,
            alpha=0.88,
            zorder=8,
        )

    rgb, alpha = net_heatmap(avg)
    core_r = 1.05 + 0.28 * alpha
    ax.add_patch(Circle((0.0, 0.0), 1.22, fill=False, edgecolor=_FOL, linewidth=1.0, alpha=0.35, zorder=6))
    if alpha > 0.02:
        ax.add_patch(Circle((0.0, 0.0), core_r * 1.12, facecolor=rgb, edgecolor="none", alpha=0.22 * alpha, zorder=7))
        ax.add_patch(Circle((0.0, 0.0), core_r * 0.78, facecolor=rgb, edgecolor="none", alpha=0.55 * alpha, zorder=8))
        ax.add_patch(Circle((0.0, 0.0), core_r * 0.52, facecolor=rgb, edgecolor="white", linewidth=0.6, alpha=float(np.clip(0.55 + 0.45 * alpha, 0.55, 1.0)), zorder=9))
    rim = GATE_COLORS.get(gate, _GOLD)
    ax.add_patch(Circle((0.0, 0.0), core_r * 0.52, fill=False, edgecolor=rim, linewidth=2.0, alpha=0.9, zorder=10))
    return overall


def _heatmap_legend(ax, *, y: float = -10.22) -> None:
    x0, width, height = -2.4, 4.8, 0.16
    n = 64
    step = width / n
    for i in range(n):
        net = -10.0 + 20.0 * (i + 0.5) / n
        rgb, alpha = net_heatmap(net)
        ax.add_patch(
            Rectangle(
                (x0 + i * step, y),
                step,
                height,
                facecolor=rgb,
                edgecolor="none",
                alpha=max(alpha, 0.06),
                zorder=8,
            )
        )
    ax.text(x0, y - 0.26, "−10", ha="center", va="top", fontsize=6.5, color=_NEUTRAL, zorder=8)
    ax.text(0.0, y - 0.26, "0", ha="center", va="top", fontsize=6.5, color=_NEUTRAL, zorder=8)
    ax.text(x0 + width, y - 0.26, "+10", ha="center", va="top", fontsize=6.5, color=_NEUTRAL, zorder=8)


def _draw_meter(
    ax,
    net: Sequence[float],
    *,
    prev_net: Optional[Sequence[float]] = None,
    cx: float = _METER_CX,
    cy: float = _METER_CY,
    radius: float = _METER_R,
) -> dict:
    """Classic semicircle needle gauge under the flower."""
    prev_mean = mean_ray_net(prev_net) if prev_net is not None else None
    info = meter_score(net, prev_mean)
    ax.add_patch(
        Wedge(
            (cx, cy),
            radius * 1.22,
            0,
            180,
            facecolor="#0c0e12",
            edgecolor="#4a5160",
            linewidth=1.3,
            zorder=12,
        )
    )
    ax.plot(
        [cx - radius * 1.18, cx + radius * 1.18],
        [cy, cy],
        color="#4a5160",
        linewidth=1.1,
        zorder=12,
        solid_capstyle="round",
    )
    nseg = 52
    for i in range(nseg):
        t0, t1 = i / nseg, (i + 1) / nseg
        a0 = np.degrees(np.pi * (1.0 - t0))
        a1 = np.degrees(np.pi * (1.0 - t1))
        ax.add_patch(
            Wedge(
                (cx, cy),
                radius,
                min(a0, a1),
                max(a0, a1),
                width=radius * 0.16,
                facecolor=_meter_rgb(0.5 * (t0 + t1)),
                edgecolor="none",
                zorder=13,
            )
        )
    for val in (-10, -5, 0, 5, 10):
        ang = meter_angle(val)
        inner, outer = radius * 0.78, radius * 1.02
        ax.plot(
            [cx + inner * np.cos(ang), cx + outer * np.cos(ang)],
            [cy + inner * np.sin(ang), cy + outer * np.sin(ang)],
            color="#e6edf3",
            linewidth=1.15 if val in (-10, 0, 10) else 0.7,
            zorder=14,
            solid_capstyle="round",
        )
    ax.text(cx - radius - 0.12, cy - 0.12, "misaligned", ha="right", va="top", fontsize=6.5, color="#f85149", zorder=15)
    ax.text(cx, cy + radius + 0.18, "transitional", ha="center", va="bottom", fontsize=6.5, color="#d29922", zorder=15)
    ax.text(cx + radius + 0.12, cy - 0.12, "aligned", ha="left", va="top", fontsize=6.5, color="#3fb950", zorder=15)

    if prev_net is not None:
        ghost = meter_score(prev_net, None)
        pang = meter_angle(ghost["score"])
        ax.plot(
            [cx, cx + radius * 0.86 * np.cos(pang)],
            [cy, cy + radius * 0.86 * np.sin(pang)],
            color="#8b949e",
            linewidth=1.4,
            alpha=0.32,
            zorder=15,
            solid_capstyle="round",
        )

    ang = meter_angle(info["score"])
    tip = (cx + radius * 0.90 * np.cos(ang), cy + radius * 0.90 * np.sin(ang))
    left = (cx + 0.09 * np.cos(ang + np.pi / 2), cy + 0.09 * np.sin(ang + np.pi / 2))
    right = (cx + 0.09 * np.cos(ang - np.pi / 2), cy + 0.09 * np.sin(ang - np.pi / 2))
    tail = (cx - 0.22 * np.cos(ang), cy - 0.22 * np.sin(ang))
    ax.add_patch(Polygon([left, tip, right, tail], closed=True, facecolor="#f0f3f6", edgecolor="#111111", linewidth=0.45, zorder=16))
    ax.add_patch(Circle((cx, cy), 0.13, facecolor="#f0f3f6", edgecolor="#111111", linewidth=0.6, zorder=17))
    return info


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
    prev_net: Optional[Sequence[float]] = None,
) -> Path:
    del precession
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    v, w, net, gate = state["v"], state["w"], state["net"], state["gate"]
    fig, ax = plt.subplots(figsize=(10.0, 12.6), facecolor=_BG)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.05)
    ax.set_facecolor(_BG)
    ax.set_aspect("equal")
    ax.set_xlim(-7.9, 7.9)
    ax.set_ylim(-10.85, 8.4)
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
    _draw_meter(ax, net, prev_net=prev_net)
    _heatmap_legend(ax)

    ax.text(0, 7.85, title, ha="center", fontsize=15, color=_TEXT, fontweight="bold")
    idx = state.get("index")
    n = state.get("n") or 1
    cursor = ""
    if idx is not None and n:
        cursor = f"   {int(idx) + 1}/{int(n)}"
    ax.text(
        0,
        7.22,
        f"{gate}   mean(v)={state['mean_v']:.2f}   mean(net₇)={mean_ray_net(net):.2f}   {state.get('label') or ''}{cursor}",
        ha="center",
        fontsize=9,
        color=ring_col,
    )
    ax.text(
        0,
        -10.58,
        "Needle: misaligned → aligned  ·  figures: frequency heatmap  ·  origin = mean(net₇)",
        ha="center",
        fontsize=7.5,
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
    prev_net: Optional[List[float]] = None
    for i, state in enumerate(frames):
        t = i / max(n - 1, 1)
        breath = 1.0 + 0.06 * np.sin(2.0 * np.pi * 2.0 * t) * (0.35 + 0.65 * max(0.0, state["mean_v"]) / 10.0)
        if state["gate"] in {"REFUSE", "STOP_FEEDING"}:
            breath *= 0.90
        p = tmp / f"frame_{i:04d}.png"
        render_aura_frame(
            state,
            save_path=p,
            title=f"{title}  ·  @{who}",
            breath=breath,
            dpi=dpi,
            tight=False,
            prev_net=prev_net,
        )
        prev_net = list(state["net"])
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
