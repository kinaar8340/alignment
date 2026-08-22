"""Stack unit-circle frames along time into a torus, plus 2D trend charts."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .slices import ALIGNED_NAMES, N_SLICES, ROMAN
from .timeline import TimedScore, trajectory_stats

PathLike = Union[str, Path]

_BG = "#0d1117"
_PANEL = "#161b22"
_TEXT = "#e6edf3"
_ALIGNED = "#3fb950"
_MISALIGNED = "#f85149"
_NEUTRAL = "#8b949e"
_GOLD = "#d29922"
_ACCENT = "#58a6ff"

GATE_COLORS = {
    "FLOW": _ALIGNED,
    "CORRECT": _GOLD,
    "STOP_FEEDING": "#f0883e",
    "REFUSE": _MISALIGNED,
}


def _net_color(net: float) -> tuple:
    t = (float(net) + 10.0) / 20.0
    t = min(1.0, max(0.0, t))
    # red → gold → green
    if t < 0.5:
        u = t * 2.0
        return (0.97, 0.32 + 0.28 * u, 0.29 + 0.25 * u, 0.95)
    u = (t - 0.5) * 2.0
    return (0.25 + 0.0 * (1 - u), 0.55 + 0.20 * u, 0.25 + 0.07 * u, 0.95)


def torus_frame(net: Sequence[float], theta: float, *, R: float = 3.2, r0: float = 0.55, rscale: float = 0.07):
    """One unit-circle frame at toroidal angle theta. Returns (xyz n×3, colors)."""
    pts = []
    cols = []
    n = len(net)
    for j, val in enumerate(list(net) + [net[0]]):
        phi = 2.0 * np.pi * (j % n) / n
        mag = abs(float(val))
        r = r0 + rscale * mag
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        pts.append((x, y, z))
        cols.append(_net_color(net[j % n]))
    return np.asarray(pts), cols


def render_torus_dashboard(
    rows: List[TimedScore],
    *,
    title: str = "Alignment torus — stacked unit-circle frames",
    save_path: PathLike = "outputs/torus_dashboard.png",
    max_rings: int = 72,
    dpi: int = 140,
) -> Path:
    if not rows:
        raise ValueError("No scored posts to visualize")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    stats = trajectory_stats(rows)
    n = len(rows)
    idx = np.linspace(0, n - 1, num=min(n, max_rings), dtype=int)
    rings = [rows[i] for i in idx]

    fig = plt.figure(figsize=(16.5, 11), facecolor=_BG)
    gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1.15, 1], height_ratios=[0.18, 1.15, 1.0], hspace=0.38, wspace=0.22)

    ax_head = fig.add_subplot(gs[0, :])
    ax_head.set_facecolor(_PANEL)
    ax_head.set_xlim(0, 10)
    ax_head.set_ylim(0, 3)
    ax_head.axis("off")
    who = stats.get("username") or "user"
    ax_head.text(0.25, 2.15, title, fontsize=16, color=_TEXT, fontweight="bold")
    ax_head.text(
        0.25,
        0.85,
        f"@{who}   n={stats['n']}   posts={stats['posts']}   replies={stats['replies']}   "
        f"avg mean(v)={stats['avg_mean_v']:.2f}   avg net={stats['avg_net']:.2f}   "
        f"{stats['start'][:10]} → {stats['end'][:10]}",
        fontsize=10,
        color=_NEUTRAL,
    )
    gates = stats.get("gate_counts") or {}
    gx = 0.25
    for name, count in sorted(gates.items()):
        ax_head.text(gx, 0.15, f"{name} {count}", fontsize=10, color=GATE_COLORS.get(name, _TEXT), fontweight="bold")
        gx += 1.6

    ax3 = fig.add_subplot(gs[1, 0], projection="3d")
    ax3.set_facecolor(_BG)
    ax3.xaxis.pane.fill = False
    ax3.yaxis.pane.fill = False
    ax3.zaxis.pane.fill = False
    ax3.tick_params(colors=_NEUTRAL, labelsize=7)
    ax3.set_title("Unit circles stacked on a torus (time → θ)", color=_ACCENT, fontsize=11, pad=8)

    # faint torus core
    R, r_core = 3.2, 0.55
    u = np.linspace(0, 2 * np.pi, 90)
    v = np.linspace(0, 2 * np.pi, 40)
    u, v = np.meshgrid(u, v)
    cx = (R + r_core * np.cos(v)) * np.cos(u)
    cy = (R + r_core * np.cos(v)) * np.sin(u)
    cz = r_core * np.sin(v)
    ax3.plot_wireframe(cx, cy, cz, color="#30363d", linewidth=0.3, alpha=0.45)

    for k, row in enumerate(rings):
        theta = 2.0 * np.pi * k / max(len(rings), 1)
        xyz, cols = torus_frame(row.result.net, theta)
        ax3.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=GATE_COLORS.get(row.result.gate, _GOLD), linewidth=1.4, alpha=0.85)
        ax3.scatter(xyz[:-1, 0], xyz[:-1, 1], xyz[:-1, 2], c=cols[:-1], s=8, linewidths=0, depthshade=False)

    # ribbons: each slice over time
    for j in range(N_SLICES):
        ribbon = []
        for k, row in enumerate(rings):
            theta = 2.0 * np.pi * k / max(len(rings), 1)
            phi = 2.0 * np.pi * j / N_SLICES
            mag = abs(row.result.net[j])
            r = 0.55 + 0.07 * mag
            ribbon.append(
                (
                    (R + r * np.cos(phi)) * np.cos(theta),
                    (R + r * np.cos(phi)) * np.sin(theta),
                    r * np.sin(phi),
                )
            )
        ribbon = np.asarray(ribbon)
        segs = np.stack([ribbon[:-1], ribbon[1:]], axis=1)
        colors = [_net_color(rings[k].result.net[j]) for k in range(len(rings) - 1)]
        ax3.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=1.6, alpha=0.7))

    ax3.set_box_aspect((1, 1, 0.55))
    ax3.view_init(elev=22, azim=38)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])

    ax_tr = fig.add_subplot(gs[1, 1])
    ax_tr.set_facecolor(_PANEL)
    times = np.arange(n)
    mean_v = [r.result.mean_v for r in rows]
    mean_w = [r.result.mean_w for r in rows]
    net = [sum(r.result.net) / 10.0 for r in rows]
    ax_tr.plot(times, mean_v, color=_ALIGNED, linewidth=2.0, label="mean(v) aligned")
    ax_tr.plot(times, [-w for w in mean_w], color=_MISALIGNED, linewidth=2.0, label="|mean(w)| misaligned")
    ax_tr.plot(times, net, color=_GOLD, linewidth=1.6, linestyle="--", label="mean(net)")
    ax_tr.axhline(0, color=_NEUTRAL, linewidth=0.6, alpha=0.5)
    ax_tr.set_title("Trend toward aligned / misaligned", color=_TEXT, fontsize=11)
    ax_tr.set_ylabel("score", color=_TEXT)
    ax_tr.tick_params(colors=_NEUTRAL)
    ax_tr.legend(facecolor=_PANEL, edgecolor=_NEUTRAL, labelcolor=_TEXT, fontsize=8)
    for spine in ax_tr.spines.values():
        spine.set_color(_NEUTRAL)
    ax_tr.yaxis.grid(True, color=_NEUTRAL, alpha=0.2)
    ax_tr.set_axisbelow(True)

    ax_heat = fig.add_subplot(gs[2, 0])
    ax_heat.set_facecolor(_PANEL)
    heat = np.array([r.result.net for r in rows]).T  # 10 x n
    im = ax_heat.imshow(
        heat,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-10,
        vmax=10,
        interpolation="nearest",
        origin="upper",
    )
    ax_heat.set_yticks(range(N_SLICES))
    ax_heat.set_yticklabels([f"{rom} {name}" for rom, name in zip(ROMAN, ALIGNED_NAMES)], color=_TEXT, fontsize=8)
    ax_heat.set_xlabel("post index (time →)", color=_TEXT)
    ax_heat.set_title("Slice net over time  (−10 misaligned → +10 aligned)", color=_TEXT, fontsize=11)
    ax_heat.tick_params(colors=_NEUTRAL)
    cb = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=_NEUTRAL)
    cb.outline.set_edgecolor(_NEUTRAL)

    ax_g = fig.add_subplot(gs[2, 1])
    ax_g.set_facecolor(_PANEL)
    for i, row in enumerate(rows):
        ax_g.scatter(i, 0.5, color=GATE_COLORS.get(row.result.gate, _TEXT), s=28, zorder=3)
    ax_g.set_ylim(0, 1)
    ax_g.set_yticks([])
    ax_g.set_title("Gate along the year", color=_TEXT, fontsize=11)
    ax_g.set_xlabel("post index (time →)", color=_TEXT)
    ax_g.tick_params(colors=_NEUTRAL)
    for spine in ax_g.spines.values():
        spine.set_color(_NEUTRAL)
    labels = "   ".join(f"{g}={c}" for g, c in sorted(gates.items()))
    ax_g.text(0.02, 0.82, labels, transform=ax_g.transAxes, color=_NEUTRAL, fontsize=9)

    fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return save_path
