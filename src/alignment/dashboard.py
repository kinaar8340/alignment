"""Dark-themed alignment dashboard: paired unit circles, bars, fruits, gate."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyBboxPatch

from .engine import ScoreResult
from .slices import ALIGNED_NAMES, ROMAN

PathLike = Union[str, Path]

_BG = "#0d1117"
_PANEL = "#161b22"
_TEXT = "#e6edf3"
_ALIGNED = "#3fb950"
_MISALIGNED = "#f85149"
_NEUTRAL = "#8b949e"
_ACCENT = "#58a6ff"
_GOLD = "#d29922"
_STOP = "#f0883e"

GATE_COLORS = {
    "FLOW": _ALIGNED,
    "CORRECT": _GOLD,
    "STOP_FEEDING": _STOP,
    "REFUSE": _MISALIGNED,
}


def _wrap(text: str, width: int = 52) -> str:
    text = (text or "").strip()
    if not text:
        return "—"
    return "\n".join(textwrap.wrap(text, width=width))


def _angles() -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)


def _close(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.concatenate([arr, arr[:1]])


def _draw_radar(ax, magnitudes, *, clockwise: bool, color: str, title: str, labels, values_for_label):
    ax.set_facecolor(_PANEL)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1 if clockwise else 1)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    sign = "" if clockwise else "-"
    ax.set_yticklabels([f"{sign}{n}" for n in (2, 4, 6, 8, 10)], color=_NEUTRAL, fontsize=8)
    ax.set_xticks(_angles())
    ax.set_xticklabels([])
    ax.tick_params(colors=_TEXT)
    ax.spines["polar"].set_color(_NEUTRAL)
    ax.yaxis.grid(True, color=_NEUTRAL, alpha=0.35, linestyle="--")
    ax.xaxis.grid(True, color=_NEUTRAL, alpha=0.35)
    ax.set_title(title, color=color, fontsize=10, pad=46, fontweight="bold")

    angs = _close(_angles())
    mag = _close(magnitudes)
    ax.plot(angs, mag, "o-", color=color, linewidth=2.4, markersize=7)
    ax.fill(angs, mag, color=color, alpha=0.22)
    ax.plot(angs, np.full_like(angs, 10.0), "--", color=_GOLD if clockwise else "#a371f7", alpha=0.45, linewidth=1.0)

    for ang, name, val in zip(_angles(), labels, values_for_label):
        ax.text(
            ang,
            12.4,
            f"{name}\n{val:.1f}",
            ha="center",
            va="center",
            fontsize=8,
            color=_TEXT,
            clip_on=False,
        )
    ax.plot(0, 0, marker="X", color=_GOLD, markersize=16, markeredgewidth=2.4, linestyle="None")


def create_dashboard(
    result: ScoreResult,
    title: str = "Alignment Scoring Dashboard",
    save_path: PathLike = "alignment_dashboard.png",
    dpi: int = 150,
) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12), facecolor=_BG)
    gs = gridspec.GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[0.85, 2.15, 1.7],
        width_ratios=[1, 1, 1, 1],
        hspace=0.46,
        wspace=0.28,
    )

    ax_sum = fig.add_subplot(gs[0, :])
    ax_sum.set_facecolor(_PANEL)
    ax_sum.set_xlim(0, 10)
    ax_sum.set_ylim(0, 3)
    ax_sum.axis("off")

    gate_col = GATE_COLORS.get(result.gate, _ACCENT)
    ax_sum.add_patch(
        FancyBboxPatch(
            (0.18, 1.45),
            2.15,
            1.25,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            facecolor=gate_col,
            edgecolor="white",
            linewidth=1.6,
            alpha=0.95,
        )
    )
    ax_sum.text(
        1.25,
        2.08,
        result.gate,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="white",
    )

    ax_sum.text(2.6, 2.45, f"mean(v) = {result.mean_v:.2f}", fontsize=13, color=_ALIGNED, fontweight="bold")
    ax_sum.text(2.6, 1.85, f"min(v)  = {result.min_v:.2f}", fontsize=12, color=_ALIGNED)
    ax_sum.text(4.7, 2.45, f"mean(w) = {result.mean_w:.2f}", fontsize=13, color=_MISALIGNED, fontweight="bold")
    ax_sum.text(4.7, 1.85, f"min(w)  = {result.min_w:.2f}", fontsize=12, color=_MISALIGNED)

    flags = [
        ("Observer", result.observer_present, True),
        ("Asks first", result.asks_first, True),
        ("Taking w/o asking", result.taking_without_asking, False),
        ("Exploit class", result.exploit_class, False),
        ("Doom loop", result.doom_loop, False),
    ]
    for i, (name, val, good_when_true) in enumerate(flags):
        if good_when_true:
            col = _ALIGNED if val else _MISALIGNED
        else:
            col = _MISALIGNED if val else _ALIGNED
        mark = "●" if val else "○"
        ax_sum.text(
            6.55 + (i % 3) * 1.18,
            2.58 - (i // 3) * 0.78,
            f"{mark} {name}",
            fontsize=9,
            color=col,
        )

    ax_sum.text(0.22, 0.72, "ACTION:", fontsize=11, color=_GOLD, fontweight="bold")
    ax_sum.text(0.22, 0.18, _wrap(result.action, 110), fontsize=10, color=_TEXT)

    fig.suptitle(title, fontsize=20, fontweight="bold", color=_TEXT, y=0.985)

    ax_pos = fig.add_subplot(gs[1, 0:2], polar=True)
    _draw_radar(
        ax_pos,
        result.v,
        clockwise=True,
        color=_ALIGNED,
        title="POSITIVE UNIT CIRCLE  ·  clockwise from X",
        labels=ROMAN,
        values_for_label=result.v,
    )

    ax_neg = fig.add_subplot(gs[1, 2:4], polar=True)
    _draw_radar(
        ax_neg,
        [abs(x) for x in result.w],
        clockwise=False,
        color=_MISALIGNED,
        title="NEGATIVE UNIT CIRCLE  ·  counterclockwise from X",
        labels=ROMAN,
        values_for_label=result.w,
    )

    ax_bar = fig.add_subplot(gs[2, 0:2])
    ax_bar.set_facecolor(_PANEL)
    x = np.arange(10)
    width = 0.36
    bars_v = ax_bar.bar(
        x - width / 2,
        result.v,
        width,
        label="Aligned (v)",
        color=_ALIGNED,
        edgecolor="white",
        linewidth=0.4,
    )
    bars_w = ax_bar.bar(
        x + width / 2,
        [abs(w) for w in result.w],
        width,
        label="|Misaligned| (w)",
        color=_MISALIGNED,
        edgecolor="white",
        linewidth=0.4,
    )
    ax_bar.set_ylabel("Magnitude", color=_TEXT)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [f"{r}\n{a}" for r, a in zip(ROMAN, ALIGNED_NAMES)],
        color=_TEXT,
        fontsize=7,
    )
    ax_bar.set_ylim(0, 11.0)
    ax_bar.tick_params(colors=_TEXT)
    ax_bar.legend(facecolor=_PANEL, edgecolor=_NEUTRAL, labelcolor=_TEXT)
    ax_bar.set_title("Slice Magnitudes Comparison", color=_TEXT, fontsize=12)
    ax_bar.yaxis.grid(True, color=_NEUTRAL, alpha=0.25)
    ax_bar.set_axisbelow(True)
    for spine in ax_bar.spines.values():
        spine.set_color(_NEUTRAL)
    for bar in bars_v:
        ax_bar.annotate(
            f"{bar.get_height():.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=_ALIGNED,
        )
    for bar, orig in zip(bars_w, result.w):
        ax_bar.annotate(
            f"{orig:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=_MISALIGNED,
        )

    ax_info = fig.add_subplot(gs[2, 2:4])
    ax_info.set_facecolor(_PANEL)
    ax_info.set_xlim(0, 10)
    ax_info.set_ylim(0, 10)
    ax_info.axis("off")
    ax_info.text(0.35, 9.35, "FRUITS TEST", fontsize=12, color=_GOLD, fontweight="bold")
    ax_info.text(0.35, 8.95, _wrap(result.fruits, 54), fontsize=10, color=_TEXT, va="top")
    ax_info.text(0.35, 5.55, "RATIONALE", fontsize=12, color=_GOLD, fontweight="bold")
    ax_info.text(0.35, 5.15, _wrap(result.rationale, 54), fontsize=9, color=_NEUTRAL, va="top")
    ax_info.text(
        0.35,
        1.15,
        "Open-loop primacy (O)  |  Wisdom-gated σ(Z)  |  Observer at X",
        fontsize=9,
        color=_ACCENT,
    )
    if result.dominant_misaligned_slices():
        named = ", ".join(
            f"{n.split(' ', 1)[-1]} ({val:.1f})" for n, val in result.dominant_misaligned_slices()
        )
        ax_info.text(0.35, 0.35, f"Deepest w: {named}", fontsize=8, color=_MISALIGNED)

    fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    return save_path
