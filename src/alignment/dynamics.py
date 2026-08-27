"""Two different machines. They are not one argument.

Machine 1 — Lorenz (1963), ``render_lorenz``:
  ẋ = σ(y − x),  ẏ = x(ρ − z) − y,  ż = xy − βz
  textbook chaos: σ = 10, ρ = 28, β = 8/3.
  Equilibria C± are convection fixed points, not a measured road.
  Lorenz σ is a fluid parameter. It is not alignment σ(Z).

Machine 2 — slice motion, ``render_motion_regimes``:
  ∂Ψ/∂t = O[Ψ] + σ(Z)·C[Ψ, E] + (1−σ(Z))·A[Ψ]
  linear pull toward constitutional vectors in slices.py.
  That ODE does not live on the butterfly unless a caption forces it.

Lorenz plots are visualizers of sensitivity and two basins. They are not
a measured road, not a proof of the poles, and not a reversal of reveal
A/B/C. Slice motion is a separate ODE toward constitutional vectors.
Coloring +x green and −x red is paint; swapping the map swaps the story.
obtained remains false: still moving is the picture, not north.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import solve_ivp

from .slices import HUMAN, JESUS, N_SLICES, SATAN

PathLike = Union[str, Path]

LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0


def lorenz_rhs(t: float, state: Sequence[float], sigma: float = LORENZ_SIGMA, rho: float = LORENZ_RHO, beta: float = LORENZ_BETA):
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z,
    ]


def lorenz_equilibria(rho: float = LORENZ_RHO, beta: float = LORENZ_BETA) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (origin, C+, C−) of the convection model. Names are not theology."""
    if rho <= 1:
        zero = np.zeros(3)
        return zero, zero, zero
    r = np.sqrt(beta * (rho - 1.0))
    major = np.array([r, r, rho - 1.0])
    minor = np.array([-r, -r, rho - 1.0])
    return np.zeros(3), major, minor


@dataclass
class LorenzTrajectory:
    t: np.ndarray
    xyz: np.ndarray  # shape (3, N)

    @property
    def lobe_sign(self) -> np.ndarray:
        return np.sign(self.xyz[0])


def integrate_lorenz(
    y0: Sequence[float] = (0.0, 1.0, 1.05),
    t_span: Tuple[float, float] = (0.0, 40.0),
    n_eval: int = 8000,
    **params,
) -> LorenzTrajectory:
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(
        lambda t, y: lorenz_rhs(t, y, **params),
        t_span,
        y0,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return LorenzTrajectory(t=sol.t, xyz=sol.y)


def motion_rhs(
    psi: np.ndarray,
    sigma_z: float,
    k_o: float = 0.45,
    k_c: float = 0.85,
    k_adapt: float = 0.70,
) -> np.ndarray:
    """dΨ/dt = O[Ψ] + σ(Z)·C[Ψ, E] + (1−σ(Z))·A[Ψ].

    O: unconditional feed-forward toward the fully aligned reference.
    C: wisdom-gated correction toward the same reference (consequences).
    A: adaptation-without-asking toward the fully misaligned reference.
    """
    jesus = np.asarray(JESUS, dtype=float)
    satan = np.asarray(SATAN, dtype=float)
    o_term = k_o * (jesus - psi) / 10.0
    c_term = k_c * (jesus - psi) / 10.0
    a_term = k_adapt * (satan - psi) / 10.0
    sigma_z = float(np.clip(sigma_z, 0.0, 1.0))
    return o_term + sigma_z * c_term + (1.0 - sigma_z) * a_term


def integrate_motion(
    psi0: Optional[Sequence[float]] = None,
    sigma_z: float = 1.0,
    t_span: Tuple[float, float] = (0.0, 12.0),
    n_eval: int = 400,
    **params,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the 10-slice motion equation. Returns (t, psi) with psi shape (10, N)."""
    y0 = np.asarray(HUMAN if psi0 is None else psi0, dtype=float)
    if y0.shape != (N_SLICES,):
        raise ValueError(f"psi0 must have shape ({N_SLICES},)")
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)

    def rhs(t, y):
        return motion_rhs(y, sigma_z, **params)

    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-8)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y


def mean_alignment(psi: np.ndarray) -> np.ndarray:
    """Mean of the 10-vector over time. psi shape (10, N)."""
    return psi.mean(axis=0)


def render_lorenz(
    save_path: PathLike = "lorenz_attractor.png",
    seeds: Optional[Iterable[Sequence[float]]] = None,
    dpi: int = 150,
) -> Path:
    """Butterfly plot. Visual only: two basins and sensitivity. Not a road."""
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = (
            (0.0, 1.0, 1.05),
            (0.1, -0.2, 0.3),
            (-0.05, 0.4, 0.2),
        )

    origin, major, minor = lorenz_equilibria()
    fig = plt.figure(figsize=(11, 8), facecolor="#0d1117")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#30363d")
    ax.yaxis.pane.set_edgecolor("#30363d")
    ax.zaxis.pane.set_edgecolor("#30363d")

    for seed in seeds:
        traj = integrate_lorenz(y0=seed)
        x, y, z = traj.xyz
        # Color by lobe. The map is paint; it is not a Trinity.
        lobe = np.where(x >= 0, 1, 0)
        for mask, c, alpha in (
            (lobe == 1, "#3fb950", 0.75),
            (lobe == 0, "#f85149", 0.75),
        ):
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            # split contiguous runs so jumps across the fold are not drawn
            splits = np.where(np.diff(idx) > 1)[0] + 1
            for run in np.split(idx, splits):
                if run.size < 2:
                    continue
                ax.plot(x[run], y[run], z[run], color=c, linewidth=0.7, alpha=alpha)

    ax.scatter(*major, color="#3fb950", s=80, label="C+  (x>0 equilibrium)", depthshade=False)
    ax.scatter(*minor, color="#f85149", s=80, label="C−  (x<0 equilibrium)", depthshade=False)
    ax.scatter(*origin, color="#d29922", s=50, marker="X", label="origin (unstable at these params)", depthshade=False)

    ax.set_xlabel("x", color="#e6edf3")
    ax.set_ylabel("y", color="#e6edf3")
    ax.set_zlabel("z", color="#e6edf3")
    ax.set_title("Lorenz (1963)  ·  visualizer, not a measured road", color="#e6edf3", pad=12)
    ax.legend(facecolor="#161b22", edgecolor="#8b949e", labelcolor="#e6edf3", loc="upper left")
    fig.text(
        0.5,
        0.02,
        "Two basins, sensitive to a nudge, origin unstable. Analogies only. "
        "Does not reverse reveal A/B/C. Lorenz σ ≠ alignment σ(Z).",
        ha="center",
        color="#8b949e",
        fontsize=8,
    )
    fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)


def render_motion_regimes(
    save_path: PathLike = "motion_regimes.png",
    psi0: Optional[Sequence[float]] = None,
    dpi: int = 150,
) -> Path:
    """Mean alignment over time for wisdom-gate σ(Z). Not Lorenz. Not a discovered pole."""
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start = np.asarray(psi0 if psi0 is not None else [2.0, 1.0, -1.0, 0.0, 3.0, -2.0, 1.5, -0.5, 0.0, 1.0], dtype=float)
    regimes = (
        (1.0, "#3fb950", "σ(Z) = 1  correction on  → constitutional pole"),
        (0.5, "#d29922", "σ(Z) = 0.5  mixed / wisdom-gated"),
        (0.0, "#f85149", "σ(Z) = 0  adaptation walks the other pole"),
    )

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    for sigma_z, color, label in regimes:
        t, psi = integrate_motion(psi0=start, sigma_z=sigma_z)
        ax.plot(t, mean_alignment(psi), color=color, linewidth=2.2, label=label)

    ax.axhline(10.0, color="#3fb950", linestyle="--", alpha=0.4, linewidth=1, label="constitutional pole [10…10]")
    ax.axhline(0.0, color="#d29922", linestyle=":", alpha=0.5, linewidth=1, label="human origin (not a third road)")
    ax.axhline(-10.0, color="#f85149", linestyle="--", alpha=0.4, linewidth=1, label="constitutional pole [−10…−10]")
    ax.set_xlabel("t", color="#e6edf3")
    ax.set_ylabel("mean(Ψ)", color="#e6edf3")
    ax.set_title("Slice motion (Machine 2)  ·  gated gradient, not Lorenz", color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#8b949e")
    ax.legend(facecolor="#161b22", edgecolor="#8b949e", labelcolor="#e6edf3", fontsize=8)
    ax.set_ylim(-11, 11)
    ax.yaxis.grid(True, color="#8b949e", alpha=0.2)
    fig.text(
        0.5,
        0.01,
        "Walks toward poles declared in slices.py. Does not discover them. "
        "σ(Z) is a wisdom switch, not Lorenz’s Prandtl σ.",
        ha="center",
        color="#8b949e",
        fontsize=8,
    )
    fig.savefig(save_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)
