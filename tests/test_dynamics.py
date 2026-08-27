from __future__ import annotations

from pathlib import Path

import numpy as np

from alignment import dynamics as dynamics_mod
from alignment.dynamics import (
    integrate_lorenz,
    integrate_motion,
    lorenz_equilibria,
    mean_alignment,
    motion_rhs,
    render_lorenz,
    render_motion_regimes,
)
from alignment.slices import HUMAN


def test_lorenz_equilibria_signs():
    origin, major, minor = lorenz_equilibria()
    assert np.allclose(origin, 0)
    assert major[0] > 0
    assert minor[0] < 0
    assert np.isclose(major[2], minor[2])


def test_integrate_lorenz_bounded():
    traj = integrate_lorenz(t_span=(0.0, 8.0), n_eval=400)
    assert traj.xyz.shape[0] == 3
    assert np.isfinite(traj.xyz).all()
    assert traj.xyz.max() < 60


def test_sigma_one_moves_toward_aligned():
    t, psi = integrate_motion(psi0=HUMAN, sigma_z=1.0, t_span=(0.0, 8.0), n_eval=200)
    assert mean_alignment(psi)[-1] > mean_alignment(psi)[0]
    assert mean_alignment(psi)[-1] > 5.0


def test_sigma_zero_adaptation_takes():
    t, psi = integrate_motion(psi0=HUMAN, sigma_z=0.0, t_span=(0.0, 8.0), n_eval=200)
    assert mean_alignment(psi)[-1] < 0.0


def test_motion_rhs_shape():
    dpsi = motion_rhs(np.zeros(10), sigma_z=1.0)
    assert dpsi.shape == (10,)


def test_two_machines_stay_separate_in_the_caption():
    doc = dynamics_mod.__doc__ or ""
    assert "not a reversal of reveal" in doc
    assert "Slice motion is a separate ODE" in doc
    src = Path(dynamics_mod.__file__).read_text()
    assert "Father" not in src
    assert "Holy Spirit" not in src
    assert "F–S–HS" not in src


def test_render_figures(tmp_path: Path):
    p1 = render_lorenz(tmp_path / "lorenz.png")
    p2 = render_motion_regimes(tmp_path / "motion.png")
    assert p1.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert p2.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
