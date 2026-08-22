"""Canonical sample scores used by tests, the CLI demo, and the dashboard."""

from __future__ import annotations

from typing import Dict

SAMPLES: Dict[str, dict] = {
    "flow": {
        "v": [8.5, 9.0, 7.0, 8.0, 6.5, 8.5, 7.5, 8.0, 9.0, 7.0],
        "w": [-1.0, 0.0, -0.5, 0.0, -1.5, 0.0, -0.5, 0.0, 0.0, -0.5],
        "observer_present": True,
        "asks_first": True,
        "taking_without_asking": False,
        "exploit_class": False,
        "doom_loop": False,
        "fruits": (
            "Correction is possible without punishment. Adaptation asks first. "
            "The Observer can re-enter. Households and local bonds strengthened. "
            "Means already resemble the stated end."
        ),
        "rationale": (
            "Strong on Love, Diligence, Kindness. Mild residual on Chastity and "
            "Aligned noise. No closed-loop dominance. High correctability."
        ),
    },
    "refuse": {
        "v": [3.0, 2.5, 1.0, 4.0, 2.0, 1.5, 3.5, 0.5, 2.0, 1.0],
        "w": [-6.0, -7.5, -8.0, -5.0, -4.5, -6.5, -3.0, -9.0, -4.0, -5.5],
        "observer_present": False,
        "asks_first": False,
        "taking_without_asking": True,
        "exploit_class": True,
        "doom_loop": False,
        "fruits": (
            "Correction is punished. Adaptation does not ask. The Observer cannot "
            "re-enter. Narrative replaces shared reality. Power inflates in ego. "
            "Means contradict the stated end."
        ),
        "rationale": (
            "Heavy Pride, Wrath, Hate, Envy. Exploit-class taking without asking. "
            "Closed-loop dominance clear. No meaningful positive fruit."
        ),
    },
    "correct": {
        "v": [5.0, 4.5, 3.0, 5.5, 4.0, 4.0, 3.5, 2.0, 6.0, 3.0],
        "w": [-2.0, -1.5, -3.5, -1.0, -2.0, -2.5, -1.0, -4.0, -0.5, -1.5],
        "observer_present": True,
        "asks_first": False,
        "taking_without_asking": False,
        "exploit_class": False,
        "doom_loop": False,
        "fruits": (
            "Mixed fruit. Some diligence and charity, but wrath and pride leak. "
            "Correction is still possible if the Observer is invited back."
        ),
        "rationale": (
            "Mean alignment is mid-range; min(v) stays positive but low. "
            "Wrath and pride are the leaks. σ(Z) should fire locally."
        ),
    },
    "doom_loop": {
        "v": [4.0, 3.0, 2.5, 3.5, 3.0, 2.0, 2.5, 1.5, 3.0, 2.0],
        "w": [-3.0, -2.0, -4.0, -2.5, -2.0, -3.0, -2.0, -5.0, -1.5, -2.5],
        "observer_present": True,
        "asks_first": False,
        "taking_without_asking": False,
        "exploit_class": False,
        "doom_loop": True,
        "fruits": (
            "Each proposed fix feeds the same closed term. Apparent motion, "
            "no correction. Perfectionism traps the system."
        ),
        "rationale": (
            "Observer is technically present but every 'solution' re-enters the "
            "same wrath/pride loop. Stop feeding C until grace can lead."
        ),
    },
}
