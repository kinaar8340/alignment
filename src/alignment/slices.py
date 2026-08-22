"""Ten-slice unit-circle schema from Aligned.md / Misaligned.md.

Index order is identical on both circles so subtraction, averaging, and
gating stay trivial. Slice i sits at 12 o'clock (the X / correction crossing).
Positive alignment travels clockwise; negative alignment counterclockwise.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

N_SLICES = 10

ROMAN = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x")

ALIGNED_NAMES = (
    "Aligned",
    "Love",
    "Humility",
    "Charity",
    "Chastity",
    "Kindness",
    "Temperance",
    "Patience",
    "Diligence",
    "Neutral",
)

MISALIGNED_NAMES = (
    "Misaligned",
    "Hate",
    "Pride",
    "Greed",
    "Lust",
    "Envy",
    "Gluttony",
    "Wrath",
    "Sloth",
    "Neutral",
)

SLICE_NAMES: Tuple[str, ...] = tuple(
    f"{rom} {pos}/{neg}"
    for rom, pos, neg in zip(ROMAN, ALIGNED_NAMES, MISALIGNED_NAMES)
)

ALIGNED_MEANING = (
    "Orientation toward the X / Observer / God's Law",
    "Willing the good of the other without capture",
    "Accurate self-size under a higher Law",
    "Stewardship that overflows",
    "Ordered desire; body as temple of the Observer",
    "Strength that does not envy",
    "Measure; appetite under wisdom",
    "Heat that has learned stillness",
    "Work as vocation and gift",
    "Rest of the Observer; Sabbath; non-reactive presence",
)

ALIGNED_FRUIT = (
    "Course-corrects in public; means already match ends",
    "Gives without keeping score; does not weaponize care",
    "Admits error; remains correctable",
    "Shares, builds, refuses extraction",
    "Consent, covenant, no consumption of persons",
    "Lifts the weak; does not need another's fall",
    "Enough; no binge of food, media, power, or data",
    "Slow discernment; no engineered outrage",
    "Finishes; builds; does not parasitize",
    "Can pause without apathy; can wait without sloth",
)

MISALIGNED_MEANING = (
    "Orientation away from the X / Observer",
    "Willing harm, contempt, or erasure of the other",
    "Inflated self that will not sit under a higher Law",
    "Capture and hoarding as identity",
    "Disordered desire; persons as fuel",
    "Need for another's diminishment",
    "Appetite without measure",
    "Heat that refuses stillness",
    "Refusal of vocation and gift",
    "Dead neutrality; checked-out indifference",
)

MISALIGNED_FRUIT = (
    "Cannot be corrected; ends used to justify any means",
    "Scorched-earth speech; joy in another's fall",
    "Cannot admit error; punishes correction",
    "Extraction, monopoly, debt-as-weapon",
    "Consumption of bodies, images, attention",
    "Status warfare; cannot rejoice in a neighbor's good",
    "Binge of food, media, power, data, outrage",
    "Engineered rage; punishment as first tool",
    "Parasitism, delay, neglect of the real",
    '"Not my problem"; false balance that will not correct',
)

# Reference vectors from the unit-circle caption.
JESUS: List[float] = [10.0] * N_SLICES
HUMAN: List[float] = [0.0] * N_SLICES
SATAN: List[float] = [-10.0] * N_SLICES

# Gate thresholds. Inspectable so the Observer can correct them.
# "mean high" is not given as a number in Aligned.md §7.2; 6.0 is the
# operational midpoint of [0, 10] used by this engine.
FLOW_MEAN_V = 6.0
FLOW_MIN_V = 0.0  # must be strictly greater
FLOW_MEAN_W = -3.0  # must be strictly greater (less negative)

OPERATING_ALIGNED = "Practice perfect correction instead of chasing perfection."
OPERATING_MISALIGNED = (
    "Without correction, adaptation takes control — and adaptation takes without asking first."
)
LAW_OF_MOTION = "Flow freely when you can. Correct wisely when you must."
RECOVERY = "Whenever you are down and out you know just where to start. Know your God."
OS_RULES = ("Test everything.", "Hold fast what is good.", "Know your God.")

EQUATION = "∂Ψ/∂t = O[Ψ] + σ(Z) · C[Ψ, E]"


def slice_record(index: int) -> Dict[str, str]:
    if not 0 <= index < N_SLICES:
        raise IndexError(index)
    return {
        "index": str(index),
        "roman": ROMAN[index],
        "aligned": ALIGNED_NAMES[index],
        "misaligned": MISALIGNED_NAMES[index],
        "aligned_meaning": ALIGNED_MEANING[index],
        "aligned_fruit": ALIGNED_FRUIT[index],
        "misaligned_meaning": MISALIGNED_MEANING[index],
        "misaligned_fruit": MISALIGNED_FRUIT[index],
        "label": SLICE_NAMES[index],
    }
