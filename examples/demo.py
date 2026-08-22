#!/usr/bin/env python3
"""Render sample dashboards and Lorenz / motion figures into outputs/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alignment.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", "-o", str(ROOT / "outputs")]))
