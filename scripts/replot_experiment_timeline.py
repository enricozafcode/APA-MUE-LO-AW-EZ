#!/usr/bin/env python3
"""Rebuild experiment progress plots from logs/meta_agent/experiment_timeline.json."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_tracker import DEFAULT_TIMELINE, replot_from_timeline  # noqa: E402


def main() -> None:
    path = replot_from_timeline(DEFAULT_TIMELINE)
    if path is None:
        print(f"No timeline found at {DEFAULT_TIMELINE}")
        raise SystemExit(1)
    print(f"Plots rebuilt from {path}")


if __name__ == "__main__":
    main()
