#!/usr/bin/env python3
"""
Run Phase 3 (final full-data training) only, using saved best hyperparameters.

Typical sources (auto-detected in order):
  - logs/search_comparison.json → final_params
  - submission/best_params.json

Examples:
  python scripts/run_phase3_final.py
  python scripts/run_phase3_final.py --best-params logs/my_winner.json
  python scripts/run_phase3_final.py --config configs/agent_config.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent Phase 3 final training only.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to agent_config.json (default: <root>/configs/agent_config.json).",
    )
    parser.add_argument(
        "--best-params",
        type=Path,
        dest="best_params",
        default=None,
        help="JSON file with hyperparameters (object of keys, or {\"params\": {...}}).",
    )
    args = parser.parse_args()

    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from src.agent import run_phase3_final_only

    run_phase3_final_only(
        project_root=root,
        config_path=args.config,
        best_params_path=args.best_params,
    )


if __name__ == "__main__":
    main()
