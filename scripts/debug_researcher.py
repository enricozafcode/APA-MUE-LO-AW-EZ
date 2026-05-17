#!/usr/bin/env python3
"""One-shot Perch researcher call — same prompts as step 1a, with optional live streaming."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_client import LLMClient, llm_response_failed
from perch_agent import (
    PERCH_RESEARCHER_SYSTEM_PROMPT,
    PERCH_SEARCH_SPACE,
    PerchResearcher,
    _resolve_researcher_timeout_seconds,
)
from perch_memory import PerchExperimentMemory


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Perch researcher LLM call")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "agent_config.json",
    )
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=ROOT / "logs" / "meta_agent" / "perch" / "light",
        help="Perch experiment memory directory (aug baseline subfolder)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens live (also enabled by researcher.stream_debug in config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt sizes only, do not call Ollama",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    rc = config.get("researcher") or {}
    llm_rc = config.get("llm_researcher") or {}
    provider = llm_rc.get("provider") or config.get("llm", {}).get("provider", "ollama")
    model = rc.get("model") or llm_rc.get("model", "deepseek-r1:8b")
    temp = float(rc.get("temperature", 0.6))
    timeout = _resolve_researcher_timeout_seconds(config)
    stream_debug = bool(args.stream or rc.get("stream_debug") or llm_rc.get("stream_debug"))

    mem_dir = args.memory_dir
    if not mem_dir.is_dir():
        print(f"Memory dir not found: {mem_dir}", file=sys.stderr)
        sys.exit(1)

    ranking = (config.get("meta_agent") or {}).get("primary_metric", "macro_average_precision")
    memory = PerchExperimentMemory(mem_dir, ranking_metric=ranking)
    batch = max(1, int(rc.get("batch_size") or config.get("perch", {}).get("experiments_per_researcher_call", 1)))
    researcher = PerchResearcher(
        LLMClient(
            provider=provider,
            model=model,
            timeout_seconds=timeout,
            stream_debug=stream_debug,
        ),
        memory,
        temperature=temp,
        batch_size=batch,
    )

    history = memory.researcher_context()
    user_preview = (
        f"{history}\n\n[search space + instructions — same as perch_agent.next_experiment]"
    )
    print(f"Model          : {model}")
    print(f"Provider       : {provider}")
    print(f"Timeout        : {timeout:.0f}s")
    print(f"Stream debug   : {stream_debug}")
    print(f"Memory runs    : {memory.total()} @ {mem_dir}")
    print(f"System chars   : {len(PERCH_RESEARCHER_SYSTEM_PROMPT)}")
    print(f"User chars     : {len(user_preview)} (+ {len(json.dumps(PERCH_SEARCH_SPACE))} search space)")
    print("-" * 60)
    print(history[:1200])
    if len(history) > 1200:
        print("… [truncated]")
    print("-" * 60)

    if args.dry_run:
        return

    t0 = time.time()
    if batch > 1:
        specs = researcher.next_experiments()
        print(f"\nElapsed: {time.time() - t0:.1f}s — {len(specs)} experiments")
        print(json.dumps(specs, indent=2))
    else:
        spec = researcher.next_experiment()
        print(f"\nElapsed: {time.time() - t0:.1f}s")
        print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
