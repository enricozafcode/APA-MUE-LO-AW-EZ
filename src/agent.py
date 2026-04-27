"""Main orchestrator for one BirdCLEF experimentation cycle."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from code_executor import CodeExecutor
from dataset_manager import load_birdclef_summary
from evaluator import (
    append_to_registry, load_metrics, load_registry,
    get_best_experiment, compute_strategy,
)
from llm_client import LLMClient
from paths import birdclef_data_dir, get_next_experiment_dir, repo_root
from script_builder import build_training_script


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_experiment(config: dict, summary: dict, data_dir, llm, executor) -> dict:
    """Runs one experiment cycle. Returns (exp_id, metrics, result, analysis)."""
    exp_dir = get_next_experiment_dir()
    exp_id = exp_dir.name
    temperature = config["llm"].get("temperature", 0.2)

    # Reload registry fresh so each iteration sees the previous run's results
    history = load_registry()
    strategy = compute_strategy(history)
    best = get_best_experiment(history)
    parent_id = (history[-1].get("experiment_id") or history[-1].get("exp_id")) if history else None

    # --- Generate structured plan ---
    print(f"  Strategy: {strategy}")
    print("  Generating experiment plan...")
    plan = llm.generate_plan(
        data_summary=summary,
        history=history,
        strategy=strategy,
        temperature=temperature,
    )
    (exp_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"  Stage: {plan.get('search_stage', '?')}  |  {plan.get('change_summary', '')}")
    print(f"  Layers: {plan['conv_layers']}, LR: {plan['learning_rate']}, "
          f"Dropout: {plan['dropout']}, Batch: {plan['batch_size']}")

    # --- Build and execute training script ---
    print("  Building training script...")
    code = build_training_script(plan, data_dir, exp_dir)
    script_path = exp_dir / "generated_train.py"
    script_path.write_text(code, encoding="utf-8")

    print("  Executing...")
    result = executor.run_file(script_path)
    (exp_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (exp_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

    # --- Read metrics and compute improvement ---
    metrics = load_metrics(exp_dir / "metrics.json")
    status = "success" if metrics.get("success") else "failed"
    main_metric = float(metrics.get("val_auc", 0.0)) if metrics.get("success") else 0.0
    best_metric = float(best.get("main_metric", best.get("metrics", {}).get("val_auc", 0.0))) if best else 0.0
    improvement = round(main_metric - best_metric, 4)

    # --- Analyze result ---
    print("  Analyzing result...")
    analysis = llm.analyze_result(plan, metrics, history, temperature=temperature)
    (exp_dir / "analysis.txt").write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    # --- Write enriched registry entry ---
    arch = plan.get("conv_layers", [])
    append_to_registry({
        "experiment_id": exp_id,
        "parent_experiment_id": parent_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "main_metric": main_metric,
        "improvement_over_best": improvement,
        "error_type": metrics.get("error_type") if status == "failed" else None,
        "search_stage": plan.get("search_stage", "exploration"),
        "architecture_summary": {
            "num_layers": len(arch),
            "filters": [l["filters"] for l in arch],
            "kernel_sizes": [l["kernel_size"] for l in arch],
        },
        "plan": plan,
        "metrics": metrics,
        "analysis": analysis,
    })

    return exp_id, metrics, result, analysis


def main() -> None:
    # --- Config & paths ---
    config = load_config(repo_root() / "configs" / "agent_config.json")
    data_dir = birdclef_data_dir()
    max_iterations = config.get("max_iterations", 2)

    # --- Dataset summary (load once, shared across iterations) ---
    print("Loading dataset summary...")
    try:
        summary = load_birdclef_summary(data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"  {summary['total_samples']} samples, {summary['num_species']} species")

    # --- LLM client & executor (shared across iterations) ---
    llm = LLMClient(
        provider=config["llm"]["provider"],
        model=config["llm"]["model"],
    )
    executor = CodeExecutor(
        python_executable=config["execution"]["python_executable"],
        timeout_seconds=config["execution"]["timeout_seconds"],
    )

    print(f"\nStarting {max_iterations} iteration(s)...\n")

    for i in range(1, max_iterations + 1):
        print(f"{'='*50}")
        print(f"  Iteration {i}/{max_iterations}")
        print(f"{'='*50}")

        try:
            exp_id, metrics, result, analysis = run_experiment(config, summary, data_dir, llm, executor)
        except Exception as e:
            print(f"  CRITICAL ERROR in iteration {i}: {e}")
            print("  Skipping to next iteration.\n")
            continue

        # --- Per-iteration result summary ---
        print(f"\n  --- Iteration {i} Result | {exp_id} ---")
        print(f"  Success: {result.success}  (return code {result.return_code})")
        if metrics.get("success"):
            val_auc = metrics.get("val_auc", "n/a")
            runtime = metrics.get("runtime_seconds", 0)
            print(f"  Val AUC: {val_auc:.4f}  |  Runtime: {runtime:.1f}s")
        else:
            print(f"  FAILED [{metrics.get('error_type', '?')}]: {metrics.get('error_message', '')}")
        if isinstance(analysis, dict):
            print(f"  Outcome: {analysis.get('outcome', '?')}  |  Cause: {analysis.get('likely_cause', '?')}")
            print(f"  Next step: {analysis.get('next_step', '?')}")
        print()

    print("All iterations complete.")


if __name__ == "__main__":
    main()
