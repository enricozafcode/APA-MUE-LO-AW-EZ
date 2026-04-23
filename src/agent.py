"""Main orchestrator for one BirdCLEF experimentation cycle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from code_executor import CodeExecutor
from dataset_manager import load_birdclef_summary
from evaluator import load_metrics, summarize_metrics
from llm_client import LLMClient
from paths import birdclef_data_dir, get_next_experiment_dir, repo_root


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    # --- 1. Config & paths ---
    config = load_config(repo_root() / "configs" / "agent_config.json")
    exp_dir = get_next_experiment_dir()
    data_dir = birdclef_data_dir()
    print(f"Experiment directory: {exp_dir}")

    # --- 2. Dataset summary ---
    print("Loading dataset summary...")
    try:
        summary = load_birdclef_summary(data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    (exp_dir / "data_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"  {summary['total_samples']} samples, {summary['num_species']} species")

    # --- 3. LLM client ---
    llm = LLMClient(
        provider=config["llm"]["provider"],
        model=config["llm"]["model"],
    )
    temperature = config["llm"].get("temperature", 0.2)

    # --- 4. Generate experiment plan ---
    print("Generating experiment plan...")
    plan = llm.generate_plan(
        data_summary=summary,
        instructions=(
            "Propose a minimal BirdCLEF baseline experiment. "
            "Use mel-spectrograms, a small CNN, sigmoid output with binary cross-entropy. "
            "Train on a tiny subset (max 200 samples, max 3 epochs). Keep it simple and fast."
        ),
        temperature=temperature,
    )
    if not plan:
        print("ERROR: LLM returned an empty plan.")
        sys.exit(1)
    (exp_dir / "plan.txt").write_text(plan, encoding="utf-8")
    print("  Plan saved.")

    # --- 5. Generate training code ---
    print("Generating training code...")
    code = llm.generate_code(
        plan=plan,
        data_dir=data_dir,
        exp_dir=exp_dir,
        temperature=temperature,
    )
    if not code:
        print("ERROR: LLM returned empty code.")
        sys.exit(1)
    script_path = exp_dir / "generated_train.py"
    script_path.write_text(code, encoding="utf-8")
    print(f"  Script saved: {script_path}")

    # --- 6. Execute generated code ---
    print("\n--- EXECUTING GENERATED CODE ---")
    executor = CodeExecutor(
        python_executable=config["execution"]["python_executable"],
        timeout_seconds=config["execution"]["timeout_seconds"],
    )
    result = executor.run_file(script_path)
    (exp_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (exp_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")

    # --- 7. Evaluate ---
    metrics = load_metrics(exp_dir / "metrics.json")

    # --- 8. Print summary ---
    print("\n--- EXPERIMENT RESULTS ---")
    print(f"Execution success: {result.success}  (return code {result.return_code})")
    print(summarize_metrics(metrics))
    print(f"\nFull experiment saved to: {exp_dir}")


if __name__ == "__main__":
    main()